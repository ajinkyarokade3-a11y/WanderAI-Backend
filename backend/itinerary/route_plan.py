"""Generalized, destination-agnostic route planning for day-wise itineraries.

Pure coordinate-driven logic: no per-region, per-state, per-country, or
per-name rules exist anywhere in this module. Decisions derive only from
stored coordinates, estimated travel, activity durations, day counts, and
the configuration below. Named places appear solely in labels copied from
caller-supplied data, never in logic.

Pipeline position (acyclic, single pass)::

    activity selection
        -> proximity day assignment (this module)
        -> overnight anchor calculation (generator, from placed days)
        -> hotel assignment (hotel_assignment, unchanged behavior)
        -> route-leg evaluation (this module, read-only over final items)

Day assignment never depends on hotel selection, so no circular logic and
guaranteed termination. Anything this module cannot verify is reported
with an explicit warning or ``unable_to_verify`` status — activities are
never dropped, coordinates never invented, and estimates never labeled
verified (no routing provider exists).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.itinerary.hotel_assignment import (
    HOTEL_ASSIGNMENT_CONFIG,
    centroid,
    estimate_travel_minutes,
    haversine_km,
)

ROUTE_PLAN_CONFIG: Dict[str, Any] = {
    # Longest acceptable single hop between consecutive stops (minutes,
    # estimated). Breaches keep the activity with a warning, never drop it.
    "max_one_way_travel_minutes": 120.0,
    # Daily estimated-travel budget per sightseeing day (minutes).
    "max_daily_travel_minutes": 240.0,
    # Buffer added per activity when totaling a day's travel load.
    "activity_travel_buffer_minutes": 30.0,
    # Hard cap on sightseeing activities per day (pace profiles override).
    "max_activities_per_day": 4,
    # Earliest usable sightseeing start, minutes since midnight (09:00).
    # Day 1 keeps a later floor (arrival + check-in); see below.
    "schedule_day_start_minutes": 540,
    # Day-1 floor preserves the historical opener: transport plus hotel
    # check-in occupy the morning, so sightseeing starts mid-afternoon.
    "schedule_arrival_day_start_minutes": 990,  # 04:30 PM
    # Latest permissible activity end; overruns spill to another day.
    "schedule_day_end_minutes": 1350,  # 10:30 PM
    # Fallback road speed, estimate-only (shared with the hotel engine so
    # both systems assume the same speed; never presented as measured data).
    "default_travel_speed_kmh": HOTEL_ASSIGNMENT_CONFIG["assumed_speed_kmh"],
    "pace_profiles": {
        "relaxed": {"max_activities_per_day": 2, "max_daily_travel_minutes": 150.0},
        "balanced": {"max_activities_per_day": 4, "max_daily_travel_minutes": 240.0},
        "packed": {"max_activities_per_day": 6, "max_daily_travel_minutes": 330.0},
    },
}

ESTIMATION_METHOD = "haversine_speed_estimate"

# Validation statuses for day routes. Assigned honestly from implemented
# checks only; operating-hours validation does not exist and is never
# claimed.
STATUS_VALID = "Valid"
STATUS_WARNING = "Valid-with-Warning"
STATUS_REVISION = "Requires-Revision"
STATUS_UNVERIFIED = "Unable-to-Verify"


@dataclass
class RouteEndpoint:
    """One end of a travel leg (names copied, never interpreted)."""

    item_id: str
    item_type: str
    name: str
    coordinates_available: bool


@dataclass
class RouteLeg:
    """One estimated hop between consecutive stops on the same day."""

    origin: RouteEndpoint
    destination: RouteEndpoint
    distance_km: Optional[float] = None
    estimated_duration_minutes: Optional[float] = None
    estimation_method: Optional[str] = None
    is_estimated: bool = True
    status: str = "unable_to_verify"
    coordinates_available: bool = False
    uncertainty_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": {
                "item_id": self.origin.item_id,
                "item_type": self.origin.item_type,
                "name": self.origin.name,
                "coordinates_available": self.origin.coordinates_available,
            },
            "destination": {
                "item_id": self.destination.item_id,
                "item_type": self.destination.item_type,
                "name": self.destination.name,
                "coordinates_available": self.destination.coordinates_available,
            },
            "distance_km": self.distance_km,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "estimation_method": self.estimation_method,
            "is_estimated": self.is_estimated,
            "status": self.status,
            "coordinates_available": self.coordinates_available,
            "uncertainty_reason": self.uncertainty_reason,
        }


@dataclass
class DayRoute:
    """One sightseeing day: ordered activities, legs, totals, warnings."""

    day: int
    activity_ids: List[str] = field(default_factory=list)
    legs: List[RouteLeg] = field(default_factory=list)
    daily_travel_minutes: Optional[float] = None
    max_one_way_minutes: Optional[float] = None
    activity_minutes: float = 0.0
    warnings: List[str] = field(default_factory=list)
    status: str = STATUS_VALID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "activity_ids": list(self.activity_ids),
            "legs": [leg.to_dict() for leg in self.legs],
            "daily_travel_minutes": self.daily_travel_minutes,
            "max_one_way_minutes": self.max_one_way_minutes,
            "activity_minutes": self.activity_minutes,
            "warnings": list(self.warnings),
            "status": self.status,
        }


@dataclass
class RoutePlan:
    """Full assignment: day per activity plus per-day route detail."""

    activity_day: Dict[str, int] = field(default_factory=dict)
    days: Dict[int, DayRoute] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def resolve_pace(pace: Any, config: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Merge the pace profile over the defaults (unknown pace -> balanced)."""
    cfg = dict(ROUTE_PLAN_CONFIG)
    if config:
        cfg.update(config)
    profiles = cfg.get("pace_profiles") or {}
    profile = profiles.get(str(pace or "").casefold(), None)
    if not isinstance(profile, Mapping):
        profile = profiles.get("balanced", {})
    merged = {
        "max_activities_per_day": cfg.get("max_activities_per_day", 4),
        "max_daily_travel_minutes": cfg.get("max_daily_travel_minutes", 240.0),
    }
    for key, value in (profile or {}).items():
        if key in merged:
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue
    return merged


def _valid_point(value: Any) -> Optional[Tuple[float, float]]:
    try:
        lat = float(value[0])
        lng = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    if math.isnan(lat) or math.isnan(lng):
        return None
    return lat, lng


def _activity_point(activity: Any) -> Optional[Tuple[float, float]]:
    if isinstance(activity, Mapping):
        return _valid_point(
            (activity.get("latitude"), activity.get("longitude"))
        )
    return _valid_point(
        (getattr(activity, "latitude", None), getattr(activity, "longitude", None))
    )


def _activity_id(activity: Any, fallback: str) -> str:
    if isinstance(activity, Mapping):
        value = activity.get("id", fallback)
    else:
        value = getattr(activity, "id", fallback)
    return str(value)


def _activity_minutes(activity: Any) -> float:
    if isinstance(activity, Mapping):
        raw = activity.get("duration_hours", 0)
    else:
        raw = getattr(activity, "duration_hours", 0)
    try:
        return max(0.0, float(raw or 0)) * 60.0
    except (TypeError, ValueError):
        return 0.0


def _speed(cfg: Mapping[str, Any]) -> float:
    try:
        speed = float(cfg.get("default_travel_speed_kmh", 50.0))
    except (TypeError, ValueError):
        speed = 50.0
    return speed if speed > 0 else 50.0


def build_leg(
    origin: RouteEndpoint,
    destination: RouteEndpoint,
    origin_point: Optional[Tuple[float, float]],
    destination_point: Optional[Tuple[float, float]],
    speed_kmh: float,
) -> RouteLeg:
    """One leg between two endpoints.

    Coordinates present -> ``estimated`` (even when the distance is
    genuinely zero); anything missing -> ``unable_to_verify`` with a
    reason. Zero is never used as a substitute for missing data: absent
    inputs yield ``None`` values, not ``0.0``.
    """
    if origin_point is None or destination_point is None:
        missing = []
        if origin_point is None:
            missing.append("origin")
        if destination_point is None:
            missing.append("destination")
        return RouteLeg(
            origin=origin,
            destination=destination,
            estimation_method=None,
            is_estimated=True,
            status="unable_to_verify",
            coordinates_available=False,
            uncertainty_reason=(
                "Missing origin or destination coordinates: "
                + " and ".join(missing)
            ),
        )
    distance = haversine_km(
        origin_point[0], origin_point[1], destination_point[0], destination_point[1]
    )
    if distance is None:  # pragma: no cover - guarded by _valid_point
        return RouteLeg(
            origin=origin,
            destination=destination,
            estimation_method=None,
            is_estimated=True,
            status="unable_to_verify",
            coordinates_available=False,
            uncertainty_reason="Coordinates failed validation",
        )
    minutes = estimate_travel_minutes(distance, speed_kmh)
    return RouteLeg(
        origin=origin,
        destination=destination,
        distance_km=round(distance, 1),
        estimated_duration_minutes=round(minutes, 1) if minutes is not None else None,
        estimation_method=ESTIMATION_METHOD,
        is_estimated=True,
        status="estimated",
        coordinates_available=True,
        uncertainty_reason=None,
    )


def summarize_legs(legs: Sequence[RouteLeg]) -> Tuple[Optional[float], Optional[float]]:
    """(total estimated minutes, max one-way minutes); None when unverifiable."""
    estimated = [
        leg.estimated_duration_minutes
        for leg in legs
        if leg.status == "estimated" and leg.estimated_duration_minutes is not None
    ]
    if not estimated:
        return None, None
    return round(sum(estimated), 1), round(max(estimated), 1)


def format_clock_time(minutes: Any) -> Optional[str]:
    """Format minutes-since-midnight like "04:30 PM" (matches _end_time style).

    Returns None for invalid or out-of-day input instead of wrapping around
    midnight, so unrepresentable times can never silently become morning.
    """
    try:
        total = int(round(float(minutes)))
    except (TypeError, ValueError):
        return None
    if total < 0 or total >= 24 * 60:
        return None
    hour, minute = divmod(total, 60)
    period = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display:02d}:{minute:02d} {period}"


def _merge_blockers(blockers: Sequence[Tuple[Any, Any]]) -> List[Tuple[int, int]]:
    """Normalize fixed commitments to a sorted, merged [(start, end)] list.

    Unparseable or inverted entries are skipped, never invented.
    """
    if isinstance(blockers, Mapping):
        pairs = list(blockers.items())
        spans = [(s, e) for _, (s, e) in pairs]
    else:
        spans = list(blockers or [])
    cleaned: List[Tuple[int, int]] = []
    for span in spans:
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= start < end <= 24 * 60:
            cleaned.append((start, end))
    cleaned.sort()
    merged: List[List[int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def schedule_day_stops(
    stops: Sequence[Tuple[str, Optional[Tuple[float, float]], Any]],
    slot_floors: Sequence[int],
    blockers: Sequence[Tuple[int, int]],
    *,
    day_start_floor: int,
    day_end: int,
    speed_kmh: float,
    buffer_min: float,
) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    """Schedule one day's stops sequentially without overlaps.

    ``stops`` is [(aid, point|None, duration_hours)] in desired display
    order; ``slot_floors`` holds legacy template earliest-starts per
    position. Durations follow the same rule as the generator's
    ``_end_time`` (at least 60 minutes). Each stop starts at ``max(template
    floor, previous end + travel + buffer)``, pushed past any fixed
    ``blockers``. A stop whose end would pass ``day_end`` (or midnight)
    spills for reassignment instead of overlapping. Returns ([(aid,
    start_min, end_min)], [spilled aids]). Pure and deterministic;
    nothing is dropped.
    """
    try:
        floor_default = int(day_start_floor)
    except (TypeError, ValueError):
        floor_default = 540
    try:
        ceiling = int(day_end)
    except (TypeError, ValueError):
        ceiling = 1350
    try:
        pace_buffer = max(0.0, float(buffer_min))
    except (TypeError, ValueError):
        pace_buffer = 0.0
    fixed = _merge_blockers(blockers)
    scheduled: List[Tuple[str, int, int]] = []
    spilled: List[str] = []
    cursor = floor_default
    prev_point: Optional[Tuple[float, float]] = None
    for position, (aid, point, duration_hours) in enumerate(stops):
        try:
            floor = int(slot_floors[position]) if position < len(slot_floors) else floor_default
        except (TypeError, ValueError, IndexError):
            floor = floor_default
        try:
            need = max(60, int(round(float(duration_hours or 0) * 60)))
        except (TypeError, ValueError):
            need = 60
        earliest = max(floor, cursor)
        if prev_point is not None:
            # Transition padding always applies; the distance part only
            # when both endpoints are known (never invented).
            travel = 0.0
            if point is not None:
                leg_min = estimate_travel_minutes(
                    haversine_km(prev_point[0], prev_point[1], point[0], point[1]) or 0.0,
                    speed_kmh,
                )
                if leg_min is not None:
                    travel = leg_min
            earliest = max(earliest, int(round(cursor + travel + pace_buffer)))
        start = earliest
        for block_start, block_end in fixed:
            if start < block_end and block_start < start + need:
                start = block_end
        end = start + need
        if end > ceiling or end >= 24 * 60:
            spilled.append(aid)
            continue
        scheduled.append((aid, start, end))
        cursor = end
        prev_point = point
    return scheduled, spilled


def plan_activity_days(
    activities: Sequence[Any],
    days: int,
    pace: Any = None,
    config: Optional[Mapping[str, Any]] = None,
) -> RoutePlan:
    """Assign activities to days by geographic proximity (deterministic).

    Ranked input order is significant: it breaks every tie, so identical
    inputs always produce identical assignments. Locatable activities are
    inserted greedily where they add the least travel, subject to per-day
    sightseeing time windows (arrival day is shorter), a per-day activity
   Minutes target that spreads load across all usable days, and estimated
    travel budgets. Unlocatable ones keep legacy round-robin slots with
    explicit warnings. Nothing is dropped: overflow beyond every feasible
    day keeps a slot and is flagged ``Requires-Revision``.
    """
    cfg = dict(ROUTE_PLAN_CONFIG)
    if config:
        cfg.update(config)
    profile = resolve_pace(pace, cfg)
    span = max(1, int(days or 1))
    speed = _speed(cfg)
    max_per_day = max(1, int(profile["max_activities_per_day"]))
    max_daily = float(profile["max_daily_travel_minutes"])
    max_one_way = float(cfg.get("max_one_way_travel_minutes", 120.0))
    buffer_min = float(cfg.get("activity_travel_buffer_minutes", 30.0))
    try:
        window_start = int(float(cfg.get("schedule_day_start_minutes", 540)))
        arrival_floor = int(float(cfg.get("schedule_arrival_day_start_minutes", 990)))
        window_end = int(float(cfg.get("schedule_day_end_minutes", 1350)))
    except (TypeError, ValueError):
        window_start, arrival_floor, window_end = 540, 990, 1350

    def _day_window(day: int) -> float:
        floor = arrival_floor if day == 1 else window_start
        return max(0.0, float(window_end - floor))

    indexed = list(activities or [])
    points: Dict[str, Optional[Tuple[float, float]]] = {}
    order_index: Dict[str, int] = {}
    for position, activity in enumerate(indexed):
        aid = _activity_id(activity, f"activity-{position}")
        order_index.setdefault(aid, position)
        points.setdefault(aid, _activity_point(activity))

    locatable = [aid for aid in order_index if points[aid] is not None]
    unlocatable = [aid for aid in order_index if points[aid] is None]

    members: Dict[int, List[str]] = {day: [] for day in range(1, span + 1)}
    day_warnings: Dict[int, List[str]] = {day: [] for day in range(1, span + 1)}
    overflow: List[str] = []
    opener_excess: Dict[str, float] = {}
    # Reference stop for each opener warning (activity id of the nearest
    # already-placed stop), so the warning names a real origin instead of
    # reading like a travel time for the final validated legs.
    opener_ref: Dict[str, str] = {}

    if not locatable:
        # No coordinates anywhere: legacy round-robin, byte-identical
        # day mapping to the previous distributor.
        aids = [aid for aid in order_index]
        for position, aid in enumerate(aids):
            members[(position % span) + 1].append(aid)
    else:
        # Load target: total sightseeing minutes spread evenly. Days
        # already at/above target yield to emptier days so trips use all
        # usable days instead of clustering early; days below target keep
        # a clustering bias (join a started day over opening a new one).
        total_minutes = sum(
            _activity_minutes_by_id(a, indexed, order_index) for a in locatable
        )
        target_load = total_minutes / span if span else total_minutes
        day_load: Dict[int, float] = {day: 0.0 for day in range(1, span + 1)}
        placed_points: Dict[str, Tuple[float, float]] = {}
        ranked_locatable = sorted(locatable, key=lambda a: order_index[a])
        for aid in ranked_locatable:
            point = points[aid]
            assert point is not None  # locatable by construction
            duration_min = _activity_minutes_by_id(aid, indexed, order_index)
            # Remoteness proxy: distance to the closest already-placed
            # stop. Genuinely far-flung activities are flagged even when
            # they open their own day (no hotel location exists yet to
            # compare against, so this is the only honest reference).
            nearest_placed_km: Optional[float] = None
            nearest_placed_aid: Optional[str] = None
            for placed_aid, placed in placed_points.items():
                dist = haversine_km(point[0], point[1], placed[0], placed[1])
                if dist is not None and (
                    nearest_placed_km is None or dist < nearest_placed_km
                ):
                    nearest_placed_km = dist
                    nearest_placed_aid = placed_aid
            opener_min = estimate_travel_minutes(nearest_placed_km, speed) \
                if nearest_placed_km is not None else 0.0
            best_day: Optional[int] = None
            best_key: Optional[Tuple[float, float, float, float, int]] = None
            for day in range(1, span + 1):
                current = members[day]
                if len(current) >= max_per_day:
                    continue
                window = _day_window(day)
                if not current:
                    # A lone activity must fit the day's usable window;
                    # travel is zero for a single-stop day.
                    if duration_min > window:
                        continue
                    incremental_min = opener_min or 0.0
                    load_after = duration_min
                    time_after = duration_min
                    occupied = False
                else:
                    nearest_km = min(
                        haversine_km(point[0], point[1], points[m][0], points[m][1])
                        for m in current
                    )
                    if nearest_km is None:  # pragma: no cover - validated
                        continue
                    nearest_min = estimate_travel_minutes(nearest_km, speed)
                    if nearest_min is None or nearest_min > max_one_way:
                        continue
                    day_travel = _day_travel_estimate(
                        [points[m] for m in current] + [point], speed, buffer_min
                    )
                    if day_travel is not None and day_travel > max_daily:
                        continue
                    load_after = day_load[day] + duration_min
                    time_after = load_after + (day_travel or 0.0)
                    if time_after > window + 1e-6:
                        continue
                    incremental_min = nearest_min
                    occupied = True
                # Prefer days below the load target (spread), then started
                # days (clustering), then least extra travel, lighter
                # resulting load, and lower day numbers. Deterministic.
                over_target = 0.0 if day_load[day] < target_load else 1.0
                key = (
                    over_target,
                    0.0 if occupied else 1.0,
                    round(incremental_min, 3),
                    round(load_after, 3),
                    day,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_day = day
            if best_day is None:
                overflow.append(aid)
                continue
            if not members[best_day]:
                if opener_min is not None and opener_min > max_one_way:
                    opener_excess[aid] = opener_min
                    if nearest_placed_aid is not None:
                        opener_ref[aid] = nearest_placed_aid
            members[best_day].append(aid)
            day_load[best_day] += duration_min
            placed_points[aid] = point
        # Overflow first: locatable but infeasible on every budgeted day.
        # Repair geographically instead of round-robin mingling: an empty
        # day with a fitting window avoids inter-stop legs entirely;
        # otherwise join the nearest day (still flagged Requires-Revision
        # via overflow_set — proximity minimizes the damage, never hides
        # it). Deterministic; nothing is dropped.
        for aid in overflow:
            point = points[aid]
            duration_min = _activity_minutes_by_id(aid, indexed, order_index)
            best_day: Optional[int] = None
            best_key: Optional[Tuple[float, float, int]] = None
            for day in range(1, span + 1):
                current = members[day]
                if not current:
                    window = _day_window(day)
                    key = (0.0 if duration_min <= window else 1.0,
                           0.0, day)
                else:
                    nearest_km: Optional[float] = None
                    for member in current:
                        member_point = points.get(member)
                        if member_point is None or point is None:
                            continue
                        dist = haversine_km(point[0], point[1],
                                            member_point[0], member_point[1])
                        if dist is not None and (
                                nearest_km is None or dist < nearest_km):
                            nearest_km = dist
                    if nearest_km is None:
                        # Joins only unlocatable members: unverifiable
                        # either way; prefer the emptiest, lowest day.
                        key = (2.0, float(len(current)), day)
                    else:
                        key = (2.0, round(nearest_km, 3), day)
                if best_key is None or key < best_key:
                    best_key = key
                    best_day = day
            if best_day is None:  # pragma: no cover - span >= 1 always
                best_day = 1
            members[best_day].append(aid)
            day_load[best_day] += duration_min
        # Unlocatable activities have no coordinates to judge: spread them
        # onto the emptiest days (a solo unlocatable stop creates no
        # inter-stop leg at all) with legacy determinism. Placement stays
        # explicitly unverified via warnings below.
        for aid in unlocatable:
            target = min(range(1, span + 1),
                         key=lambda d: (len(members[d]), d))
            members[target].append(aid)
            day_load[target] += _activity_minutes_by_id(
                aid, indexed, order_index)

        # Rebalance: fill empty usable days from overloaded donor days so
        # trips use all days instead of leaving hotel-only gaps.
        # Geography-aware: the donor is the day most above the load
        # target, and the moved stop is the one farthest from that day's
        # centroid (the outlier), keeping each day's core cluster intact.
        # Deterministic; each move strictly reduces the empty-day count,
        # so this always terminates. Never drops.
        while True:
            empty_days = [d for d in range(1, span + 1) if not members[d]]
            if not empty_days:
                break
            donor_days = [d for d in range(1, span + 1) if len(members[d]) >= 2]
            if not donor_days:
                break
            donor = max(
                donor_days,
                key=lambda d: (
                    day_load[d] - target_load,
                    len(members[d]),
                    -d,
                ),
            )
            target = min(empty_days)
            target_window = _day_window(target)
            donor_pts = [
                points[a] for a in members[donor] if points[a] is not None
            ]
            center = centroid(donor_pts) if donor_pts else None

            def _outlier_rank(aid: str) -> Tuple[float, float, int]:
                dist = 0.0
                if center is not None and points[aid] is not None:
                    dist = (
                        haversine_km(
                            center[0], center[1],
                            points[aid][0], points[aid][1],
                        )
                        or 0.0
                    )
                return (
                    -round(dist, 3),
                    -round(
                        _activity_minutes_by_id(aid, indexed, order_index), 3
                    ),
                    order_index[aid],
                )

            movable = sorted(members[donor], key=_outlier_rank)
            moved = False
            for cand in movable:
                if _activity_minutes_by_id(cand, indexed, order_index) <= target_window:
                    members[donor].remove(cand)
                    members[target].append(cand)
                    day_load[donor] -= _activity_minutes_by_id(
                        cand, indexed, order_index
                    )
                    day_load[target] += _activity_minutes_by_id(
                        cand, indexed, order_index
                    )
                    if cand in opener_excess:
                        del opener_excess[cand]
                    opener_ref.pop(cand, None)
                    moved = True
                    break
            if not moved:
                break

    # Deterministic geographic order within each day: nearest-neighbor from
    # the previous day's end (day 1 starts at its first-ranked member).
    ordered: Dict[int, List[str]] = {}
    previous_end: Optional[Tuple[float, float]] = None
    for day in range(1, span + 1):
        day_aids = list(members[day])
        if not day_aids:
            ordered[day] = []
            continue
        placed: List[str] = []
        remaining = sorted(day_aids, key=lambda a: order_index[a])
        if previous_end is not None and any(points[a] is not None for a in remaining):
            remaining.sort(
                key=lambda a: (
                    haversine_km(
                        previous_end[0],
                        previous_end[1],
                        points[a][0],
                        points[a][1],
                    )
                    if points[a] is not None
                    else float("inf"),
                    order_index[a],
                )
            )
        while remaining:
            current_aid = remaining.pop(0)
            placed.append(current_aid)
            current_point = points[current_aid]
            if current_point is None:
                continue
            remaining.sort(
                key=lambda a: (
                    haversine_km(
                        current_point[0],
                        current_point[1],
                        points[a][0],
                        points[a][1],
                    )
                    if points[a] is not None
                    else float("inf"),
                    order_index[a],
                )
            )
        ordered[day] = placed
        locatable_placed = [a for a in placed if points[a] is not None]
        if locatable_placed:
            previous_end = points[locatable_placed[-1]]

    plan = RoutePlan()
    overflow_set = set(overflow)
    unlocatable_set = set(unlocatable)
    for day in range(1, span + 1):
        route = DayRoute(day=day, activity_ids=list(ordered[day]))
        legs: List[RouteLeg] = []
        day_points = [points[aid] for aid in ordered[day]]
        for prev_aid, aid in zip(ordered[day], ordered[day][1:]):
            legs.append(
                build_leg(
                    _endpoint(prev_aid, indexed, order_index, "activity"),
                    _endpoint(aid, indexed, order_index, "activity"),
                    points[prev_aid],
                    points[aid],
                    speed,
                )
            )
        route.legs = legs
        total, longest = summarize_legs(legs)
        route.daily_travel_minutes = total
        route.max_one_way_minutes = longest
        route.activity_minutes = round(
            sum(
                _activity_minutes_by_id(aid, indexed, order_index)
                for aid in ordered[day]
            ),
            1,
        )
        day_unable = [leg for leg in legs if leg.status != "estimated"]
        if day_unable and ordered[day]:
            route.warnings.append(
                f"Day {day} has {len(day_unable)} travel leg(s) that cannot "
                "be verified from available coordinates."
            )
        if total is not None and total > max_daily:
            route.warnings.append(
                f"Day {day} estimated travel ({total:.0f} min) exceeds the "
                f"{max_daily:.0f} min daily budget."
            )
        if longest is not None and longest > max_one_way:
            route.warnings.append(
                f"Day {day} has a one-way leg of about {longest:.0f} min, "
                f"above the {max_one_way:.0f} min limit."
            )
        flagged = [aid for aid in ordered[day] if aid in overflow_set]
        if flagged:
            route.warnings.append(
                f"Day {day} could not confidently place {len(flagged)} "
                "activit(ies); kept for revision, not dropped."
            )
        for aid in ordered[day]:
            if aid in opener_excess:
                # Name the stop by title: raw activity ids in user-facing
                # warnings are stale-looking and unverifiable downstream.
                # The reference stop is named too, so the estimate reads
                # as remoteness from a real stop — not as a validated leg.
                stop_name = _endpoint(
                    aid, indexed, order_index, "activity").name
                ref_name = _endpoint(
                    opener_ref[aid], indexed, order_index, "activity").name \
                    if opener_ref.get(aid) else None
                if ref_name:
                    route.warnings.append(
                        f"{stop_name} is about "
                        f"{opener_excess[aid]:.0f} min from the nearest other "
                        f"stop ({ref_name}) — a remote destination, above the "
                        f"{max_one_way:.0f} min one-way limit."
                    )
                else:
                    route.warnings.append(
                        f"Reaching {stop_name} takes about "
                        f"{opener_excess[aid]:.0f} min, "
                        f"above the {max_one_way:.0f} min one-way limit."
                    )
        if [aid for aid in ordered[day] if aid in unlocatable_set]:
            route.warnings.append(
                f"Day {day} includes activities without coordinates; "
                "placement is unverified."
            )
        route.status = _day_status(route, ordered[day], unlocatable_set, overflow_set)
        plan.days[day] = route
        for aid in ordered[day]:
            plan.activity_day[aid] = day
    if overflow:
        plan.warnings.append(
            f"{len(overflow)} activit(ies) exceeded every feasible day and "
            "were kept with revision warnings."
        )
    return plan


def _endpoint(
    aid: str,
    indexed: Sequence[Any],
    order_index: Dict[str, int],
    item_type: str,
) -> RouteEndpoint:
    position = order_index.get(aid, 0)
    source = indexed[position] if 0 <= position < len(indexed) else None
    if isinstance(source, Mapping):
        name = str(source.get("title", source.get("name", aid)))
    else:
        name = str(getattr(source, "title", getattr(source, "name", aid)))
    return RouteEndpoint(
        item_id=aid, item_type=item_type, name=name, coordinates_available=True
    )


def _activity_minutes_by_id(
    aid: str, indexed: Sequence[Any], order_index: Dict[str, int]
) -> float:
    position = order_index.get(aid)
    if position is None or not 0 <= position < len(indexed):
        return 0.0
    return _activity_minutes(indexed[position])


def _day_travel_estimate(
    pts: Sequence[Optional[Tuple[float, float]]],
    speed_kmh: float,
    buffer_min: float,
) -> Optional[float]:
    """Chain length + per-stop buffer for a candidate day order."""
    ordered = [p for p in pts if p is not None]
    if len(ordered) < 2:
        return 0.0 if ordered else None
    total = 0.0
    for first, second in zip(ordered, ordered[1:]):
        dist = haversine_km(first[0], first[1], second[0], second[1])
        if dist is None:
            return None
        minutes = estimate_travel_minutes(dist, speed_kmh)
        if minutes is None:
            return None
        total += minutes + buffer_min
    return round(total, 1)


def _day_status(
    route: DayRoute,
    day_aids: Sequence[str],
    unlocatable: set,
    overflow: set,
) -> str:
    if any(aid in overflow for aid in day_aids):
        return STATUS_REVISION
    if not day_aids:
        return STATUS_VALID
    if all(aid in unlocatable for aid in day_aids):
        return STATUS_UNVERIFIED
    if route.warnings:
        return STATUS_WARNING
    return STATUS_VALID


_CLOCK_PATTERN = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([APap][Mm])?\s*$")


def parse_clock_time(text: Any) -> Optional[int]:
    """Parse "09:00 AM" / "13:00" to minutes since midnight, else None.

    Unparseable values yield None so callers skip validation instead of
    guessing — never invent a time.
    """
    if not isinstance(text, str):
        return None
    match = _CLOCK_PATTERN.match(text)
    if not match:
        return None
    try:
        hour = int(match.group(1))
        minute = int(match.group(2))
    except (TypeError, ValueError):
        return None
    period = (match.group(3) or "").upper()
    if period:
        if hour < 1 or hour > 12 or minute > 59:
            return None
        hour = hour % 12
        if period == "PM":
            hour += 12
    elif hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def detect_overlaps(
    stops: Sequence[Tuple[str, Any, Any]],
) -> Dict[str, List[str]]:
    """Find same-day activity pairs with overlapping [start, end) windows.

    Returns {stop_id: [conflicting stop ids]}. Stops with unparseable or
    missing times are skipped (no false positives); nothing is dropped.
    """
    parsed: List[Tuple[str, int, int]] = []
    for stop_id, start, end in stops:
        start_min = parse_clock_time(start)
        end_min = parse_clock_time(end)
        if start_min is None or end_min is None or end_min <= start_min:
            continue
        parsed.append((str(stop_id), start_min, end_min))
    conflicts: Dict[str, List[str]] = {}
    for index, (aid, a_start, a_end) in enumerate(parsed):
        for bid, b_start, b_end in parsed[index + 1:]:
            if a_start < b_end and b_start < a_end:
                conflicts.setdefault(aid, []).append(bid)
                conflicts.setdefault(bid, []).append(aid)
    return conflicts
