"""Proximity-based, day-wise overnight stay planning.

Generic, location-independent logic: every decision is driven by stored
coordinates, estimated travel time, the itinerary structure, and caller
supplied budgets. There are no per-region, per-state, or per-country rules.

Model of a trip: night ``i`` (1-based) follows sightseeing day ``i``. The
overnight anchor for night ``i`` is the centroid of the next morning's
activities (day ``i + 1``), falling back to day ``i`` and then to the trip
destination. A night keeps its current hotel unless the next anchor is
significantly far away *and* a closer verified hotel exists.

Distance is straight-line (haversine) and travel time is estimated from a
configurable assumed road speed. These are planning estimates, never exact
routing: reasons shown to travelers always say "about". When coordinates
are missing anywhere in the comparison, the current hotel is retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HOTEL_ASSIGNMENT_CONFIG: Dict[str, float] = {
    "max_preferred_distance_km": 40.0,
    "max_preferred_travel_time_minutes": 90.0,
    "significant_distance_km": 60.0,
    "significant_travel_time_minutes": 120.0,
    # Overnight-region linkage scale (km, straight-line). Anchor points
    # chained within this distance belong to one travel region, so
    # consecutive nights serving the same region reuse one stay instead of
    # churning. Deliberately roomier than the per-leg preferred limit:
    # nearby stops (e.g. same-day-trip sights ~100 km apart) form one
    # region even though no single hotel serves every stop in it.
    "region_link_km": 120.0,
    # Fallback road speed used ONLY to estimate travel time from
    # straight-line distance. Never presented as measured routing data.
    "assumed_speed_kmh": 50.0,
}

_EARTH_RADIUS_KM = 6371.0


@dataclass
class DayAnchor:
    """Where the traveler wakes up for (or ends) a sightseeing day."""

    latitude: float
    longitude: float
    label: str


@dataclass
class OvernightStay:
    """One planned night: night ``i`` follows sightseeing day ``i``."""

    night: int
    hotel: Optional[Any]  # Hotel row, or None when nothing affordable fits
    reason: str
    distance_km: Optional[float] = None
    travel_minutes: Optional[float] = None
    retained: bool = True
    # Machine-readable outcome for this night (pinned/opening/move/retain…).
    decision: str = ""
    # Per-candidate audit trail: every ranked candidate evaluated for this
    # night with its distance, affordability, and concrete outcome/reason.
    # In-memory for debugging/testing; callers serialize a compact subset.
    evaluations: List[Dict[str, Any]] = field(default_factory=list)


def haversine_km(
    lat_a: Any, lng_a: Any, lat_b: Any, lng_b: Any
) -> Optional[float]:
    """Straight-line distance in km, or None when any coordinate is missing."""
    try:
        lat_a, lng_a, lat_b, lng_b = (
            float(lat_a),
            float(lng_a),
            float(lat_b),
            float(lng_b),
        )
    except (TypeError, ValueError):
        return None
    for value in (lat_a, lng_a, lat_b, lng_b):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
    if not (-90.0 <= lat_a <= 90.0 and -90.0 <= lat_b <= 90.0):
        return None
    if not (-180.0 <= lng_a <= 180.0 and -180.0 <= lng_b <= 180.0):
        return None
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lam = math.radians(lng_b - lng_a)
    arc = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(
        delta_lam / 2.0
    ) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(max(0.0, min(1.0, arc))))


def estimate_travel_minutes(
    distance_km: Any, speed_kmh: Any = HOTEL_ASSIGNMENT_CONFIG["assumed_speed_kmh"]
) -> Optional[float]:
    """Rough drive-time estimate from straight-line distance.

    Clearly a fallback: straight-line distance underestimates real roads,
    so callers must present the result as an estimate ("about"), never as
    measured routing data.
    """
    try:
        distance = float(distance_km)
        speed = float(speed_kmh)
    except (TypeError, ValueError):
        return None
    if distance < 0 or speed <= 0:
        return None
    return distance / speed * 60.0


def centroid(points: Sequence[Tuple[Any, Any]]) -> Optional[Tuple[float, float]]:
    """Mean position of valid coordinate pairs, or None when there are none."""
    latitudes: List[float] = []
    longitudes: List[float] = []
    for lat, lng in points or []:
        try:
            latitudes.append(float(lat))
            longitudes.append(float(lng))
        except (TypeError, ValueError):
            continue
    if not latitudes:
        return None
    return sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)


def _hotel_coords(hotel: Any) -> Optional[Tuple[float, float]]:
    point = centroid(
        [(getattr(hotel, "latitude", None), getattr(hotel, "longitude", None))]
    )
    return point


def _fmt_distance(distance_km: Optional[float]) -> str:
    if distance_km is None:
        return "an unknown distance"
    return f"about {distance_km:.0f} km"


def _fmt_time(travel_minutes: Optional[float]) -> str:
    if travel_minutes is None:
        return "an unknown travel time"
    return f"about {travel_minutes:.0f} min"


def _anchors_complete(
    anchors: Mapping[int, Optional[DayAnchor]],
) -> Dict[int, DayAnchor]:
    return {night: anchor for night, anchor in (anchors or {}).items() if anchor is not None}


def validate_stay_assignment(
    stay: OvernightStay,
    anchor: Optional[DayAnchor],
    candidates: Sequence[Any],
    config: Optional[Mapping[str, float]] = None,
    affordable_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Audit one decided overnight stay against live inputs (no side effects).

    Recomputes hotel-to-anchor distance/duration from stored coordinates,
    checks the preferred limits, and verifies the retention is consistent:
    a retained stay far beyond the significant threshold requires that no
    affordable, located, nearer candidate exists (otherwise the report
    flags ``inconsistent`` with the offending candidate). Pass
    ``affordable_ids`` to exclude over-budget candidates from the nearer
    check (when omitted, every located candidate counts). Pinned stays and
    missing coordinates are reported honestly, never guessed. Returns a
    structured report; raises nothing.
    """
    cfg = dict(HOTEL_ASSIGNMENT_CONFIG)
    if config:
        cfg.update(config)
    report: Dict[str, Any] = {
        "night": getattr(stay, "night", None),
        "hotel_id": getattr(getattr(stay, "hotel", None), "id", None),
        "retained": bool(getattr(stay, "retained", True)),
        "decision": getattr(stay, "decision", "") or "",
        "anchor": None,
        "distance_km": None,
        "travel_minutes": None,
        "within_preferred": False,
        "retention_consistent": True,
        "issues": [],
    }
    try:
        anchor_point = None
        if anchor is not None:
            anchor_point = _valid_anchor_point(
                (anchor.latitude, anchor.longitude))
        if anchor_point is not None:
            report["anchor"] = {
                "latitude": anchor_point[0],
                "longitude": anchor_point[1],
                "label": getattr(anchor, "label", None),
            }
        hotel_point = _hotel_coords(getattr(stay, "hotel", None))
        distance = None
        minutes = None
        if hotel_point is not None and anchor_point is not None:
            distance = haversine_km(hotel_point[0], hotel_point[1],
                                    anchor_point[0], anchor_point[1])
            if distance is not None:
                minutes = estimate_travel_minutes(
                    distance, cfg["assumed_speed_kmh"])
        report["distance_km"] = round(distance, 1) if distance is not None else None
        report["travel_minutes"] = round(minutes, 1) if minutes is not None else None
        if distance is not None and minutes is not None:
            report["within_preferred"] = bool(
                distance <= cfg["max_preferred_distance_km"]
                and minutes <= cfg["max_preferred_travel_time_minutes"])
        if not report["retained"]:
            return report
        if distance is None or minutes is None:
            return report
        far = bool(
            distance >= cfg["significant_distance_km"]
            or minutes >= cfg["significant_travel_time_minutes"])
        if not far:
            return report
        nearer: List[str] = []
        allowed = None if affordable_ids is None else set(affordable_ids)
        for candidate in candidates or []:
            if getattr(candidate, "id", None) == report["hotel_id"]:
                continue
            if allowed is not None and getattr(candidate, "id", None) not in allowed:
                continue
            candidate_point = _hotel_coords(candidate)
            if candidate_point is None or anchor_point is None:
                continue
            candidate_dist = haversine_km(
                candidate_point[0], candidate_point[1],
                anchor_point[0], anchor_point[1])
            if candidate_dist is not None and candidate_dist < distance:
                nearer.append(str(getattr(candidate, "name", None)
                                  or getattr(candidate, "id", None)))
        if nearer:
            report["retention_consistent"] = False
            report["issues"].append(
                "Retained stay is beyond the significant threshold while "
                f"nearer located candidates exist: {', '.join(sorted(set(nearer)))}."
            )
    except Exception as exc:
        report["issues"].append(f"validation failed: {exc}")
    return report


def cluster_anchor_points(
    ordered_points: Sequence[Tuple[Any, Tuple[float, float]]],
    link_km: Optional[float] = None,
) -> Dict[Any, int]:
    """Group anchor points into travel regions (deterministic).

    Single-linkage chaining over straight-line distance: points linked
    within ``link_km`` (default ``region_link_km``) belong to one region,
    so nearby stops form a region while far-flung ones stay separate.
    Input order decides ties: region ids follow first appearance, so
    identical inputs always produce identical assignments. Points without
    valid coordinates are skipped, never invented. Returns {key: region_id}.

    Chaining note (deliberate, verified by tests): A-B-C with A-B and B-C
    each within range but A-C beyond it still forms ONE region. This is
    safe for stay planning because regions only order candidates and
    record incumbents — every move/retain decision is re-derived from
    real hotel-to-anchor distances, so a chained region can never force
    or block a hotel change. Tightening the rule (e.g. complete-linkage)
    would fragment natural day-trip corridors without improving any
    decision, so single-linkage is retained.
    """
    try:
        threshold = float(
            link_km
            if link_km is not None
            else HOTEL_ASSIGNMENT_CONFIG["region_link_km"]
        )
    except (TypeError, ValueError):
        threshold = 120.0
    keys: List[Any] = []
    pts: List[Tuple[float, float]] = []
    for key, point in ordered_points or []:
        valid = _valid_anchor_point(point)
        if valid is not None and key not in keys:
            keys.append(key)
            pts.append(valid)
    parent = list(range(len(keys)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dist = haversine_km(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
            if dist is not None and dist <= threshold:
                parent[find(i)] = find(j)
    region_of_root: Dict[int, int] = {}
    regions: Dict[Any, int] = {}
    for key, idx in zip(keys, range(len(keys))):
        root = find(idx)
        if root not in region_of_root:
            region_of_root[root] = len(region_of_root)
        regions[key] = region_of_root[root]
    return regions


def _valid_anchor_point(point: Any) -> Optional[Tuple[float, float]]:
    try:
        lat = float(point[0])
        lng = float(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    if math.isnan(lat) or math.isnan(lng):
        return None
    return lat, lng


def plan_overnight_stays(
    *,
    nights: int,
    anchors: Mapping[int, Optional[DayAnchor]],
    hotels: Sequence[Any],
    pinned_hotels: Optional[Mapping[int, Any]] = None,
    nightly_price: Optional[Callable[[Any], float]] = None,
    total_pot: Optional[float] = None,
    config: Optional[Mapping[str, float]] = None,
    night_regions: Optional[Mapping[int, Any]] = None,
    arrival_point: Optional[Tuple[Any, Any]] = None,
) -> Tuple[List[OvernightStay], float]:
    """Assign one hotel per overnight stay using proximity logic.

    Accommodation follows activity geography, not the calendar: nights
    serving the same travel region reuse one stay (cluster-based, so no
    churning between nearby stops), while a genuinely new region triggers
    a fresh proximity evaluation. The opening night additionally respects
    the arrival location instead of blindly taking the top-ranked hotel.

    Args:
        nights: number of overnight stays (night ``i`` follows day ``i``).
        anchors: overnight anchor per night (where the traveler wakes up
            next); nights without anchors retain their current hotel.
        hotels: ranked hotel candidates (best first); ranking is preserved
            as the tiebreak so preference matching still matters.
        pinned_hotels: traveler-fixed stays (e.g. previously confirmed
            hotel items) keyed by night; these are never reassigned.
        nightly_price: cost of one night per hotel; defaults to the row's
            ``price_per_night``.
        total_pot: total remaining budget for all newly assigned nights;
            ``None`` means unconstrained. Pinned nights cost nothing extra
            here because preserved items are already counted upstream.
        config: threshold overrides; defaults to HOTEL_ASSIGNMENT_CONFIG.
        night_regions: travel-region id per night (see
            ``cluster_anchor_points``); consecutive nights sharing a region
            prefer that region's incumbent stay, so cluster runs never
            churn. ``None`` (or nights missing from the map) keep the
            legacy per-night evaluation.
        arrival_point: (latitude, longitude) where the traveler arrives
            (e.g. the destination point). The opening stay is chosen among
            affordable hotels near the arrival area instead of blindly
            taking the top-ranked hotel, so night 1 can never strand the
            traveler hours from where they land. ``None`` keeps the legacy
            top-ranked opening.

    Returns:
        (decisions in night order, newly committed spend). Spend covers
        every non-pinned night that receives a hotel (new assignments and
        retained stays alike, since retained stays still occupy paid
        nights); pinned nights cost nothing extra because preserved items
        are already counted upstream. A decision carries ``hotel=None``
        when no candidate fits the running budget; callers translate that
        into a generation error.
    """
    cfg = dict(HOTEL_ASSIGNMENT_CONFIG)
    if config:
        cfg.update(config)
    complete_anchors = _anchors_complete(anchors)
    pinned = dict(pinned_hotels or {})
    ranked = list(hotels or [])
    regions = dict(night_regions or {})
    arrival = _valid_anchor_point(arrival_point) if arrival_point is not None else None
    # Incumbent stay per travel region: once a region is served by a
    # verified hotel, later nights in the same region prefer it, so a
    # cluster run never churns between nearby stays.
    region_stay: Dict[Any, Any] = {}

    def _note_region(hotel: Any, night: int) -> None:
        if regions.get(night) is not None and hotel is not None:
            region_stay[regions[night]] = hotel

    def price_of(hotel: Any) -> float:
        if nightly_price is not None:
            try:
                return float(nightly_price(hotel))
            except (TypeError, ValueError):
                return float("inf")
        try:
            return float(getattr(hotel, "price_per_night", 0) or 0)
        except (TypeError, ValueError):
            return float("inf")

    remaining = total_pot
    spent = 0.0
    decisions: List[OvernightStay] = []
    current: Optional[Any] = None
    # Non-pinned nights still to serve (including tonight): picks must fit
    # *all* of them at their nightly rate, mirroring the previous
    # whole-trip total check while still allowing different hotels nightly.
    open_nights = [n for n in range(1, max(0, int(nights or 0)) + 1) if n not in pinned]
    nights_left = {
        night: sum(1 for later in open_nights if later >= night) for night in open_nights
    }

    def _evaluate(
        decided: Optional[Any],
        branch: str,
        anchor_point: Optional[Tuple[float, float]],
        affordable: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """Per-candidate audit for one night (see OvernightStay.evaluations).

        Every ranked candidate gets a concrete, factual outcome: the
        decided hotel carries the branch decision, everyone else a
        rejection reason (budget, coordinates, preferred band, rank order,
        or distance). Pure and deterministic; never invents data.
        """
        affordable_ids = {getattr(h, "id", None) for h in affordable}
        decided_id = getattr(decided, "id", None) if decided is not None else None
        decided_point = _hotel_coords(decided) if decided is not None else None
        decided_dist: Optional[float] = None
        if decided_point is not None and anchor_point is not None:
            decided_dist = haversine_km(
                decided_point[0], decided_point[1],
                anchor_point[0], anchor_point[1],
            )
        evals: List[Dict[str, Any]] = []
        for hotel in ranked:
            hid = getattr(hotel, "id", None)
            if decided is not None and hid == decided_id:
                continue
            point = _hotel_coords(hotel)
            dist: Optional[float] = None
            mins: Optional[float] = None
            if point is not None and anchor_point is not None:
                dist = haversine_km(
                    point[0], point[1], anchor_point[0], anchor_point[1])
                if dist is not None:
                    mins = estimate_travel_minutes(
                        dist, cfg["assumed_speed_kmh"])
            if hid not in affordable_ids:
                # Failed this night's budget lookahead (price × remaining
                # nights); `affordable` was computed from the pre-spend
                # remainder, so this is exact.
                reason = "rejected_unaffordable"
            elif dist is None or mins is None:
                reason = "rejected_no_coords"
            elif (dist <= cfg["max_preferred_distance_km"]
                    and mins <= cfg["max_preferred_travel_time_minutes"]):
                reason = "rejected_rank_order"
            elif (decided_dist is not None and dist > decided_dist):
                reason = "rejected_farther"
            else:
                reason = "rejected_outside_preferred"
            evals.append({
                "hotel_id": hid,
                "name": getattr(hotel, "name", None),
                "distance_km": round(dist, 1) if dist is not None else None,
                "travel_minutes": round(mins, 1) if mins is not None else None,
                "affordable": hid in affordable_ids,
                "decision": branch if hid == decided_id else reason,
            })
        return evals

    for night in range(1, max(0, int(nights or 0)) + 1):
        anchor = complete_anchors.get(night)
        anchor_point = (
            (anchor.latitude, anchor.longitude) if anchor is not None else None
        )
        need = max(1, nights_left.get(night, 1))
        affordable = [
            hotel
            for hotel in ranked
            if remaining is None or price_of(hotel) * need <= remaining
        ]
        pinned_hotel = pinned.get(night)
        if pinned_hotel is not None:
            current = pinned_hotel
            if regions.get(night) is not None:
                region_stay[regions[night]] = pinned_hotel
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=pinned_hotel,
                    reason=f"Kept your chosen stay at {getattr(pinned_hotel, 'name', 'the selected hotel')}.",
                    retained=True,
                    decision="pinned",
                    evaluations=_evaluate(pinned_hotel, "pinned",
                                          anchor_point, affordable),
                )
            )
            continue

        if not affordable:
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=None,
                    reason="No verified stay fits the remaining budget.",
                    retained=False,
                    decision="unaffordable",
                    evaluations=_evaluate(None, "unaffordable",
                                          anchor_point, affordable),
                )
            )
            continue

        if (
            current is not None
            and remaining is not None
            and price_of(current) > remaining
        ):
            # Currently held stay no longer fits even tonight (e.g. an
            # expensive traveler-confirmed pick): fall back to the best
            # affordable option rather than blowing the trip budget.
            forced = affordable[0]
            cost = price_of(forced)
            remaining = remaining - cost
            current = forced
            _note_region(forced, night)
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=forced,
                    reason=(
                        f"Switching to {getattr(forced, 'name', 'a verified stay')} "
                        "to stay within the trip budget."
                    ),
                    retained=False,
                    decision="budget_forced",
                    evaluations=_evaluate(forced, "budget_forced",
                                          anchor_point, affordable),
                )
            )
            continue

        def _dist_to(point: Optional[Tuple[float, float]], hotel: Any) -> Optional[float]:
            hotel_point = _hotel_coords(hotel)
            if point is None or hotel_point is None:
                return None
            return haversine_km(
                hotel_point[0], hotel_point[1], point[0], point[1]
            )

        def _dist_or_inf(value: Optional[float]) -> float:
            # NOTE: 0.0 km is a real distance (hotel on the anchor), never
            # missing data — only None means unverifiable.
            return float("inf") if value is None else value

        # Cluster-first ordering: the incumbent stay of this night's
        # travel region (when affordable) leads the pool, so a cluster run
        # reuses one verified stay instead of churning night by night.
        # Rank order still breaks every remaining tie.
        pool = list(affordable)
        incumbent = None
        if regions.get(night) is not None:
            incumbent = region_stay.get(regions[night])
            if incumbent is not None and incumbent in pool:
                pool = [incumbent] + [h for h in pool if h != incumbent]
            else:
                incumbent = None

        if current is None:
            arrival_constrained = False
            if arrival is not None:
                reachable = [
                    hotel for hotel in pool
                    if _dist_or_inf(_dist_to(arrival, hotel))
                    <= cfg["significant_distance_km"]
                ]
                if reachable:
                    # Near the arrival area: prefer the reachable stay
                    # closest to tonight's anchor (the next sightseeing
                    # cluster), rank order breaking ties.
                    def _anchor_rank(hotel: Any) -> Tuple[float, int]:
                        dist = _dist_to(anchor_point, hotel)
                        return (
                            dist if dist is not None else float("inf"),
                            pool.index(hotel),
                        )
                    chosen = min(reachable, key=_anchor_rank)
                    if chosen != pool[0]:
                        arrival_constrained = True
                else:
                    # Nothing affordable near the arrival area: minimize the
                    # traveler's total night-1 burden (arrival transfer +
                    # first-cluster reach) instead of optimizing either leg
                    # alone, with arrival distance as the tiebreak so the
                    # landing area keeps priority on ties.
                    def _opening_cost(hotel: Any) -> Tuple[float, float, int]:
                        arrival_dist = _dist_or_inf(
                            _dist_to(arrival, hotel))
                        anchor_dist = _dist_or_inf(
                            _dist_to(anchor_point, hotel))
                        return (
                            arrival_dist + anchor_dist,
                            arrival_dist,
                            pool.index(hotel),
                        )
                    chosen = min(pool, key=_opening_cost)
                    arrival_constrained = True
            else:
                chosen = pool[0]
            cost = price_of(chosen)
            remaining = None if remaining is None else remaining - cost
            current = chosen
            _note_region(chosen, night)
            label = anchor.label if anchor else "the trip area"
            reason = f"Opening stay for {label}."
            opening_distance = None
            opening_time = None
            chosen_point = _hotel_coords(chosen)
            if anchor is not None and chosen_point is not None:
                opening_distance = haversine_km(
                    chosen_point[0], chosen_point[1],
                    anchor.latitude, anchor.longitude,
                )
                opening_time = (
                    estimate_travel_minutes(opening_distance, cfg["assumed_speed_kmh"])
                    if opening_distance is not None
                    else None
                )
            if opening_distance is not None and opening_time is not None:
                if (
                    opening_distance <= cfg["max_preferred_distance_km"]
                    and opening_time <= cfg["max_preferred_travel_time_minutes"]
                ):
                    reason = f"Opening stay near {label}."
                else:
                    reason = (
                        f"Opening stay {_fmt_distance(opening_distance)} "
                        f"({_fmt_time(opening_time)}) from {label}."
                    )
            if arrival_constrained:
                arrival_dist = _dist_or_inf(_dist_to(arrival, chosen))
                if arrival_dist <= cfg["significant_distance_km"]:
                    reason = (
                        f"{reason} Chosen near the arrival area so night 1 "
                        "never strands the traveler far from where they land."
                    )
                else:
                    reason = (
                        f"{reason} No affordable stay is near the arrival "
                        "area; the stay minimizing the combined arrival "
                        "transfer and first-cluster reach was kept "
                        "for night 1."
                    )
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=chosen,
                    reason=reason,
                    distance_km=opening_distance,
                    travel_minutes=opening_time,
                    retained=False,
                    decision="opening",
                    evaluations=_evaluate(chosen, "opening",
                                          anchor_point, affordable),
                )
            )
            continue

        current_point = _hotel_coords(current)
        distance = None
        travel_time = None
        if current_point is not None and anchor_point is not None:
            distance = haversine_km(
                current_point[0], current_point[1], anchor_point[0], anchor_point[1]
            )
            travel_time = (
                estimate_travel_minutes(distance, cfg["assumed_speed_kmh"])
                if distance is not None
                else None
            )

        label = anchor.label if anchor else "the next sightseeing area"
        current_name = getattr(current, "name", "the current stay")
        if distance is None or travel_time is None:
            remaining = None if remaining is None else remaining - price_of(current)
            _note_region(current, night)
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=current,
                    reason=(
                        f"Staying at {current_name} — not enough location "
                        "detail to justify a move."
                    ),
                    retained=True,
                    decision="retained_no_detail",
                    evaluations=_evaluate(current, "retained_no_detail",
                                          anchor_point, affordable),
                )
            )
            continue

        with_coords = [
            hotel for hotel in pool if _hotel_coords(hotel) is not None
        ]
        if not with_coords:
            remaining = None if remaining is None else remaining - price_of(current)
            _note_region(current, night)
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=current,
                    reason=(
                        f"Staying at {current_name} — nearby options have "
                        "no usable location data."
                    ),
                    distance_km=distance,
                    travel_minutes=travel_time,
                    retained=True,
                    decision="retained_no_coords",
                    evaluations=_evaluate(current, "retained_no_coords",
                                          anchor_point, affordable),
                )
            )
            continue

        def within_preferred(hotel: Any) -> bool:
            point = _hotel_coords(hotel)
            if point is None or anchor_point is None:
                return False
            dist = haversine_km(point[0], point[1], anchor_point[0], anchor_point[1])
            if dist is None:
                return False
            time_needed = estimate_travel_minutes(dist, cfg["assumed_speed_kmh"])
            time_ok = (
                time_needed if time_needed is not None else float("inf")
            ) <= cfg["max_preferred_travel_time_minutes"]
            return dist <= cfg["max_preferred_distance_km"] and time_ok

        current_time_ok = (
            travel_time if travel_time is not None else float("inf")
        ) <= cfg["max_preferred_travel_time_minutes"]
        if (
            distance <= cfg["max_preferred_distance_km"]
            and current_time_ok
        ):
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=current,
                    reason=(
                        f"Staying at {current_name} — {label} is "
                        f"{_fmt_distance(distance)} ({_fmt_time(travel_time)} away), "
                        "a practical day trip."
                    ),
                    distance_km=distance,
                    travel_minutes=travel_time,
                    retained=True,
                    decision="retained_preferred",
                    evaluations=_evaluate(current, "retained_preferred",
                                          anchor_point, affordable),
                )
            )
            remaining = None if remaining is None else remaining - price_of(current)
            _note_region(current, night)
            continue

        move_needed = distance >= cfg["significant_distance_km"] or (
            travel_time is not None
            and travel_time >= cfg["significant_travel_time_minutes"]
        )
        if move_needed:
            # Move only to a stay genuinely near the anchor (rank order
            # wins among those). Switching to another far-away hotel
            # burns budget without cutting travel, so in that case the
            # current stay is retained instead.
            near = [
                hotel
                for hotel in with_coords
                if getattr(hotel, "id", None) != getattr(current, "id", None)
                and within_preferred(hotel)
            ]
            if near:
                best = near[0]
                cost = price_of(best)
                remaining = None if remaining is None else remaining - cost
                current = best
                _note_region(best, night)
                decisions.append(
                    OvernightStay(
                        night=night,
                        hotel=best,
                        reason=(
                            f"{label} is {_fmt_distance(distance)} "
                            f"({_fmt_time(travel_time)} from {current_name}); "
                            f"new stay near {label} to cut travel time."
                        ),
                        distance_km=distance,
                        travel_minutes=travel_time,
                        retained=False,
                        decision="moved",
                        evaluations=_evaluate(best, "moved",
                                              anchor_point, affordable),
                    )
                )
                continue
            nearest_id = None
            nearest_dist: Optional[float] = None
            for hotel in with_coords:
                point = _hotel_coords(hotel)
                if point is None or anchor_point is None:
                    continue
                dist = haversine_km(point[0], point[1], anchor_point[0], anchor_point[1])
                if dist is not None and (nearest_dist is None or dist < nearest_dist):
                    nearest_dist = dist
                    nearest_id = getattr(hotel, "id", None)
            if nearest_id is not None and nearest_id != getattr(current, "id", None):
                reason = (
                    f"Staying at {current_name} — no verified stay near "
                    f"{label}, so moving would not cut the trip."
                )
            elif len(with_coords) <= 1:
                reason = (
                    f"Staying at {current_name} — verified hotel inventory "
                    f"is limited near {label}; the available hotel is "
                    "retained and may require longer travel."
                )
            else:
                reason = (
                    f"Staying at {current_name} — the closest verified "
                    f"option for {label}."
                )
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=current,
                    reason=reason,
                    distance_km=distance,
                    travel_minutes=travel_time,
                    retained=True,
                    decision="retained_no_near_option",
                    evaluations=_evaluate(current, "retained_no_near_option",
                                          anchor_point, affordable),
                )
            )
            remaining = None if remaining is None else remaining - price_of(current)
            _note_region(current, night)
            continue

        decisions.append(
            OvernightStay(
                night=night,
                hotel=current,
                reason=(
                    f"Staying at {current_name} — {label} is within "
                    "comfortable reach, so no move was made."
                ),
                distance_km=distance,
                travel_minutes=travel_time,
                retained=True,
                decision="retained_comfortable",
                evaluations=_evaluate(current, "retained_comfortable",
                                      anchor_point, affordable),
            )
        )
        remaining = None if remaining is None else remaining - price_of(current)
        _note_region(current, night)
        continue

    # One paid night per non-pinned decided stay (new and retained alike);
    # pinned nights were already counted upstream.
    spent = 0.0
    for decided in decisions:
        if decided.hotel is not None and decided.night not in pinned:
            try:
                spent += float(price_of(decided.hotel))
            except (TypeError, ValueError):
                pass

    return decisions, spent
