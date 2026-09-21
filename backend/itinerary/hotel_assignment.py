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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

HOTEL_ASSIGNMENT_CONFIG: Dict[str, float] = {
    "max_preferred_distance_km": 40.0,
    "max_preferred_travel_time_minutes": 90.0,
    "significant_distance_km": 60.0,
    "significant_travel_time_minutes": 120.0,
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


def plan_overnight_stays(
    *,
    nights: int,
    anchors: Mapping[int, Optional[DayAnchor]],
    hotels: Sequence[Any],
    pinned_hotels: Optional[Mapping[int, Any]] = None,
    nightly_price: Optional[Callable[[Any], float]] = None,
    total_pot: Optional[float] = None,
    config: Optional[Mapping[str, float]] = None,
) -> Tuple[List[OvernightStay], float]:
    """Assign one hotel per overnight stay using proximity logic.

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

    for night in range(1, max(0, int(nights or 0)) + 1):
        pinned_hotel = pinned.get(night)
        if pinned_hotel is not None:
            current = pinned_hotel
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=pinned_hotel,
                    reason=f"Kept your chosen stay at {getattr(pinned_hotel, 'name', 'the selected hotel')}.",
                    retained=True,
                )
            )
            continue

        need = max(1, nights_left.get(night, 1))
        affordable = [
            hotel
            for hotel in ranked
            if remaining is None or price_of(hotel) * need <= remaining
        ]
        if not affordable:
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=None,
                    reason="No verified stay fits the remaining budget.",
                    retained=False,
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
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=forced,
                    reason=(
                        f"Switching to {getattr(forced, 'name', 'a verified stay')} "
                        "to stay within the trip budget."
                    ),
                    retained=False,
                )
            )
            continue

        anchor = complete_anchors.get(night)

        if current is None:
            chosen = affordable[0]
            cost = price_of(chosen)
            remaining = None if remaining is None else remaining - cost
            current = chosen
            label = anchor.label if anchor else "the trip area"
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=chosen,
                    reason=f"Opening stay near {label}.",
                    retained=False,
                )
            )
            continue

        current_point = _hotel_coords(current)
        anchor_point = (
            (anchor.latitude, anchor.longitude) if anchor is not None else None
        )
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
            decisions.append(
                OvernightStay(
                    night=night,
                    hotel=current,
                    reason=(
                        f"Staying at {current_name} — not enough location "
                        "detail to justify a move."
                    ),
                    retained=True,
                )
            )
            continue

        with_coords = [
            hotel for hotel in affordable if _hotel_coords(hotel) is not None
        ]
        if not with_coords:
            remaining = None if remaining is None else remaining - price_of(current)
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
                )
            )
            remaining = None if remaining is None else remaining - price_of(current)
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
                )
            )
            remaining = None if remaining is None else remaining - price_of(current)
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
            )
        )
        remaining = None if remaining is None else remaining - price_of(current)
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
