"""Persist catalog-grounded generated and optimized trip itinerary items."""

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

from sqlalchemy.orm import Session
from types import SimpleNamespace

from backend.models.models import Activity, Hotel, ItineraryItem, TransportOption, Trip
from backend.recommendation.engine import RecommendationEngine
from backend.itinerary.hotel_assignment import (
    DayAnchor,
    HOTEL_ASSIGNMENT_CONFIG,
    OvernightStay,
    centroid,
    cluster_anchor_points,
    estimate_travel_minutes,
    haversine_km,
    plan_overnight_stays,
)
from backend.itinerary.route_plan import (
    RouteEndpoint,
    RoutePlan,
    build_leg,
    detect_overlaps,
    format_clock_time,
    parse_clock_time,
    plan_activity_days,
    resolve_pace,
    schedule_day_stops,
    summarize_legs,
)


class ItineraryGenerationError(Exception):
    """Raised when a trip itinerary cannot be generated from verified catalog records."""


logger = logging.getLogger(__name__)


class ItineraryGenerator:
    """Build proposed itinerary items from deterministic catalog recommendations."""

    def __init__(self, db: Session):
        self.db = db

    def generate_for_trip(self, trip_id: str) -> List[ItineraryItem]:
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        if not trip:
            raise ItineraryGenerationError("Trip not found")
        existing_items = self._trip_items(trip)
        if existing_items:
            return existing_items
        try:
            return self._persist_ranked_items(trip, [])
        except ItineraryGenerationError as exc:
            raise self._with_budget_hint(trip, exc) from exc

    def _with_budget_hint(self, trip: Trip, exc: ItineraryGenerationError) -> ItineraryGenerationError:
        """Append the cheapest workable total to budget failures so travelers
        know by how much to raise the budget (or what to cut) instead of
        guessing. Non-budget errors pass through untouched."""
        if "fits the trip budget" not in str(exc):
            return exc
        try:
            estimate = self.minimum_viable_estimate(trip)
        except Exception:
            return exc
        if estimate is None:
            return exc
        cur = (trip.currency or "INR").upper()
        return ItineraryGenerationError(
            f"{exc} (cheapest workable plan ≈ {cur} {estimate:,.0f} for "
            f"{max(1, int(trip.traveler_count or 1))} traveler(s), "
            f"{max(1, int(trip.duration_days or 1))} days — "
            "raise the budget or shorten the trip)."
        )

    def minimum_viable_estimate(self, trip: Trip) -> Optional[float]:
        """Cheapest possible trip total from real inventory, or None when any
        required piece is missing. Mirrors validator scoping."""
        if not trip.destination_id:
            return None
        currency = (trip.currency or "INR").upper()
        travelers = max(1, int(trip.traveler_count or 1))
        duration = max(1, int(trip.duration_days or 1))
        nights = max(1, duration - 1)
        required = max(0, duration - 1)
        session_id = trip.discovery_session_id

        def _scoped(query, model):
            query = query.filter(
                model.destination_id == trip.destination_id,
                model.currency == currency,
                model.is_active == True,  # noqa: E712
            )
            if session_id:
                return query.filter(
                    model.inventory_source == "discovered",
                    model.discovery_session_id == session_id,
                    model.verification_status == "verified_candidate",
                )
            return query.filter(model.inventory_source.in_(["catalog", "live"]))

        hotel_prices = sorted(
            float(h.price_per_night or 0)
            for h in _scoped(self.db.query(Hotel), Hotel).all()
        )
        if not hotel_prices:
            return None
        transport_prices = sorted(
            float(o.price or 0)
            for o in _scoped(self.db.query(TransportOption), TransportOption)
            .filter(TransportOption.capacity >= travelers).all()
        )
        act_prices = sorted(
            float(a.price_per_person or 0) * travelers
            for a in _scoped(self.db.query(Activity), Activity).all()
        )
        if len(act_prices) < required:
            return None
        return (hotel_prices[0] * nights
                + (transport_prices[0] if transport_prices else 0.0)
                + sum(act_prices[:required]))

    def optimize_for_trip(self, trip_id: str) -> List[ItineraryItem]:
        """Replace proposed catalog selections with a budget-bounded ranked selection."""
        trip = self.db.query(Trip).filter(Trip.id == trip_id).first()
        if not trip:
            raise ItineraryGenerationError("Trip not found")

        existing_items = self._trip_items(trip)
        replaceable = [
            item for item in existing_items
            if item.status == "proposed" and (item.hotel_id or item.transport_id or item.activity_id)
        ]
        preserved_items = [item for item in existing_items if item not in replaceable]
        for item in replaceable:
            self.db.delete(item)
        if replaceable:
            self.db.flush()

        try:
            new_items = self._persist_ranked_items(trip, preserved_items, commit=False)
        except ItineraryGenerationError as exc:
            raise self._with_budget_hint(trip, exc) from exc
        self.db.commit()
        return self._sorted_items([*preserved_items, *new_items])

    def _persist_ranked_items(
        self,
        trip: Trip,
        preserved_items: Sequence[ItineraryItem],
        *,
        commit: bool = True,
    ) -> List[ItineraryItem]:
        new_items = self._ranked_items(trip, preserved_items)
        if not new_items:
            raise ItineraryGenerationError("No catalog-backed itinerary items could be generated")
        self.db.add_all(new_items)
        if commit:
            self.db.commit()
        # Sorted like _trip_items so callers see days in order regardless of
        # build sequence (multi-night stays are emitted before activities).
        return self._sorted_items(new_items)

    def _ranked_items(
        self,
        trip: Trip,
        preserved_items: Sequence[ItineraryItem],
    ) -> List[ItineraryItem]:
        if not trip.destination_id or not trip.destination:
            raise ItineraryGenerationError("Trip must reference a valid catalog destination")

        recommendations = RecommendationEngine(self.db).get_recommendations(
            trip.destination_id,
            self._preferences(trip),
            discovery_session_id=trip.discovery_session_id,
        )
        hotels = self._ranked_catalog(Hotel, recommendations["recommended_hotels"], trip)
        activities = self._ranked_catalog(Activity, recommendations["recommended_activities"], trip)
        transport_options = [
            option for option in self._ranked_catalog(
                TransportOption,
                recommendations["recommended_transport"],
                trip,
            )
            if option.capacity >= self._traveler_count(trip)
        ]

        duration_days = self._duration_days(trip)
        traveler_count = self._traveler_count(trip)
        remaining_budget = self._remaining_budget(trip, preserved_items)
        preserved_transport_ids = {item.transport_id for item in preserved_items if item.transport_id}
        preserved_activity_ids = {item.activity_id for item in preserved_items if item.activity_id}

        activity_slots = max(0, self._activity_slots(duration_days, trip.pace) - len(preserved_activity_ids))
        required_activity_days = max(0, duration_days - 1 - len(preserved_activity_ids))
        # Budget-honest selection: the hotel must leave room for transfers
        # + required activities, otherwise trips silently blow past the
        # traveler's budget (hotel eating 96% was a real case). Minimums are
        # computed from the cheapest available options.
        min_transport_cost = (
            min((float(o.price or 0) for o in transport_options), default=0.0)
            if (not preserved_transport_ids and transport_options) else 0.0
        )
        cheapest_activity_costs = sorted(
            float(a.price_per_person or 0) * traveler_count for a in activities
        )
        min_activities_cost = sum(cheapest_activity_costs[:required_activity_days])

        hotel_nights = max(1, duration_days - 1)
        # NOTE: hotel economics keep the legacy order: a base stay is
        # reserved here (before transfers/activities pick), so activity
        # selection sees exactly the remainder it always has. Proximity
        # planning later spends within that same reserved envelope, and any
        # unspent part is refunded — trip totals can only match or beat the
        # previous single-hotel totals. The minimums below only feed the
        # transport reservation.
        hotel_pot = (remaining_budget - min_transport_cost - min_activities_cost
                     if remaining_budget is not None else None)
        pinned_days = {
            item.day_number
            for item in preserved_items
            if item.item_type == "hotel" and 1 <= item.day_number <= hotel_nights
        }
        open_nights = [n for n in range(1, hotel_nights + 1) if n not in pinned_days]
        base_hotel, base_total = self._reserve_base_hotel(
            hotels, open_nights, hotel_pot
        )
        remaining_budget = self._subtract_budget(remaining_budget, base_total)

        transport = None
        if not preserved_transport_ids and transport_options:
            # Transfers are optional: skip when the destination has no
            # transport inventory (or none fits the budget) instead of
            # failing trip generation. Selection leaves the reserved
            # activity minimum untouched.
            transport_pot = (remaining_budget - min_activities_cost
                             if remaining_budget is not None else None)
            transport = self._first_within_budget(
                transport_options,
                lambda candidate: float(candidate.price or 0),
                transport_pot,
            )
            if transport:
                remaining_budget = self._subtract_budget(remaining_budget, float(transport.price or 0))

        if activity_slots > 0 and not activities:
            raise ItineraryGenerationError("No active catalog activities are available for this destination and currency")
        selected_activities = []
        selected_activity_ids = set(preserved_activity_ids)
        priced = []
        seen_ids = set(selected_activity_ids)
        for a in activities:
            if a.id in seen_ids:
                continue
            seen_ids.add(a.id)
            priced.append((a, float(a.price_per_person or 0) * traveler_count))
        # Geographic coherence gate: individually relevant activities can be
        # collectively infeasible (more clusters than usable days). Admit
        # whole clusters in ranked order so the trip prefers a coherent
        # subset over maximizing unrelated stops. The required minimum is
        # always backfilled (recorded), and slot caps still apply below.
        coherent_pool, coherence_notes = self._coherent_subset(
            priced, required_activity_days, duration_days
        )
        priced = coherent_pool
        # Affordable in ranked order first (preference match wins). Only to
        # reach the required minimum, the cheapest remaining ones follow —
        # never an expensive forced pick that blows the budget.
        affordable = [(a, c) for a, c in priced
                      if remaining_budget is None or c <= remaining_budget]
        pricey_ids = {a.id for a, c in priced
                      if remaining_budget is not None and c > remaining_budget}
        pricey = sorted(((a, c) for a, c in priced if a.id in pricey_ids),
                        key=lambda t: t[1])
        for activity, cost in affordable + pricey:
            if len(selected_activities) >= activity_slots:
                break
            if activity.id in pricey_ids:
                if len(selected_activities) >= required_activity_days:
                    continue
            elif remaining_budget is not None and cost > remaining_budget:
                continue
            selected_activities.append(activity)
            selected_activity_ids.add(activity.id)
            remaining_budget = self._subtract_budget(remaining_budget, cost)
        if len(selected_activities) < required_activity_days:
            raise ItineraryGenerationError("Not enough distinct active catalog activities fit the requested duration and budget")

        occupied_orders = {(item.day_number, item.order_index) for item in preserved_items}
        # New stops also occupy orders so same-day items never collide.
        used_orders = set(occupied_orders)
        new_items: List[ItineraryItem] = []
        if transport:
            new_items.append(
                ItineraryItem(
                    trip_id=trip.id,
                    day_number=1,
                    order_index=self._available_order(1, 1, used_orders),
                    item_type="transport",
                    title=transport.name,
                    description=self._transport_description(transport),
                    start_time="09:30 AM",
                    end_time="01:00 PM",
                    cost=float(transport.price or 0),
                    status="proposed",
                    transport_id=transport.id,
                    location=transport.route_to,
                    meta_data={"ui": self._entity_ui_meta(transport)},
                )
            )

        # Day assignment is proximity-driven (deterministic geographic
        # grouping); times are duration-aware (see _route_placements).
        route_plan = plan_activity_days(
            selected_activities, duration_days, pace=trip.pace
        )
        # Trip-level warnings channel: selection-coherence notes plus any
        # plan-level warnings serialize as `itinerary_warnings` so nothing
        # excluded is ever silent. Transient (same-process responses carry
        # it; fresh loads default to []).
        try:
            setattr(trip, "_itinerary_warnings",
                    list(coherence_notes) + list(route_plan.warnings or []))
        except Exception:
            pass
        # Regional hotel top-up: discovery runs once per destination, so
        # multi-cluster itineraries otherwise see a single base hotel.
        # Live provider results around uncovered clusters are persisted as
        # ordinary "live" rows and appended to the ranked candidates below;
        # ranking, budget allowance, and proximity assignment are untouched.
        # Fail-safe: provider trouble keeps the existing candidates.
        hotels = self._top_up_regional_hotels(
            trip, hotels, route_plan, selected_activities, traveler_count
        )
        day_blockers = self._day_blockers(transport, preserved_items, hotel_nights)
        try:
            max_per_day = max(
                1, int(resolve_pace(trip.pace)["max_activities_per_day"])
            )
        except (TypeError, ValueError, KeyError):
            max_per_day = 4
        placements, unrestored_ids, placement_warnings = self._route_placements(
            route_plan, selected_activities, day_blockers, max_per_day
        )

        # True hotel allowance: base_total was reserved up front, so add it
        # back to the live remainder. Spending within this allowance keeps
        # the trip total budget-honest while allowing pricier nearby stays.
        hotel_allowance = (
            remaining_budget + base_total
            if remaining_budget is not None
            else None
        )
        stays, hotel_spent, night_regions = self._plan_stays(
            trip, hotels, hotel_nights, base_hotel, base_total,
            hotel_allowance, preserved_items, placements
        )
        # Global repair loop (plan → evaluate → repair → recalculate):
        # per-day planning above never sees the actual stay plan, so an
        # activity can still sit a brutal hotel-leg away from its day's
        # stays. Relocate such activities to feasible days, re-time, and
        # re-run stay assignment on the repaired geography. Bounded and
        # strictly improving; leftovers keep honest warnings.
        placements, stays, night_regions, unrestored_ids, placement_warnings, hotel_spent = \
            self._repair_global_route(
                placements, stays, night_regions,
                route_plan, selected_activities, day_blockers, max_per_day,
                trip, hotels, hotel_nights, base_hotel, base_total,
                hotel_allowance, preserved_items, unrestored_ids,
                placement_warnings, hotel_spent,
            )
        # Base total was reserved up front; only the difference between the
        # proximity plan's actual spend and the reservation moves the budget.
        # hotel_spent always reflects the FINAL stay plan above.
        remaining_budget = self._subtract_budget(
            remaining_budget, hotel_spent - base_total
        )
        covered_nights = {
            item.day_number
            for item in preserved_items
            if item.item_type == "hotel" and 1 <= item.day_number <= hotel_nights
        }
        for stay in stays:
            if stay.night in covered_nights or stay.hotel is None:
                continue
            hotel_ui = self._entity_ui_meta(stay.hotel)
            hotel_ui["hotel_assignment_reason"] = stay.reason
            # Accommodation is not sightseeing: it sorts after the day's
            # activities as the overnight stay and carries an explicit
            # flag so renderers never mistake it for a timed activity.
            # The authoritative overnight region rides along so every
            # downstream view derives from this same stay plan.
            hotel_ui["is_accommodation"] = True
            if night_regions.get(stay.night) is not None:
                hotel_ui["overnight_region"] = night_regions[stay.night]
            # Compact per-night candidate audit (debugging/testing): who
            # was considered, what won, and why each rejected candidate
            # lost — nearest rejections first, capped. Full detail lives
            # on OvernightStay.evaluations for unit tests.
            stay_evals = [dict(e) for e in
                          (getattr(stay, "evaluations", None) or [])]

            def _audit_rank(entry: Dict[str, Any]) -> Tuple[float, float, str]:
                dist = entry.get("distance_km")
                if dist is None:
                    return (1.0, float("inf"), str(entry.get("hotel_id") or ""))
                try:
                    return (0.0, float(dist), str(entry.get("hotel_id") or ""))
                except (TypeError, ValueError):
                    return (1.0, float("inf"), str(entry.get("hotel_id") or ""))

            stay_evals.sort(key=_audit_rank)
            rejected = [
                {"hotel_id": e.get("hotel_id"), "name": e.get("name"),
                 "distance_km": e.get("distance_km"),
                 "travel_minutes": e.get("travel_minutes"),
                 "reason": e.get("decision")}
                for e in stay_evals[:5]
            ]
            hotel_ui["stay_audit"] = {
                "night": stay.night,
                "decision": getattr(stay, "decision", "") or "",
                "candidates_considered": len(stay_evals) + 1,
                "rejected": rejected,
                "rejected_truncated": max(0, len(stay_evals) - len(rejected)),
            }
            new_items.append(
                ItineraryItem(
                    trip_id=trip.id,
                    day_number=stay.night,
                    order_index=self._available_order(stay.night, 50, used_orders),
                    item_type="hotel",
                    title=stay.hotel.name,
                    description=stay.hotel.description,
                    start_time=None,
                    end_time=None,
                    cost=float(stay.hotel.price_per_night or 0),
                    status="proposed",
                    hotel_id=stay.hotel.id,
                    location=stay.hotel.address or stay.hotel.name,
                    meta_data={"ui": hotel_ui},
                )
            )

        # Proportional distribution: activities spread across the whole
        # trip, up to 2 per day (morning + afternoon). Thin inventories
        # yield ~1/day; rich ones fill both daily slots.
        for activity, day_number, preferred_order, start_time in placements:
            new_items.append(
                ItineraryItem(
                    trip_id=trip.id,
                    day_number=day_number,
                    order_index=self._available_order(day_number, preferred_order, used_orders),
                    item_type="activity",
                    title=activity.title,
                    description=activity.description,
                    start_time=start_time,
                    end_time=self._end_time(start_time, activity.duration_hours),
                    cost=float(activity.price_per_person or 0) * traveler_count,
                    status="proposed",
                    activity_id=activity.id,
                    location=activity.meeting_point or trip.destination.name,
                    meta_data={"ui": self._entity_ui_meta(activity)},
                )
            )
        departure_note = self._departure_note(
            trip, preserved_items, new_items,
            [stay.hotel for stay in stays], transport,
        )
        if departure_note is not None:
            departure_note.order_index = self._available_order(
                departure_note.day_number, departure_note.order_index, occupied_orders
            )
            new_items.append(departure_note)
        self._attach_route_metadata(
            new_items, placements, stays, preserved_items, trip, route_plan,
            placement_warnings, unrestored_ids, night_regions,
        )
        return new_items

    def _reserve_base_hotel(
        self,
        hotels: Sequence[Any],
        open_nights: Sequence[int],
        hotel_pot: Optional[float],
    ) -> Tuple[Optional[Any], float]:
        """Legacy whole-trip reservation: best-ranked hotel whose nightly
        rate fits every uncovered night. Same selection and errors as the
        previous single-hotel logic; the returned total is reserved up
        front and proximity planning later spends within it."""
        if not open_nights:
            return None, 0.0
        if not hotels:
            raise ItineraryGenerationError("No active catalog hotel is available for this destination and currency")
        for hotel in hotels:
            try:
                price = float(hotel.price_per_night or 0)
            except (TypeError, ValueError):
                continue
            if hotel_pot is None or price * len(open_nights) <= hotel_pot:
                return hotel, price * len(open_nights)
        raise ItineraryGenerationError("No active catalog hotel fits the trip budget")

    @staticmethod
    def _located_activities(activities: Sequence[Any]) -> List[Any]:
        """Activities with usable coordinates (validated range, no NaN).

        Never invents or repairs coordinates; invalid entries are skipped
        so they cannot poison a centroid mean.
        """
        located = []
        for activity in activities or []:
            try:
                lat = float(getattr(activity, "latitude", None))
                lng = float(getattr(activity, "longitude", None))
            except (TypeError, ValueError):
                continue
            if (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0
                    and lat == lat and lng == lng):
                located.append(activity)
        return located

    @staticmethod
    def _overnight_anchor(
        day_acts_next: Sequence[Any],
        day_acts_same: Sequence[Any],
        dest_name: str,
        dest_point: Tuple[Any, Any],
        night: int,
    ) -> Tuple[Optional[DayAnchor], Dict[str, Any]]:
        """Build the overnight anchor for one night with diagnostics.

        Source priority is unchanged: next morning's located activities,
        else tonight's, else the destination point. A plain centroid is
        only accepted when every source activity sits within
        ``significant_distance_km`` of it; otherwise the anchor falls back
        to the centroid of the largest same-region subgroup (deterministic:
        most members, ties broken by earliest day order), so the hotel
        search optimizes for a real activity cluster rather than an
        artificial midpoint between distant cities. Returns
        (anchor-or-None, diagnostics dict for logging only).
        """
        try:
            spread_limit = float(
                HOTEL_ASSIGNMENT_CONFIG.get("significant_distance_km", 60.0))
        except (TypeError, ValueError):
            spread_limit = 60.0
        source = (ItineraryGenerator._located_activities(day_acts_next)
                  or ItineraryGenerator._located_activities(day_acts_same))
        diagnostics: Dict[str, Any] = {
            "night": night,
            "method": "none",
            "titles": [],
            "coordinates": [],
            "centroid": None,
            "activity_distances_km": [],
            "max_spread_km": None,
        }
        if not source:
            anchor = None
            if dest_point[0] is not None and dest_point[1] is not None:
                try:
                    anchor = DayAnchor(
                        latitude=float(dest_point[0]),
                        longitude=float(dest_point[1]),
                        label=dest_name,
                    )
                    diagnostics["method"] = "destination-fallback"
                    diagnostics["centroid"] = (
                        anchor.latitude, anchor.longitude)
                except (TypeError, ValueError):
                    anchor = None
            logger.debug("overnight anchor: %r", diagnostics)
            return anchor, diagnostics
        points = [(a.latitude, a.longitude) for a in source]
        center = centroid(points)
        diagnostics["titles"] = [
            str(getattr(a, "title", None) or "") for a in source]
        diagnostics["coordinates"] = [
            (float(a.latitude), float(a.longitude)) for a in source]
        diagnostics["centroid"] = center
        per_act = []
        for lat, lng in points:
            dist = haversine_km(center[0], center[1], lat, lng) \
                if center is not None else None
            per_act.append(round(dist, 1) if dist is not None else None)
        diagnostics["activity_distances_km"] = per_act
        known = [d for d in per_act if d is not None]
        diagnostics["max_spread_km"] = max(known) if known else None
        label = str(getattr(source[0], "title", None) or dest_name)
        if len(source) > 1:
            label = f"{label} (+{len(source) - 1} more)"
        if (center is not None
                and diagnostics["max_spread_km"] is not None
                and diagnostics["max_spread_km"] <= spread_limit):
            diagnostics["method"] = "centroid"
            logger.debug("overnight anchor: %r", diagnostics)
            return DayAnchor(latitude=center[0], longitude=center[1],
                             label=label), diagnostics
        # Scattered day: anchor the largest regional subgroup instead of
        # a midpoint that serves nobody.
        ordered = [((idx, 0), (float(a.latitude), float(a.longitude)))
                   for idx, a in enumerate(source)]
        try:
            subgroups = cluster_anchor_points(ordered)
        except Exception:
            subgroups = {}
        by_region: Dict[Any, List[int]] = {}
        for idx in range(len(source)):
            by_region.setdefault(subgroups.get((idx, 0), idx), []).append(idx)
        best = max(by_region.values(),
                   key=lambda idxs: (len(idxs), -min(idxs)))
        kept = [source[idx] for idx in sorted(best)]
        fallback_center = centroid(
            [(a.latitude, a.longitude) for a in kept])
        if fallback_center is None:
            diagnostics["method"] = "centroid-unreliable"
            logger.debug("overnight anchor: %r", diagnostics)
            return (DayAnchor(latitude=center[0], longitude=center[1],
                              label=label)
                    if center is not None else None), diagnostics
        diagnostics["method"] = "subgroup-fallback"
        diagnostics["centroid"] = fallback_center
        diagnostics["kept_titles"] = [
            str(getattr(a, "title", None) or "") for a in kept]
        sub_label = str(getattr(kept[0], "title", None) or dest_name)
        if len(source) > 1:
            sub_label = f"{sub_label} (+{len(source) - 1} more)"
        logger.debug("overnight anchor: %r", diagnostics)
        return DayAnchor(latitude=fallback_center[0],
                         longitude=fallback_center[1],
                         label=sub_label), diagnostics

    def _plan_stays(
        self,
        trip: Trip,
        hotels: Sequence[Any],
        hotel_nights: int,
        base_hotel: Optional[Any],
        base_total: float,
        hotel_allowance: Optional[float],
        preserved_items: Sequence[ItineraryItem],
        placements: Sequence[Tuple[Any, int, int, str]],
    ) -> Tuple[List[OvernightStay], float, Dict[int, int]]:
        """Plan one hotel per overnight stay using proximity logic.

        Nights already covered by preserved hotel items (including
        traveler-confirmed picks and live selections) are pinned and never
        reassigned. Newly assigned nights spend within ``hotel_allowance``
        (the true post-transfer/activity hotel budget), so trip totals stay
        budget-honest; preserved nights cost nothing extra here because
        their cost is already counted upstream. Also returns the
        authoritative night→region map backing every stay decision.
        """
        nights = max(1, int(hotel_nights or 1))
        pinned: Dict[int, Any] = {}
        for item in preserved_items:
            if item.item_type != "hotel" or not 1 <= item.day_number <= nights:
                continue
            if item.hotel_id:
                hotel_row = self.db.query(Hotel).filter(Hotel.id == item.hotel_id).first()
                if hotel_row is not None:
                    pinned[item.day_number] = hotel_row
                    continue
            ui = (item.meta_data or {}).get("ui", {}) or {}
            pinned[item.day_number] = SimpleNamespace(
                id=f"custom:{item.id}",
                name=item.title,
                description=item.description,
                address=item.location,
                latitude=ui.get("latitude"),
                longitude=ui.get("longitude"),
                price_per_night=0.0,
            )

        activities_by_day: Dict[int, List[Any]] = {}
        for activity, day_number, _, _ in placements:
            activities_by_day.setdefault(day_number, []).append(activity)
        preserved_activity_ids = {
            item.activity_id
            for item in preserved_items
            if item.item_type == "activity" and item.activity_id
        }
        if preserved_activity_ids:
            for row in (
                self.db.query(Activity)
                .filter(Activity.id.in_(sorted(preserved_activity_ids)))
                .all()
            ):
                day = next(
                    (
                        item.day_number
                        for item in preserved_items
                        if item.item_type == "activity" and item.activity_id == row.id
                    ),
                    None,
                )
                if day is not None:
                    activities_by_day.setdefault(day, []).append(row)

        dest_name = trip.destination.name if trip.destination else "the trip area"
        dest_point = (
            (trip.destination.latitude, trip.destination.longitude)
            if trip.destination
            else (None, None)
        )
        anchors: Dict[int, DayAnchor] = {}
        for night in range(1, nights + 1):
            anchor, _diagnostics = self._overnight_anchor(
                activities_by_day.get(night + 1, []),
                activities_by_day.get(night, []),
                dest_name,
                dest_point,
                night,
            )
            if anchor is not None:
                anchors[night] = anchor

        needs_assignment = any(
            night not in pinned for night in range(1, nights + 1)
        )
        if needs_assignment and base_hotel is None and not hotels:
            raise ItineraryGenerationError("No active catalog hotel is available for this destination and currency")
        night_regions = self._night_regions(activities_by_day, nights)
        stays, spent = plan_overnight_stays(
            nights=nights,
            anchors=anchors,
            hotels=hotels,
            pinned_hotels=pinned,
            total_pot=hotel_allowance,
            night_regions=night_regions,
            arrival_point=dest_point,
        )
        uncovered = [
            stay for stay in stays
            if stay.night not in pinned and stay.hotel is None
        ]
        if uncovered:
            if base_hotel is None:
                raise ItineraryGenerationError("No active catalog hotel fits the trip budget")
            # Proximity moves left a later night unaffordable: keep one
            # retained base stay (exactly the reserved total) instead of
            # failing the trip.
            try:
                nightly = float(base_hotel.price_per_night or 0)
            except (TypeError, ValueError):
                nightly = 0.0
            rebuilt: List[OvernightStay] = []
            for stay in stays:
                if stay.night in pinned:
                    rebuilt.append(stay)
                    continue
                rebuilt.append(
                    OvernightStay(
                        night=stay.night,
                        hotel=base_hotel,
                        reason=(
                            f"Keeping one base stay at {base_hotel.name} to hold "
                            "the trip budget."
                        ),
                        distance_km=stay.distance_km,
                        travel_minutes=stay.travel_minutes,
                        retained=True,
                    )
                )
            open_count = sum(1 for stay in stays if stay.night not in pinned)
            stays, spent = rebuilt, nightly * open_count
        for stay in stays:
            if stay.night not in pinned and stay.hotel is None:
                raise ItineraryGenerationError("No active catalog hotel fits the trip budget")
        return stays, spent, night_regions

    def _top_up_regional_hotels(
        self,
        trip: Trip,
        hotels: Sequence[Any],
        route_plan: RoutePlan,
        selected_activities: Sequence[Any],
        traveler_count: int,
    ) -> List[Any]:
        """Append live hotels discovered around uncovered clusters.

        Discovery runs once per destination, so multi-cluster itineraries
        otherwise choose from a single base hotel. This fail-safe step
        searches the existing live provider around each significant
        cluster centroid lacking nearby coverage and persists ordinary
        "live" rows (same gates as live fill); ranking, budget allowance,
        and proximity assignment downstream are untouched. Any provider
        trouble keeps the existing candidates.
        """
        try:
            from backend.live_fill.regional_hotels import discover_regional_hotels
            from backend.live_fill.service import _search_window
        except Exception:
            return list(hotels)
        try:
            if trip.destination is None:
                return list(hotels)
            regions = self._search_regions(route_plan, selected_activities)
            if not regions:
                return list(hotels)
            check_in, check_out = _search_window(
                trip.start_date, trip.end_date, self._duration_days(trip)
            )
            report = discover_regional_hotels(
                self.db,
                destination=trip.destination,
                currency=trip.currency or "INR",
                traveler_count=traveler_count,
                check_in=check_in,
                check_out=check_out,
                region_centroids=regions,
                existing_hotels=list(hotels),
            )
            added = report.get("added") or []
            # Previously topped-up live rows for this destination stay
            # visible too (regenerates must see the same universe, not
            # just freshly searched rows). Destination-scoped, active,
            # currency-matched — no cross-destination mixing.
            live_rows = (
                self.db.query(Hotel)
                .filter(
                    Hotel.destination_id == trip.destination.id,
                    Hotel.inventory_source == "live",
                    Hotel.is_active.is_(True),
                )
                .all()
            )
            if trip.currency:
                live_rows = [
                    h for h in live_rows if h.currency == trip.currency
                ]
            seen = {getattr(h, "id", None) for h in hotels}
            extra = []
            for row in list(added) + list(live_rows):
                if getattr(row, "id", None) not in seen:
                    seen.add(getattr(row, "id", None))
                    extra.append(row)
            if not extra:
                return list(hotels)
            return [*hotels, *extra]
        except Exception:
            return list(hotels)

    @staticmethod
    def _search_regions(
        route_plan: RoutePlan,
        selected_activities: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """One search centroid per planned sightseeing day.

        Planned days are proximity groups, so each day centroid is already
        a meaningful cluster anchor (town-level reverse-geocoding resolves
        much better for "Hampi" than for a merged multi-stop centroid).
        Derived only from selected-activity coordinates — no
        destination-specific logic. Returns [{"latitude", "longitude",
        "label", "activity_minutes"}].
        """
        by_id: Dict[str, Any] = {}
        for activity in selected_activities or []:
            aid = str(getattr(activity, "id", "") or "")
            if aid and aid not in by_id:
                by_id[aid] = activity
        regions: List[Dict[str, Any]] = []
        for day in sorted(getattr(route_plan, "days", {}) or {}):
            pts: List[Tuple[float, float]] = []
            minutes = 0.0
            first_title = ""
            for aid in route_plan.days[day].activity_ids or []:
                activity = by_id.get(str(aid))
                if activity is None:
                    continue
                try:
                    lat = float(getattr(activity, "latitude", None))
                    lng = float(getattr(activity, "longitude", None))
                except (TypeError, ValueError):
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                    continue
                pts.append((lat, lng))
                try:
                    minutes += max(0.0, float(
                        getattr(activity, "duration_hours", 0) or 0)) * 60.0
                except (TypeError, ValueError):
                    pass
                if not first_title:
                    first_title = str(
                        getattr(activity, "title", None)
                        or getattr(activity, "name", None) or "")
            if not pts:
                continue
            center = centroid(pts)
            if center is None:
                continue
            regions.append({
                "latitude": center[0],
                "longitude": center[1],
                "label": f"{first_title} area" if first_title else "the area",
                "activity_minutes": round(minutes, 1),
            })
        return regions

    @staticmethod
    def _coherent_subset(
        priced: Sequence[Tuple[Any, float]],
        required_min: int,
        duration_days: int,
    ) -> Tuple[List[Tuple[Any, float]], List[str]]:
        """Filter ranked candidates to a geographically coherent subset.

        Locatable activities are chained into coordinate-driven regions
        (shared ``region_link_km`` scale — no destination logic); whole
        regions are admitted in ranked order while the region count fits
        the usable days, since each region needs at least one sightseeing
        day. Activities without coordinates are exempt (kept, warned
        downstream) rather than judged. If coherence would breach the
        required minimum, the cheapest excluded stops are backfilled and
        recorded. Returns (kept in ranked order, notes). Never raises.
        """
        try:
            link_km = float(
                HOTEL_ASSIGNMENT_CONFIG.get("region_link_km", 120.0))
        except (TypeError, ValueError):
            link_km = 120.0
        try:
            usable_days = max(1, int(duration_days or 1))
        except (TypeError, ValueError):
            usable_days = 1
        try:
            need_min = max(0, int(required_min or 0))
        except (TypeError, ValueError):
            need_min = 0
        entries = list(priced or [])
        if not entries:
            return [], []
        locatable_idx: List[int] = []
        points: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []
        for pos, (activity, _cost) in enumerate(entries):
            try:
                lat = float(getattr(activity, "latitude", None))
                lng = float(getattr(activity, "longitude", None))
            except (TypeError, ValueError):
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                continue
            if lat != lat or lng != lng:  # NaN guard without imports
                continue
            locatable_idx.append(pos)
            points.append(((pos, 0), (lat, lng)))
        try:
            key_regions = cluster_anchor_points(points, link_km=link_km)
        except Exception:
            return entries, []
        region_of_pos: Dict[int, int] = {}
        for (pos, _), region in key_regions.items():
            region_of_pos[pos] = region
        # Order regions by best (lowest) ranked position; deterministic.
        region_best: Dict[int, int] = {}
        for pos in locatable_idx:
            region = region_of_pos.get(pos)
            if region is None:
                continue
            if region not in region_best or pos < region_best[region]:
                region_best[region] = pos
        admitted_regions = set(
            sorted(region_best, key=lambda r: region_best[r])[:usable_days]
        )
        locatable = set(locatable_idx)
        kept: List[Tuple[Any, float]] = []
        dropped: List[Tuple[Any, float]] = []
        dropped_positions: List[int] = []
        for pos, entry in enumerate(entries):
            if pos not in locatable:
                kept.append(entry)  # exempt: no coordinates to judge
                continue
            region = region_of_pos.get(pos)
            if region is None or region in admitted_regions:
                kept.append(entry)
            else:
                dropped.append(entry)
                dropped_positions.append(pos)
        notes: List[str] = []
        if dropped:
            notes.append(
                f"{len(dropped)} relevant stop(s) were not selected: their "
                f"region(s) do not fit {usable_days} usable day(s) "
                "(geographic coherence preferred over unrelated stops)."
            )
        # Minimum viability beats coherence: backfill cheapest excluded.
        if len(kept) < need_min and dropped:
            by_price = sorted(dropped, key=lambda e: (e[1], str(e[0])))
            while len(kept) < need_min and by_price:
                activity, cost = by_price.pop(0)
                kept.append((activity, cost))
                notes.append(
                    f"{getattr(activity, 'title', activity)} kept for minimum "
                    "trip viability despite its region; routing warnings apply."
                )
            # Restore ranked order after backfill.
            order = {id(e[0]): i for i, e in enumerate(entries)}
            kept.sort(key=lambda e: order.get(id(e[0]), len(entries)))
        return kept, notes

    @staticmethod
    def _night_regions(
        activities_by_day: Mapping[int, Sequence[Any]],
        nights: int,
    ) -> Dict[int, int]:
        """Travel-region id per overnight night from placed activities.

        All placed sightseeing stops are chained into coordinate-driven
        regions (see ``cluster_anchor_points``); each night inherits the
        majority region of its anchor source days (next morning's
        activities, falling back to the same day's), so consecutive nights
        serving one travel region share an id. Nights without any located
        activity are left out (legacy per-night evaluation). Deterministic;
        coordinates are never invented.
        """
        ordered: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []
        act_region_key: Dict[str, int] = {}
        position = 0
        for day in sorted(activities_by_day):
            for activity in activities_by_day[day] or []:
                try:
                    lat = float(getattr(activity, "latitude", None))
                    lng = float(getattr(activity, "longitude", None))
                except (TypeError, ValueError):
                    continue
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                    continue
                key = (day, position)
                position += 1
                ordered.append((key, (lat, lng)))
                aid = str(getattr(activity, "id", "") or "")
                if aid:
                    act_region_key.setdefault(aid, len(ordered) - 1)
        if not ordered:
            return {}
        key_regions = cluster_anchor_points(ordered)
        act_regions: Dict[str, int] = {}
        for aid, idx in act_region_key.items():
            region = key_regions.get(ordered[idx][0])
            if region is not None:
                act_regions[aid] = region
        night_regions: Dict[int, int] = {}
        for night in range(1, max(1, int(nights or 1)) + 1):
            # Same fallback as anchor construction: next morning's LOCATED
            # activities, else tonight's. An empty-looking day with only
            # unlocatable stops must not block the fallback (a non-empty
            # list of coord-less stops is still unusable for regions).
            located_by_day: Dict[int, List[Any]] = {}
            for day, acts in (activities_by_day or {}).items():
                located = []
                for activity in acts or []:
                    try:
                        lat = float(getattr(activity, "latitude", None))
                        lng = float(getattr(activity, "longitude", None))
                    except (TypeError, ValueError):
                        continue
                    if (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0
                            and lat == lat and lng == lng):
                        located.append(activity)
                located_by_day[day] = located
            source = (located_by_day.get(night + 1, []) or []
                      or located_by_day.get(night, []) or [])
            votes: Dict[int, int] = {}
            for activity in source:
                region = act_regions.get(str(getattr(activity, "id", "") or ""))
                if region is not None:
                    votes[region] = votes.get(region, 0) + 1
            if votes:
                night_regions[night] = min(votes, key=lambda r: (-votes[r], r))
        return night_regions

    @staticmethod
    def _route_slot(day_number: int, position: int) -> tuple[int, int, str]:
        """Start-time template for the position-th stop of a day.

        Same vocabulary as the legacy slot table (04:30 PM opener,
        09:00 AM / 03:00 PM cores, 06:00 PM evening stacking); only WHICH
        activity lands in each template changed (geographic order instead
        of rank order). Day 1 always has a first stop when activities were
        selected, so the historical opener is preserved.
        """
        if day_number == 1 and position == 0:
            return day_number, 3, "04:30 PM"
        if day_number == 1:
            evening = ["06:00 PM", "07:30 PM", "08:30 PM"]
            slot = position - 1
            return day_number, 4 + slot, evening[min(slot, 2)]
        if position == 0:
            return day_number, 1, "09:00 AM"
        if position == 1:
            return day_number, 2, "03:00 PM"
        return day_number, 3 + (position - 2), "06:00 PM"

    @staticmethod
    def _day_blockers(
        transport: Optional[Any],
        preserved_items: Sequence[ItineraryItem],
        hotel_nights: int,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """Fixed commitments per day that sightseeing must work around.

        New transfers always use the same clock times as emission below
        (09:30 AM–01:00 PM); preserved items contribute their stored times
        when parseable. Overnight hotel stays are deliberately NOT blockers:
        a hotel is where the traveler sleeps, not a midday appointment, so
        blocking 01:30–03:00 PM every day only shrinks sightseeing windows
        and pushes activities into the evening. Arrival-day check-in is
        covered by the later sightseeing floor. Unparseable times are
        skipped, never invented.
        """
        blockers: Dict[int, List[Tuple[int, int]]] = {}
        if transport is not None:
            blockers.setdefault(1, []).append((570, 780))
        for item in preserved_items or []:
            try:
                day = int(item.day_number)
            except (TypeError, ValueError):
                continue
            start = parse_clock_time(item.start_time)
            end = parse_clock_time(item.end_time)
            if start is not None and end is not None and end > start:
                blockers.setdefault(day, []).append((start, end))
        return blockers

    def _route_placements(
        self,
        route_plan: RoutePlan,
        selected_activities: Sequence[Any],
        blockers: Mapping[int, Sequence[Tuple[int, int]]],
        max_per_day: int,
    ) -> Tuple[
        List[Tuple[Any, int, int, str]],
        List[str],
        Dict[str, List[str]],
    ]:
        """Order each planned day geographically and stamp feasible times.

        Template floors keep the legacy vocabulary (arrival opener, morning
        / afternoon / evening slots); actual starts honor durations, travel
        estimates, and fixed blockers, with overflow relocated to another
        geographically feasible day. Relocation candidates are tried in
        order of increasing extra travel and must satisfy the one-way and
        daily travel budgets — a spilled stop is never glued onto a distant
        day just because time fits there. Returns (placements,
        unrestored_ids, placement_warnings); unrestored activities keep
        duration-aware, non-overlapping times past the day end and are
        flagged downstream — never dropped, shortened, or hidden.
        """
        from backend.itinerary.route_plan import ROUTE_PLAN_CONFIG

        try:
            day_start = int(float(ROUTE_PLAN_CONFIG.get("schedule_day_start_minutes", 540)))
            arrival_floor = int(float(ROUTE_PLAN_CONFIG.get("schedule_arrival_day_start_minutes", 990)))
            day_end = int(float(ROUTE_PLAN_CONFIG.get("schedule_day_end_minutes", 1350)))
            speed = float(ROUTE_PLAN_CONFIG.get("default_travel_speed_kmh", 50.0))
            buffer_min = float(ROUTE_PLAN_CONFIG.get("activity_travel_buffer_minutes", 30.0))
            max_one_way = float(ROUTE_PLAN_CONFIG.get("max_one_way_travel_minutes", 120.0))
            max_daily = float(ROUTE_PLAN_CONFIG.get("max_daily_travel_minutes", 240.0))
        except (TypeError, ValueError):
            day_start, arrival_floor, day_end, speed, buffer_min = 540, 990, 1350, 50.0, 30.0
            max_one_way, max_daily = 120.0, 240.0
        if speed <= 0:
            speed = 50.0
        try:
            per_day_cap = max(1, int(max_per_day))
        except (TypeError, ValueError):
            per_day_cap = 4
        by_id: Dict[str, Any] = {}
        meta: Dict[str, Tuple[Optional[Tuple[float, float]], Any]] = {}
        for activity in selected_activities:
            aid = str(getattr(activity, "id", ""))
            if aid and aid not in by_id:
                by_id[aid] = activity
                try:
                    lat = float(getattr(activity, "latitude", None))
                    lng = float(getattr(activity, "longitude", None))
                    point: Optional[Tuple[float, float]] = (lat, lng)
                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                        point = None
                except (TypeError, ValueError):
                    point = None
                meta[aid] = (point, getattr(activity, "duration_hours", 0))

        def template_floors(day: int, count: int) -> List[int]:
            floors: List[int] = []
            for position in range(max(0, count)):
                _, _, stamp = self._route_slot(day, position)
                parsed = parse_clock_time(stamp)
                floors.append(
                    parsed if parsed is not None
                    else (arrival_floor if day == 1 else day_start)
                )
            return floors

        def day_travel_for(aids: Sequence[str]) -> Optional[float]:
            total = 0.0
            legs = 0
            for first, second in zip(aids, aids[1:]):
                first_pt, second_pt = meta[first][0], meta[second][0]
                if first_pt is None or second_pt is None:
                    return None
                dist = haversine_km(
                    first_pt[0], first_pt[1], second_pt[0], second_pt[1]
                )
                minutes = estimate_travel_minutes(dist, speed)
                if minutes is None:
                    return None
                total += minutes
                legs += 1
            return round(total + buffer_min * legs, 1)

        def nearest_minutes(aid: str, aids: Sequence[str]) -> Optional[float]:
            point = meta[aid][0]
            if point is None:
                return None
            best: Optional[float] = None
            for other in aids:
                other_pt = meta[other][0]
                if other_pt is None:
                    return None
                minutes = estimate_travel_minutes(
                    haversine_km(point[0], point[1], other_pt[0], other_pt[1]),
                    speed,
                )
                if minutes is None:
                    return None
                if best is None or minutes < best:
                    best = minutes
            return best if best is not None else 0.0

        scheduled: Dict[int, List[Tuple[str, int, int]]] = {}
        spilled: List[Tuple[str, int]] = []
        for day in sorted(route_plan.days):
            aids = [a for a in route_plan.days[day].activity_ids if a in by_id]
            stops = [
                (aid, meta[aid][0], meta[aid][1]) for aid in aids
            ]
            done, spilled_here = schedule_day_stops(
                stops,
                template_floors(day, len(aids)),
                list(blockers.get(day, [])),
                day_start_floor=arrival_floor if day == 1 else day_start,
                day_end=day_end,
                speed_kmh=speed,
                buffer_min=buffer_min,
            )
            scheduled[day] = done
            spilled += [(aid, day) for aid in spilled_here]

        placement_warnings: Dict[str, List[str]] = {}

        def note(aid: str, warning: str) -> None:
            placement_warnings.setdefault(aid, [])
            if warning not in placement_warnings[aid]:
                placement_warnings[aid].append(warning)

        # Validate → repair: relocate spilled stops only onto days where
        # they are geographically plausible (one-way + daily budgets hold).
        # Candidates are tried nearest-first so regions stay consecutive.
        for aid, from_day in list(spilled):
            point = meta[aid][0]
            options: List[Tuple[float, int]] = []
            for target in sorted(scheduled):
                if target == from_day or len(scheduled[target]) >= per_day_cap:
                    continue
                trial_aids = [a for (a, _, _) in scheduled[target]] + [aid]
                travel = day_travel_for(trial_aids)
                near = nearest_minutes(aid, [a for (a, _, _) in scheduled[target]])
                if near is None:
                    continue
                if near > max_one_way:
                    continue
                if travel is not None and travel > max_daily:
                    continue
                options.append((near, target))
            options.sort()
            placed = False
            for _, target in options:
                trial_aids = [a for (a, _, _) in scheduled[target]] + [aid]
                trial = [
                    (a, meta[a][0], meta[a][1]) for a in trial_aids
                ]
                rescheduled, still_out = schedule_day_stops(
                    trial,
                    template_floors(target, len(trial_aids)),
                    list(blockers.get(target, [])),
                    day_start_floor=arrival_floor if target == 1 else day_start,
                    day_end=day_end,
                    speed_kmh=speed,
                    buffer_min=buffer_min,
                )
                if not still_out:
                    scheduled[target] = rescheduled
                    placed = True
                    break
            if placed:
                spilled = [(a, d) for (a, d) in spilled if a != aid]

        # Swap pass: exchange an unrestored long stop with a short scheduled
        # stop elsewhere when both fit better swapped (e.g. a 7h hike moves
        # to a morning, a short stroll takes the evening). Same travel
        # budgets and caps as the move pass; counts per day never change.
        # Trial days are ordered longest-first so the long stop actually
        # lands on a morning floor instead of inheriting another evening.
        def _longest_first(aids):
            def _dur(a):
                try:
                    return -float(meta[a][1] or 0)
                except (TypeError, ValueError):
                    return 0.0
            return sorted(aids, key=_dur)

        for aid, from_day in list(spilled):
            if aid not in [a for (a, d) in spilled]:
                continue
            try:
                need = float(meta[aid][1] or 0)
            except (TypeError, ValueError):
                continue
            swapped = False
            for target in sorted(scheduled):
                if target == from_day:
                    continue
                for (cand, _, _) in list(scheduled[target]):
                    try:
                        cand_need = float(meta[cand][1] or 0)
                    except (TypeError, ValueError):
                        cand_need = 0.0
                    if cand_need >= need:
                        continue  # only trade down in duration
                    trial_target = _longest_first(
                        [a for (a, _, _) in scheduled[target] if a != cand] + [aid])
                    near = nearest_minutes(aid, [a for (a, _, _) in scheduled[target] if a != cand])
                    if near is None or near > max_one_way:
                        continue
                    if day_travel_for(trial_target) is not None and day_travel_for(trial_target) > max_daily:
                        continue
                    resched_t, out_t = schedule_day_stops(
                        [(a, meta[a][0], meta[a][1]) for a in trial_target],
                        template_floors(target, len(trial_target)),
                        list(blockers.get(target, [])),
                        day_start_floor=arrival_floor if target == 1 else day_start,
                        day_end=day_end, speed_kmh=speed, buffer_min=buffer_min)
                    if out_t:
                        continue
                    trial_from = _longest_first(
                        [a for (a, _, _) in scheduled[from_day] if a != aid] + [cand])
                    resched_f, out_f = schedule_day_stops(
                        [(a, meta[a][0], meta[a][1]) for a in trial_from],
                        template_floors(from_day, len(trial_from)),
                        list(blockers.get(from_day, [])),
                        day_start_floor=arrival_floor if from_day == 1 else day_start,
                        day_end=day_end, speed_kmh=speed, buffer_min=buffer_min)
                    if out_f:
                        continue
                    scheduled[target] = resched_t
                    scheduled[from_day] = resched_f
                    spilled = [(a, d) for (a, d) in spilled if a != aid]
                    swapped = True
                    break
                if swapped:
                    break

        # Anything still spilled keeps its original day with honest,
        # duration-aware times placed after the day's last stop (never
        # overlapping) and an overtime warning — never dropped.
        unrestored: List[str] = []
        for aid, from_day in spilled:
            if aid in unrestored:
                continue
            unrestored.append(aid)
            try:
                need = max(60, int(round(float(meta[aid][1] or 0) * 60)))
            except (TypeError, ValueError):
                need = 60
            day_sched = scheduled.setdefault(from_day, [])
            last_end = max(
                [end for (_, _, end) in day_sched] or
                [arrival_floor if from_day == 1 else day_start]
            )
            try:
                pace_buffer = max(0, int(round(buffer_min)))
            except (TypeError, ValueError):
                pace_buffer = 0
            start = last_end + pace_buffer
            for block_start, block_end in sorted(blockers.get(from_day, [])):
                if start < block_end and block_start < start + need:
                    start = block_end
            end = start + need
            # Persisted clock times must stay representable: clamp the start
            # inside the day (the final validation pass will flag any
            # resulting clash as Requires-Revision) instead of falling back
            # to an unrelated template stamp that would overlap silently.
            if start >= 24 * 60:
                start = 24 * 60 - 1
                end = start + need
            day_sched.append((aid, start, end))
            start_text = format_clock_time(start) or "day end"
            overtime_note = (
                f"{getattr(by_id.get(aid), 'title', aid)} on Day {from_day} "
                f"runs {start_text} past the usual sightseeing hours — "
                "the day is too full, review the schedule."
            )
            if end >= 24 * 60:
                overtime_note += " It ends after midnight."
            note(aid, overtime_note)

        # Final validation: activities on the same day must never overlap.
        # The scheduler guarantees this by construction; this pass keeps
        # the generate → validate → repair loop honest.
        for day in sorted(scheduled):
            ordered_by_start = sorted(scheduled[day], key=lambda t: (t[1], t[2]))
            for (first_aid, _, first_end), (second_aid, second_start, _) in zip(
                ordered_by_start, ordered_by_start[1:]
            ):
                if second_start < first_end:
                    note(
                        second_aid,
                        f"{getattr(by_id.get(second_aid), 'title', second_aid)} "
                        f"overlaps another stop on Day {day} — times clash, "
                        "review the schedule.",
                    )
            scheduled[day] = ordered_by_start

        placements: List[Tuple[Any, int, int, str]] = []
        for day in sorted(scheduled):
            for position, (aid, start, _end) in enumerate(scheduled[day]):
                _, order_index, _ = self._route_slot(day, position)
                start_text = format_clock_time(start)
                if start_text is None:  # pragma: no cover - scheduler bounds output
                    _, _, stamp = self._route_slot(day, position)
                    start_text = stamp
                placements.append((by_id[aid], day, order_index, start_text))
        return placements, unrestored, placement_warnings

        placements: List[Tuple[Any, int, int, str]] = []
        for day in sorted(scheduled):
            for position, (aid, start, _end) in enumerate(scheduled[day]):
                _, order_index, _ = self._route_slot(day, position)
                start_text = format_clock_time(start)
                if start_text is None:  # pragma: no cover - scheduler bounds output
                    _, _, stamp = self._route_slot(day, position)
                    start_text = stamp
                placements.append((by_id[aid], day, order_index, start_text))
        return placements, unrestored, placement_warnings

    def _repair_global_route(
        self,
        placements: Sequence[Tuple[Any, int, int, str]],
        stays: Sequence[OvernightStay],
        night_regions: Mapping[int, int],
        route_plan: RoutePlan,
        selected_activities: Sequence[Any],
        day_blockers: Mapping[int, Sequence[Tuple[int, int]]],
        max_per_day: int,
        trip: Trip,
        hotels: Sequence[Any],
        hotel_nights: int,
        base_hotel: Optional[Any],
        base_total: float,
        hotel_allowance: Optional[float],
        preserved_items: Sequence[ItineraryItem],
        unrestored_ids: Sequence[str],
        placement_warnings: Dict[str, List[str]],
        hotel_spent: float,
    ) -> Tuple[
        List[Tuple[Any, int, int, str]],
        List[OvernightStay],
        Dict[int, int],
        List[str],
        Dict[str, List[str]],
        float,
    ]:
        """Repair avoidable route extremes, then recalculate everything.

        Per-day planning never sees the actual stay plan, so this pass
        evaluates the COMPLETE route (morning hotel legs, inter-activity
        legs, evening hotel legs) and relocates activities whose placement
        creates avoidable extreme travel. After each accepted move it
        re-times the touched days, re-runs stay assignment on the repaired
        geography, and re-scores; the loop is bounded and strictly
        improving, so it always terminates. Anything unrepairable keeps
        honest warnings via the normal metadata path. Returns
        (placements, stays, night_regions, unrestored_ids,
        placement_warnings, hotel_spent). Never raises: on any internal
        failure the inputs pass through unchanged.
        """
        from backend.itinerary.route_plan import ROUTE_PLAN_CONFIG

        try:
            day_start = int(float(ROUTE_PLAN_CONFIG.get("schedule_day_start_minutes", 540)))
            arrival_floor = int(float(ROUTE_PLAN_CONFIG.get("schedule_arrival_day_start_minutes", 990)))
            day_end = int(float(ROUTE_PLAN_CONFIG.get("schedule_day_end_minutes", 1350)))
            speed = float(ROUTE_PLAN_CONFIG.get("default_travel_speed_kmh", 50.0))
            buffer_min = float(ROUTE_PLAN_CONFIG.get("activity_travel_buffer_minutes", 30.0))
            max_one_way = float(ROUTE_PLAN_CONFIG.get("max_one_way_travel_minutes", 120.0))
        except (TypeError, ValueError):
            day_start, arrival_floor, day_end, speed, buffer_min = 540, 990, 1350, 50.0, 30.0
            max_one_way = 120.0
        if speed <= 0:
            speed = 50.0
        try:
            per_day_cap = max(1, int(max_per_day))
        except (TypeError, ValueError):
            per_day_cap = 4
        try:
            max_iters = 3
            audit: List[str] = []
            by_id: Dict[str, Any] = {}
            for activity in selected_activities or []:
                aid = str(getattr(activity, "id", "") or "")
                if aid and aid not in by_id:
                    by_id[aid] = activity

            def _point_of(aid: str) -> Optional[Tuple[float, float]]:
                activity = by_id.get(aid)
                if activity is None:
                    return None
                try:
                    lat = float(getattr(activity, "latitude", None))
                    lng = float(getattr(activity, "longitude", None))
                except (TypeError, ValueError):
                    return None
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                    return None
                return lat, lng

            def _need_of(aid: str) -> int:
                activity = by_id.get(aid)
                try:
                    return max(60, int(round(float(
                        getattr(activity, "duration_hours", 0) or 0) * 60)))
                except (TypeError, ValueError):
                    return 60

            def _title_of(aid: str) -> str:
                activity = by_id.get(aid)
                return str(getattr(activity, "title", None)
                           or getattr(activity, "name", None) or aid)

            # Schedule state rebuilt from placements (same duration rule as
            # the scheduler, so times round-trip exactly). Every planned
            # day is seeded — including empty ones — so repair can use a
            # free day as a target; placements only carry acts, which
            # would otherwise hide empty days from repair entirely.
            sched: Dict[int, List[Tuple[str, int, int]]] = {}
            try:
                for empty_day in sorted((route_plan.days or {})):
                    sched.setdefault(int(empty_day), [])
            except (TypeError, ValueError):
                pass
            for activity, day_number, _order, start_text in placements or []:
                aid = str(getattr(activity, "id", "") or "")
                if not aid:
                    continue
                start = parse_clock_time(start_text)
                if start is None:
                    continue
                sched.setdefault(int(day_number), []).append(
                    (aid, start, start + _need_of(aid)))
            for day_stops in sched.values():
                day_stops.sort(key=lambda t: (t[1], t[2]))

            preserved_hotels: Dict[int, Any] = {}
            try:
                preserved_ids = {
                    item.hotel_id for item in preserved_items or []
                    if item.item_type == "hotel" and item.hotel_id
                }
                if preserved_ids:
                    for row in (
                        self.db.query(Hotel)
                        .filter(Hotel.id.in_(sorted(preserved_ids)))
                        .all()
                    ):
                        for item in preserved_items or []:
                            if (item.item_type == "hotel"
                                    and item.hotel_id == row.id
                                    and item.day_number not in preserved_hotels):
                                preserved_hotels[item.day_number] = row
            except Exception:
                preserved_hotels = {}

            def _stay_hotel(night: int, stays_list: Sequence[OvernightStay]) -> Optional[Any]:
                for stay in stays_list:
                    if stay.night == night and stay.hotel is not None:
                        return stay.hotel
                return preserved_hotels.get(night)

            def _coords_of_hotel(hotel: Any) -> Optional[Tuple[float, float]]:
                if hotel is None:
                    return None
                try:
                    lat = float(getattr(hotel, "latitude", None))
                    lng = float(getattr(hotel, "longitude", None))
                except (TypeError, ValueError):
                    return None
                return lat, lng

            def _leg_minutes(from_pt: Optional[Tuple[float, float]],
                             to_pt: Optional[Tuple[float, float]]) -> Optional[float]:
                if from_pt is None or to_pt is None:
                    return None
                dist = haversine_km(from_pt[0], from_pt[1], to_pt[0], to_pt[1])
                return estimate_travel_minutes(dist, speed)

            def _morning_hotel(day: int, stays_list: Sequence[OvernightStay]) -> Optional[Any]:
                # Same rule as metadata: previous night's actual stay.
                if day <= 1:
                    return _stay_hotel(1, stays_list)
                return (_stay_hotel(day - 1, stays_list)
                        or _stay_hotel(day, stays_list))

            def _score(sched_state: Mapping[int, Sequence[Tuple[str, int, int]]],
                       stays_list: Sequence[OvernightStay]
                       ) -> Tuple[int, float, set, Dict[tuple, float]]:
                extremes = 0
                total = 0.0
                keys: set = set()
                minutes_by_key: Dict[tuple, float] = {}
                for day in sorted(sched_state):
                    aids = [aid for (aid, _, _) in sched_state[day]]
                    if not aids:
                        continue
                    morning = _coords_of_hotel(_morning_hotel(day, stays_list))
                    tonight = _coords_of_hotel(
                        _stay_hotel(day, stays_list) or _morning_hotel(day, stays_list))
                    hops: List[Tuple[str, Optional[Tuple[float, float]],
                                      str, Optional[Tuple[float, float]]]] = []
                    hops.append((f"hotel:{day - 1 if day > 1 else 1}",
                                 morning, aids[0], _point_of(aids[0])))
                    for prev_aid, aid in zip(aids, aids[1:]):
                        hops.append((prev_aid, _point_of(prev_aid),
                                     aid, _point_of(aid)))
                    hops.append((aids[-1], _point_of(aids[-1]),
                                 f"hotel:{day}", tonight))
                    for origin_id, origin_pt, dest_id, dest_pt in hops:
                        minutes = _leg_minutes(origin_pt, dest_pt)
                        if minutes is None:
                            continue
                        total += minutes
                        if minutes > max_one_way:
                            extremes += 1
                            keys.add((day, origin_id, dest_id))
                            minutes_by_key[(day, origin_id, dest_id)] = minutes
                return extremes, round(total, 1), keys, minutes_by_key

            def _floors(day: int, count: int) -> List[int]:
                floors: List[int] = []
                for position in range(max(0, count)):
                    _, _, stamp = self._route_slot(day, position)
                    parsed = parse_clock_time(stamp)
                    floors.append(
                        parsed if parsed is not None
                        else (arrival_floor if day == 1 else day_start)
                    )
                return floors

            def _retime(day: int, ordered_aids: Sequence[str]
                        ) -> Optional[List[Tuple[str, int, int]]]:
                fixed = [(aid, _point_of(aid),
                          getattr(by_id[aid], "duration_hours", 0))
                         for aid in ordered_aids if aid in by_id]
                done, spilled = schedule_day_stops(
                    fixed,
                    _floors(day, len(fixed)),
                    list((day_blockers or {}).get(day, [])),
                    day_start_floor=arrival_floor if day == 1 else day_start,
                    day_end=day_end,
                    speed_kmh=speed,
                    buffer_min=buffer_min,
                )
                if spilled:
                    return None
                return done

            def _nearest_order(aids: Sequence[str], newcomer: str) -> List[str]:
                # Best-effort geographic insert position for the newcomer.
                base = [a for a in aids if a != newcomer]
                point = _point_of(newcomer)
                if point is None or not base:
                    return base + [newcomer]
                best_pos = len(base)
                best_dist: Optional[float] = None
                for pos in range(len(base) + 1):
                    neighbors = ([base[pos - 1]] if pos > 0 else []) + (
                        [base[pos]] if pos < len(base) else [])
                    worst: Optional[float] = None
                    ok = True
                    for other in neighbors:
                        other_pt = _point_of(other)
                        if other_pt is None:
                            continue
                        dist = haversine_km(point[0], point[1],
                                            other_pt[0], other_pt[1])
                        if dist is None:
                            ok = False
                            break
                        worst = dist if worst is None else max(worst, dist)
                    if not ok:
                        continue
                    if worst is None:
                        worst = 0.0
                    if best_dist is None or worst < best_dist:
                        best_dist = worst
                        best_pos = pos
                return base[:best_pos] + [newcomer] + base[best_pos:]

            cur_stays = list(stays or [])
            cur_regions = dict(night_regions or {})
            try:
                cur_spent = float(hotel_spent or 0.0)
            except (TypeError, ValueError):
                cur_spent = 0.0
            base_extremes, base_total_min, base_keys, base_minutes = _score(sched, cur_stays)

            for _ in range(max_iters):
                # Worst extreme leg drives candidacy (highest minutes,
                # deterministic ties); unverifiable legs are never repair
                # candidates — their warnings stand honestly.
                worst: Optional[Tuple[str, int, str]] = None  # (aid, day, role)
                for day, origin_id, dest_id in sorted(
                        base_keys,
                        key=lambda k: (-base_minutes.get(k, 0.0),
                                       k[0], str(k[1]), str(k[2]))):
                    aids = [aid for (aid, _, _) in sched.get(day, [])]
                    if not aids:
                        continue
                    if origin_id.startswith("hotel:") and dest_id == aids[0]:
                        worst = (dest_id, day, "morning")
                        break
                    if dest_id.startswith("hotel:") and origin_id == aids[-1]:
                        worst = (origin_id, day, "evening")
                        break
                    if dest_id in aids:
                        worst = (dest_id, day, "inter")
                        break
                if worst is None:
                    break
                aid, day, _role = worst
                title = _title_of(aid)
                best_move = None  # (score, target, new_sched)
                for target in sorted(sched):
                    if target == day:
                        continue
                    if len(sched.get(target, [])) + 1 > per_day_cap:
                        continue
                    if len(sched.get(day, [])) - 1 < 1:
                        break  # never vacate the source day entirely
                    trial_order = _nearest_order(
                        [a for (a, _, _) in sched.get(target, [])], aid)
                    new_target = _retime(target, trial_order)
                    if new_target is None:
                        continue
                    new_source = _retime(
                        day, [a for (a, _, _) in sched.get(day, [])
                              if a != aid])
                    if new_source is None:  # pragma: no cover - removal frees time
                        continue
                    trial = {d: list(v) for d, v in sched.items()}
                    trial[target] = new_target
                    trial[day] = new_source
                    extremes, total, keys, _trial_minutes = _score(trial, cur_stays)
                    if keys - base_keys:
                        continue  # never trade for a brand-new extreme
                    if (extremes, total) < (base_extremes, base_total_min):
                        if best_move is None or (extremes, total) < best_move[0]:
                            best_move = ((extremes, total), target, trial)
                if best_move is None:
                    audit.append(
                        f"Kept {title} on day {day}: no feasible nearer day "
                        "without creating a new extreme leg.")
                    break
                (new_extremes, new_total), target, trial = best_move
                # Accept: re-time, clean its overtime note (it fits now),
                # relocate its opener warning, re-run stays, re-score.
                # Snapshot everything first: if the stay recalculation
                # makes the global route worse, revert rather than keep a
                # bad repair (never leave half-applied state behind).
                snapshot = {
                    "sched": {d: list(v) for d, v in sched.items()},
                    "stays": list(cur_stays),
                    "regions": dict(cur_regions),
                    "spent": cur_spent,
                    "score": (base_extremes, base_total_min),
                    "unrestored": list(unrestored_ids),
                    "warnings": {k: list(v)
                                 for k, v in placement_warnings.items()},
                    "plan_warnings": {
                        d: list(r.warnings)
                        for d, r in (route_plan.days or {}).items()},
                }
                sched = trial
                if aid in unrestored_ids:
                    unrestored_ids = [u for u in unrestored_ids if u != aid]
                if aid in placement_warnings:
                    placement_warnings[aid] = [
                        w for w in placement_warnings[aid]
                        if "past the usual sightseeing hours" not in w
                    ]
                    if not placement_warnings[aid]:
                        del placement_warnings[aid]
                for plan_day, route in (route_plan.days or {}).items():
                    kept_warnings = []
                    moved_warnings = []
                    for warning in route.warnings:
                        # Opener (remoteness) warnings travel with their stop
                        # in both wordings: legacy "Reaching ..." and the
                        # "nearest other stop" form.
                        if ("Reaching" in warning
                                or "nearest other stop" in warning) \
                                and title in warning:
                            moved_warnings.append(warning)
                        else:
                            kept_warnings.append(warning)
                    if moved_warnings:
                        route.warnings = kept_warnings
                        for warning in moved_warnings:
                            if warning not in route_plan.days[target].warnings:
                                route_plan.days[target].warnings.append(warning)

                def _restore_snapshot() -> None:
                    sched.clear()
                    sched.update(snapshot["sched"])
                    cur_stays[:] = snapshot["stays"]
                    cur_regions.clear()
                    cur_regions.update(snapshot["regions"])
                    unrestored_ids[:] = snapshot["unrestored"]
                    placement_warnings.clear()
                    placement_warnings.update(snapshot["warnings"])
                    for plan_day, saved in snapshot["plan_warnings"].items():
                        if plan_day in route_plan.days:
                            route_plan.days[plan_day].warnings = saved

                try:
                    rerun = self._plan_stays(
                        trip, hotels, hotel_nights, base_hotel, base_total,
                        hotel_allowance, preserved_items,
                        [(by_id[a], d, 0, "") for d in sorted(sched)
                         for (a, _, _) in sched[d] if a in by_id],
                    )
                    cur_stays, cur_spent, cur_regions = rerun
                except Exception as exc:
                    _restore_snapshot()
                    cur_spent = snapshot["spent"]
                    audit.append(
                        f"Reverted moving {title} to day {target}: stay "
                        f"recalculation failed ({exc}).")
                    break
                rescored = _score(sched, cur_stays)
                base_extremes, base_total_min, base_keys, base_minutes = rescored
                if (base_extremes, base_total_min) > snapshot["score"]:
                    _restore_snapshot()
                    cur_spent = snapshot["spent"]
                    base_extremes, base_total_min = snapshot["score"]
                    base_keys, base_minutes = _score(sched, cur_stays)[2:]
                    audit.append(
                        f"Reverted moving {title} to day {target}: combined "
                        "route got worse after stay recalculation.")
                    break
                audit.append(
                    f"Repaired: moved {title} from day {day} to day {target}; "
                    f"extreme legs now {base_extremes}, travel {base_total_min} min.")
            final_placements: List[Tuple[Any, int, int, str]] = []
            for day in sorted(sched):
                for position, (aid, start, _end) in enumerate(sched[day]):
                    if aid not in by_id:
                        continue
                    _, order_index, _ = self._route_slot(day, position)
                    start_text = format_clock_time(start)
                    if start_text is None:  # pragma: no cover - retimed bounds
                        _, _, stamp = self._route_slot(day, position)
                        start_text = stamp
                    final_placements.append(
                        (by_id[aid], day, order_index, start_text))
            try:
                existing = getattr(trip, "_itinerary_warnings", None) or []
                existing.extend(a for a in audit[:8] if a not in existing)
                setattr(trip, "_itinerary_warnings", existing)
            except Exception:
                pass
            return (final_placements, list(cur_stays), dict(cur_regions),
                    list(unrestored_ids), placement_warnings, cur_spent)
        except Exception:
            # Fail-safe: inputs pass through with the ORIGINAL spend so
            # budget math stays exact; metadata validation still runs.
            try:
                original_spent = float(hotel_spent or 0.0)
            except (TypeError, ValueError):
                original_spent = 0.0
            return (list(placements), list(stays or []), dict(night_regions or {}),
                    list(unrestored_ids or []), placement_warnings, original_spent)

    def _attach_route_metadata(
        self,
        new_items: List[ItineraryItem],
        placements: Sequence[Tuple[Any, int, int, str]],
        stays: Sequence[OvernightStay],
        preserved_items: Sequence[ItineraryItem],
        trip: Trip,
        route_plan: RoutePlan,
        placement_warnings: Optional[Mapping[str, Sequence[str]]] = None,
        unrestored_ids: Optional[Sequence[str]] = None,
        night_regions: Optional[Mapping[int, int]] = None,
    ) -> None:
        """Attach legs, day summaries, and warnings to new items' ``ui``.

        Read-only over everything else: preserved items are never touched,
        and all values flow through the existing ``meta_data.ui``
        flattening, so API shape stays compatible.
        """
        act_rows: Dict[str, Any] = {}
        for activity, _, _, _ in placements:
            aid = str(getattr(activity, "id", ""))
            if aid:
                act_rows[aid] = activity
        preserved_act_ids = {
            item.activity_id for item in preserved_items if item.activity_id
        }
        if preserved_act_ids:
            for row in (
                self.db.query(Activity)
                .filter(Activity.id.in_(sorted(preserved_act_ids)))
                .all()
            ):
                act_rows.setdefault(row.id, row)

        hotel_by_night: Dict[int, Any] = {}
        for stay in stays:
            if stay.hotel is not None:
                hotel_by_night[stay.night] = stay.hotel
        # Traveler-confirmed (preserved) stays are authoritative for their
        # nights too: without them, legs on/after a confirmed night would
        # reference the wrong hotel. Only DB-backed rows (real coordinates)
        # qualify — custom live picks without a catalog row stay unknown
        # rather than invented.
        if preserved_items:
            preserved_hotel_ids = {
                item.hotel_id
                for item in preserved_items
                if item.item_type == "hotel" and item.hotel_id
            }
            if preserved_hotel_ids:
                for row in (
                    self.db.query(Hotel)
                    .filter(Hotel.id.in_(sorted(preserved_hotel_ids)))
                    .all()
                ):
                    for item in preserved_items:
                        if (
                            item.item_type == "hotel"
                            and item.hotel_id == row.id
                            and item.day_number not in hotel_by_night
                        ):
                            hotel_by_night[item.day_number] = row

        def coords_of(kind: str, ref_id: Optional[str]) -> Optional[Tuple[float, float]]:
            row = None
            if kind == "activity" and ref_id:
                row = act_rows.get(ref_id)
            elif kind == "hotel" and ref_id:
                for hotel in hotel_by_night.values():
                    if str(getattr(hotel, "id", "")) == ref_id:
                        row = hotel
                        break
            if row is None:
                return None
            try:
                lat = float(getattr(row, "latitude", None))
                lng = float(getattr(row, "longitude", None))
            except (TypeError, ValueError):
                return None
            return lat, lng

        def endpoint(kind: str, ref_id: str, name: str) -> RouteEndpoint:
            point = coords_of(kind, ref_id)
            return RouteEndpoint(
                item_id=ref_id,
                item_type=kind,
                name=name,
                coordinates_available=point is not None,
            )

        # Chain order per day: preserved activities (existing day/order)
        # interleaved with new placements, all by (order_index, id).
        chain: Dict[int, List[ItineraryItem]] = {}
        for item in preserved_items:
            if item.item_type == "activity":
                chain.setdefault(item.day_number, []).append(item)
        new_act_items = [i for i in new_items if i.item_type == "activity"]
        for item in new_act_items:
            chain.setdefault(item.day_number, []).append(item)
        for day_items in chain.values():
            day_items.sort(key=lambda i: (i.order_index, i.id))

        speed = 50.0
        max_one_way = 120.0
        try:
            from backend.itinerary.route_plan import ROUTE_PLAN_CONFIG

            speed = float(ROUTE_PLAN_CONFIG.get("default_travel_speed_kmh", 50.0))
            max_one_way = float(ROUTE_PLAN_CONFIG.get("max_one_way_travel_minutes", 120.0))
        except (TypeError, ValueError):
            speed = 50.0
            max_one_way = 120.0

        new_ids = {item.id for item in new_items}
        first_new_of_day: Dict[int, ItineraryItem] = {}
        for day in sorted(chain):
            for item in chain[day]:
                if item.id in new_ids:
                    first_new_of_day.setdefault(day, item)
                    break

        # Final accommodation validation: exactly one authoritative stay
        # per night. Duplicates (e.g. a manually added hotel item on a
        # night the planner already covers) are flagged, never silently
        # merged or deleted. Hotel items are counted directly (the day
        # chain above holds sightseeing only, by design).
        hotel_nights: Dict[int, List[str]] = {}
        for item in list(preserved_items) + [i for i in new_items if i.id in new_ids]:
            if item.item_type == "hotel":
                try:
                    hotel_nights.setdefault(int(item.day_number), []).append(
                        str(item.title))
                except (TypeError, ValueError):
                    continue
        dup_night_warnings: Dict[int, str] = {}
        for day in sorted(hotel_nights):
            hotel_titles = hotel_nights[day]
            if len(hotel_titles) > 1:
                dup_night_warnings[day] = (
                    f"Day {day} lists {len(hotel_titles)} stays "
                    f"({', '.join(hotel_titles)}); the first confirmed stay "
                    "applies and the duplicate needs review."
                )

        for day in sorted(chain):
            # Evening stay (checkout next morning) for the day summary.
            evening_hotel = hotel_by_night.get(day)
            # Morning origin: the traveler wakes up in the PREVIOUS night's
            # stay, not tonight's. Day 1 keeps tonight's stay as the
            # arrival approximation (checked in before sightseeing starts).
            if day <= 1:
                morning_hotel = hotel_by_night.get(1)
            else:
                morning_hotel = hotel_by_night.get(day - 1) or hotel_by_night.get(day)
            day_hotel = morning_hotel
            hotel_name = str(getattr(day_hotel, "name", "")) if day_hotel else ""
            hotel_ref = str(getattr(day_hotel, "id", "")) if day_hotel else ""
            evening_name = str(getattr(evening_hotel, "name", "")) if evening_hotel else ""
            # Day-level schedule validation (read-only): overlapping stops
            # are flagged, never removed or retimed.
            day_titles = {
                i.id: i.title
                for i in chain[day]
                if i.item_type == "activity" and i.activity_id
            }
            overlap_map = detect_overlaps([
                (i.id, i.start_time, i.end_time)
                for i in chain[day]
                if i.item_type == "activity" and i.activity_id
            ])
            day_overlap_warnings: List[str] = []
            seen_pairs = set()
            for aid, others in overlap_map.items():
                for oid in others:
                    key = tuple(sorted((aid, oid)))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    day_overlap_warnings.append(
                        f"{day_titles.get(aid, aid)} overlaps with "
                        f"{day_titles.get(oid, oid)} on Day {day} — times "
                        "clash, review the schedule."
                    )
            previous_kind: Optional[str] = None
            previous_ref: Optional[str] = None
            previous_name = ""
            if day_hotel is not None:
                previous_kind, previous_ref, previous_name = "hotel", hotel_ref, hotel_name
            # Closing leg: last activity back to the evening stay. Validated
            # read-only like every other leg — extreme returns are warned,
            # never hidden or auto-moved.
            evening_leg_warnings: List[str] = []
            day_acts = [i for i in chain[day]
                        if i.item_type == "activity" and i.activity_id]
            if day_acts and evening_hotel is not None:
                last = day_acts[-1]
                evening_ref = str(getattr(evening_hotel, "id", "") or "")
                evening_leg = build_leg(
                    endpoint("activity", last.activity_id or "", last.title),
                    endpoint("hotel", evening_ref, evening_name),
                    coords_of("activity", last.activity_id),
                    coords_of("hotel", evening_ref),
                    speed,
                )
                if (evening_leg.status == "estimated"
                        and (evening_leg.estimated_duration_minutes or 0)
                        > max_one_way):
                    evening_leg_warnings.append(
                        f"Day {day} ends about "
                        f"{evening_leg.estimated_duration_minutes:.0f} min "
                        f"from {evening_name}; a closer stay would cut travel."
                    )
            for item in chain[day]:
                if item.item_type != "activity" or not item.activity_id:
                    continue
                leg = build_leg(
                    endpoint(previous_kind or "unknown", previous_ref or "", previous_name),
                    endpoint("activity", item.activity_id, item.title),
                    coords_of(previous_kind or "", previous_ref),
                    coords_of("activity", item.activity_id),
                    speed,
                )
                previous_kind, previous_ref, previous_name = (
                    "activity",
                    item.activity_id,
                    item.title,
                )
                if item.id not in new_ids:
                    continue
                ui = item.meta_data.get("ui", {}) if item.meta_data else {}
                ui["route_leg_in"] = leg.to_dict()
                day_warnings = list(route_plan.days[day].warnings) if day in route_plan.days else []
                for extra in (placement_warnings or {}).get(item.activity_id or "", []):
                    if extra not in day_warnings:
                        day_warnings.append(extra)
                if day in dup_night_warnings and dup_night_warnings[day] not in day_warnings:
                    day_warnings.append(dup_night_warnings[day])
                for evening_warning in evening_leg_warnings:
                    if evening_warning not in day_warnings:
                        day_warnings.append(evening_warning)
                if leg.status != "estimated":
                    day_warnings.append(
                        f"Travel to {item.title} cannot be verified from "
                        "available coordinates."
                    )
                for other_id in overlap_map.get(item.id, []):
                    other_title = day_titles.get(other_id, "another stop")
                    day_warnings.append(
                        f"{item.title} overlaps with {other_title} on Day "
                        f"{day} — times clash, review the schedule."
                    )
                if (
                    leg.origin.item_type == "hotel"
                    and leg.status == "estimated"
                    and (leg.estimated_duration_minutes or 0) > max_one_way
                ):
                    day_warnings.append(
                        f"Day {day} starts about {leg.estimated_duration_minutes:.0f} min "
                        f"from {leg.origin.name}; a closer stay would cut travel."
                    )
                if day_warnings:
                    existing = ui.get("route_warnings", [])
                    ui["route_warnings"] = list(existing) + [
                        w for w in day_warnings if w not in existing
                    ]
                if first_new_of_day.get(day) is item:
                    total, longest = summarize_legs(
                        [
                            build_leg(
                                endpoint("activity", a.activity_id or "", a.title),
                                endpoint("activity", b.activity_id or "", b.title),
                                coords_of("activity", a.activity_id),
                                coords_of("activity", b.activity_id),
                                speed,
                            )
                            for a, b in zip(chain[day], chain[day][1:])
                            if a.item_type == "activity" and b.item_type == "activity"
                        ]
                    )
                    status = route_plan.days[day].status if day in route_plan.days else "Unable-to-Verify"
                    summary_warnings = list(day_warnings)
                    for warning in day_overlap_warnings:
                        if warning not in summary_warnings:
                            summary_warnings.append(warning)
                    unrestored_here = [
                        i for i in chain[day]
                        if i.item_type == "activity"
                        and (unrestored_ids or [])
                        and i.activity_id in set(unrestored_ids or [])
                    ]
                    if overlap_map or unrestored_here:
                        status = "Requires-Revision"
                    elif status == "Valid" and (
                        leg.status != "estimated" or summary_warnings
                    ):
                        status = "Valid-with-Warning"
                    ui["route_day_summary"] = {
                        "day": day,
                        "activity_count": sum(
                            1 for i in chain[day] if i.item_type == "activity"
                        ),
                        "daily_travel_minutes": total,
                        "max_one_way_minutes": longest,
                        "hotel": evening_name or hotel_name or None,
                        # Same fallback as the hotel name: beyond the last
                        # night, the previous night's region still applies.
                        "overnight_region": (night_regions or {}).get(
                            day, (night_regions or {}).get(day - 1)),
                        "warnings": summary_warnings,
                        "status": status,
                    }
                item.meta_data = {**(item.meta_data or {}), "ui": ui}

    def _trip_items(self, trip: Trip) -> List[ItineraryItem]:
        return self._sorted_items(
            self.db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id).all()
        )

    @staticmethod
    def _sorted_items(items: Iterable[ItineraryItem]) -> List[ItineraryItem]:
        return sorted(items, key=lambda item: (item.day_number, item.order_index, item.id))

    @staticmethod
    def _preferences(trip: Trip) -> Dict[str, Any]:
        preferences = trip.preferences
        if not preferences:
            return {}
        return {
            "budget_tier": preferences.budget_tier,
            "interests": preferences.interests or [],
            "accommodation_types": preferences.accommodation_types or [],
            "transport_preferences": preferences.transport_preferences or [],
            "travel_companions": preferences.travel_companions,
            "dietary_requirements": preferences.dietary_requirements or [],
            "special_requests": preferences.special_requests,
        }

    def _ranked_catalog(
        self,
        model: Type[Any],
        recommendations: Sequence[Dict[str, Any]],
        trip: Trip,
    ) -> List[Any]:
        ranked_ids = [entry.get("id") for entry in recommendations if entry.get("id")]
        if not ranked_ids:
            return []
        query = self.db.query(model).filter(
            model.id.in_(ranked_ids),
            model.destination_id == trip.destination_id,
            model.is_active.is_(True),
        )
        if trip.discovery_session_id:
            query = query.filter(
                model.inventory_source == "discovered",
                model.discovery_session_id == trip.discovery_session_id,
                model.verification_status == "verified_candidate",
            )
        else:
            # Curated catalog plus live-provider rows (SerpApi/OSM fills).
            query = query.filter(model.inventory_source.in_(["catalog", "live"]))
        if trip.currency:
            query = query.filter(model.currency == trip.currency)
        catalog_by_id = {item.id: item for item in query.all()}
        missing_ids = sorted({item_id for item_id in ranked_ids if item_id not in catalog_by_id})
        if missing_ids:
            raise ItineraryGenerationError(
                f"Recommendation selected invalid catalog IDs for {model.__name__}: {', '.join(missing_ids)}"
            )
        return [catalog_by_id[item_id] for item_id in ranked_ids if item_id in catalog_by_id]

    @staticmethod
    def _duration_days(trip: Trip) -> int:
        return max(1, int(trip.duration_days or 1))

    @staticmethod
    def _traveler_count(trip: Trip) -> int:
        return max(1, int(trip.traveler_count or 1))

    @classmethod
    def _remaining_budget(cls, trip: Trip, preserved_items: Sequence[ItineraryItem]) -> Optional[float]:
        if trip.total_budget is None:
            return None
        return float(trip.total_budget) - sum(float(item.cost or 0) for item in preserved_items)

    @staticmethod
    def _activity_slots(duration_days: int, pace: Optional[str]) -> int:
        if str(pace or "").casefold() == "relaxed":
            return duration_days
        if str(pace or "").casefold() == "packed":
            return duration_days * 2
        return 1 + max(0, duration_days - 1) * 2

    @staticmethod
    def _first_within_budget(
        candidates: Sequence[Any],
        cost: Any,
        remaining_budget: Optional[float],
    ) -> Optional[Any]:
        for candidate in candidates:
            if remaining_budget is None or cost(candidate) <= remaining_budget:
                return candidate
        return None

    @staticmethod
    def _subtract_budget(remaining_budget: Optional[float], cost: float) -> Optional[float]:
        if remaining_budget is None:
            return None
        return remaining_budget - cost

    @staticmethod
    def _available_order(day_number: int, preferred_order: int, occupied_orders: set[tuple[int, int]]) -> int:
        order = preferred_order
        while (day_number, order) in occupied_orders:
            order += 1
        occupied_orders.add((day_number, order))
        return order

    @staticmethod
    def _transport_description(transport: TransportOption) -> Optional[str]:
        if transport.route_from and transport.route_to:
            return f"{transport.route_from} to {transport.route_to}"
        return None

    @staticmethod
    def _entity_ui_meta(entity: Any) -> Dict[str, Any]:
        meta = {
            "image_url": (getattr(entity, "images", None) or [None])[0],
            "latitude": getattr(entity, "latitude", None),
            "longitude": getattr(entity, "longitude", None),
            "source_url": getattr(entity, "source_url", None),
            "evidence": getattr(entity, "evidence", None) or [],
        }
        return {key: value for key, value in meta.items() if value not in (None, [], "")}

    def _departure_note(
        self,
        trip: Trip,
        preserved_items: Sequence[ItineraryItem],
        new_items: Sequence[ItineraryItem],
        stay_hotels: Sequence[Any],
        transport: Any,
    ) -> Optional[ItineraryItem]:
        """Append a derived check-out/departure note when the last day is empty.

        Content references only persisted trip facts (stay/transfer names);
        never inventory. Returns None when the last day already has items.
        """
        duration_days = self._duration_days(trip)
        if duration_days < 2:
            return None
        if any(item.day_number == duration_days for item in preserved_items):
            return None
        if any(item.day_number == duration_days for item in new_items):
            return None
        stay_name = next(
            (hotel.name for hotel in reversed(list(stay_hotels)) if hotel is not None),
            None,
        )
        if stay_name is None:
            stay_name = next(
                (item.title for item in preserved_items if item.item_type == "hotel"), None
            )
        ride_name = transport.name if transport is not None else next(
            (item.title for item in preserved_items if item.item_type == "transport"), None
        )
        details = []
        if stay_name:
            details.append(f"check out from {stay_name}")
        if ride_name:
            details.append(f"return transfer via {ride_name}")
        dest_name = trip.destination.name if trip.destination else "destination"
        description = (
            ("; ".join(details) + f"; homeward departure from {dest_name}.")
            if details else f"Leisure morning and homeward departure from {dest_name}."
        )
        return ItineraryItem(
            trip_id=trip.id,
            day_number=duration_days,
            order_index=1,
            item_type="note",
            title=f"Check-out & homeward departure ({dest_name})",
            description=description,
            start_time="10:00 AM",
            end_time="12:00 PM",
            cost=0.0,
            status="proposed",
            location=dest_name,
            meta_data={"ui": {}},
        )

    @staticmethod
    def _end_time(start_time: str, duration_hours: Any) -> str:
        start_hour, minute_period = start_time.split(":", maxsplit=1)
        start_minute, period = minute_period.split(" ", maxsplit=1)
        hour = int(start_hour) % 12
        minute = int(start_minute)
        if period == "PM":
            hour += 12
        duration_minutes = max(60, round(float(duration_hours or 0) * 60))
        end_minutes = (hour * 60 + minute + duration_minutes) % (24 * 60)
        end_hour, end_minute = divmod(end_minutes, 60)
        display_period = "AM" if end_hour < 12 else "PM"
        display_hour = end_hour % 12 or 12
        return f"{display_hour:02d}:{end_minute:02d} {display_period}"
