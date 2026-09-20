"""Persist catalog-grounded generated and optimized trip itinerary items."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Type

from sqlalchemy.orm import Session

from backend.models.models import Activity, Hotel, ItineraryItem, TransportOption, Trip
from backend.recommendation.engine import RecommendationEngine


class ItineraryGenerationError(Exception):
    """Raised when a trip itinerary cannot be generated from verified catalog records."""


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
        return self._persist_ranked_items(trip, [])

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

        new_items = self._persist_ranked_items(trip, preserved_items, commit=False)
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
        return new_items

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
        preserved_hotel_ids = {item.hotel_id for item in preserved_items if item.hotel_id}
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
        hotel = None
        if not preserved_hotel_ids:
            if not hotels:
                raise ItineraryGenerationError("No active catalog hotel is available for this destination and currency")
            hotel_pot = (remaining_budget - min_transport_cost - min_activities_cost
                         if remaining_budget is not None else None)
            hotel = self._first_within_budget(
                hotels,
                lambda candidate: float(candidate.price_per_night or 0) * hotel_nights,
                hotel_pot,
            )
            if not hotel:
                raise ItineraryGenerationError("No active catalog hotel fits the trip budget")
            if hotel:
                remaining_budget = self._subtract_budget(
                    remaining_budget,
                    float(hotel.price_per_night or 0) * hotel_nights,
                )

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

        if hotel:
            new_items.append(
                ItineraryItem(
                    trip_id=trip.id,
                    day_number=1,
                    order_index=self._available_order(1, 2, used_orders),
                    item_type="hotel",
                    title=hotel.name,
                    description=hotel.description,
                    start_time="01:30 PM",
                    end_time="03:00 PM",
                    cost=float(hotel.price_per_night or 0) * hotel_nights,
                    status="proposed",
                    hotel_id=hotel.id,
                    location=hotel.address or hotel.name,
                    meta_data={"ui": self._entity_ui_meta(hotel)},
                )
            )

        # Proportional distribution: activities spread across the whole
        # trip, up to 2 per day (morning + afternoon). Thin inventories
        # yield ~1/day; rich ones fill both daily slots.
        total_selected = len(selected_activities)
        day_slot_counts: Dict[int, int] = {}
        for index, activity in enumerate(selected_activities):
            day_number, preferred_order, start_time = self._activity_slot(index, duration_days)
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
        departure_note = self._departure_note(trip, preserved_items, new_items, hotel, transport)
        if departure_note is not None:
            departure_note.order_index = self._available_order(
                departure_note.day_number, departure_note.order_index, occupied_orders
            )
            new_items.append(departure_note)
        return new_items

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
        hotel: Any,
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
        stay_name = hotel.name if hotel is not None else next(
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
    def _activity_slot(index: int, duration_days: int) -> tuple[int, int, str]:
        # Round-robin across the whole trip so every sightseeing day receives an
        # activity before any day receives a second one. Sequential 2-per-day
        # packing clustered the minimum required activities on days 1-3 and left
        # later days empty. Index 0 keeps its historical day-1 evening slot;
        # later day-1 stops go further into the evening so times stay sorted.
        span = max(1, int(duration_days or 1))
        round_number, day_offset = divmod(index, span)
        day_number = day_offset + 1
        if index == 0:
            return day_number, 3, "04:30 PM"
        if day_number == 1:
            evening = ["06:00 PM", "07:30 PM", "08:30 PM"]
            return day_number, 4 + round_number, evening[min(round_number - 1, 2)]
        if round_number == 0:
            return day_number, 1, "09:00 AM"
        if round_number == 1:
            return day_number, 2, "03:00 PM"
        return day_number, 3 + round_number, "06:00 PM"

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
