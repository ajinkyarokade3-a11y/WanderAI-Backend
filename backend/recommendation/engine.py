"""Deterministic, catalog-grounded recommendations for the existing API."""

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.ai.gemini_service import gemini_service
from backend.models.models import Activity, Hotel, TransportOption


class RecommendationEngine:
    """Rank active catalog candidates while keeping Gemini supplementary."""

    # Proximity component of the hotel score (added to the existing 100).
    # Weight 10 matches the smallest existing components, so geography can
    # decide close calls but never outrank budget + category + rating
    # combined. Distances use straight-line haversine and decay linearly
    # to zero at PROXIMITY_SCALE_KM; hotels without coordinates receive
    # the neutral midpoint so missing data neither helps nor hurts them.
    PROXIMITY_WEIGHT = 10.0
    PROXIMITY_SCALE_KM = 200.0
    PROXIMITY_NEUTRAL = 5.0

    def __init__(self, db: Session):
        self.db = db

    def get_recommendations(
        self,
        destination_id: Optional[str],
        preferences: Dict[str, Any],
        discovery_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        preferences = preferences or {}
        hotels_query = self.db.query(Hotel).filter(Hotel.is_active.is_(True))
        activities_query = self.db.query(Activity).filter(Activity.is_active.is_(True))
        transport_query = self.db.query(TransportOption).filter(TransportOption.is_active.is_(True))
        if destination_id is not None:
            hotels_query = hotels_query.filter(Hotel.destination_id == destination_id)
            activities_query = activities_query.filter(Activity.destination_id == destination_id)
            transport_query = transport_query.filter(TransportOption.destination_id == destination_id)
        if discovery_session_id:
            hotels_query = hotels_query.filter(
                Hotel.inventory_source == "discovered",
                Hotel.discovery_session_id == discovery_session_id,
                Hotel.verification_status == "verified_candidate",
            )
            activities_query = activities_query.filter(
                Activity.inventory_source == "discovered",
                Activity.discovery_session_id == discovery_session_id,
                Activity.verification_status == "verified_candidate",
            )
            transport_query = transport_query.filter(
                TransportOption.inventory_source == "discovered",
                TransportOption.discovery_session_id == discovery_session_id,
                TransportOption.verification_status == "verified_candidate",
            )
        else:
            # Curated catalog plus live-provider rows (SerpApi/OSM fills).
            # Discovery sessions stay strictly isolated above.
            hotels_query = hotels_query.filter(Hotel.inventory_source.in_(["catalog", "live"]))
            activities_query = activities_query.filter(Activity.inventory_source.in_(["catalog", "live"]))
            transport_query = transport_query.filter(TransportOption.inventory_source.in_(["catalog", "live"]))

        ranked_hotels = self._rank_hotels(
            hotels_query.all(), preferences, activities_query.all()
        )[:5]
        # Up to 12 activities so multi-day trips can fill ~2 stops/day;
        # the generator still caps selection by pace slots and budget.
        ranked_activities = self._rank_activities(activities_query.all(), preferences)[:12]
        ranked_transport = self._rank_transport(transport_query.all(), preferences)[:4]
        return {
            "ai_insights": self._ai_insights(preferences, destination_id),
            "recommended_hotels": [
                self._hotel_response(hotel, score, proximity_km)
                for hotel, score, proximity_km in ranked_hotels
            ],
            "recommended_activities": [
                self._activity_response(activity, score) for activity, score in ranked_activities
            ],
            "recommended_transport": [
                self._transport_response(option, score) for option, score in ranked_transport
            ],
        }

    def _rank_hotels(
        self,
        hotels: Sequence[Hotel],
        preferences: Dict[str, Any],
        activities: Optional[Sequence[Any]] = None,
    ) -> List[Tuple[Hotel, float, Optional[float]]]:
        prices = [self._number(hotel.price_per_night) for hotel in hotels]
        accommodation_types = self._values(preferences, "accommodation_types")
        interests = self._values(preferences, "interests")
        dietary_requirements = self._values(preferences, "dietary_requirements")
        companions = self._values(preferences, "travel_companions")
        special_requests = self._values(preferences, "special_requests")
        anchor_points = self._anchor_points(activities)
        scored = []
        for hotel in hotels:
            context = self._catalog_text(
                hotel.name, hotel.category, hotel.description, hotel.amenities, hotel.address
            )
            price_position = self._relative_position(hotel.price_per_night, prices)
            proximity_score, proximity_km = self._proximity_terms(hotel, anchor_points)
            score = (
                self._match_score(accommodation_types, [hotel.category], 30)
                + self._match_score(interests, context, 10)
                + self._match_score(dietary_requirements, context, 10)
                + self._match_score(companions, context, 5)
                + self._match_score(special_requests, context, 5)
                + self._budget_score(preferences.get("budget_tier"), price_position, 15)
                + self._rating_score(hotel.rating, 15)
                + self._price_efficiency(price_position, 10)
                + proximity_score
            )
            scored.append((hotel, round(score, 2), proximity_km))
        ranked = self._sort_scored(
            [(hotel, score) for hotel, score, _ in scored],
            lambda hotel: (-self._number(hotel.rating), self._number(hotel.price_per_night), str(hotel.id)),
        )
        proximity_by_id = {str(hotel.id): proximity_km for hotel, _, proximity_km in scored}
        return [
            (hotel, score, proximity_by_id.get(str(hotel.id)))
            for hotel, score in ranked
        ]

    @staticmethod
    def _anchor_points(activities: Optional[Sequence[Any]]) -> List[Tuple[float, float]]:
        """Valid activity coordinates for proximity scoring.

        Accepts Activity rows, mappings, or raw (lat, lng) pairs. Invalid
        or missing coordinates are skipped (never invented, never crash).
        """
        points: List[Tuple[float, float]] = []
        for item in activities or []:
            if isinstance(item, Mapping):
                raw = (item.get("latitude"), item.get("longitude"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                raw = (item[0], item[1])
            else:
                raw = (getattr(item, "latitude", None), getattr(item, "longitude", None))
            try:
                lat = float(raw[0])
                lng = float(raw[1])
            except (TypeError, ValueError, IndexError):
                continue
            if -90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0:
                points.append((lat, lng))
        return points

    def _proximity_terms(
        self, hotel: Any, anchor_points: Sequence[Tuple[float, float]]
    ) -> Tuple[float, Optional[float]]:
        """(proximity score 0..10, average distance km or None).

        Average straight-line distance from the hotel to every valid
        activity anchor: the better a base a hotel makes, the higher the
        score. No valid anchors at all -> (0.0, None) so legacy behavior
        is byte-identical. Hotel without coordinates -> neutral midpoint.
        """
        if not anchor_points:
            return 0.0, None
        # Deferred import: backend.itinerary imports the generator, which
        # imports this engine - a top-level import would be circular.
        from backend.itinerary.hotel_assignment import haversine_km

        try:
            hotel_lat = float(getattr(hotel, "latitude", None))
            hotel_lng = float(getattr(hotel, "longitude", None))
        except (TypeError, ValueError):
            return self.PROXIMITY_NEUTRAL, None
        if not (-90.0 <= hotel_lat <= 90.0 and -180.0 <= hotel_lng <= 180.0):
            return self.PROXIMITY_NEUTRAL, None
        distances = [
            haversine_km(hotel_lat, hotel_lng, lat, lng) for lat, lng in anchor_points
        ]
        distances = [d for d in distances if d is not None]
        if not distances:
            return self.PROXIMITY_NEUTRAL, None
        average = sum(distances) / len(distances)
        score = self.PROXIMITY_WEIGHT * max(
            0.0, 1.0 - average / self.PROXIMITY_SCALE_KM
        )
        return round(score, 2), round(average, 1)

    def _rank_activities(
        self,
        activities: Sequence[Activity],
        preferences: Dict[str, Any],
    ) -> List[Tuple[Activity, float]]:
        prices = [self._number(activity.price_per_person) for activity in activities]
        durations = [self._number(activity.duration_hours) for activity in activities]
        interests = self._values(preferences, "interests")
        dietary_requirements = self._values(preferences, "dietary_requirements")
        companions = self._values(preferences, "travel_companions")
        special_requests = self._values(preferences, "special_requests")
        scored = []
        for activity in activities:
            context = self._catalog_text(
                activity.title,
                activity.category,
                activity.description,
                activity.meeting_point,
                activity.difficulty_level,
            )
            price_position = self._relative_position(activity.price_per_person, prices)
            duration_position = self._relative_position(activity.duration_hours, durations)
            score = (
                self._match_score(interests, [activity.category], 30)
                + self._match_score(interests, context, 10)
                + self._match_score(dietary_requirements, context, 10)
                + self._match_score(companions, context, 5)
                + self._match_score(special_requests, context, 5)
                + self._budget_score(preferences.get("budget_tier"), price_position, 15)
                + self._rating_score(activity.rating, 15)
                + self._price_efficiency(price_position, 5)
                + self._duration_efficiency(duration_position, 5)
            )
            scored.append((activity, round(score, 2)))
        return self._sort_scored(
            scored,
            lambda activity: (
                -self._number(activity.rating),
                self._number(activity.price_per_person),
                self._number(activity.duration_hours),
                str(activity.id),
            ),
        )

    def _rank_transport(
        self,
        options: Sequence[TransportOption],
        preferences: Dict[str, Any],
    ) -> List[Tuple[TransportOption, float]]:
        prices = [self._number(option.price) for option in options]
        durations = [self._number(option.duration_hours) for option in options]
        transport_preferences = self._values(preferences, "transport_preferences")
        companions = self._values(preferences, "travel_companions")
        special_requests = self._values(preferences, "special_requests")
        scored = []
        for option in options:
            context = self._catalog_text(
                option.type, option.name, option.route_from, option.route_to, option.features
            )
            price_position = self._relative_position(option.price, prices)
            duration_position = self._relative_position(option.duration_hours, durations)
            score = (
                self._match_score(transport_preferences, [option.type], 35)
                + self._match_score(transport_preferences, context, 5)
                + self._match_score(companions, context, 5)
                + self._match_score(special_requests, context, 5)
                + self._budget_score(preferences.get("budget_tier"), price_position, 25)
                + self._price_efficiency(price_position, 15)
                + self._duration_efficiency(duration_position, 10)
            )
            scored.append((option, round(score, 2)))
        return self._sort_scored(
            scored,
            lambda option: (self._number(option.price), self._number(option.duration_hours), str(option.id)),
        )

    @staticmethod
    def _sort_scored(scored: Sequence[Tuple[Any, float]], tie_breaker: Any) -> List[Tuple[Any, float]]:
        return sorted(scored, key=lambda entry: (-entry[1], *tie_breaker(entry[0])))

    @staticmethod
    def _values(preferences: Dict[str, Any], key: str) -> List[str]:
        value = preferences.get(key)
        if value is None:
            return []
        values = value if isinstance(value, (list, tuple, set)) else [value]
        return [normalized for item in values if (normalized := RecommendationEngine._normalize(item))]

    @staticmethod
    def _catalog_text(*values: Any) -> List[str]:
        text = []
        for value in values:
            entries = value if isinstance(value, (list, tuple, set)) else [value]
            text.extend(normalized for item in entries if (normalized := RecommendationEngine._normalize(item)))
        return text

    @staticmethod
    def _normalize(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").replace("_", " ").replace("-", " ").casefold()).strip()

    @classmethod
    def _match_score(cls, preferences: Iterable[str], catalog_values: Iterable[str], weight: float) -> float:
        requested = list(preferences)
        if not requested:
            return 0.0
        matched = sum(1 for preference in requested if any(cls._matches(preference, value) for value in catalog_values))
        return weight * matched / len(requested)

    @staticmethod
    def _matches(preference: str, catalog_value: str) -> bool:
        preference_words = set(preference.split())
        catalog_words = set(catalog_value.split())
        return bool(preference_words) and preference_words.issubset(catalog_words)

    @classmethod
    def _budget_score(cls, budget_tier: Any, price_position: float, weight: float) -> float:
        tier = cls._normalize(budget_tier)
        if tier == "budget":
            suitability = 1 - price_position
        elif tier == "moderate":
            suitability = 1 - abs(price_position - 0.5) * 2
        elif tier in {"premium", "luxury", "ultra luxury"}:
            suitability = price_position
        else:
            suitability = 0.0
        return weight * max(0.0, suitability)

    @staticmethod
    def _rating_score(rating: Any, weight: float) -> float:
        return weight * max(0.0, min(RecommendationEngine._number(rating), 5.0)) / 5.0

    @staticmethod
    def _price_efficiency(price_position: float, weight: float) -> float:
        return weight * (1 - price_position)

    @staticmethod
    def _duration_efficiency(duration_position: float, weight: float) -> float:
        return weight * (1 - duration_position)

    @staticmethod
    def _relative_position(value: Any, values: Sequence[float]) -> float:
        if not values:
            return 0.5
        minimum, maximum = min(values), max(values)
        if minimum == maximum:
            return 0.5
        return max(0.0, min(1.0, (RecommendationEngine._number(value) - minimum) / (maximum - minimum)))

    @staticmethod
    def _number(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _ai_insights(preferences: Dict[str, Any], destination_id: Optional[str]) -> Dict[str, Any]:
        try:
            return gemini_service.recommend(preferences=preferences, destination_id=destination_id)
        except Exception:
            return {
                "status": "unavailable",
                "message": "Catalog recommendations are available; supplementary AI insights are unavailable.",
            }

    @staticmethod
    def _hotel_response(
        hotel: Hotel, score: float, proximity_km: Optional[float] = None
    ) -> Dict[str, Any]:
        return {
            "id": hotel.id,
            "destination_id": hotel.destination_id,
            "vendor_id": hotel.vendor_id,
            "name": hotel.name,
            "category": hotel.category,
            "price_per_night": hotel.price_per_night,
            "currency": hotel.currency,
            "rating": hotel.rating,
            "address": hotel.address,
            "amenities": hotel.amenities or [],
            "images": hotel.images or [],
            "description": hotel.description,
            "is_active": hotel.is_active,
            "created_at": hotel.created_at,
            "inventory_source": hotel.inventory_source,
            "verification_status": hotel.verification_status,
            "discovery_session_id": hotel.discovery_session_id,
            "match_score": score,
            "proximity_km": proximity_km,
        }

    @staticmethod
    def _activity_response(activity: Activity, score: float) -> Dict[str, Any]:
        return {
            "id": activity.id,
            "destination_id": activity.destination_id,
            "vendor_id": activity.vendor_id,
            "title": activity.title,
            "category": activity.category,
            "duration_hours": activity.duration_hours,
            "price_per_person": activity.price_per_person,
            "currency": activity.currency,
            "difficulty_level": activity.difficulty_level,
            "rating": activity.rating,
            "images": activity.images or [],
            "description": activity.description,
            "meeting_point": activity.meeting_point,
            "is_active": activity.is_active,
            "created_at": activity.created_at,
            "inventory_source": activity.inventory_source,
            "verification_status": activity.verification_status,
            "discovery_session_id": activity.discovery_session_id,
            "match_score": score,
        }

    @staticmethod
    def _transport_response(option: TransportOption, score: float) -> Dict[str, Any]:
        return {
            "id": option.id,
            "destination_id": option.destination_id,
            "vendor_id": option.vendor_id,
            "type": option.type,
            "name": option.name,
            "route_from": option.route_from,
            "route_to": option.route_to,
            "duration_hours": option.duration_hours,
            "price": option.price,
            "currency": option.currency,
            "capacity": option.capacity,
            "features": option.features or [],
            "is_active": option.is_active,
            "created_at": option.created_at,
            "inventory_source": option.inventory_source,
            "verification_status": option.verification_status,
            "discovery_session_id": option.discovery_session_id,
            "match_score": score,
        }
