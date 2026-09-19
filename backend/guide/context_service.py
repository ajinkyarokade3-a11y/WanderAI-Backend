"""GuideContextService — builds compact, grounded context for the AI Guide.

Single responsibility: load ONLY the fields the Guide needs, scoped by
(authenticated user_id, trip_id) with trip-ownership verification.
Never invents data; missing data is reported as unavailable.
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, joinedload

from backend.models.models import (
    Booking,
    GuideConversationSummary,
    GuideMessage,
    ItineraryItem,
    Trip,
    User,
)

RECENT_MESSAGE_LIMIT = 15
SUMMARY_REFRESH_EVERY = 20

GUIDE_SYSTEM_INSTRUCTION = """You are TourFlow AI, a persistent travel assistant for the currently authenticated traveler.

Use the supplied user, trip, itinerary, booking, budget, and conversation context.
Answer based on real application data.
Never invent bookings, hotels, restaurants, prices, itinerary items, dates, or user information.
If information is unavailable, explicitly say that it is unavailable.
Remember the current conversation and maintain continuity.
When the user refers to 'tomorrow', 'today', 'there', 'that hotel', 'the restaurant', etc., resolve the reference using the current trip and recent conversation.
Do not confuse previous trips with the active trip.
If the user asks for a modification to the itinerary, identify the relevant itinerary item and return a structured change request that the backend can process."""


def _current_trip_day(trip: Trip) -> Optional[int]:
    """1-based trip day derived from start_date, or None when unknown."""
    if not trip.start_date:
        return None
    try:
        start = trip.start_date.date() if isinstance(trip.start_date, datetime) else trip.start_date
        today = date.today()
        delta = (today - start).days + 1
        if delta < 1:
            return 1
        if trip.duration_days and delta > trip.duration_days:
            return trip.duration_days
        return delta
    except Exception:
        return None


def _item_dict(item: ItineraryItem) -> Dict[str, Any]:
    return {
        "id": item.id,
        "day_number": item.day_number,
        "order_index": item.order_index,
        "item_type": item.item_type,
        "title": item.title,
        "description": item.description,
        "start_time": item.start_time,
        "end_time": item.end_time,
        "cost": float(item.cost or 0),
        "status": item.status,
        "location": item.location,
    }


class GuideContextService:
    def __init__(self, db: Session):
        self.db = db

    # -- ownership -----------------------------------------------------
    def get_owned_trip(self, user_id: str, trip_id: str) -> Optional[Trip]:
        """Trip only when owned by user_id (isolation)."""
        return (
            self.db.query(Trip)
            .filter(Trip.id == trip_id, Trip.user_id == user_id)
            .first()
        )

    def get_active_trip(self, user_id: str) -> Optional[Trip]:
        """Most recently updated non-cancelled trip for the user."""
        return (
            self.db.query(Trip)
            .filter(Trip.user_id == user_id, Trip.status != "cancelled")
            .order_by(Trip.updated_at.desc())
            .first()
        )

    # -- main builder --------------------------------------------------
    def buildContext(self, userId: str, tripId: Optional[str] = None) -> Dict[str, Any]:
        """Structured context: {user, trip, itinerary, bookings, budget,
        conversationSummary, recentMessages}. All queries scoped by userId."""
        user = self.db.query(User).filter(User.id == userId).first()
        trip: Optional[Trip] = None
        if tripId:
            trip = self.get_owned_trip(userId, tripId)
        else:
            trip = self.get_active_trip(userId)

        user_ctx: Dict[str, Any] = {}
        if user:
            first = (user.full_name or "").strip().split(" ")[0] if user.full_name else ""
            user_ctx = {
                "id": user.id,
                "full_name": user.full_name,
                "first_name": first or None,
                "email": user.email,
            }

        if trip is None:
            return {
                "user": user_ctx,
                "trip": None,
                "itinerary": {"today": [], "tomorrow": [], "upcoming": [], "completed": [], "available": False},
                "bookings": [],
                "budget": None,
                "conversationSummary": "",
                "recentMessages": [],
                "current_trip_day": None,
            }

        trip_id = trip.id
        # Minimal field loads: itinerary + bookings for THIS trip only.
        items = (
            self.db.query(ItineraryItem)
            .filter(ItineraryItem.trip_id == trip_id)
            .order_by(ItineraryItem.day_number.asc(), ItineraryItem.order_index.asc())
            .limit(300)
            .all()
        )
        bookings = (
            self.db.query(Booking)
            .filter(Booking.trip_id == trip_id)
            .order_by(Booking.booking_date.desc())
            .limit(100)
            .all()
        )
        # Conversation memory scoped by (user, trip).
        recent_rows = (
            self.db.query(GuideMessage)
            .filter(GuideMessage.user_id == userId, GuideMessage.trip_id == trip_id)
            .order_by(GuideMessage.created_at.desc())
            .limit(RECENT_MESSAGE_LIMIT)
            .all()
        )
        recent_rows = list(reversed(recent_rows))
        summary_row = (
            self.db.query(GuideConversationSummary)
            .filter(
                GuideConversationSummary.user_id == userId,
                GuideConversationSummary.trip_id == trip_id,
            )
            .first()
        )

        current_day = _current_trip_day(trip)
        today_items = [ _item_dict(i) for i in items if current_day is not None and i.day_number == current_day ]
        tomorrow_items = [ _item_dict(i) for i in items if current_day is not None and i.day_number == current_day + 1 ]
        upcoming_items = [ _item_dict(i) for i in items if current_day is None or i.day_number >= (current_day or 1) ][:30]
        completed_items = [ _item_dict(i) for i in items if i.status == "completed" ][:20]

        # Open-question support: destination facts, traveler preferences,
        # and the full bookable catalog (compact) so the Guide can answer
        # "anything" grounded — never from thin air.
        dest = trip.destination
        destination_info: Dict[str, Any] = {}
        if dest is not None:
            destination_info = {
                "description": dest.description,
                "best_time_to_visit": dest.best_time_to_visit,
                "tags": dest.tags or [],
                "state_region": dest.state_region,
                "country": dest.country,
            }
        prefs = getattr(trip, "preferences", None)
        preferences: Dict[str, Any] = {}
        if prefs is not None:
            preferences = {
                "budget_tier": prefs.budget_tier,
                "interests": prefs.interests or [],
                "travel_companions": prefs.travel_companions,
                "accommodation_types": prefs.accommodation_types or [],
                "transport_preferences": prefs.transport_preferences or [],
                "dietary_requirements": prefs.dietary_requirements or [],
                "special_requests": prefs.special_requests,
            }
        catalog = self._catalog_for_trip(trip)

        # Companions: prefer TripPreference.travel_companions + traveler_count.
        companions = None
        companion_label = None
        try:
            prefs = trip.preferences
            if prefs is not None:
                companions = prefs.travel_companions
        except Exception:
            companions = None
        traveler_count = trip.traveler_count or 1
        if companions:
            companion_label = companions
        elif traveler_count > 1:
            companion_label = f"{traveler_count} travelers"

        booking_total = round(sum(float(b.amount or 0) for b in bookings if b.status in ("confirmed", "pending")), 2)
        total_budget = float(trip.total_budget or 0)
        budget_ctx = {
            "total_budget": total_budget,
            "currency": trip.currency or "INR",
            "current_spend": booking_total,
            "remaining_budget": round(total_budget - booking_total, 2),
        }

        dest = trip.destination
        trip_ctx = {
            "id": trip.id,
            "title": trip.title,
            "name": trip.title,
            "destination": dest.name if dest else None,
            "destination_id": trip.destination_id,
            "status": trip.status,
            "start_date": trip.start_date.isoformat() if trip.start_date else None,
            "end_date": trip.end_date.isoformat() if trip.end_date else None,
            "duration_days": trip.duration_days,
            "current_trip_day": current_day,
            "traveler_count": traveler_count,
            "companions": companions,
            "companion_label": companion_label,
            "pace": trip.pace,
        }

        return {
            "user": user_ctx,
            "trip": trip_ctx,
            "itinerary": {
                "today": today_items,
                "tomorrow": tomorrow_items,
                "upcoming": upcoming_items,
                "completed": completed_items,
                "available": len(items) > 0,
                "total_items": len(items),
            },
            "bookings": [
                {
                    "id": b.id,
                    "booking_reference": b.booking_reference,
                    "item_type": b.item_type,
                    "amount": float(b.amount or 0),
                    "currency": b.currency,
                    "status": b.status,
                    "payment_status": b.payment_status,
                }
                for b in bookings
            ],
            "budget": budget_ctx,
            "destination_info": destination_info,
            "preferences": preferences,
            "catalog": catalog,
            "conversationSummary": summary_row.summary if summary_row else "",
            "recentMessages": [
                {"role": r.role, "message": r.message, "created_at": r.created_at.isoformat() if r.created_at else None}
                for r in recent_rows
            ],
        }

    def _catalog_for_trip(self, trip: Trip) -> Dict[str, Any]:
        """Compact bookable catalog for the trip destination (open questions).

        Only the fields the Guide needs; prompts stay small.
        """
        from backend.models.models import Activity, Hotel

        activities: List[Dict[str, Any]] = []
        hotels: List[Dict[str, Any]] = []
        try:
            if trip.destination_id:
                for a in (
                    self.db.query(Activity)
                    .filter(Activity.destination_id == trip.destination_id,
                            Activity.is_active == True)  # noqa: E712
                    .order_by(Activity.rating.desc()).limit(12).all()
                ):
                    activities.append({
                        "id": a.id, "title": a.title, "category": a.category,
                        "price_per_person": float(a.price_per_person or 0),
                        "currency": a.currency, "rating": a.rating,
                        "difficulty": a.difficulty_level,
                    })
                for h in (
                    self.db.query(Hotel)
                    .filter(Hotel.destination_id == trip.destination_id,
                            Hotel.is_active == True)  # noqa: E712
                    .order_by(Hotel.rating.desc()).limit(8).all()
                ):
                    hotels.append({
                        "id": h.id, "name": h.name, "category": h.category,
                        "price_per_night": float(h.price_per_night or 0),
                        "currency": h.currency, "rating": h.rating,
                    })
        except Exception:
            pass
        return {"activities": activities, "hotels": hotels}

    # -- greeting ------------------------------------------------------
    def build_greeting(self, ctx: Dict[str, Any]) -> str:
        user = ctx.get("user") or {}
        trip = ctx.get("trip")
        if trip is None:
            return "I don't see an active trip yet. Create or select a trip and I'll help you plan it."
        first = user.get("first_name")
        dest = trip.get("destination")
        title = trip.get("title") or trip.get("name")
        if first and dest and title:
            return f"Hi {first}! I'm your TourFlow Guide. You're on your {title} trip in {dest}. How can I help today?"
        if dest and title:
            return f"Hello! I'm your TourFlow Guide for your {title} trip in {dest}. How can I help today?"
        if title:
            return f"Hello! I'm your TourFlow Guide for your {title} trip. How can I help today?"
        return "Hello! I'm your TourFlow Guide. How can I help today?"

    # -- summary maintenance -------------------------------------------
    def maybe_update_summary(self, user_id: str, trip_id: str) -> None:
        """Periodically condense older history into the rolling summary.

        Cheap extractive summary (no LLM call): keeps prompt compact and fast.
        """
        count = (
            self.db.query(GuideMessage)
            .filter(GuideMessage.user_id == user_id, GuideMessage.trip_id == trip_id)
            .count()
        )
        row = (
            self.db.query(GuideConversationSummary)
            .filter(
                GuideConversationSummary.user_id == user_id,
                GuideConversationSummary.trip_id == trip_id,
            )
            .first()
        )
        prev_count = row.message_count if row else 0
        if count - prev_count < SUMMARY_REFRESH_EVERY and row is not None:
            return
        # Summarize all but the latest window.
        older = (
            self.db.query(GuideMessage)
            .filter(GuideMessage.user_id == user_id, GuideMessage.trip_id == trip_id)
            .order_by(GuideMessage.created_at.asc())
            .limit(max(0, count - RECENT_MESSAGE_LIMIT))
            .all()
        )
        if not older:
            return
        topics: List[str] = []
        for m in older[-30:]:
            snippet = (m.message or "").strip().replace("\n", " ")[:120]
            if snippet:
                topics.append(f"{m.role}: {snippet}")
        summary = "Earlier conversation: " + " | ".join(topics)
        summary = summary[:2000]
        if row is None:
            from backend.models.models import generate_uuid
            row = GuideConversationSummary(
                id=generate_uuid(), user_id=user_id, trip_id=trip_id,
                summary=summary, message_count=count,
            )
            self.db.add(row)
        else:
            row.summary = summary
            row.message_count = count
        self.db.commit()
