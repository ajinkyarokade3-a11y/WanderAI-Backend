"""Unified traveler trip activity feed.

Sources (existing tables only):
- Notification (traveler-visible)
- Alert (disruption)
- ChangeHistory (sanitized)
- Trip.confirmed_at / Trip.status (trip_confirmed)

TripMessage is intentionally excluded: internal operator messages
are never exposed to travelers.

Sanitization:
- ChangeHistory old_value/new_value/internal metadata not exposed.
- Only traveler-safe title/message derived from reason/action.
- No vendor private info, no operator-only fields.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from backend.models.models import Trip, Notification, Alert, ChangeHistory

ALLOWED_TYPES = {
    "itinerary_updated",
    "booking_updated",
    "transport_updated",
    "accommodation_updated",
    "notification",
    "disruption_alert",
    "trip_confirmed",
    "trip_replanned",
}

# Action -> activity type mapping (ChangeHistory)
_CHANGE_TYPE_MAP = {
    "trip_created": "itinerary_updated",
    "preferences_updated": "itinerary_updated",
    "item_added": "itinerary_updated",
    "item_deleted": "itinerary_updated",
    "activity_swapped": "itinerary_updated",
    "activity_edited": "itinerary_updated",
    "restaurant_added": "itinerary_updated",
    "restaurant_removed": "itinerary_updated",
    "update_title": "itinerary_updated",
    "update_status": "itinerary_updated",
    "update_start_date": "itinerary_updated",
    "update_end_date": "itinerary_updated",
    "update_duration_days": "itinerary_updated",
    "update_total_budget": "itinerary_updated",
    "update_pace": "itinerary_updated",
    "transport_changed": "transport_updated",
    "hotel_changed": "accommodation_updated",
    "hotel_selected": "accommodation_updated",
    "booking_cancelled": "booking_updated",
    "booking_confirmed": "booking_updated",
    "booking_confirm": "booking_updated",
    "booking_cancel": "booking_updated",
    "booking_rebook": "booking_updated",
    "replan_applied": "trip_replanned",
    "disruption_triggered": "disruption_alert",
    "alert_resolved": "disruption_alert",
    "request_confirmed": "trip_confirmed",
    "request_cancelled": "itinerary_updated",
}


def _map_change_type(action: str) -> str:
    if action in _CHANGE_TYPE_MAP:
        return _CHANGE_TYPE_MAP[action]
    if action.startswith("booking_"):
        return "booking_updated"
    if action.startswith("update_"):
        return "itinerary_updated"
    return "itinerary_updated"


def _sanitized_change_message(ch: ChangeHistory) -> str:
    # Prefer reason (already traveler-safe), fallback to action description
    if ch.reason and ch.reason.strip():
        return ch.reason.strip()
    # Sanitized fallback without leaking old/new values
    if ch.field_changed:
        return f"{ch.action} ({ch.field_changed})"
    return ch.action


def _severity_for_change(action: str) -> str:
    if action in ("replan_applied", "disruption_triggered"):
        return "warning"
    if action in ("booking_cancelled",):
        return "warning"
    return "info"


def _get_owned_trip(db: Session, user_id: str, trip_id: str) -> Trip:
    trip = db.query(Trip).filter(Trip.id == trip_id, Trip.user_id == user_id).first()
    if not trip:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


def build_activity_events(db: Session, trip: Trip) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    # Notifications
    for n in db.query(Notification).filter(Notification.trip_id == trip.id, Notification.user_id == trip.user_id).all():
        sev = "info"
        if n.type == "warning":
            sev = "warning"
        elif n.type == "success":
            sev = "success"
        events.append({
            "id": f"notification-{n.id}",
            "type": "notification",
            "title": n.title,
            "message": n.message,
            "trip_id": trip.id,
            "created_at": n.created_at.isoformat() if n.created_at else datetime.utcnow().isoformat(),
            "severity": sev,
            "is_read": bool(n.is_read),
            "_sort_key": n.created_at or datetime.min,
        })

    # Alerts -> disruption_alert
    for a in db.query(Alert).filter(Alert.trip_id == trip.id).all():
        sev = a.severity or "warning"
        # is_read = resolved means read
        events.append({
            "id": f"alert-{a.id}",
            "type": "disruption_alert",
            "title": a.title,
            "message": a.description,
            "trip_id": trip.id,
            "created_at": a.created_at.isoformat() if a.created_at else datetime.utcnow().isoformat(),
            "severity": sev,
            "is_read": bool(a.is_resolved),
            "_sort_key": a.created_at or datetime.min,
        })

    # ChangeHistory -> sanitized
    for ch in db.query(ChangeHistory).filter(ChangeHistory.trip_id == trip.id).all():
        typ = _map_change_type(ch.action)
        # Skip actions that would duplicate notification/alert already covered? Keep all.
        events.append({
            "id": f"change-{ch.id}",
            "type": typ,
            "title": ch.action.replace("_", " ").title(),
            "message": _sanitized_change_message(ch),
            "trip_id": trip.id,
            "created_at": ch.timestamp.isoformat() if ch.timestamp else datetime.utcnow().isoformat(),
            "severity": _severity_for_change(ch.action),
            "is_read": False,  # change history is feed, not separately readable; treat as unread unless notification covers it
            "_sort_key": ch.timestamp or datetime.min,
        })

    # Trip confirmed synthetic event
    if trip.confirmed_at:
        events.append({
            "id": f"trip_confirmed-{trip.id}",
            "type": "trip_confirmed",
            "title": "Trip Confirmed",
            "message": "Your trip has been confirmed.",
            "trip_id": trip.id,
            "created_at": trip.confirmed_at.isoformat() if isinstance(trip.confirmed_at, datetime) else str(trip.confirmed_at),
            "severity": "success",
            "is_read": False,
            "_sort_key": trip.confirmed_at if isinstance(trip.confirmed_at, datetime) else datetime.min,
        })

    # Sort newest first
    events.sort(key=lambda e: e["_sort_key"], reverse=True)
    for e in events:
        e.pop("_sort_key", None)
    return events


def list_activity(db: Session, user_id: str, trip_id: str, limit: int = 20, offset: int = 0, type_filter: Optional[str] = None):
    trip = _get_owned_trip(db, user_id, trip_id)
    events = build_activity_events(db, trip)
    if type_filter:
        events = [e for e in events if e["type"] == type_filter]
    total = len(events)
    unread = sum(1 for e in events if not e["is_read"])
    paged = events[offset: offset + limit]
    return paged, total, unread


def unread_count(db: Session, user_id: str, trip_id: str) -> int:
    trip = _get_owned_trip(db, user_id, trip_id)
    events = build_activity_events(db, trip)
    return sum(1 for e in events if not e["is_read"])
