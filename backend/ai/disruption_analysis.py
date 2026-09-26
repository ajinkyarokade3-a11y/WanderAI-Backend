"""AI-powered disruption impact analysis.

Uses the failover router (Gemini -> OpenRouter -> Grok) to generate
practical alternative plans for trip disruptions. The AI receives real
trip data (itinerary, bookings, catalog activities, transport options)
and returns a structured analysis with alternatives that are validated
against the database before being returned to the operator.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.ai.ai_types import AIErrorCategory, AIProviderError, AIRequest
from backend.ai.router import generate_with_failover
from backend.models.models import (
    Activity, Alert, Booking, ItineraryItem, TransportOption, Trip, Vendor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON Schema for the AI response — enforces structured output.
# ---------------------------------------------------------------------------

DISRUPTION_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "disruption_cause": {"type": "string"},
        "affected_day": {"type": ["integer", "null"]},
        "affected_activity": {"type": ["string", "null"]},
        "severity": {"type": "string"},
        "impact_summary": {"type": "string"},
        "affected_bookings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "vendor_name": {"type": "string"},
                    "impact": {"type": "string"},
                },
                "required": ["id", "vendor_name", "impact"],
            },
        },
        "affected_vendors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "vendor_type": {"type": "string"},
                    "impact": {"type": "string"},
                },
                "required": ["id", "name", "vendor_type", "impact"],
            },
        },
        "transport_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "required": {"type": "boolean"},
                },
                "required": ["description", "required"],
            },
        },
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "activity_id": {"type": "string"},
                    "title": {"type": "string"},
                    "location": {"type": ["string", "null"]},
                    "cost": {"type": "number"},
                    "duration_hours": {"type": ["number", "null"]},
                    "rationale": {"type": "string"},
                    "estimated_time_impact": {"type": ["string", "null"]},
                    "estimated_cost_impact": {"type": ["number", "null"]},
                    "risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["activity_id", "title", "cost", "rationale"],
            },
        },
        "estimated_schedule_impact": {"type": ["string", "null"]},
        "estimated_cost_impact": {"type": ["number", "null"]},
        "risks_limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "disruption_cause", "severity", "impact_summary",
        "affected_bookings", "affected_vendors", "transport_changes",
        "alternatives", "risks_limitations",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are TourFlow AI's disruption impact analyzer for Indian travel. \
Analyze the given trip disruption and generate practical alternative plans. \
Only suggest alternatives from the provided catalog activities. \
Never invent vendors, bookings, prices, or schedules. \
If no suitable alternatives exist, return an empty alternatives array. \
Be specific about the disruption cause and impact. \
Consider geographic and time feasibility. \
All alternatives are suggestions pending operator approval."""


def _gather_trip_context(db: Session, trip: Trip) -> Dict[str, Any]:
    """Gather all relevant trip data for the AI prompt."""
    # Itinerary items with their related entities
    itinerary = []
    for item in trip.itinerary:
        entry = {
            "id": item.id,
            "day_number": item.day_number,
            "order_index": item.order_index,
            "item_type": item.item_type,
            "title": item.title,
            "description": item.description,
            "location": item.location,
            "cost": item.cost,
            "status": item.status,
        }
        if item.activity:
            entry["activity"] = {
                "id": item.activity.id,
                "title": item.activity.title,
                "category": item.activity.category,
                "duration_hours": item.activity.duration_hours,
                "price_per_person": item.activity.price_per_person,
                "meeting_point": item.activity.meeting_point,
                "difficulty_level": item.activity.difficulty_level,
            }
        if item.hotel:
            entry["hotel"] = {
                "id": item.hotel.id,
                "name": item.hotel.name,
                "category": item.hotel.category,
                "price_per_night": item.hotel.price_per_night,
            }
        if item.transport:
            entry["transport"] = {
                "id": item.transport.id,
                "name": item.transport.name,
                "type": item.transport.type,
                "route_from": item.transport.route_from,
                "route_to": item.transport.route_to,
                "duration_hours": item.transport.duration_hours,
                "price": item.transport.price,
            }
        itinerary.append(entry)

    # Bookings with vendor info
    bookings = []
    for booking in trip.bookings:
        entry = {
            "id": booking.id,
            "item_type": booking.item_type,
            "amount": booking.amount,
            "currency": booking.currency,
            "status": booking.status,
            "payment_status": booking.payment_status,
        }
        if booking.vendor:
            entry["vendor"] = {
                "id": booking.vendor.id,
                "name": booking.vendor.name,
                "vendor_type": booking.vendor.vendor_type,
            }
        bookings.append(entry)

    # Catalog activities for this destination (potential alternatives)
    catalog_activities = []
    if trip.destination_id:
        activities = db.query(Activity).filter(
            Activity.destination_id == trip.destination_id,
            Activity.is_active == True,  # noqa: E712
        ).all()
        for act in activities:
            catalog_activities.append({
                "id": act.id,
                "title": act.title,
                "category": act.category,
                "duration_hours": act.duration_hours,
                "price_per_person": act.price_per_person,
                "meeting_point": act.meeting_point,
                "difficulty_level": act.difficulty_level,
            })

    # Transport options for this destination
    transport_options = []
    if trip.destination_id:
        transports = db.query(TransportOption).filter(
            TransportOption.destination_id == trip.destination_id,
            TransportOption.is_active == True,  # noqa: E712
        ).all()
        for t in transports:
            transport_options.append({
                "id": t.id,
                "name": t.name,
                "type": t.type,
                "route_from": t.route_from,
                "route_to": t.route_to,
                "duration_hours": t.duration_hours,
                "price": t.price,
            })

    return {
        "trip_id": trip.id,
        "destination": trip.destination.name if trip.destination else None,
        "duration_days": trip.duration_days,
        "traveler_count": trip.traveler_count,
        "currency": trip.currency,
        "pace": trip.pace,
        "start_date": trip.start_date.isoformat() if trip.start_date else None,
        "end_date": trip.end_date.isoformat() if trip.end_date else None,
        "itinerary": itinerary,
        "bookings": bookings,
        "catalog_activities": catalog_activities,
        "transport_options": transport_options,
    }


def _build_prompt(trip_context: Dict[str, Any], disruption: Dict[str, Any]) -> str:
    """Build the AI prompt with trip context and disruption details."""
    return (
        f"Analyze this trip disruption and generate practical alternatives.\n\n"
        f"Trip context:\n{json.dumps(trip_context, indent=2)}\n\n"
        f"Disruption:\n{json.dumps(disruption, indent=2)}\n\n"
        f"Return JSON matching the provided schema."
    )


def _validate_alternatives(
    db: Session,
    trip: Trip,
    alternatives: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate AI-suggested alternatives against the database.

    Only returns alternatives that exist in the catalog, are active,
    belong to the trip's destination, and are not already in the itinerary.
    """
    if not trip.destination_id:
        return []

    active_activity_ids = {item.activity_id for item in trip.itinerary if item.activity_id}
    validated = []

    for alt in alternatives:
        activity_id = alt.get("activity_id")
        if not activity_id or activity_id in active_activity_ids:
            continue

        activity = db.query(Activity).filter(
            Activity.id == activity_id,
            Activity.destination_id == trip.destination_id,
            Activity.is_active == True,  # noqa: E712
        ).first()

        if not activity:
            continue

        validated.append({
            "activity_id": activity.id,
            "title": activity.title,
            "location": activity.meeting_point,
            "cost": round(activity.price_per_person * trip.traveler_count, 2),
            "currency": activity.currency,
            "duration_hours": activity.duration_hours,
            "rationale": alt.get("rationale", "Catalog alternative for this destination."),
            "estimated_time_impact": alt.get("estimated_time_impact"),
            "estimated_cost_impact": alt.get("estimated_cost_impact"),
            "risks": alt.get("risks", []),
        })

    return validated


def _enrich_affected_bookings(
    db: Session,
    trip: Trip,
    affected_item_ids: set[str],
) -> List[Dict[str, Any]]:
    """Find bookings linked to affected itinerary items."""
    affected = []
    for booking in trip.bookings:
        if booking.item_id and booking.item_id in affected_item_ids:
            entry = {
                "id": booking.id,
                "vendor_name": booking.vendor.name if booking.vendor else "Unknown",
                "impact": f"Booking for affected item may need rescheduling or cancellation.",
            }
            affected.append(entry)
    return affected


def _enrich_affected_vendors(
    db: Session,
    trip: Trip,
    affected_item_ids: set[str],
) -> List[Dict[str, Any]]:
    """Find vendors linked to affected itinerary items."""
    vendors = {}
    for item in trip.itinerary:
        if item.id not in affected_item_ids:
            continue
        if item.activity and item.activity.vendor:
            v = item.activity.vendor
            vendors[v.id] = {
                "id": v.id,
                "name": v.name,
                "vendor_type": v.vendor_type,
                "impact": f"Activity vendor for Day {item.day_number} item.",
            }
        if item.hotel and item.hotel.vendor:
            v = item.hotel.vendor
            vendors[v.id] = {
                "id": v.id,
                "name": v.name,
                "vendor_type": v.vendor_type,
                "impact": f"Hotel vendor for Day {item.day_number} item.",
            }
        if item.transport and item.transport.vendor:
            v = item.transport.vendor
            vendors[v.id] = {
                "id": v.id,
                "name": v.name,
                "vendor_type": v.vendor_type,
                "impact": f"Transport vendor for Day {item.day_number} item.",
            }
    return list(vendors.values())


def analyze_disruption(
    db: Session,
    trip: Trip,
    disruption: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a comprehensive disruption impact analysis using AI.

    :param db: Database session.
    :param trip: The trip being analyzed.
    :param disruption: Disruption details (type, severity, title, description).
    :raises AIProviderError: If all providers fail or the response is malformed.
    """
    trip_context = _gather_trip_context(db, trip)
    prompt = _build_prompt(trip_context, disruption)

    request = AIRequest(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=DISRUPTION_ANALYSIS_SCHEMA,
        max_output_tokens=2048,
    )

    # First attempt: strict schema enforcement
    try:
        logger.info("Disruption analysis: attempting strict schema request for trip %s", trip.id)
        response = generate_with_failover(request)
        data = _parse_response(response.text)
        data["source"] = response.provider
        logger.info(
            "Disruption analysis: strict schema succeeded via %s/%s for trip %s",
            response.provider, response.model, trip.id,
        )
        return _enrich_analysis(db, trip, data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Strict schema disruption analysis failed (%s), retrying with json_object mode",
            exc,
        )
    except AIProviderError as exc:
        logger.warning(
            "Disruption analysis: strict schema attempt failed for trip %s (%s), retrying with json_object mode",
            trip.id, exc.public_message,
        )

    # Retry: JSON mode only (no strict schema)
    request = AIRequest(
        prompt=prompt,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=None,
        max_output_tokens=2048,
    )
    try:
        logger.info("Disruption analysis: attempting json_object fallback for trip %s", trip.id)
        response = generate_with_failover(request)
        data = _parse_response(response.text)
        data["source"] = response.provider
        logger.info(
            "Disruption analysis: json_object fallback succeeded via %s/%s for trip %s",
            response.provider, response.model, trip.id,
        )
        return _enrich_analysis(db, trip, data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "Disruption analysis: json_object fallback returned malformed JSON for trip %s: %s",
            trip.id, exc,
        )
        raise AIProviderError(
            f"Disruption analysis failed: malformed JSON response: {exc}",
            category=AIErrorCategory.SERVER_ERROR,
            provider="",
            model="",
        ) from exc


def _parse_response(text: str) -> Dict[str, Any]:
    """Parse and validate the AI response."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Disruption analysis response must be a JSON object")
    return data


def _enrich_analysis(
    db: Session,
    trip: Trip,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Enrich the AI analysis with database-validated data.

    - Validates alternatives against the catalog
    - Enriches affected bookings and vendors from the database
    - Identifies affected itinerary items
    """
    # Determine affected items from the disruption
    affected_item_ids = _identify_affected_items(trip, data)

    # Validate alternatives against the database
    raw_alternatives = data.get("alternatives", [])
    validated_alternatives = _validate_alternatives(db, trip, raw_alternatives)

    # Enrich affected bookings and vendors
    affected_bookings = _enrich_affected_bookings(db, trip, affected_item_ids)
    affected_vendors = _enrich_affected_vendors(db, trip, affected_item_ids)

    # Build affected items list
    affected_items = []
    for item in trip.itinerary:
        if item.id in affected_item_ids:
            affected_items.append({
                "id": item.id,
                "day_number": item.day_number,
                "item_type": item.item_type,
                "title": item.title,
                "location": item.location,
                "cost": item.cost,
                "status": item.status,
            })

    return {
        "trip_id": trip.id,
        "disruption_cause": data.get("disruption_cause", "Unknown disruption"),
        "affected_day": data.get("affected_day"),
        "affected_activity": data.get("affected_activity"),
        "severity": data.get("severity", "warning"),
        "impact_summary": data.get("impact_summary", ""),
        "affected_items": affected_items,
        "affected_bookings": affected_bookings,
        "affected_vendors": affected_vendors,
        "transport_changes": data.get("transport_changes", []),
        "alternatives": validated_alternatives,
        "estimated_schedule_impact": data.get("estimated_schedule_impact"),
        "estimated_cost_impact": data.get("estimated_cost_impact"),
        "risks_limitations": data.get("risks_limitations", []),
        "source": data.get("source", "ai"),
        "status": "pending_approval",
    }


def _identify_affected_items(trip: Trip, data: Dict[str, Any]) -> set[str]:
    """Identify which itinerary items are affected by the disruption."""
    affected_day = data.get("affected_day")
    affected_activity = data.get("affected_activity")

    affected_ids = set()
    for item in trip.itinerary:
        if item.status not in ("proposed", "confirmed"):
            continue
        if affected_day is not None and item.day_number == affected_day:
            affected_ids.add(item.id)
        elif affected_activity and affected_activity.lower() in (item.title or "").lower():
            affected_ids.add(item.id)

    # If no specific items matched, fall back to outdoor items on the affected day
    if not affected_ids and affected_day is not None:
        for item in trip.itinerary:
            if item.status in ("proposed", "confirmed") and item.day_number == affected_day:
                affected_ids.add(item.id)

    return affected_ids
