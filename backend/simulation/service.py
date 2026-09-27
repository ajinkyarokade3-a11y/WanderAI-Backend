"""Digital Twin trip simulation service.

Deterministic what-if analysis: given a trip's itinerary and weather context,
identifies affected items using causal, activity-specific impact logic.
Never modifies the real itinerary.

Impact is determined by:
1. Selected weather scenario
2. Activity/transport type
3. Indoor vs outdoor classification
4. Transportation dependency
5. Safety considerations
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.models import Activity, ItineraryItem, TransportOption, Trip

logger = logging.getLogger(__name__)

# Impact severity levels
IMPACT_NONE = "none"
IMPACT_LOW = "low"
IMPACT_MODERATE = "moderate"
IMPACT_HIGH = "high"
IMPACT_SEVERE = "severe"

# Impact types
TYPE_NONE = "none"
TYPE_DELAY = "delay"
TYPE_MODIFIED = "modified"
TYPE_POSTPONED = "postponed"
TYPE_CANCELLED = "cancelled"

# Recommended actions
ACTION_NO_CHANGE = "no_change"
ACTION_MONITOR = "monitor"
ACTION_RESCHEDULE = "reschedule"
ACTION_MOVE_INDOORS = "move_indoors"
ACTION_CANCEL = "cancel"

# Weather severity levels for internal use
SEVERITY_LEVELS = {"low": 0, "moderate": 1, "high": 2, "severe": 3, "unknown": -1}


def _parse_time(time_str: Optional[str]) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        return datetime.strptime(time_str.strip(), "%H:%M")
    except (ValueError, TypeError):
        return None


def _time_to_minutes(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    return dt.hour * 60 + dt.minute


def _get_severity_level(severity: Optional[str]) -> int:
    if not severity:
        return -1
    return SEVERITY_LEVELS.get(severity.lower(), -1)


def _is_outdoor_activity(activity: Optional[Activity]) -> bool:
    """Determine if an activity is outdoor based on its category."""
    if not activity:
        return True
    outdoor_categories = {"adventure", "nature"}
    return (activity.category or "").lower() in outdoor_categories


def _is_outdoor_transport(transport: Optional[TransportOption]) -> bool:
    """Determine if a transport type is weather-sensitive."""
    if not transport:
        return True
    weather_sensitive = {"flight", "boat", "self_drive", "private_cab"}
    return (transport.type or "").lower() in weather_sensitive


def _compute_item_impact(
    item: ItineraryItem,
    scenario_severity: str,
    activity: Optional[Activity] = None,
    transport: Optional[TransportOption] = None,
) -> Dict[str, Any]:
    """Compute causal weather impact for a single itinerary item.

    Returns dict with: impact_severity, impact_type, recommended_action,
    explanation, estimated_delay_minutes (only when applicable).
    """
    item_type = (item.item_type or "").lower()
    is_outdoor = True

    if item_type == "activity" and activity:
        is_outdoor = _is_outdoor_activity(activity)
    elif item_type == "transport" and transport:
        is_outdoor = _is_outdoor_transport(transport)

    # Clear/low weather: no impact for anything
    if scenario_severity == "low":
        return {
            "impact_severity": IMPACT_NONE,
            "impact_type": TYPE_NONE,
            "recommended_action": ACTION_NO_CHANGE,
            "explanation": "No weather impact",
            "estimated_delay_minutes": None,
        }

    if item_type == "transport":
        return _compute_transport_impact(item, transport, scenario_severity, is_outdoor)
    elif item_type == "activity":
        return _compute_activity_impact(item, activity, scenario_severity, is_outdoor)
    elif item_type == "meal":
        return _compute_meal_impact(item, scenario_severity, is_outdoor)
    elif item_type == "hotel":
        return _compute_hotel_impact(scenario_severity)
    else:
        return {
            "impact_severity": IMPACT_NONE,
            "impact_type": TYPE_NONE,
            "recommended_action": ACTION_NO_CHANGE,
            "explanation": "No weather impact",
            "estimated_delay_minutes": None,
        }


def _compute_transport_impact(
    item: ItineraryItem,
    transport: Optional[TransportOption],
    severity: str,
    is_outdoor: bool,
) -> Dict[str, Any]:
    """Compute impact for transport items based on type and weather."""
    transport_type = (transport.type or "").lower() if transport else ""
    title = item.title or "Transport"

    if transport_type == "flight":
        if severity == "moderate":
            return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                    "explanation": f"Flight may experience minor delays due to {severity} weather conditions",
                    "estimated_delay_minutes": 15}
        elif severity == "high":
            return {"impact_severity": IMPACT_HIGH, "impact_type": TYPE_DELAY, "recommended_action": ACTION_RESCHEDULE,
                    "explanation": f"Flight may experience significant delays due to adverse weather",
                    "estimated_delay_minutes": 45}
        elif severity == "severe":
            return {"impact_severity": IMPACT_HIGH, "impact_type": TYPE_DELAY, "recommended_action": ACTION_RESCHEDULE,
                    "explanation": f"Flight faces high disruption risk; delays or cancellation possible due to severe weather",
                    "estimated_delay_minutes": 120}
    elif transport_type == "boat":
        if severity == "moderate":
            return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                    "explanation": "Boat ride may experience minor delays",
                    "estimated_delay_minutes": 15}
        elif severity == "high":
            return {"impact_severity": IMPACT_MODERATE, "impact_type": TYPE_MODIFIED, "recommended_action": ACTION_RESCHEDULE,
                    "explanation": "Boat ride may be delayed or rescheduled due to weather conditions",
                    "estimated_delay_minutes": 30}
        elif severity == "severe":
            return {"impact_severity": IMPACT_SEVERE, "impact_type": TYPE_CANCELLED, "recommended_action": ACTION_CANCEL,
                    "explanation": "Boat ride likely cancelled due to severe weather and safety concerns",
                    "estimated_delay_minutes": None}
    elif transport_type in ("private_cab", "self_drive"):
        if severity == "moderate":
            return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                    "explanation": "Road transport may experience minor delays",
                    "estimated_delay_minutes": 10}
        elif severity == "high":
            return {"impact_severity": IMPACT_MODERATE, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                    "explanation": "Road transport may experience delays due to weather conditions",
                    "estimated_delay_minutes": 20}
        elif severity == "severe":
            return {"impact_severity": IMPACT_HIGH, "impact_type": TYPE_DELAY, "recommended_action": ACTION_RESCHEDULE,
                    "explanation": "Road transport may experience significant delays due to severe weather",
                    "estimated_delay_minutes": 45}
    elif transport_type in ("volvo_bus", "train"):
        if severity == "moderate":
            return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
                    "explanation": "No weather impact",
                    "estimated_delay_minutes": None}
        elif severity == "high":
            return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                    "explanation": "Bus/train may experience minor delays",
                    "estimated_delay_minutes": 15}
        elif severity == "severe":
            return {"impact_severity": IMPACT_MODERATE, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                    "explanation": "Bus/train may experience delays due to severe weather",
                    "estimated_delay_minutes": 30}
    else:
        # Unknown transport type — treat as outdoor
        return _compute_generic_outdoor_impact(severity, "Transport")


def _compute_activity_impact(
    item: ItineraryItem,
    activity: Optional[Activity],
    severity: str,
    is_outdoor: bool,
) -> Dict[str, Any]:
    """Compute impact for activity items based on indoor/outdoor and weather."""
    title = item.title or "Activity"

    if not is_outdoor:
        # Indoor activities: generally unaffected by weather itself
        if severity in ("moderate", "high"):
            return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
                    "explanation": "No weather impact (indoor activity)",
                    "estimated_delay_minutes": None}
        elif severity == "severe":
            return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_NONE, "recommended_action": ACTION_MONITOR,
                    "explanation": "Indoor activity generally available; monitor for transport/safety impacts",
                    "estimated_delay_minutes": None}

    # Outdoor activities
    if severity == "moderate":
        return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                "explanation": "Outdoor activity may experience minor delays",
                "estimated_delay_minutes": 15}
    elif severity == "high":
        return {"impact_severity": IMPACT_MODERATE, "impact_type": TYPE_MODIFIED, "recommended_action": ACTION_MOVE_INDOORS,
                "explanation": "Outdoor activity may be delayed, shortened, or moved indoors due to weather",
                "estimated_delay_minutes": 30}
    elif severity == "severe":
        return {"impact_severity": IMPACT_HIGH, "impact_type": TYPE_POSTPONED, "recommended_action": ACTION_RESCHEDULE,
                "explanation": "Outdoor activity likely postponed or cancelled due to severe weather and safety concerns",
                "estimated_delay_minutes": None}

    return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
            "explanation": "No weather impact", "estimated_delay_minutes": None}


def _compute_meal_impact(
    item: ItineraryItem,
    severity: str,
    is_outdoor: bool,
) -> Dict[str, Any]:
    """Compute impact for meal items."""
    if not is_outdoor:
        return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
                "explanation": "No weather impact (indoor dining)",
                "estimated_delay_minutes": None}
    if severity == "moderate":
        return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                "explanation": "Outdoor dining may experience minor delays",
                "estimated_delay_minutes": 10}
    elif severity == "high":
        return {"impact_severity": IMPACT_MODERATE, "impact_type": TYPE_MODIFIED, "recommended_action": ACTION_MOVE_INDOORS,
                "explanation": "Outdoor dining may be moved indoors due to weather",
                "estimated_delay_minutes": None}
    elif severity == "severe":
        return {"impact_severity": IMPACT_HIGH, "impact_type": TYPE_MODIFIED, "recommended_action": ACTION_MOVE_INDOORS,
                "explanation": "Outdoor dining likely moved indoors due to severe weather",
                "estimated_delay_minutes": None}
    return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
            "explanation": "No weather impact", "estimated_delay_minutes": None}


def _compute_hotel_impact(severity: str) -> Dict[str, Any]:
    """Hotels are indoor accommodation — generally unaffected by weather."""
    if severity == "severe":
        return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_NONE, "recommended_action": ACTION_MONITOR,
                "explanation": "Hotel stay generally unaffected; monitor for transport disruptions",
                "estimated_delay_minutes": None}
    return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
            "explanation": "No weather impact", "estimated_delay_minutes": None}


def _compute_generic_outdoor_impact(severity: str, label: str) -> Dict[str, Any]:
    """Generic outdoor impact for unknown types."""
    if severity == "moderate":
        return {"impact_severity": IMPACT_LOW, "impact_type": TYPE_DELAY, "recommended_action": ACTION_MONITOR,
                "explanation": f"{label} may experience minor delays due to weather",
                "estimated_delay_minutes": 15}
    elif severity == "high":
        return {"impact_severity": IMPACT_MODERATE, "impact_type": TYPE_MODIFIED, "recommended_action": ACTION_RESCHEDULE,
                "explanation": f"{label} may be delayed or modified due to weather",
                "estimated_delay_minutes": 30}
    elif severity == "severe":
        return {"impact_severity": IMPACT_HIGH, "impact_type": TYPE_POSTPONED, "recommended_action": ACTION_RESCHEDULE,
                "explanation": f"{label} likely postponed due to severe weather",
                "estimated_delay_minutes": None}
    return {"impact_severity": IMPACT_NONE, "impact_type": TYPE_NONE, "recommended_action": ACTION_NO_CHANGE,
            "explanation": "No weather impact", "estimated_delay_minutes": None}


def _detect_dependencies(
    items: List[ItineraryItem],
    affected_ids: set,
) -> List[Dict[str, Any]]:
    """Detect dependency chains between affected and downstream items."""
    dependencies = []
    by_day: Dict[int, List[ItineraryItem]] = {}
    for item in items:
        by_day.setdefault(item.day_number, []).append(item)

    for day_num, day_items in by_day.items():
        day_items_sorted = sorted(day_items, key=lambda x: (x.order_index, x.start_time or ""))

        affected_transports = [
            i for i in day_items_sorted
            if i.id in affected_ids and i.item_type == "transport"
        ]

        for transport in affected_transports:
            transport_time = _time_to_minutes(_parse_time(transport.start_time))
            if transport_time is None:
                continue

            for downstream in day_items_sorted:
                if downstream.id == transport.id:
                    continue
                if downstream.order_index <= transport.order_index:
                    continue

                downstream_time = _time_to_minutes(_parse_time(downstream.start_time))
                if downstream_time is None:
                    continue

                if downstream_time - transport_time <= 120:
                    dependencies.append({
                        "source_item_id": transport.id,
                        "source_title": transport.title,
                        "target_item_id": downstream.id,
                        "target_title": downstream.title,
                        "dependency_type": "cascading_delay",
                        "description": f"Transport delay may cause late arrival for {downstream.title}",
                    })

        affected_activities = [
            i for i in day_items_sorted
            if i.id in affected_ids and i.item_type == "activity"
        ]

        for activity in affected_activities:
            activity_time = _time_to_minutes(_parse_time(activity.start_time))
            if activity_time is None:
                continue

            for meal in day_items_sorted:
                if meal.item_type != "meal":
                    continue
                if meal.id in affected_ids:
                    continue

                meal_time = _time_to_minutes(_parse_time(meal.start_time))
                if meal_time is None:
                    continue

                if 0 <= meal_time - activity_time <= 60:
                    dependencies.append({
                        "source_item_id": activity.id,
                        "source_title": activity.title,
                        "target_item_id": meal.id,
                        "target_title": meal.title,
                        "dependency_type": "timing_conflict",
                        "description": f"Activity delay may affect meal timing for {meal.title}",
                    })

    return dependencies


def _detect_conflicts(
    items: List[ItineraryItem],
    affected_ids: set,
) -> List[Dict[str, Any]]:
    """Detect schedule conflicts caused by weather delays."""
    conflicts = []

    by_day: Dict[int, List[ItineraryItem]] = {}
    for item in items:
        by_day.setdefault(item.day_number, []).append(item)

    for day_num, day_items in by_day.items():
        day_items_sorted = sorted(day_items, key=lambda x: (x.start_time or "", x.order_index))

        for i, item in enumerate(day_items_sorted):
            if item.id not in affected_ids:
                continue

            item_end = _time_to_minutes(_parse_time(item.end_time))
            if item_end is None:
                continue

            if i + 1 < len(day_items_sorted):
                next_item = day_items_sorted[i + 1]
                next_start = _time_to_minutes(_parse_time(next_item.start_time))
                if next_start is None:
                    continue

                # Get the delay for this item to check for overlap
                for affected in items:
                    if affected.id == item.id:
                        # Recompute delay from impact
                        break

                # Use a simple heuristic: if next item starts within 30 min of current end
                if next_start < item_end + 30:
                    conflicts.append({
                        "item_id": next_item.id,
                        "item_title": next_item.title,
                        "conflict_type": "timing_overlap",
                        "description": f"Weather delay may cause overlap with {next_item.title}",
                        "severity": "moderate",
                    })

    return conflicts


def simulate_trip_weather_impact(
    db: Session,
    trip: Trip,
    weather_context: Dict[str, Any],
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """Run deterministic weather impact simulation for a trip.

    Uses causal, activity-specific impact logic. Never modifies the real itinerary.
    """
    items = trip.itinerary or []
    if not items:
        return {
            "trip_id": trip.id,
            "weather_context": weather_context,
            "scenario": scenario,
            "affected_items": [],
            "dependencies": [],
            "conflicts": [],
            "replanning_required": False,
            "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Determine effective severity from scenario or weather context
    current = weather_context.get("current") or {}
    effective_severity = current.get("severity") or "unknown"

    if scenario:
        scenario_lower = scenario.lower()
        if scenario_lower == "unavailable":
            return {
                "trip_id": trip.id,
                "weather_context": {"available": False, "reason": "scenario_unavailable"},
                "scenario": scenario,
                "affected_items": [],
                "dependencies": [],
                "conflicts": [],
                "replanning_required": False,
                "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        scenario_severity_map = {
            "severe": "severe",
            "heavy_rain": "high",
            "high_wind": "high",
            "moderate": "moderate",
            "low": "low",
        }
        effective_severity = scenario_severity_map.get(scenario_lower, effective_severity)

    severity_level = _get_severity_level(effective_severity)

    # If weather is unavailable or severity is low, no impact
    if not weather_context.get("available", False):
        return {
            "trip_id": trip.id,
            "weather_context": weather_context,
            "scenario": scenario,
            "affected_items": [],
            "all_items": [],
            "dependencies": [],
            "conflicts": [],
            "replanning_required": False,
            "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if severity_level <= 0:
        # Build all_items with no impact for each item
        no_impact_items = []
        for item in items:
            if item.status in ("skipped", "cancelled"):
                continue
            no_impact_items.append({
                "item_id": item.id,
                "day_number": item.day_number,
                "order_index": item.order_index,
                "item_type": item.item_type,
                "title": item.title,
                "reason": "No weather impact",
                "severity": IMPACT_NONE,
                "impact_type": TYPE_NONE,
                "recommended_action": ACTION_NO_CHANGE,
                "estimated_delay_minutes": None,
            })
        return {
            "trip_id": trip.id,
            "weather_context": weather_context,
            "scenario": scenario,
            "affected_items": [],
            "all_items": no_impact_items,
            "dependencies": [],
            "conflicts": [],
            "replanning_required": False,
            "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Build lookup maps for activity and transport details
    # Use ORM relationships first (loaded via joinedload), fall back to DB query
    activity_map = {}
    transport_map = {}

    for item in items:
        if item.activity_id:
            if item.activity:
                activity_map[item.activity_id] = item.activity
            else:
                # Fallback: query the database
                activity = db.query(Activity).filter(Activity.id == item.activity_id).first()
                if activity:
                    activity_map[item.activity_id] = activity
        if item.transport_id:
            if item.transport:
                transport_map[item.transport_id] = item.transport
            else:
                # Fallback: query the database
                transport = db.query(TransportOption).filter(TransportOption.id == item.transport_id).first()
                if transport:
                    transport_map[item.transport_id] = transport

    # Compute impact for each item using causal logic
    affected_items = []
    all_items_with_impact = []
    affected_ids = set()

    for item in items:
        if item.status in ("skipped", "cancelled"):
            continue

        activity = activity_map.get(item.activity_id) if item.activity_id else None
        transport = transport_map.get(item.transport_id) if item.transport_id else None

        impact = _compute_item_impact(item, effective_severity, activity, transport)

        item_result = {
            "item_id": item.id,
            "day_number": item.day_number,
            "order_index": item.order_index,
            "item_type": item.item_type,
            "title": item.title,
            "reason": impact["explanation"],
            "severity": impact["impact_severity"],
            "impact_type": impact["impact_type"],
            "recommended_action": impact["recommended_action"],
            "estimated_delay_minutes": impact["estimated_delay_minutes"],
        }
        all_items_with_impact.append(item_result)

        # Only count items with actual impact as affected
        if impact["impact_severity"] != IMPACT_NONE:
            affected_items.append(item_result)
            affected_ids.add(item.id)

    # Detect dependencies and conflicts
    dependencies = _detect_dependencies(items, affected_ids)
    conflicts = _detect_conflicts(items, affected_ids)

    # Replanning is required if any item has high/severe impact or there are conflicts
    has_high_impact = any(
        i["severity"] in (IMPACT_HIGH, IMPACT_SEVERE) for i in affected_items
    )
    replanning_required = has_high_impact or len(conflicts) > 0 or len(affected_ids) >= 3

    return {
        "trip_id": trip.id,
        "weather_context": weather_context,
        "scenario": scenario,
        "affected_items": affected_items,
        "all_items": all_items_with_impact,
        "dependencies": dependencies,
        "conflicts": conflicts,
        "replanning_required": replanning_required,
        "simulation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
