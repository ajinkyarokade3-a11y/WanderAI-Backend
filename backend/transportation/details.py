"""Real operator/schedule snapshots for itinerary transport items.

Every value comes from the TransportOption row (or its attached vendor);
unknown stays None so the UI hides it instead of inventing schedules,
prices, availability, providers, or booking links. ``booking_url`` falls
back to ``source_url`` — both columns hold real provider URLs only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def provider_display_name(transport: Any) -> Optional[str]:
    """Operator name: explicit column first, attached vendor second."""
    explicit = (getattr(transport, "operator_name", None) or "").strip() if getattr(
        transport, "operator_name", None) else None
    if explicit:
        return explicit
    try:
        vendor = getattr(transport, "vendor", None)
        name = getattr(vendor, "name", None) if vendor is not None else None
        clean = str(name or "").strip()
        return clean or None
    except Exception:
        return None


def transport_details_snapshot(transport: Any) -> Dict[str, Any]:
    """Build the transport_details snapshot stored on itinerary items."""
    stops = getattr(transport, "stops", None) or []
    features = getattr(transport, "features", None) or []
    return {
        "transport_id": getattr(transport, "id", None),
        "mode": getattr(transport, "type", None),
        "name": getattr(transport, "name", None),
        "operator": provider_display_name(transport),
        "service_number": getattr(transport, "service_number", None),
        "route_from": getattr(transport, "route_from", None),
        "route_to": getattr(transport, "route_to", None),
        "departure_time": getattr(transport, "departure_time", None),
        "arrival_time": getattr(transport, "arrival_time", None),
        "duration_hours": getattr(transport, "duration_hours", None),
        "stops": list(stops) if isinstance(stops, list) else [],
        "price": getattr(transport, "price", None),
        "currency": getattr(transport, "currency", None),
        "travel_class": getattr(transport, "travel_class", None),
        "capacity": getattr(transport, "capacity", None),
        "features": list(features) if isinstance(features, list) else [],
        "availability": getattr(transport, "availability_status", None),
        "booking_url": getattr(transport, "booking_url", None) or getattr(
            transport, "source_url", None),
        "inventory_source": getattr(transport, "inventory_source", None),
    }
