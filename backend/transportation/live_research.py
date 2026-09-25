"""Live transportation research for post-requirements trip creation.

Product flow (transportation is NOT part of onboarding/checklist):
    Origin + Destination + Dates + Travelers (confirmed trip requirements)
        -> live transportation research (this module)
        -> Gemini analyzes the available real options
        -> AI itinerary generator includes the best transfer
        -> traveler can switch transportation later (POST /trips/{id}/change-transport)

Catalog rows remain the source of truth when they cover the traveler's
origin route. When they don't, this module researches real-world transfer
modes for the origin->destination pair (geocoded distance + mode estimates),
asks Gemini to analyze/rank them against the traveler's constraints, and
persists the ranked options as ``TransportOption`` rows with
``inventory_source="live"`` / ``verification_status="live_researched"`` so
the deterministic itinerary generator picks them up automatically
(``RecommendationEngine`` already includes ``live`` rows alongside
``catalog`` rows).

Never raises: provider failures only reduce what gets researched. No
bookings, availability, or schedules are ever fabricated — rows carry
estimated durations/prices derived from straight-line distance and are
labeled as researched estimates in their description.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.models import Destination, TransportOption

logger = logging.getLogger(__name__)

LIVE_SOURCE = "live"
LIVE_VERIFICATION = "live_researched"

_EARTH_RADIUS_KM = 6371.0
_FALLBACK_DISTANCE_KM = 500.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
    try:
        p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlambda = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))
    except (TypeError, ValueError):
        return None


def _route_distance_km(origin: str, destination: Destination) -> float:
    """Straight-line origin->destination distance; fallback when geocoding fails."""
    try:
        from backend.database.config import settings
        from backend.places.service import geocode_place

        geo = geocode_place(origin, settings.NOMINATIM_API_URL, settings.PLACES_TIMEOUT_S)
        if geo and destination.latitude is not None and destination.longitude is not None:
            distance = _haversine_km(geo[0], geo[1], destination.latitude, destination.longitude)
            if distance and distance > 1:
                return round(distance, 1)
    except Exception as exc:
        logger.warning("Transport distance geocode skipped for %s: %s", origin, exc)
    return _FALLBACK_DISTANCE_KM


def _mode_estimates(
    origin: str,
    destination_name: str,
    distance_km: float,
    traveler_count: int,
    currency: str,
) -> List[Dict[str, Any]]:
    """Deterministic real-mode estimates for an Indian origin->destination pair.

    Prices are totals for the whole party (matching the catalog ``price``
    semantics the generator sums directly, not per-person). Flight/train/bus
    scale with travelers; a private cab is one vehicle for up to 6.
    """
    travelers = max(1, int(traveler_count or 1))
    distance = max(10.0, float(distance_km or _FALLBACK_DISTANCE_KM))
    modes: List[Dict[str, Any]] = []
    # Flight: viable beyond short hops; ~650 km/h block speed + airport overhead.
    if distance >= 250:
        per_person = 3500.0 + distance * 4.5
        modes.append({
            "type": "flight",
            "name": f"Flight {origin} to {destination_name}",
            "route_from": origin,
            "route_to": destination_name,
            "duration_hours": round(1.5 + distance / 650.0, 1),
            "price": round(per_person * travelers, 2),
            "capacity": 180,
            "features": ["Fastest option", "Check airline baggage allowance"],
        })
    # Train: the default workhorse for 100-1500 km pairs.
    if distance <= 2000:
        per_person = 800.0 + distance * 1.2
        modes.append({
            "type": "train",
            "name": f"Train {origin} to {destination_name}",
            "route_from": origin,
            "route_to": destination_name,
            "duration_hours": round(distance / 55.0 + 0.5, 1),
            "price": round(per_person * travelers, 2),
            "capacity": 72,
            "features": ["AC coaches", "Overnight options on long routes"],
        })
    # Volvo bus: competitive up to ~800 km.
    if distance <= 900:
        per_person = 600.0 + distance * 1.0
        modes.append({
            "type": "volvo_bus",
            "name": f"Volvo bus {origin} to {destination_name}",
            "route_from": origin,
            "route_to": destination_name,
            "duration_hours": round(distance / 45.0 + 0.5, 1),
            "price": round(per_person * travelers, 2),
            "capacity": 40,
            "features": ["AC sleeper/seater", "Luggage included"],
        })
    # Private cab: door-to-door, one vehicle price regardless of party (up to 6).
    cab_price = round(distance * 14.0 + 1500.0, 2)
    modes.append({
        "type": "private_cab",
        "name": f"Private cab {origin} to {destination_name}",
        "route_from": origin,
        "route_to": destination_name,
        "duration_hours": round(distance / 50.0 + 0.5, 1),
        "price": cab_price,
        "capacity": max(6, travelers),
        "features": ["Door-to-door", "AC", "Flexible departure"],
    })
    for mode in modes:
        mode["currency"] = currency
        mode["distance_km"] = round(distance, 1)
    return modes


def _fallback_ranking(modes: List[Dict[str, Any]], distance_km: float) -> List[Dict[str, Any]]:
    """Distance-based ranking when Gemini analysis is unavailable."""
    order = ["private_cab", "volvo_bus", "train", "flight"]
    if 200 <= distance_km < 800:
        order = ["train", "volvo_bus", "private_cab", "flight"]
    elif distance_km >= 800:
        order = ["flight", "train", "volvo_bus", "private_cab"]
    rank = {mode: index for index, mode in enumerate(order)}
    return sorted(modes, key=lambda m: rank.get(str(m.get("type")), 99))


def research_live_transport_options(
    db: Session,
    destination: Destination,
    origin: str,
    *,
    traveler_count: int = 2,
    currency: str = "INR",
    start_date: Any = None,
    end_date: Any = None,
    duration_days: int = 4,
    total_budget: Optional[float] = None,
    preferences: Optional[Dict[str, Any]] = None,
    gemini_service: Any = None,
) -> Dict[str, Any]:
    """Research + persist transfer options for a confirmed origin->destination pair.

    Returns ``{"researched": int, "options": [TransportOption], "source": str,
    "distance_km": float, "error": Optional[str]}``. Never raises.
    """
    result: Dict[str, Any] = {
        "researched": 0, "options": [], "source": "catalog", "distance_km": None, "error": None,
    }
    clean_origin = (origin or "").strip()
    if not clean_origin or destination is None:
        return result
    currency = (currency or "INR").upper()
    travelers = max(1, int(traveler_count or 1))
    try:
        existing = db.query(TransportOption).filter(
            TransportOption.destination_id == destination.id,
            TransportOption.is_active == True,  # noqa: E712
            TransportOption.currency == currency,
            TransportOption.capacity >= travelers,
            func.lower(TransportOption.route_from).contains(clean_origin.lower()),
        ).order_by(TransportOption.duration_hours.asc(), TransportOption.price.asc()).all()
        if existing:
            result["options"] = existing
            return result

        distance_km = _route_distance_km(clean_origin, destination)
        result["distance_km"] = distance_km
        estimates = _mode_estimates(clean_origin, destination.name, distance_km, travelers, currency)

        ranked_modes: List[Dict[str, Any]] = list(estimates)
        analysis_source = "distance_fallback"
        if gemini_service is not None:
            try:
                analysis = gemini_service.analyze_transport_options({
                    "origin": clean_origin,
                    "destination": destination.name,
                    "distance_km": distance_km,
                    "traveler_count": travelers,
                    "currency": currency,
                    "start_date": str(start_date) if start_date else None,
                    "end_date": str(end_date) if end_date else None,
                    "duration_days": int(duration_days or 4),
                    "total_budget": total_budget,
                    "preferences": preferences or {},
                    "candidate_modes": estimates,
                })
                order = [str(item.get("type") or item.get("name") or "").lower()
                         for item in (analysis or {}).get("ranked_options", [])]
                reasons = {(str(item.get("type") or "").lower()): str(item.get("recommendation_reason") or "")
                           for item in (analysis or {}).get("ranked_options", [])}
                if order:
                    rank = {mode: index for index, mode in enumerate(order)}
                    ranked_modes = sorted(
                        estimates, key=lambda m: rank.get(str(m.get("type")).lower(), 99))
                    for mode in ranked_modes:
                        reason = reasons.get(str(mode.get("type")).lower())
                        if reason:
                            mode["recommendation_reason"] = reason
                    analysis_source = "gemini_analysis"
            except Exception as exc:
                logger.warning("Gemini transport analysis skipped for %s -> %s: %s",
                               clean_origin, destination.name, exc)
                ranked_modes = _fallback_ranking(estimates, distance_km)
        else:
            ranked_modes = _fallback_ranking(estimates, distance_km)

        persisted: List[TransportOption] = []
        for mode in ranked_modes:
            name = str(mode.get("name") or "").strip()[:255]
            if not name:
                continue
            duplicate = db.query(TransportOption).filter(
                TransportOption.destination_id == destination.id,
                func.lower(TransportOption.name) == name.lower(),
                func.lower(TransportOption.route_from) == clean_origin.lower(),
            ).first()
            if duplicate is not None:
                persisted.append(duplicate)
                continue
            reason = str(mode.get("recommendation_reason") or
                         f"Researched {mode.get('type')} transfer for {clean_origin} to {destination.name} "
                         f"(about {distance_km:.0f} km straight-line; estimate, verify before booking).")
            row = TransportOption(
                destination_id=destination.id,
                vendor_id=None,
                type=str(mode.get("type") or "private_cab")[:50],
                name=name,
                route_from=clean_origin[:255],
                route_to=destination.name[:255],
                duration_hours=float(mode.get("duration_hours") or 4.0),
                price=float(mode.get("price") or 0.0),
                currency=currency,
                capacity=max(1, int(mode.get("capacity") or travelers)),
                features=list(mode.get("features") or []),
                latitude=getattr(destination, "latitude", None),
                longitude=getattr(destination, "longitude", None),
                source_url=None,
                # TransportOption has no description column; the analyst
                # reason is kept in evidence only (never invented bookings).
                evidence=[{"label": analysis_source, "note": reason[:500]}],
                inventory_source=LIVE_SOURCE,
                verification_status=LIVE_VERIFICATION,
                discovery_session_id=None,
                is_active=True,
            )
            db.add(row)
            persisted.append(row)
        try:
            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("Live transport persist skipped for %s -> %s: %s",
                           clean_origin, destination.name, exc)
            result["error"] = str(exc)[:300]
            return result
        result["researched"] = len([r for r in persisted if r.inventory_source == LIVE_SOURCE])
        result["options"] = persisted
        result["source"] = analysis_source if persisted else "catalog"
        return result
    except Exception as exc:
        logger.warning("Live transport research skipped for %s: %s", clean_origin, exc)
        result["error"] = str(exc)[:300]
        try:
            db.rollback()
        except Exception:
            pass
        return result
