"""Region-aware live hotel top-up for itinerary generation.

Problem: hotel discovery runs once per destination (research returns one
hotel; live fill searches one destination name and skips when any hotel
exists), so multi-cluster itineraries end up with a single base hotel and
300-600 minute warnings. This module searches the EXISTING live hotel
provider around each significant activity cluster instead, reusing the
live pipeline's gates verbatim:

    per-region provider search (SerpApi, existing client)
        -> normalize (existing normalize_property)
        -> require name + price + coordinates (existing live-fill gates;
           missing coordinates get one honest Nominatim geocode attempt,
           then the record is skipped — never invented)
        -> persist as inventory_source="live" rows (same verification
           level as every other live-filled hotel; never catalog,
           never verified_candidate)
        -> destination-scoped, idempotent by name (re-runs add nothing)

Callers (the itinerary generator) append the returned rows to the hotel
candidates they already ranked; budget, ranking-tiebreak, proximity
assignment, session isolation, and API shapes are untouched. Never raises:
provider failures only reduce what gets topped up.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.database.config import settings
from backend.models.models import Destination, Hotel

logger = logging.getLogger(__name__)

REGIONAL_HOTEL_CONFIG: Dict[str, Any] = {
    # Upper bound on searched day-clusters per generation (SerpApi quota;
    # one search per planned day at most, coverage skips cost nothing).
    "max_regions_per_trip": 6,
    # Raw provider results requested per region (5-10 is the reasonable
    # pool; existing verification/filtering reduces it naturally).
    "max_results_per_region": 8,
    # Geographic verification: a result farther than this from the
    # searched cluster centroid is provider filler (e.g. another state's
    # villa for a small-town query), not a nearby stay — rejected with a
    # "too_far" count, never persisted. Generous on purpose: real nearby
    # towns stay inside, cross-country junk does not.
    "max_result_distance_km": 100.0,
}


def _valid_point(latitude: Any, longitude: Any) -> Optional[Tuple[float, float]]:
    try:
        lat, lng = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return lat, lng


def _regional_id(destination_id: Any, name: str) -> str:
    """Stable id fitting the 36-char Hotel PK: sha1(destination|name).

    Deterministic so reruns and multi-region batches dedupe to one row
    per property per destination (a key only, never an identity claim).
    """
    import hashlib

    digest = hashlib.sha1(
        f"{destination_id}|{str(name or '').strip().casefold()}".encode("utf-8")
    ).hexdigest()[:12]
    return f"live-htl-{digest}"


def discover_regional_hotels(
    db: Session,
    *,
    destination: Destination,
    currency: str,
    traveler_count: int,
    check_in: str,
    check_out: str,
    region_centroids: Sequence[Mapping[str, Any]],
    existing_hotels: Sequence[Any],
    search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    reverse_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    geocode_fn: Optional[Callable[..., Optional[Tuple[float, float, str]]]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Top up live hotels around uncovered activity clusters.

    Args:
        destination: destination row the new hotels belong to (scoping).
        currency: ISO currency for provider quotes and persisted rows.
        traveler_count: adults for the provider quote.
        check_in/check_out: YYYY-MM-DD quote window.
        region_centroids: [{"latitude", "longitude", "label",
            "activity_minutes"}]; searched most-sightseeing-first.
        existing_hotels: current candidate rows (coverage check + dedup).
        search_fn/reverse_fn/geocode_fn: injectable provider seams
            (defaults hit SerpApi/Nominatim via existing clients).
        config: overrides for REGIONAL_HOTEL_CONFIG.

    Returns {"added": [Hotel rows], "per_region": {label: counts},
        "skipped_covered": [labels], "errors": [messages]}. Never raises.
    """
    from backend.itinerary.hotel_assignment import (
        HOTEL_ASSIGNMENT_CONFIG, haversine_km,
    )
    from backend.live_fill.service import _cached_hotel_search, _category_for_class
    from backend.places.service import geocode_place, reverse_geocode

    cfg = dict(REGIONAL_HOTEL_CONFIG)
    if config:
        cfg.update(config)
    report: Dict[str, Any] = {"added": [], "per_region": {},
                              "skipped_covered": [], "errors": []}
    try:
        max_regions = max(0, int(cfg.get("max_regions_per_trip", 4)))
        max_results = max(1, int(cfg.get("max_results_per_region", 8)))
    except (TypeError, ValueError):
        max_regions, max_results = 4, 8
    try:
        coverage_km = float(
            HOTEL_ASSIGNMENT_CONFIG.get("max_preferred_distance_km", 40.0))
    except (TypeError, ValueError):
        coverage_km = 40.0
    try:
        max_result_km = float(cfg.get("max_result_distance_km", 100.0))
    except (TypeError, ValueError):
        max_result_km = 100.0

    api_key = (settings.SERPAPI_API_KEY or "").strip()
    if not api_key and search_fn is None:
        return report
    currency = (currency or "INR").upper()
    adults = max(1, min(int(traveler_count or 2), 16))

    located_existing = [
        (float(h.latitude), float(h.longitude))
        for h in existing_hotels or []
        if _valid_point(getattr(h, "latitude", None),
                        getattr(h, "longitude", None)) is not None
    ]
    existing_names = {str(getattr(h, "name", "") or "").strip().casefold()
                      for h in existing_hotels or []}
    persisted_names = set(existing_names)

    def _covered(point: Tuple[float, float]) -> bool:
        for plat, plng in located_existing:
            dist = haversine_km(point[0], point[1], plat, plng)
            if dist is not None and dist <= coverage_km:
                return True
        return False

    ordered = sorted(
        (r for r in region_centroids or []
         if _valid_point((r or {}).get("latitude"),
                         (r or {}).get("longitude")) is not None),
        key=lambda r: (-float((r or {}).get("activity_minutes") or 0.0),
                       str((r or {}).get("label") or "")),
    )[:max(0, max_regions)]

    for region in ordered:
        point = _valid_point(region.get("latitude"), region.get("longitude"))
        label = str(region.get("label") or "the area")
        if point is None:  # pragma: no cover - filtered above
            continue
        if _covered(point):
            report["skipped_covered"].append(label)
            continue
        stats = {"raw": 0, "kept": 0, "no_coords": 0, "no_price": 0,
                 "duplicates": 0, "too_far": 0}
        try:
            reverse = (reverse_fn or reverse_geocode)(
                point[0], point[1], settings.NOMINATIM_API_URL,
                settings.PLACES_TIMEOUT_S)
            queries: List[str] = []
            if reverse:
                if reverse.get("locality"):
                    queries.append(str(reverse["locality"]))
                if reverse.get("broader") and reverse["broader"] not in queries:
                    queries.append(str(reverse["broader"]))
            if not queries:
                report["errors"].append(
                    f"{label}: no locality resolved, search skipped")
                report["per_region"][label] = stats
                continue
            raw: List[Dict[str, Any]] = []
            search = search_fn
            for query in queries[:2]:
                try:
                    if search is not None:
                        batch = search(query) or []
                    else:
                        batch = _cached_hotel_search(
                            api_key, query, check_in, check_out, adults,
                            currency) or []
                except Exception as exc:
                    report["errors"].append(f"{label}: search failed: {exc}")
                    continue
                raw.extend(batch)
                if raw:
                    break
            # Provider entries already carry the normalized live shape
            # (name/price_per_night/latitude/longitude — the same contract
            # _fill_hotels consumes from _cached_hotel_search).
            stats["raw"] = len(raw)
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                if name.casefold() in persisted_names:
                    stats["duplicates"] += 1
                    continue
                price = item.get("price_per_night")
                try:
                    price_value = float(price)
                except (TypeError, ValueError):
                    price_value = None
                if price_value is None or price_value <= 0:
                    stats["no_price"] += 1
                    continue
                lat = item.get("latitude")
                lng = item.get("longitude")
                if _valid_point(lat, lng) is None:
                    resolved = None
                    try:
                        geocode = geocode_fn or geocode_place
                        resolved = geocode(
                            f"{name}, {queries[0]}",
                            settings.NOMINATIM_API_URL,
                            settings.PLACES_TIMEOUT_S)
                    except Exception:
                        resolved = None
                    if resolved is not None:
                        lat, lng, _display = resolved
                    else:
                        stats["no_coords"] += 1
                        continue
                # Geographic verification: the provider sometimes returns
                # far-away filler for small-town queries. A "nearby" stay
                # hundreds of kilometres from the searched cluster is not
                # a nearby stay — rejected, never persisted.
                gap = haversine_km(
                    point[0], point[1], float(lat), float(lng))
                if gap is None or gap > max_result_km:
                    stats["too_far"] += 1
                    continue
                row = Hotel(
                    id=_regional_id(destination.id, name),
                    destination_id=destination.id, vendor_id=None,
                    name=name[:255],
                    category=_category_for_class(item.get("hotel_class")),
                    price_per_night=float(price_value), currency=currency,
                    rating=item.get("rating"),
                    address=(item.get("location") or queries[0])[:255]
                    if (item.get("location") or queries[0]) else None,
                    amenities=list(item.get("amenities") or []),
                    images=([item["image_url"]] if item.get("image_url") else []),
                    description=item.get("description"),
                    latitude=float(lat), longitude=float(lng),
                    inventory_source="live", verification_status="live_provider",
                    discovery_session_id=None, is_active=True,
                )
                if db.query(Hotel).filter(Hotel.id == row.id).first() is not None:
                    stats["duplicates"] += 1
                    persisted_names.add(name.casefold())
                    continue
                db.add(row)
                persisted_names.add(name.casefold())
                located_existing.append((float(lat), float(lng)))
                report["added"].append(row)
                stats["kept"] += 1
                if stats["kept"] >= max_results:
                    break
        except Exception as exc:
            logger.warning("Regional hotel search skipped for %s: %s", label, exc)
            report["errors"].append(f"{label}: {exc}")
        report["per_region"][label] = stats
    try:
        if report["added"]:
            db.flush()
    except Exception as exc:
        logger.warning("Regional hotel persist skipped: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        report["errors"].append(f"persist: {exc}")
        report["added"] = []
    return report
