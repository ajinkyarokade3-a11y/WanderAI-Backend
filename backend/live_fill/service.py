"""Live inventory fill for trip creation.

When the curated catalog has gaps for a destination, configured live
providers (SerpApi Google Hotels, OpenStreetMap/Wikimedia places) supply
REAL data so trip creation can proceed instead of failing. Persisted rows
are marked inventory_source="live" / verification_status="live_provider"
so they never mix with curated catalog data or Gemini discovery sessions.

Transport has no live provider and cannot be filled; trips simply carry
no pre-booked transfer items when transport inventory is absent.

Never raises: provider failures only reduce what gets filled.
"""

import logging
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.database.config import settings
from backend.models.models import Activity, Destination, Hotel

logger = logging.getLogger(__name__)

LIVE_SOURCE = "live"
LIVE_VERIFICATION = "live_provider"

# Successful hotel searches cached process-locally (6h TTL) so trip
# creation and "Try Again" retries share SerpApi quota. Failures are
# never cached. Kept here (not in the search client) so the public
# /hotels/search endpoint semantics stay exactly as tested.
_HOTEL_CACHE: Dict[Tuple[Any, ...], Tuple[float, List[Dict[str, Any]]]] = {}
_HOTEL_CACHE_TTL_S = 6 * 3600


def _cached_hotel_search(api_key: str, destination: str, check_in: str,
                         check_out: str, adults: int,
                         currency: str) -> List[Dict[str, Any]]:
    from backend.hotels.service import search_serpapi_hotels

    key = (destination.lower(), check_in, check_out, adults, currency)
    hit = _HOTEL_CACHE.get(key)
    if hit is not None and hit[0] > time.monotonic():
        return [dict(item) for item in hit[1]]
    results = search_serpapi_hotels(
        api_key, settings.SERPAPI_BASE_URL, destination=destination,
        check_in_date=check_in, check_out_date=check_out,
        adults=adults, currency=currency,
        timeout_s=settings.SERPAPI_TIMEOUT_S, max_results=5,
    ) or []
    _HOTEL_CACHE[key] = (time.monotonic() + _HOTEL_CACHE_TTL_S, results)
    if len(_HOTEL_CACHE) > 200:
        now = time.monotonic()
        for stale in [k for k, (_, exp) in _HOTEL_CACHE.items() if exp <= now]:
            _HOTEL_CACHE.pop(stale, None)
    return [dict(item) for item in results]


def _category_for_kind(kind: Any) -> str:
    text = str(kind or "").lower()
    if any(word in text for word in ("museum", "historic", "monument", "temple", "church",
                                     "mosque", "heritage", "gallery", "art", "fort", "palace")):
        return "culture"
    if any(word in text for word in ("restaurant", "cafe", "food", "market", "bazaar", "cuisine")):
        return "culinary"
    if any(word in text for word in ("sport", "adventure", "climb", "dive", "surf", "ski",
                                     "trek", "paragliding", "rafting")):
        return "adventure"
    if any(word in text for word in ("spa", "resort", "leisure", "cruise", "boat", "garden")):
        return "relaxation"
    return "nature"


def _category_for_class(hotel_class: Any) -> str:
    try:
        stars = int(hotel_class)
    except (TypeError, ValueError):
        return "mid-range"
    if stars >= 5:
        return "luxury"
    if stars == 4:
        return "boutique"
    if stars == 3:
        return "mid-range"
    return "budget"


def _search_window(start_date: Any, end_date: Any, duration_days: int) -> tuple[str, str]:
    """Quote window for live hotel search: trip dates when usable, else a
    future window (live quotes need check-out after check-in)."""
    today = date.today()
    nights = max(1, int(duration_days or 2) - 1)
    check_in: Optional[date] = None
    check_out: Optional[date] = None
    try:
        if start_date is not None:
            check_in = start_date.date() if hasattr(start_date, "date") else date.fromisoformat(str(start_date)[:10])
        if end_date is not None:
            check_out = end_date.date() if hasattr(end_date, "date") else date.fromisoformat(str(end_date)[:10])
    except (ValueError, TypeError, AttributeError):
        check_in, check_out = None, None
    if check_in is None or check_in < today:
        check_in = today + timedelta(days=30)
        check_out = check_in + timedelta(days=nights)
    elif check_out is None or check_out <= check_in:
        check_out = check_in + timedelta(days=nights)
    return check_in.isoformat(), check_out.isoformat()


def ensure_live_destination(db: Session, name: str) -> Optional[Destination]:
    """Build a Destination row for an unknown place from live geocoding.

    Used when Gemini discovery is unavailable (dead key, quota, outage):
    Nominatim supplies real coordinates; SerpApi/OSM later supply the
    hotels/activities via fill_destination_inventory. Country/region come
    from the geocoder's display name, never invented. Never raises:
    returns None when the place cannot be geocoded.
    """
    clean = (name or "").strip()
    slug = re.sub(r"[^a-z0-9]+", "-", clean.lower()).strip("-")
    if not clean or not slug:
        return None
    try:
        from backend.places.service import geocode_place

        existing = db.query(Destination).filter(Destination.slug == slug).first()
        if existing is not None:
            return existing
        geo = geocode_place(clean, settings.NOMINATIM_API_URL, settings.PLACES_TIMEOUT_S)
        if not geo:
            return None
        lat, lng, display = geo
        parts = [p.strip() for p in str(display or "").split(",") if p.strip()]
        title = clean.title()[:255]
        dest = Destination(
            name=title, slug=slug[:255],
            country=(parts[-1][:100] if parts else "India"),
            state_region=(parts[-2][:100] if len(parts) >= 3 else title[:100]),
            description=(f"{title} — live-sourced destination "
                         "(SerpApi/OSM providers; curated catalog entry pending)."),
            hero_image_url=None, best_time_to_visit=None, tags=[],
            is_featured=False, latitude=float(lat), longitude=float(lng),
            inventory_source=LIVE_SOURCE, verification_status=LIVE_VERIFICATION,
            discovery_session_id=None,
        )
        db.add(dest)
        db.flush()
        return dest
    except Exception as exc:
        logger.warning("Live destination build skipped for %s: %s", name, exc)
        try:
            db.rollback()
        except Exception:
            pass
        return None


def fill_destination_inventory(
    db: Session,
    destination: Destination,
    *,
    currency: str,
    traveler_count: int,
    duration_days: int,
    start_date: Any = None,
    end_date: Any = None,
) -> Dict[str, int]:
    """Persist missing hotels/activities for destination from live providers.

    Idempotent (skips names already present). Commits what it adds so the
    rows survive even if the caller later aborts. Never raises. Provider
    failures are reported as hotel_error/activity_error strings so callers
    can explain (e.g. SerpApi quota exhausted) instead of a bare error.
    """
    filled: Dict[str, Any] = {"hotels": 0, "activities": 0,
                              "hotel_error": None, "activity_error": None}
    currency = (currency or "INR").upper()
    try:
        existing_hotels = {h.name for h in db.query(Hotel).filter(
            Hotel.destination_id == destination.id).all()}
        api_key = (settings.SERPAPI_API_KEY or "").strip()
        if api_key and not existing_hotels:
            filled["hotels"] = _fill_hotels(
                db, destination, existing_hotels, api_key, currency,
                traveler_count, duration_days, start_date, end_date)
    except Exception as exc:
        filled["hotel_error"] = str(exc)[:300]
        logger.warning("Live hotel fill skipped for %s: %s", destination.name, exc)
    try:
        existing_acts = {a.title for a in db.query(Activity).filter(
            Activity.destination_id == destination.id).all()}
        # Generator fills up to ~2 stops/day, so target that (cap 12) rather
        # than the validation minimum — thin days are what users complain of.
        days = max(1, int(duration_days or 1))
        want = min(12, max(max(1, days) - 1, 2 * days)) - len(existing_acts)
        if want > 0:
            filled["activities"] = _fill_activities(
                db, destination, existing_acts, currency, want)
    except Exception as exc:
        filled["activity_error"] = str(exc)[:300]
        logger.warning("Live activity fill skipped for %s: %s", destination.name, exc)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Live fill commit failed for %s: %s", destination.name, exc)
    return filled


def _fill_hotels(db: Session, destination: Destination, existing: set,
                 api_key: str, currency: str, traveler_count: int,
                 duration_days: int, start_date: Any, end_date: Any) -> int:
    check_in, check_out = _search_window(start_date, end_date, duration_days)
    results = _cached_hotel_search(
        api_key, destination.name, check_in, check_out,
        max(1, min(int(traveler_count or 2), 16)), currency)
    added = 0
    for item in results or []:
        name = str((item or {}).get("name") or "").strip()
        price = (item or {}).get("price_per_night")
        if not name or price is None or name in existing:
            continue
        try:
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        image = (item or {}).get("image_url")
        db.add(Hotel(
            destination_id=destination.id, vendor_id=None, name=name[:255],
            category=_category_for_class((item or {}).get("hotel_class")),
            price_per_night=price_value, currency=currency,
            rating=(item or {}).get("rating"),
            address=(item or {}).get("location") or destination.name,
            amenities=(item or {}).get("amenities") or [],
            images=[image] if image else [],
            description=(item or {}).get("description"),
            latitude=(item or {}).get("latitude"), longitude=(item or {}).get("longitude"),
            inventory_source=LIVE_SOURCE, verification_status=LIVE_VERIFICATION,
            is_active=True,
        ))
        existing.add(name)
        added += 1
    return added


def _fill_activities(db: Session, destination: Destination, existing: set,
                     currency: str, needed: int) -> int:
    from backend.places.service import get_live_places

    result = get_live_places(
        destination.name, destination.latitude, destination.longitude,
        max(int(needed or 0), 8), settings.NOMINATIM_API_URL, settings.OVERPASS_API_URL,
        settings.COMMONS_API_URL, settings.PLACES_TIMEOUT_S, settings.PLACES_RADIUS_M,
    )
    added = 0
    for place in (result or {}).get("places") or []:
        name = str((place or {}).get("name") or "").strip()
        if not name or name in existing:
            continue
        image = (place or {}).get("image_url")
        db.add(Activity(
            destination_id=destination.id, vendor_id=None, title=name[:255],
            category=_category_for_kind((place or {}).get("kind")),
            duration_hours=2.0,
            # No admission data from live listings: 0.0 = free/unknown, and
            # the description says so explicitly. Never invent a price.
            price_per_person=0.0, currency=currency,
            difficulty_level="easy", rating=None,
            images=[image] if image else [],
            description=(f"Live-sourced attraction in {destination.name}; "
                         "no admission data available, verify locally."),
            meeting_point=None,
            latitude=(place or {}).get("latitude"), longitude=(place or {}).get("longitude"),
            inventory_source=LIVE_SOURCE, verification_status=LIVE_VERIFICATION,
            is_active=True,
        ))
        existing.add(name)
        added += 1
    return added
