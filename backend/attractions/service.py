"""Live attraction/activity discovery via SerpApi Google Maps.

Backend-only: the API key is supplied by the caller (read from backend
environment) and never leaves the server. All candidates originate from the
provider ``local_results`` array -- real Google business/place records with
place IDs, coordinates and addresses. Nothing is hardcoded and nothing is
invented: missing fields yield None, nameless records are skipped, records
not verifiably associated with the requested destination are skipped, and
any transport or payload failure raises SerpApiAttractionError so callers
fall back to catalog inventory (or an honest 422) instead of fabricating
places.

Endpoint used (SerpApi Google Maps search):
  GET {SERPAPI_BASE_URL}/search
    ?engine=google_maps
    &q=<category> in <destination>
    &type=search
    &hl=en &gl=in
    [&ll=@<lat>,<lng>,14z]
    &api_key=<SERPAPI_API_KEY>

Query discipline (anti-hallucination): every query names an explicit
category AND the destination (e.g. "museums in Udaipur"). Generic
single-word queries are never issued.
"""

import logging
import math
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class SerpApiAttractionError(Exception):
    """Raised when the SerpApi attraction request fails or returns unusable data."""


# Category queries issued per destination, in priority order. The aggregator
# stops early once enough distinct candidates are collected, so quiet
# destinations cost one provider call and rich ones stay bounded.
ATTRACTION_CATEGORIES = (
    "tourist attractions",
    "museums",
    "forts and palaces",
    "parks and gardens",
    "markets and bazaars",
    "temples",
    "viewpoints",
    "cultural experiences",
)

# In-memory cache: one provider call per unique (destination, category).
_attractions_cache: Dict[str, List[Dict[str, Any]]] = {}


def clear_attraction_cache() -> None:
    """Clear the in-memory attraction cache (used by tests)."""
    _attractions_cache.clear()


def _http_get(url: str, params: Dict[str, Any], timeout_s: float) -> httpx.Response:
    return httpx.get(url, params=params, timeout=float(timeout_s))


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


def _destination_tokens(destination: str) -> List[str]:
    """Distinctive tokens of a destination name (length >= 4)."""
    return sorted({t for t in _normalized_name(destination).split() if len(t) >= 4})


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


# Max distance between a candidate and known destination coordinates to still
# count as associated (day-trip radius).
_ASSOCIATION_RADIUS_KM = 150.0


def is_associated_with_destination(
    item: Dict[str, Any],
    destination: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> bool:
    """True when a candidate is verifiably tied to the requested destination.

    Association requires either a distinctive destination token in the
    provider address, or candidate coordinates within day-trip radius of
    known destination coordinates. Anything unverifiable is rejected: a
    fuzzy provider match for a misspelled or unknown place must never enter
    an itinerary as if it were local.
    """
    tokens = _destination_tokens(destination)
    address = (item.get("address") or "").casefold()
    if tokens and address and any(token in address for token in tokens):
        return True
    try:
        dest_lat = float(latitude) if latitude is not None else None
        dest_lon = float(longitude) if longitude is not None else None
        cand_lat = float(item.get("latitude")) if item.get("latitude") is not None else None
        cand_lon = float(item.get("longitude")) if item.get("longitude") is not None else None
    except (TypeError, ValueError):
        return False
    if None in (dest_lat, dest_lon, cand_lat, cand_lon):
        return False
    return _haversine_km(dest_lat, dest_lon, cand_lat, cand_lon) <= _ASSOCIATION_RADIUS_KM


def build_attraction_query(destination: str, category: str) -> str:
    """Build a Google Maps attraction query. No hardcoded venue names."""
    dest = (destination or "").strip()
    cat = (category or "").strip()
    if not dest or not cat:
        raise ValueError("destination and category are required")
    return f"{cat} in {dest}"


def normalize_attraction_result(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize one SerpApi local_results entry; None skips records without a name.

    Only fields actually present in the provider payload are populated.
    Missing values stay None -- never fabricated.
    """
    if not isinstance(item, dict):
        return None
    name = item.get("title") or item.get("name")
    name = str(name).strip() if name else ""
    if not name:
        return None
    place_id = item.get("place_id") or item.get("data_id")
    if isinstance(place_id, list):
        place_id = place_id[0] if place_id else None
    place_id = str(place_id).strip() if place_id else None
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "attraction"
    gps = item.get("gps_coordinates") or {}
    address = item.get("address")
    address = str(address).strip() if address else None
    thumbnail = item.get("thumbnail")
    thumbnail = str(thumbnail).strip() if thumbnail else None
    website = item.get("website") or item.get("link")
    website = str(website).strip() if website else None
    rating = _to_float(item.get("rating"))
    latitude = _to_float(gps.get("latitude")) if isinstance(gps, dict) else None
    longitude = _to_float(gps.get("longitude")) if isinstance(gps, dict) else None
    types = [t for t in (item.get("types") or []) if isinstance(t, str) and t.strip()]
    maps_url = (
        "https://www.google.com/maps/search/?api=1&query=" + quote(name)
        + (f"&query_place_id={quote(place_id)}" if place_id else "")
    )
    return {
        "id": f"serpapi-attraction-{place_id}" if place_id else f"serpapi-attraction-{slug}",
        "place_id": place_id,
        "name": name,
        "address": address,
        "rating": rating,
        "latitude": latitude,
        "longitude": longitude,
        "website": website if website and website.startswith("http") else None,
        "image_url": thumbnail if thumbnail and thumbnail.startswith("http") else None,
        "maps_url": maps_url,
        "types": types,
        "source": "serpapi",
    }


def search_serpapi_attractions(
    api_key: str,
    base_url: str,
    *,
    destination: str,
    category: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timeout_s: float = 20.0,
    max_results: int = 8,
) -> List[Dict[str, Any]]:
    """Query SerpApi Google Maps for one category and return normalized places.

    Results are cached per unique query. Failures raise
    SerpApiAttractionError; callers must fall back to catalog data.
    """
    query = build_attraction_query(destination, category)
    if not (api_key or "").strip():
        raise SerpApiAttractionError("Attraction search provider is not configured")
    lat = _to_float(latitude)
    lng = _to_float(longitude)
    limit = max(1, min(int(max_results or 8), 20))
    key = "|".join([destination.strip().casefold(), category.strip().casefold(),
                    str(lat or ""), str(lng or ""), str(limit)])
    if key in _attractions_cache:
        return list(_attractions_cache[key])

    params: Dict[str, Any] = {
        "engine": "google_maps",
        "api_key": api_key.strip(),
        "q": query,
        "type": "search",
        "hl": "en",
        "gl": "in",
    }
    if lat is not None and lng is not None:
        params["ll"] = f"@{lat},{lng},14z"
    try:
        response = _http_get(base_url.rstrip("/") + "/search", params, timeout_s)
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise SerpApiAttractionError("Attraction search timed out") from exc
    except httpx.HTTPError as exc:
        raise SerpApiAttractionError(f"Attraction search failed: {exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise SerpApiAttractionError("Attraction search returned an invalid response") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise SerpApiAttractionError(f"Attraction search failed: {payload.get('error')}")
    local_results = (payload.get("local_results") if isinstance(payload, dict) else None) or []
    if not isinstance(local_results, list):
        raise SerpApiAttractionError("Attraction search returned an unexpected response")
    normalized: List[Dict[str, Any]] = []
    for entry in local_results:
        item = normalize_attraction_result(entry)
        if item is not None:
            normalized.append(item)
    result = normalized[:limit]
    _attractions_cache[key] = result
    return list(result)


def discover_live_attractions(
    destination: str,
    *,
    api_key: str = "",
    base_url: str = "https://serpapi.com",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    categories: Optional[List[str]] = None,
    per_category: int = 10,
    max_total: int = 12,
    exclude_names: Optional[List[str]] = None,
    timeout_s: float = 12.0,
) -> List[Dict[str, Any]]:
    """Collect distinct real attractions across categories. Never raises.

    Iterates categories in priority order and stops early once ``max_total``
    distinct candidates are gathered, bounding SerpApi quota per trip (one
    search per category regardless of page size). ``exclude_names`` (e.g.
    already-selected catalog titles) are skipped WITHOUT counting toward the
    target, so callers reliably receive ``max_total`` fresh candidates when
    the pool allows. No ``ll`` viewport is sent: the query already names the
    destination, viewports were observed to shrink result sets, and
    association is enforced afterwards by address token or destination-radius
    checks. Deduplicates by provider place ID, then by normalized name, and
    keeps only candidates verifiably associated with the destination. Returns
    [] when the provider is unconfigured, fails, or knows nothing -- callers
    then use catalog inventory (or an honest 422) instead of invented places.
    """
    dest = (destination or "").strip()
    if not dest or not (api_key or "").strip():
        return []
    target = max(0, int(max_total or 0))
    if target == 0:
        return []
    try:
        dest_lat = float(latitude) if latitude is not None else None
        dest_lon = float(longitude) if longitude is not None else None
    except (TypeError, ValueError):
        dest_lat, dest_lon = None, None
    collected: List[Dict[str, Any]] = []
    seen_ids: set = set()
    seen_names: set = set()
    for excluded in exclude_names or []:
        norm_excluded = _normalized_name(excluded or "")
        if norm_excluded:
            seen_names.add(norm_excluded)
    for category in categories or list(ATTRACTION_CATEGORIES):
        if len(collected) >= target:
            break
        try:
            results = search_serpapi_attractions(
                api_key, base_url, destination=dest, category=category,
                timeout_s=timeout_s, max_results=per_category,
            )
        except Exception as exc:
            # One bad category must not sink the trip; other categories and
            # the catalog still apply. Failures are never cached as results.
            logger.warning("Live attraction search failed for %r in %r: %s", category, dest, exc)
            continue
        for item in results:
            if not is_associated_with_destination(item, dest, dest_lat, dest_lon):
                continue
            place_id = item.get("place_id")
            norm_name = _normalized_name(item.get("name") or "")
            if (place_id and place_id in seen_ids) or (norm_name and norm_name in seen_names):
                continue
            if place_id:
                seen_ids.add(place_id)
            if norm_name:
                seen_names.add(norm_name)
            collected.append(item)
            if len(collected) >= target:
                break
    return collected


def count_live_attraction_candidates(
    destination: str,
    needed: int,
    *,
    api_key: str = "",
    base_url: str = "https://serpapi.com",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timeout_s: float = 12.0,
) -> int:
    """Count distinct live candidates up to ``needed``. Never raises.

    Shares the discovery cache, so a later full discovery for the same
    destination reuses these provider calls instead of consuming more quota.
    """
    if needed <= 0:
        return 0
    return len(discover_live_attractions(
        destination, api_key=api_key, base_url=base_url,
        latitude=latitude, longitude=longitude,
        max_total=needed, timeout_s=timeout_s,
    ))
