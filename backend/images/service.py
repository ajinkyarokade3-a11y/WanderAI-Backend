"""Live attraction/activity image search via SerpApi Google Images.

Backend-only: the API key is supplied by the caller (read from backend
environment) and never leaves the server. Results are real, relevance-ranked
photographs for the queried place -- never fabricated, never hardcoded.

Endpoint used:
  GET {SERPAPI_BASE_URL}/search
    ?engine=google_images
    &q=<activity title>, <destination>
    &hl=en &gl=in
    &api_key=<SERPAPI_API_KEY>

Normalization prefers the full-resolution ``original`` URL and falls back to
the ``thumbnail``; only http(s) URLs are kept. Records without a usable URL
are skipped. An in-memory per-query cache keeps one trip generation to (at
most) one provider call per unique location.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# Bounded retries for fast-fail transient provider responses (rate limits and
# upstream 5xx). A single throttled SerpApi call must not blank the photo card
# when the next attempt succeeds. Timeouts and connection errors stay
# single-shot: retrying a hung 20s call would multiply worst-case latency past
# frontend tolerance and reintroduce the gateway timeouts this proxy avoids.
MAX_IMAGE_SEARCH_ATTEMPTS = 3
_TRANSIENT_IMAGE_STATUSES = frozenset({429, 500, 502, 503, 504})


class SerpApiImageError(Exception):
    """Raised when the SerpApi image request fails or returns unusable data."""


def _redact_key(text: Any) -> str:
    """Strip credential query params from provider error text.

    httpx embeds the request URL (including ``api_key=...``) in HTTP
    errors; this content reaches API error responses and server logs, so
    the key value must never survive here.
    """
    return re.sub(r"(api_?key=)[^&\s]*", r"\1[REDACTED]", str(text), flags=re.IGNORECASE)


# Hosts that never yield displayable photos: watermarked stock comps and
# crawler links that do not hotlink reliably. Entries from these hosts are
# skipped (not merely deprioritized) so every returned URL is showable.
_BLOCKED_IMAGE_HOSTS = (
    "alamy",
    "dreamstime",
    "shutterstock",
    "gettyimages",
    "istockphoto",
    "123rf",
    "depositphotos",
    "fbsbx",
    "lookaside",
    "ytimg.com",
)


def _host_blocked(url: str) -> bool:
    host = url.casefold()
    return any(blocked in host for blocked in _BLOCKED_IMAGE_HOSTS)


# URL path fragments that indicate schematics rather than photographs
# (route maps, diagrams, timetables, logos).
_BLOCKED_URL_PATTERNS = (
    "route_map",
    "routemap",
    "locator_map",
    "locator-map",
    "diagram",
    "flowchart",
    "timetable",
    "/logo",
    "logo.",
)


def _url_blocked(url: str) -> bool:
    lowered = url.casefold()
    return _host_blocked(url) or any(pattern in lowered for pattern in _BLOCKED_URL_PATTERNS)


# Result titles that indicate non-photographs (maps, diagrams, timetables,
# logos). These are skipped so cards show real photos, not schematics.
_NON_PHOTO_TITLE_PATTERNS = (
    "route map",
    "timetable",
    "time table",
    "diagram",
    "flowchart",
    "logo",
    "icon pack",
)


def _title_blocked(title: str) -> bool:
    lowered = (title or "").casefold()
    return any(pattern in lowered for pattern in _NON_PHOTO_TITLE_PATTERNS)


_images_cache: Dict[str, List[str]] = {}


def clear_image_cache() -> None:
    """Clear the in-memory image cache (used by tests)."""
    _images_cache.clear()


def _http_get(url: str, params: Dict[str, Any], timeout_s: float) -> httpx.Response:
    return httpx.get(url, params=params, timeout=float(timeout_s))


def _usable_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return None


def normalize_image_result(item: Any) -> Optional[Dict[str, str]]:
    """Normalize one ``images_results`` entry to {title, image_url}; None skips.

    Skips records without a usable URL and URLs on blocked (watermarked or
    non-hotlinkable) hosts.
    """
    if not isinstance(item, dict):
        return None
    image_url = _usable_url(item.get("original")) or _usable_url(item.get("thumbnail"))
    if image_url is None or _url_blocked(image_url):
        return None
    title = item.get("title")
    title = str(title).strip() if title else ""
    if _title_blocked(title):
        return None
    return {"title": title, "image_url": image_url}


def build_image_query(location: str, destination: Optional[str] = None) -> str:
    """Build a relevance query from the activity title plus destination context."""
    query = (location or "").strip()
    dest = (destination or "").strip() if destination else ""
    if dest and dest.casefold() not in query.casefold():
        query = f"{query}, {dest}" if query else dest
    return query


def search_serpapi_images(
    api_key: str,
    base_url: str,
    *,
    location: str,
    destination: Optional[str] = None,
    timeout_s: float = 20.0,
    max_results: int = 3,
) -> List[Dict[str, str]]:
    """Query SerpApi Google Images and return normalized results (relevance order)."""
    query = build_image_query(location, destination)
    if not query:
        raise ValueError("location is required")
    if not (api_key or "").strip():
        raise SerpApiImageError("Image search provider is not configured")
    params: Dict[str, Any] = {
        "engine": "google_images",
        "api_key": api_key.strip(),
        "q": query,
        "hl": "en",
        "gl": "in",
    }
    response = None
    for attempt in range(1, MAX_IMAGE_SEARCH_ATTEMPTS + 1):
        try:
            response = _http_get(base_url.rstrip("/") + "/search", params, timeout_s)
            response.raise_for_status()
            break
        except httpx.TimeoutException as exc:
            raise SerpApiImageError("Image search timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in _TRANSIENT_IMAGE_STATUSES and attempt < MAX_IMAGE_SEARCH_ATTEMPTS:
                logger.warning(
                    "Image search transient failure (status %s), retrying %d/%d",
                    status, attempt, MAX_IMAGE_SEARCH_ATTEMPTS,
                )
                time.sleep(0.5 * attempt)
                continue
            raise SerpApiImageError(f"Image search failed: {_redact_key(exc)}") from exc
        except httpx.HTTPError as exc:
            raise SerpApiImageError(f"Image search failed: {_redact_key(exc)}") from exc
    assert response is not None  # loop breaks only on success or raise
    try:
        payload = response.json()
    except ValueError as exc:
        raise SerpApiImageError("Image search returned an invalid response") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise SerpApiImageError(f"Image search failed: {payload.get('error')}")
    raw = (payload.get("images_results") if isinstance(payload, dict) else None) or []
    if not isinstance(raw, list):
        raise SerpApiImageError("Image search returned an unexpected response")
    normalized: List[Dict[str, str]] = []
    for entry in raw:
        item = normalize_image_result(entry)
        if item is not None:
            normalized.append(item)
    return normalized[: max(1, min(int(max_results or 3), 10))]


# Item types that depict a real place and therefore deserve a photo.
# Transport transfers, departure notes and free-time placeholders carry no
# place identity — they render as detail cards and must never trigger a
# provider lookup.
PLACE_ITEM_TYPES = frozenset({"hotel", "activity", "meal"})


def backfill_missing_place_images(
    items: Any,
    destination_name: Optional[str],
    api_key: str = "",
    base_url: str = "https://serpapi.com",
    timeout_s: float = 10.0,
    max_calls: int = 10,
) -> int:
    """Fill missing ``ui.image_url`` on place-type itinerary items.

    Each imageless hotel/activity/meal item gets one relevance-ranked photo
    from the SerpApi Google Images provider (query: "<place>, <destination>").
    Results persist on the item's ``meta_data.ui`` so the provider is hit at
    most once per unique place — repeat reads serve the stored URL.

    Bounded (``max_calls`` unique provider calls) and never raises: without a
    key, on provider failure, or when nothing real is found the item simply
    keeps no photo and callers fall back to the destination-level image.
    Returns the number of items that gained a photo.
    """
    try:
        rows = list(items or [])
    except TypeError:
        return 0
    if not rows or not (api_key or "").strip():
        return 0
    filled = 0
    calls = 0
    resolved: dict = {}
    for item in rows:
        try:
            if getattr(item, "item_type", None) not in PLACE_ITEM_TYPES:
                continue
            meta = getattr(item, "meta_data", None) or {}
            ui = meta.get("ui", {}) if isinstance(meta, dict) else {}
            ui = ui if isinstance(ui, dict) else {}
            if isinstance(ui.get("image_url"), str) and ui["image_url"].strip():
                continue
            title = str(getattr(item, "title", "") or "").strip()
            if not title:
                continue
            query_key = f"{title.casefold()}|{(destination_name or '').strip().casefold()}"
            if query_key in resolved:
                # Same place seen before in this batch: reuse its photo
                # without another provider call.
                url = resolved[query_key]
            else:
                url = None
                if calls < max(0, int(max_calls or 0)):
                    calls += 1
                    images = get_real_images_for_location(
                        title, (destination_name or "").strip() or None,
                        api_key, base_url, timeout_s, count=1,
                    )
                    url = images[0] if images else None
                resolved[query_key] = url
            if not url:
                continue
            item.meta_data = {**(meta if isinstance(meta, dict) else {}),
                              "ui": {**ui, "image_url": url}}
            filled += 1
        except Exception as exc:
            logger.warning("Place image backfill skipped for %r: %s",
                           getattr(item, "title", "?"), exc)
    if filled:
        logger.info("Place image backfill: %d photo(s) for %r", filled, destination_name)
    return filled


def get_real_image_for_location(
    location: str,
    destination: Optional[str] = None,
    api_key: str = "",
    base_url: str = "https://serpapi.com",
    timeout_s: float = 20.0,
) -> Optional[str]:
    """Return the top real photo URL for a location, or None when unavailable.

    Cached per normalized query; never raises (provider failures yield None so
    callers keep the existing catalog image instead of a fabricated one).
    """
    images = get_real_images_for_location(
        location, destination, api_key, base_url, timeout_s, count=1,
    )
    return images[0] if images else None


def get_real_images_for_location(
    location: str,
    destination: Optional[str] = None,
    api_key: str = "",
    base_url: str = "https://serpapi.com",
    timeout_s: float = 20.0,
    count: int = 1,
) -> List[str]:
    """Return up to ``count`` distinct real photo URLs (relevance order).

    One provider call regardless of count; cached per normalized query+count;
    never raises (failures yield [] so callers fall back to catalog images).
    """
    query = (location or "").strip()
    if not query:
        return []
    limit = max(1, min(int(count or 1), 6))
    key = f"{query.casefold()}|{(destination or '').strip().casefold()}|{limit}"
    if key in _images_cache:
        return list(_images_cache[key])
    try:
        results = search_serpapi_images(
            api_key, base_url, location=query, destination=destination,
            timeout_s=timeout_s, max_results=max(limit, 5),
        )
    except Exception as exc:
        # Never cache failures: a transient provider error (e.g. rate limit)
        # must not poison the key -- the next trip retries live.
        logger.warning("Real image lookup failed for %r: %s", query, exc)
        return []
    seen: List[str] = []
    for result in results:
        url = result["image_url"]
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    _images_cache[key] = seen
    return list(seen)
