"""Bluesky provider for Social Signals.

Uses Bluesky's public searchPosts endpoint (no auth required).
https://docs.bsky.app/docs/api/app-bsky-feed-search-posts
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

BLUESKY_DEFAULT_BASE_URL = "https://public.api.bsky.app"
BLUESKY_DEFAULT_TIMEOUT_S = 15.0
BLUESKY_DEFAULT_MAX_RESULTS = 50
BLUESKY_DEFAULT_TIME_WINDOW_HOURS = 72

WEATHER_TERMS = [
    "rain", "flood", "storm", "cyclone", "hurricane", "thunderstorm",
    "cloudburst", "monsoon", "landslide", "drought", "heat wave",
    "heavy rain", "strong wind", "blizzard", "snow",
]

TRAVEL_DISRUPTION_TERMS = [
    "travel", "airport", "flight", "train", "road", "traffic",
    "disruption", "delay", "cancellation", "tourist", "hotel",
    "transport", "bus", "taxi", "closure", "landslide",
]


class BlueskyError(Exception):
    pass


class BlueskyTimeout(BlueskyError):
    pass


class BlueskyHTTPError(BlueskyError):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class BlueskyRateLimit(BlueskyError):
    pass


class BlueskyParseError(BlueskyError):
    pass


def build_query(
    destination_name: str,
    weather_condition: Optional[str] = None,
) -> str:
    """Build a Bluesky search query from destination and weather condition."""
    parts = [f'"{destination_name}"']

    if weather_condition:
        wc = weather_condition.lower()
        if "rain" in wc or "storm" in wc or "monsoon" in wc:
            parts.append(
                '(rain OR flood OR storm OR "heavy rain" OR cloudburst OR monsoon OR landslide)'
            )
        elif "wind" in wc or "cyclone" in wc:
            parts.append('(wind OR storm OR cyclone OR hurricane)')
        elif "snow" in wc or "blizzard" in wc:
            parts.append('(snow OR blizzard)')
        elif "heat" in wc or "drought" in wc:
            parts.append('(heat OR drought OR "heat wave")')
        else:
            parts.append(f'"{weather_condition}"')
    else:
        parts.append(
            '(rain OR flood OR storm OR travel OR airport OR flight OR disruption OR road OR landslide)'
        )

    return " ".join(parts)


def parse_timestamp(ts: str) -> Optional[str]:
    """Parse Bluesky timestamp to ISO 8601 format."""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts)
        return dt.isoformat()
    except (ValueError, TypeError):
        return None


def construct_post_url(uri: str, handle: str) -> str:
    """Construct a Bluesky post URL from URI and author handle."""
    if not uri or not handle:
        return ""
    parts = uri.split("/")
    post_id = parts[-1] if parts else ""
    if not post_id:
        return ""
    return f"https://bsky.app/profile/{handle}/post/{post_id}"


def is_recent(
    published_at: Optional[str],
    max_age_hours: int = BLUESKY_DEFAULT_TIME_WINDOW_HOURS,
) -> bool:
    """Check if a signal is within the time window."""
    if not published_at:
        return True
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        return dt >= cutoff
    except (ValueError, TypeError):
        return True


def mentions_destination(text: str, destination_name: str) -> bool:
    """Check if text mentions the destination."""
    if not text or not destination_name:
        return False
    return destination_name.lower() in text.lower()


def mentions_weather(text: str, weather_condition: Optional[str] = None) -> bool:
    """Check if text mentions weather-related terms."""
    if not text:
        return False
    text_lower = text.lower()

    if weather_condition and weather_condition.lower() in text_lower:
        return True

    for term in WEATHER_TERMS:
        if term in text_lower:
            return True

    return False


def determine_signal_type(text: str) -> str:
    from backend.social_signals.scoring import classify_signal_type
    return classify_signal_type(text)


def determine_weather_relation(text: str) -> str:
    """Determine weather relation from post text."""
    t = text.lower()
    for term in WEATHER_TERMS:
        if term in t:
            return "DIRECT"
    for term in TRAVEL_DISRUPTION_TERMS:
        if term in t:
            return "INDIRECT"
    return "UNKNOWN"


def normalize_post(
    post: Dict[str, Any],
    destination_name: str,
) -> Optional[Dict[str, Any]]:
    """Normalize a Bluesky post into a SocialSignal dict."""
    record = post.get("record") or {}
    author = post.get("author") or {}
    uri = post.get("uri") or ""
    handle = author.get("handle") or ""

    text = record.get("text") or ""
    created_at = record.get("createdAt") or ""

    if not text or not created_at:
        return None

    published_at = parse_timestamp(created_at)
    post_url = construct_post_url(uri, handle)

    title = text[:100] + ("..." if len(text) > 100 else "")

    return {
        "title": title,
        "summary": text,
        "source": "Bluesky",
        "category": None,
        "severity": None,
        "observed_at": published_at,
        "relevance": None,
        "source_type": "SOCIAL",
        "source_url": post_url,
        "published_at": published_at,
        "location": destination_name,
        "relevance_score": None,
        "signal_type": determine_signal_type(text),
        "sentiment": "UNKNOWN",
        "weather_relation": determine_weather_relation(text),
        "confidence": "LOW",
    }


def fetch_bluesky_signals(
    destination_name: str,
    weather_condition: Optional[str] = None,
    base_url: str = BLUESKY_DEFAULT_BASE_URL,
    timeout_s: float = BLUESKY_DEFAULT_TIMEOUT_S,
    max_results: int = BLUESKY_DEFAULT_MAX_RESULTS,
    time_window_hours: int = BLUESKY_DEFAULT_TIME_WINDOW_HOURS,
) -> List[Dict[str, Any]]:
    """Fetch and normalize signals from Bluesky.

    Raises BlueskyTimeout, BlueskyHTTPError, BlueskyRateLimit, or BlueskyParseError on failure.
    """
    query = build_query(destination_name, weather_condition)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=time_window_hours)

    params: Dict[str, Any] = {
        "q": query,
        "limit": str(min(max_results, 100)),
        "since": start_dt.isoformat(),
        "until": end_dt.isoformat(),
        "sort": "latest",
    }

    url = f"{base_url}/xrpc/app.bsky.feed.searchPosts"

    try:
        resp = httpx.get(url, params=params, timeout=timeout_s)
    except httpx.TimeoutException as e:
        raise BlueskyTimeout(f"Bluesky request timed out: {e}") from e
    except httpx.HTTPError as e:
        raise BlueskyHTTPError(f"Bluesky HTTP error: {e}") from e

    if resp.status_code == 429:
        raise BlueskyRateLimit("Bluesky rate limit exceeded")

    if resp.status_code >= 400:
        raise BlueskyHTTPError(
            f"Bluesky HTTP error: {resp.status_code}",
            status_code=resp.status_code,
        )

    try:
        payload = resp.json()
    except Exception as e:
        raise BlueskyParseError(f"Bluesky returned invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise BlueskyParseError("Bluesky returned unexpected response format")

    posts = payload.get("posts") or []

    signals: List[Dict[str, Any]] = []
    for post in posts:
        if not isinstance(post, dict):
            continue

        record = post.get("record") or {}
        text = record.get("text") or ""

        if not mentions_destination(text, destination_name):
            continue

        created_at = record.get("createdAt") or ""
        published_at = parse_timestamp(created_at)
        if not is_recent(published_at, time_window_hours):
            continue

        signal = normalize_post(post, destination_name)
        if signal:
            signals.append(signal)

    return signals
