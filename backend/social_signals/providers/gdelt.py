"""GDELT provider for Social Signals.

GDELT (Global Database of Events, Language, and Tone) monitors broadcast news,
web news, and print news from around the world. The DOC API v2 provides
free access to article search with no API key required.

API: https://api.gdeltproject.org/api/v2/doc/doc
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

GDELT_DEFAULT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DEFAULT_TIMEOUT_S = 15.0
GDELT_DEFAULT_MAX_RESULTS = 50
GDELT_DEFAULT_TIME_WINDOW_HOURS = 72

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


class GDELTError(Exception):
    pass


class GDELTTimeout(GDELTError):
    pass


class GDELTHTTPError(GDELTError):
    pass


class GDELTParseError(GDELTError):
    pass


def build_query(
    destination_name: str,
    weather_condition: Optional[str] = None,
) -> str:
    """Build a GDELT query string from destination and weather condition."""
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


def parse_seendate(seendate: str) -> Optional[str]:
    """Parse GDELT seendate (YYYYMMDDHHMMSS) to ISO 8601 format."""
    if not seendate:
        return None
    try:
        dt = datetime.strptime(seendate, "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def is_recent(
    published_at: Optional[str],
    max_age_hours: int = GDELT_DEFAULT_TIME_WINDOW_HOURS,
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


def determine_signal_type(title: str) -> str:
    from backend.social_signals.scoring import classify_signal_type
    return classify_signal_type(title)


def determine_weather_relation(title: str) -> str:
    """Determine weather relation from article title."""
    t = title.lower()
    for term in WEATHER_TERMS:
        if term in t:
            return "DIRECT"
    for term in TRAVEL_DISRUPTION_TERMS:
        if term in t:
            return "INDIRECT"
    return "UNKNOWN"


def normalize_article(
    article: Dict[str, Any],
    destination_name: str,
) -> Optional[Dict[str, Any]]:
    """Normalize a GDELT article into a SocialSignal dict."""
    title = article.get("title")
    url = article.get("url")

    if not title or not url:
        return None

    domain = article.get("domain") or "GDELT"
    seendate = article.get("seendate") or ""
    published_at = parse_seendate(seendate)

    return {
        "title": title,
        "summary": title,
        "source": domain,
        "category": None,
        "severity": None,
        "observed_at": published_at,
        "relevance": None,
        "source_type": "NEWS",
        "source_url": url,
        "published_at": published_at,
        "location": destination_name,
        "relevance_score": None,
        "signal_type": determine_signal_type(title),
        "sentiment": "UNKNOWN",
        "weather_relation": determine_weather_relation(title),
        "confidence": "LOW",
    }


def fetch_gdelt_signals(
    destination_name: str,
    weather_condition: Optional[str] = None,
    base_url: str = GDELT_DEFAULT_BASE_URL,
    timeout_s: float = GDELT_DEFAULT_TIMEOUT_S,
    max_results: int = GDELT_DEFAULT_MAX_RESULTS,
    time_window_hours: int = GDELT_DEFAULT_TIME_WINDOW_HOURS,
) -> List[Dict[str, Any]]:
    """Fetch and normalize signals from GDELT.

    Raises GDELTTimeout, GDELTHTTPError, or GDELTParseError on failure.
    """
    query = build_query(destination_name, weather_condition)

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=time_window_hours)

    params: Dict[str, Any] = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(min(max_results, 250)),
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }

    try:
        resp = httpx.get(base_url, params=params, timeout=timeout_s)
        resp.raise_for_status()
    except httpx.TimeoutException as e:
        raise GDELTTimeout(f"GDELT request timed out: {e}") from e
    except httpx.HTTPError as e:
        raise GDELTHTTPError(f"GDELT HTTP error: {e}") from e

    try:
        payload = resp.json()
    except Exception as e:
        raise GDELTParseError(f"GDELT returned invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise GDELTParseError("GDELT returned unexpected response format")

    articles = payload.get("articles") or []

    signals: List[Dict[str, Any]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue

        title = article.get("title", "")

        if not mentions_destination(title, destination_name):
            continue

        seendate = article.get("seendate") or ""
        published_at = parse_seendate(seendate)
        if not is_recent(published_at, time_window_hours):
            continue

        signal = normalize_article(article, destination_name)
        if signal:
            signals.append(signal)

    return signals
