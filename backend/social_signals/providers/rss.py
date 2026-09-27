"""Google News RSS provider for Social Signals.

Uses Google News RSS feeds (no API key required).
https://news.google.com/rss/search?q=...
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

RSS_DEFAULT_BASE_URL = "https://news.google.com/rss/search"
RSS_DEFAULT_TIMEOUT_S = 15.0
RSS_DEFAULT_MAX_RESULTS = 50
RSS_DEFAULT_TIME_WINDOW_HOURS = 72

WEATHER_TERMS = [
    "rain", "flood", "flooding", "storm", "cyclone", "hurricane",
    "thunderstorm", "cloudburst", "monsoon", "landslide", "drought",
    "heat wave", "heavy rain", "strong wind", "blizzard", "snow",
    "wind", "weather", "precipitation",
]

TRAVEL_DISRUPTION_TERMS = [
    "travel", "airport", "flight", "train", "road", "traffic",
    "disruption", "delay", "cancellation", "tourist", "tourists",
    "hotel", "transport", "bus", "taxi", "closure", "stranded",
    "traveler", "travelers", "resort", "highway", "railway", "rail",
    "aviation", "blocked",
]


class RSSError(Exception):
    pass


class RSSTimeout(RSSError):
    pass


class RSSHTTPError(RSSError):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class RSSRateLimit(RSSError):
    pass


class RSSParseError(RSSError):
    pass


def build_query(
    destination_name: str,
    weather_condition: Optional[str] = None,
) -> str:
    parts = [f'"{destination_name}"']

    if weather_condition:
        wc = weather_condition.lower()
        if "rain" in wc or "storm" in wc or "monsoon" in wc:
            parts.append("(rain OR flood OR storm OR cloudburst OR monsoon OR landslide)")
        elif "wind" in wc or "cyclone" in wc:
            parts.append("(wind OR storm OR cyclone OR hurricane)")
        elif "snow" in wc or "blizzard" in wc:
            parts.append("(snow OR blizzard)")
        elif "heat" in wc or "drought" in wc:
            parts.append("(heat OR drought OR heat wave)")
        else:
            parts.append(f'"{weather_condition}"')
    else:
        parts.append("(rain OR flood OR storm OR travel OR airport OR flight OR disruption OR road OR landslide)")

    return " ".join(parts)


def parse_rss_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        pass
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.isoformat()
    except (ValueError, TypeError):
        return None


def is_recent(
    published_at: Optional[str],
    max_age_hours: int = RSS_DEFAULT_TIME_WINDOW_HOURS,
) -> bool:
    if not published_at:
        return True
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        return dt >= cutoff
    except (ValueError, TypeError):
        return True


def mentions_destination(text: str, destination_name: str) -> bool:
    if not text or not destination_name:
        return False
    return destination_name.lower() in text.lower()


def determine_signal_type(text: str) -> str:
    from backend.social_signals.scoring import classify_signal_type
    return classify_signal_type(text)


def determine_weather_relation(text: str) -> str:
    t = text.lower()
    for term in WEATHER_TERMS:
        if term in t:
            return "DIRECT"
    for term in TRAVEL_DISRUPTION_TERMS:
        if term in t:
            return "INDIRECT"
    return "UNKNOWN"


def normalize_item(
    item: Dict[str, Any],
    destination_name: str,
) -> Optional[Dict[str, Any]]:
    title = item.get("title") or ""
    link = item.get("link") or ""
    pub_date = item.get("pubDate") or ""
    description = item.get("description") or ""

    if not title or not link:
        return None

    published_at = parse_rss_date(pub_date)
    text = f"{title} {description}"

    return {
        "title": title,
        "summary": description or title,
        "source": "Google News",
        "category": None,
        "severity": None,
        "observed_at": published_at,
        "relevance": None,
        "source_type": "NEWS",
        "source_url": link,
        "published_at": published_at,
        "location": destination_name,
        "relevance_score": None,
        "signal_type": determine_signal_type(text),
        "sentiment": "UNKNOWN",
        "weather_relation": determine_weather_relation(text),
        "confidence": "LOW",
    }


def fetch_rss_signals(
    destination_name: str,
    weather_condition: Optional[str] = None,
    base_url: str = RSS_DEFAULT_BASE_URL,
    timeout_s: float = RSS_DEFAULT_TIMEOUT_S,
    max_results: int = RSS_DEFAULT_MAX_RESULTS,
    time_window_hours: int = RSS_DEFAULT_TIME_WINDOW_HOURS,
) -> List[Dict[str, Any]]:
    query = build_query(destination_name, weather_condition)
    encoded_query = quote(query)

    params = {
        "q": query,
        "hl": "en-IN",
        "gl": "IN",
        "ceid": "IN:en",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/xml, text/xml, */*",
    }

    try:
        resp = httpx.get(base_url, params=params, headers=headers, timeout=timeout_s)
    except httpx.TimeoutException as e:
        raise RSSTimeout(f"RSS request timed out: {e}") from e
    except httpx.HTTPError as e:
        raise RSSHTTPError(f"RSS HTTP error: {e}") from e

    if resp.status_code == 429:
        raise RSSRateLimit("RSS rate limit exceeded")

    if resp.status_code >= 400:
        raise RSSHTTPError(
            f"RSS HTTP error: {resp.status_code}",
            status_code=resp.status_code,
        )

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        raise RSSParseError(f"RSS returned invalid XML: {e}") from e

    items: List[Dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub_date = item.findtext("pubDate") or ""
        description = item.findtext("description") or ""
        items.append({
            "title": title,
            "link": link,
            "pubDate": pub_date,
            "description": description,
        })

    signals: List[Dict[str, Any]] = []
    for item in items:
        title = item.get("title", "")
        if not mentions_destination(title, destination_name):
            continue

        pub_date = item.get("pubDate") or ""
        published_at = parse_rss_date(pub_date)
        if not is_recent(published_at, time_window_hours):
            continue

        signal = normalize_item(item, destination_name)
        if signal:
            signals.append(signal)
        if len(signals) >= max_results:
            break

    return signals
