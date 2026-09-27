import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

WEATHER_TERMS = {
    "rain", "flood", "flooding", "storm", "cyclone", "hurricane",
    "thunderstorm", "cloudburst", "monsoon", "landslide", "drought",
    "heat wave", "heavy rain", "strong wind", "blizzard", "snow",
    "wind", "weather", "precipitation",
}

TRAVEL_TERMS = {
    "travel", "airport", "flight", "train", "road", "traffic",
    "disruption", "delay", "cancellation", "tourist", "tourists",
    "hotel", "transport", "bus", "taxi", "closure", "stranded",
    "traveler", "travelers", "resort", "highway", "railway", "rail",
    "aviation", "blocked",
}

FLIGHT_TERMS = {"flight", "airport", "aviation", "cancelled", "delayed", "airline"}
ROAD_TERMS = {"road", "traffic", "highway", "landslide", "blocked", "closure"}
TRANSPORT_TERMS = {"train", "railway", "rail", "bus", "transport", "metro"}
TOURIST_TERMS = {"tourist", "tourists", "traveler", "travelers", "hotel", "resort", "stranded"}
WEATHER_REPORT_TERMS = {"rain", "flood", "flooding", "storm", "cyclone", "weather", "monsoon", "landslide"}
TREND_TERMS = {"trending", "viral", "popular", "breaking"}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "about", "like", "through", "after", "over",
    "between", "out", "off", "up", "down", "then", "than", "and",
    "or", "but", "not", "no", "yes", "so", "if", "because", "this",
    "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "him", "her", "us", "them", "my",
    "your", "his", "our", "their", "what", "which", "who", "when",
    "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same",
    "too", "very", "just", "now", "also", "here", "there", "again",
    "once", "further", "else", "ever", "never", "always", "often",
    "sometimes", "usually", "already", "yet", "still", "even",
}


def _tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def compute_recency_score(
    published_at: Optional[str],
    time_window_hours: int = 72,
) -> float:
    if not published_at:
        return 0.5
    try:
        ts = published_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if dt > now:
            return 0.5
        age_hours = (now - dt).total_seconds() / 3600
        if age_hours <= 24:
            return 1.0
        if age_hours <= time_window_hours:
            return 0.7
        if age_hours <= time_window_hours * 2:
            return 0.4
        return 0.1
    except (ValueError, TypeError):
        return 0.5


def compute_location_relevance(
    signal: Dict[str, Any],
    destination_name: str,
) -> float:
    if not destination_name:
        return 0.0
    dest_lower = destination_name.lower()
    location = (signal.get("location") or "").lower()
    title = (signal.get("title") or "").lower()
    summary = (signal.get("summary") or "").lower()
    text = f"{title} {summary}"

    if dest_lower in location:
        return 1.0
    if dest_lower in text:
        return 0.7
    dest_parts = dest_lower.split()
    for part in dest_parts:
        if len(part) > 3 and part in text:
            return 0.4
    return 0.0


def compute_weather_relevance(
    signal: Dict[str, Any],
    weather_condition: Optional[str] = None,
) -> float:
    relation = (signal.get("weather_relation") or "UNKNOWN").upper()
    if relation == "DIRECT":
        return 1.0
    if relation == "INDIRECT":
        return 0.6
    if relation == "NONE":
        return 0.0

    text = f"{signal.get('title') or ''} {signal.get('summary') or ''}".lower()
    if weather_condition and weather_condition.lower() in text:
        return 0.8
    for term in WEATHER_TERMS:
        if term in text:
            return 0.5
    return 0.3


def compute_travel_relevance(signal: Dict[str, Any]) -> float:
    text = f"{signal.get('title') or ''} {signal.get('summary') or ''}".lower()
    high_terms = {"flight", "airport", "road", "traffic", "landslide", "cancelled", "blocked"}
    medium_terms = {"travel", "tourist", "hotel", "delay", "disruption", "stranded"}

    for term in high_terms:
        if term in text:
            return 1.0
    for term in medium_terms:
        if term in text:
            return 0.6
    for term in TRAVEL_TERMS:
        if term in text:
            return 0.3
    return 0.0


def compute_activity_relevance(
    signal: Dict[str, Any],
    activities: Optional[List[str]] = None,
) -> float:
    if not activities:
        return 0.5
    text = f"{signal.get('title') or ''} {signal.get('summary') or ''}".lower()
    for activity in activities:
        if activity.lower() in text:
            return 1.0
    return 0.5


def compute_relevance_score(
    signal: Dict[str, Any],
    destination_name: str,
    weather_condition: Optional[str] = None,
    time_window_hours: int = 72,
    activities: Optional[List[str]] = None,
) -> float:
    dest_score = compute_location_relevance(signal, destination_name)
    weather_score = compute_weather_relevance(signal, weather_condition)
    travel_score = compute_travel_relevance(signal)
    recency_score = compute_recency_score(signal.get("published_at"), time_window_hours)
    activity_score = compute_activity_relevance(signal, activities)

    score = (
        dest_score * 0.35
        + weather_score * 0.25
        + travel_score * 0.20
        + recency_score * 0.15
        + activity_score * 0.05
    )
    return round(min(max(score, 0.0), 1.0), 4)


def classify_signal_type(text: str) -> str:
    t = text.lower()
    if any(term in t for term in TRANSPORT_TERMS):
        return "TRANSPORT_REPORT"
    if any(term in t for term in FLIGHT_TERMS):
        return "FLIGHT_DISRUPTION"
    if any(term in t for term in ROAD_TERMS):
        return "ROAD_CONDITION"
    if any(term in t for term in TOURIST_TERMS):
        return "TRAVELER_REPORT"
    if any(term in t for term in TREND_TERMS):
        return "TREND"
    if any(term in t for term in WEATHER_REPORT_TERMS):
        return "WEATHER_REPORT"
    return "EMERGING_CONDITION"


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def detect_corroboration(
    signal: Dict[str, Any],
    all_signals: List[Dict[str, Any]],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    sig_tokens = _tokenize(f"{signal.get('title') or ''} {signal.get('summary') or ''}")
    sig_source = signal.get("source") or ""
    corroborating: List[Dict[str, Any]] = []

    for other in all_signals:
        if other is signal:
            continue
        other_source = other.get("source") or ""
        if other_source == sig_source:
            continue
        other_tokens = _tokenize(f"{other.get('title') or ''} {other.get('summary') or ''}")
        similarity = _jaccard_similarity(sig_tokens, other_tokens)
        if similarity >= threshold:
            corroborating.append(other)

    return corroborating


_CONFIDENCE_LEVELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_CONFIDENCE_NAMES = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def _upgrade_confidence(level: str) -> str:
    num = _CONFIDENCE_LEVELS.get(level, 0)
    return _CONFIDENCE_NAMES.get(min(num + 1, 2), "HIGH")


def _downgrade_confidence(level: str) -> str:
    num = _CONFIDENCE_LEVELS.get(level, 0)
    return _CONFIDENCE_NAMES.get(max(num - 1, 0), "LOW")


def compute_confidence(
    signal: Dict[str, Any],
    all_signals: List[Dict[str, Any]],
) -> str:
    source_type = (signal.get("source_type") or "SOCIAL").upper()
    base = {"OFFICIAL": "MEDIUM", "NEWS": "MEDIUM", "SOCIAL": "LOW"}.get(source_type, "LOW")

    corroborating = detect_corroboration(signal, all_signals)
    if corroborating:
        base = _upgrade_confidence(base)

    sig_source = signal.get("source") or ""
    same_source_count = sum(
        1 for s in all_signals
        if s.get("source") == sig_source and s is not signal
    )
    if same_source_count >= 2:
        base = _upgrade_confidence(base)

    published_at = signal.get("published_at")
    if published_at:
        try:
            ts = published_at.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            if age_hours > 72:
                base = _downgrade_confidence(base)
        except (ValueError, TypeError):
            pass

    return base


def score_signals(
    signals: List[Dict[str, Any]],
    destination_name: str,
    weather_condition: Optional[str] = None,
    time_window_hours: int = 72,
    activities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    for sig in signals:
        sig["relevance_score"] = compute_relevance_score(
            sig, destination_name, weather_condition, time_window_hours, activities
        )
        sig["confidence"] = compute_confidence(sig, signals)
        text = f"{sig.get('title') or ''} {sig.get('summary') or ''}"
        sig["signal_type"] = classify_signal_type(text)
    return signals
