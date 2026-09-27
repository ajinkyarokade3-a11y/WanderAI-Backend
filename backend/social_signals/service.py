"""Social Signals service.

Fetches and normalizes external/public signals relevant to a travel destination.
Follows the same defensive pattern as the weather service: never fabricates data,
returns clear unavailable states on failure, caches results to avoid repeated calls.

No social/news provider is configured by default. When no provider is configured,
the service returns an unavailable state. Tests mock the provider layer.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.orm import Session

from backend.models.models import Trip

logger = logging.getLogger(__name__)


class SocialSignalsNotConfigured(Exception):
    pass


class SocialSignalsProviderError(Exception):
    pass


# Simple TTL cache
_cache: Dict[str, tuple] = {}
DEFAULT_CACHE_TTL_S = 300


def _get_cache_ttl() -> int:
    try:
        from backend.database.config import settings
        return int(getattr(settings, "SOCIAL_CACHE_TTL_S", DEFAULT_CACHE_TTL_S))
    except Exception:
        return DEFAULT_CACHE_TTL_S


def clear_social_signals_cache():
    _cache.clear()


def _cache_key(lat: float, lon: float) -> str:
    return f"social:{lat:.4f}|{lon:.4f}"


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _normalize_signal(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one raw signal from the provider into a structured dict."""
    return {
        "title": raw.get("title"),
        "summary": raw.get("summary"),
        "source": raw.get("source"),
        "category": raw.get("category"),
        "severity": raw.get("severity"),
        "observed_at": raw.get("observed_at"),
        "relevance": raw.get("relevance"),
        "source_type": raw.get("source_type"),
        "source_url": raw.get("source_url"),
        "published_at": raw.get("published_at"),
        "location": raw.get("location"),
        "relevance_score": _to_float(raw.get("relevance_score")),
        "signal_type": raw.get("signal_type"),
        "sentiment": raw.get("sentiment"),
        "weather_relation": raw.get("weather_relation"),
        "confidence": raw.get("confidence"),
    }


def _compute_overall_risk(signals: List[Dict[str, Any]]) -> str:
    """Deterministic overall risk from signal severities."""
    severity_levels = {"low": 0, "medium": 1, "high": 2}
    max_level = 0
    for sig in signals:
        sev = (sig.get("severity") or "").lower()
        level = severity_levels.get(sev, 0)
        if level > max_level:
            max_level = level
    if max_level >= 2:
        return "high"
    if max_level >= 1:
        return "medium"
    return "low"


def fetch_social_signals(
    latitude: float,
    longitude: float,
    base_url: str,
    timeout_s: float,
    api_key: str = "",
) -> Dict[str, Any]:
    """Fetch social signals from the configured provider.

    Raises SocialSignalsNotConfigured if base_url is empty.
    Raises SocialSignalsProviderError on any provider failure.
    """
    if not base_url.strip():
        raise SocialSignalsNotConfigured("Social signals provider is not configured")

    lat = _to_float(latitude)
    lon = _to_float(longitude)
    if lat is None or lon is None:
        raise ValueError("Valid latitude and longitude are required")

    cache_ttl = _get_cache_ttl()
    key = _cache_key(lat, lon)
    now = time.time()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < cache_ttl:
            return data

    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
    }
    if api_key.strip():
        params["apikey"] = api_key.strip()

    url = base_url.rstrip("/")

    try:
        resp = httpx.get(url, params=params, timeout=float(timeout_s))
        resp.raise_for_status()
    except httpx.TimeoutException as e:
        raise SocialSignalsProviderError("Social signals provider timed out") from e
    except httpx.HTTPError as e:
        raise SocialSignalsProviderError(f"Social signals provider failed: {e}") from e
    except Exception as e:
        raise SocialSignalsProviderError(f"Social signals provider failed: {e}") from e

    try:
        payload = resp.json()
    except Exception as e:
        raise SocialSignalsProviderError("Social signals provider returned invalid response") from e

    if isinstance(payload, dict) and payload.get("error"):
        raise SocialSignalsProviderError(
            f"Social signals provider error: {payload.get('reason') or payload.get('error')}"
        )

    raw_signals = []
    if isinstance(payload, list):
        raw_signals = payload
    elif isinstance(payload, dict):
        raw_signals = payload.get("signals") or payload.get("data") or payload.get("results") or []

    signals = [_normalize_signal(s) for s in raw_signals if isinstance(s, dict)]
    overall_risk = _compute_overall_risk(signals)
    now_iso = datetime.now(timezone.utc).isoformat()

    result = {
        "available": True,
        "signals": signals,
        "overall_risk": overall_risk,
        "observed_at": now_iso,
        "sources": ["external"],
        "generated_at": now_iso,
        "status": "success",
        "total": len(signals),
    }
    _cache[key] = (now, result)
    return result


def build_social_context(
    latitude: float,
    longitude: float,
    base_url: str,
    timeout_s: float,
    api_key: str = "",
) -> Dict[str, Any]:
    """Build structured social signals context for the AI/disruption pipeline.

    Never raises; all failures return an unavailable state.
    """
    try:
        data = fetch_social_signals(latitude, longitude, base_url, timeout_s, api_key)
    except SocialSignalsNotConfigured:
        return {"available": False, "reason": "not_configured"}
    except SocialSignalsProviderError:
        return {"available": False, "reason": "provider_unavailable"}
    except Exception:
        return {"available": False, "reason": "provider_unavailable"}

    return {
        "available": True,
        "signals": data.get("signals", []),
        "overall_risk": data.get("overall_risk"),
        "observed_at": data.get("observed_at"),
        "sources": data.get("sources", []),
        "generated_at": data.get("generated_at"),
        "status": data.get("status"),
        "total": data.get("total"),
    }


def get_social_signals_for_trip(
    db: Session,
    trip: Trip,
    base_url: str,
    timeout_s: float,
    api_key: str = "",
) -> Dict[str, Any]:
    """Fetch social signals for a trip's destination coordinates."""
    if not trip or not trip.destination:
        return {"available": False, "reason": "no_destination"}

    dest = trip.destination
    lat = getattr(dest, "latitude", None)
    lon = getattr(dest, "longitude", None)

    if lat is None or lon is None:
        return {"available": False, "reason": "no_coordinates"}

    return build_social_context(
        float(lat), float(lon), base_url, timeout_s, api_key=api_key
    )


def _get_gdelt_config() -> Dict[str, Any]:
    try:
        from backend.database.config import settings
        return {
            "base_url": getattr(settings, "GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc"),
            "timeout_s": float(getattr(settings, "GDELT_TIMEOUT_S", 15.0)),
            "max_results": int(getattr(settings, "GDELT_MAX_RESULTS", 50)),
            "time_window_hours": int(getattr(settings, "GDELT_TIME_WINDOW_HOURS", 72)),
        }
    except Exception:
        return {
            "base_url": "https://api.gdeltproject.org/api/v2/doc/doc",
            "timeout_s": 15.0,
            "max_results": 50,
            "time_window_hours": 72,
        }


def _gdelt_cache_key(destination_name: str) -> str:
    return f"gdelt:{destination_name.lower().strip()}"


def get_social_signals_for_trip_gdelt(
    db: Session,
    trip: Trip,
    weather_condition: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch social signals for a trip using GDELT as the provider."""
    if not trip or not trip.destination:
        return {"available": False, "reason": "no_destination"}

    dest = trip.destination
    dest_name = getattr(dest, "name", None)
    if not dest_name:
        return {"available": False, "reason": "no_destination"}

    cache_ttl = _get_cache_ttl()
    key = _gdelt_cache_key(dest_name)
    now = time.time()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < cache_ttl:
            return data

    config = _get_gdelt_config()

    try:
        from backend.social_signals.providers.gdelt import fetch_gdelt_signals
        signals = fetch_gdelt_signals(
            destination_name=dest_name,
            weather_condition=weather_condition,
            base_url=config["base_url"],
            timeout_s=config["timeout_s"],
            max_results=config["max_results"],
            time_window_hours=config["time_window_hours"],
        )
    except Exception as e:
        logger.warning("GDELT fetch failed for %s: %s", dest_name, e)
        return {"available": False, "reason": "provider_unavailable"}

    overall_risk = _compute_overall_risk(signals)
    now_iso = datetime.now(timezone.utc).isoformat()

    result = {
        "available": True,
        "signals": signals,
        "overall_risk": overall_risk,
        "observed_at": now_iso,
        "sources": ["GDELT"],
        "generated_at": now_iso,
        "status": "success",
        "total": len(signals),
    }
    _cache[key] = (now, result)
    return result


def get_social_signals_for_trip_aggregated(
    db: Session,
    trip: Trip,
    weather_condition: Optional[str] = None,
) -> Dict[str, Any]:
    from backend.social_signals.aggregator import get_aggregated_signals
    return get_aggregated_signals(db, trip, weather_condition=weather_condition)
