import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.models.models import Trip
from backend.social_signals import service as social_service

logger = logging.getLogger(__name__)

_PROVIDER_HEALTH: Dict[str, str] = {}


def get_provider_health() -> Dict[str, str]:
    return dict(_PROVIDER_HEALTH)


def _classify_error(error: str) -> str:
    err_lower = error.lower()
    if "429" in err_lower or "rate limit" in err_lower or "too many" in err_lower:
        return "RATE_LIMITED"
    if "403" in err_lower or "forbidden" in err_lower or "blocked" in err_lower:
        return "BLOCKED"
    if "timeout" in err_lower or "timed out" in err_lower:
        return "ERROR"
    if "parse" in err_lower or "invalid" in err_lower or "malformed" in err_lower:
        return "ERROR"
    return "ERROR"


class ProviderResult:
    def __init__(
        self,
        name: str,
        signals: List[Dict[str, Any]],
        error: Optional[str] = None,
    ):
        self.name = name
        self.signals = signals
        self.error = error

    @property
    def success(self) -> bool:
        return self.error is None


def _deduplicate_signals(
    signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen_urls: set = set()
    unique: List[Dict[str, Any]] = []
    for sig in signals:
        url = sig.get("source_url")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique.append(sig)
    return unique


def _sort_key(sig: Dict[str, Any]) -> Tuple[float, str]:
    score = sig.get("relevance_score") or 0.0
    published = sig.get("published_at") or sig.get("observed_at") or ""
    return (-score, published)


def aggregate_signals(
    provider_results: List[ProviderResult],
    destination_name: str = "",
    weather_condition: Optional[str] = None,
    time_window_hours: int = 72,
    activities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    successful = [p for p in provider_results if p.success]
    failed = [p for p in provider_results if not p.success]

    if not successful:
        return {
            "available": False,
            "reason": "all_providers_failed",
            "status": "unavailable",
        }

    all_signals: List[Dict[str, Any]] = []
    for p in successful:
        all_signals.extend(p.signals)

    all_signals = _deduplicate_signals(all_signals)

    if destination_name:
        from backend.social_signals.scoring import score_signals
        all_signals = score_signals(
            all_signals, destination_name, weather_condition, time_window_hours, activities
        )

    all_signals.sort(key=lambda s: s.get("published_at") or s.get("observed_at") or "", reverse=True)
    all_signals.sort(key=lambda s: s.get("relevance_score") or 0.0, reverse=True)

    sources = [p.name for p in successful]

    status = "success" if len(successful) == len(provider_results) else "partial"

    now_iso = datetime.now(timezone.utc).isoformat()

    return {
        "available": True,
        "signals": all_signals,
        "overall_risk": social_service._compute_overall_risk(all_signals),
        "observed_at": now_iso,
        "sources": sources,
        "generated_at": now_iso,
        "status": status,
        "total": len(all_signals),
    }


def _get_cache_key(destination_name: str) -> str:
    return f"agg:{destination_name.lower().strip()}"


def get_aggregated_signals(
    db: Session,
    trip: Trip,
    weather_condition: Optional[str] = None,
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not trip or not trip.destination:
        return {"available": False, "reason": "no_destination"}

    dest = trip.destination
    dest_name = getattr(dest, "name", None)
    if not dest_name:
        return {"available": False, "reason": "no_destination"}

    dest_lat = getattr(dest, "latitude", None)
    dest_lon = getattr(dest, "longitude", None)

    cache_ttl = social_service._get_cache_ttl()
    key = _get_cache_key(dest_name)
    now = time.time()
    if key in social_service._cache:
        ts, data = social_service._cache[key]
        if now - ts < cache_ttl:
            return data

    if providers is None:
        providers = ["gdelt", "bluesky", "rss"]

    results: List[ProviderResult] = []

    for provider_name in providers:
        if provider_name == "gdelt":
            result = _fetch_gdelt(dest_name, weather_condition)
        elif provider_name == "bluesky":
            result = _fetch_bluesky(dest_name, weather_condition)
        elif provider_name == "rss":
            result = _fetch_rss(dest_name, weather_condition)
        else:
            result = ProviderResult(provider_name, [], error="unknown_provider")
        results.append(result)

    response = aggregate_signals(
        results,
        destination_name=dest_name,
        weather_condition=weather_condition,
    )

    if response.get("available") and dest_lat is not None and dest_lon is not None:
        for i, sig in enumerate(response.get("signals", [])):
            if sig.get("latitude") is None or sig.get("longitude") is None:
                offset = (i - len(response["signals"]) / 2) * 0.01
                sig["latitude"] = float(dest_lat) + offset
                sig["longitude"] = float(dest_lon) + offset

    if response.get("available"):
        now_iso = datetime.now(timezone.utc).isoformat()
        response["cache_timestamp"] = now_iso
        response["cache_age_seconds"] = 0
        response["cache_fresh"] = True

    social_service._cache[key] = (now, response)
    return response


def _fetch_gdelt(
    dest_name: str,
    weather_condition: Optional[str] = None,
) -> ProviderResult:
    logger.info("[SocialSignals] GDELT -> fetching for destination: %s", dest_name)
    try:
        from backend.social_signals.providers.gdelt import fetch_gdelt_signals
        signals = fetch_gdelt_signals(
            destination_name=dest_name,
            weather_condition=weather_condition,
        )
        _PROVIDER_HEALTH["GDELT"] = "AVAILABLE"
        logger.info("[SocialSignals] GDELT -> success: %d signals", len(signals))
        return ProviderResult("GDELT", signals)
    except Exception as e:
        health = _classify_error(str(e))
        _PROVIDER_HEALTH["GDELT"] = health
        logger.warning("[SocialSignals] GDELT -> FAILED [%s]: %s: %s", health, type(e).__name__, e)
        return ProviderResult("GDELT", [], error=str(e))


def _fetch_bluesky(
    dest_name: str,
    weather_condition: Optional[str] = None,
) -> ProviderResult:
    logger.info("[SocialSignals] Bluesky -> fetching for destination: %s", dest_name)
    try:
        from backend.social_signals.providers.bluesky import fetch_bluesky_signals
        signals = fetch_bluesky_signals(
            destination_name=dest_name,
            weather_condition=weather_condition,
        )
        _PROVIDER_HEALTH["Bluesky"] = "AVAILABLE"
        logger.info("[SocialSignals] Bluesky -> success: %d signals", len(signals))
        return ProviderResult("Bluesky", signals)
    except Exception as e:
        health = _classify_error(str(e))
        _PROVIDER_HEALTH["Bluesky"] = health
        logger.warning("[SocialSignals] Bluesky -> FAILED [%s]: %s: %s", health, type(e).__name__, e)
        return ProviderResult("Bluesky", [], error=str(e))


def _fetch_rss(
    dest_name: str,
    weather_condition: Optional[str] = None,
) -> ProviderResult:
    logger.info("[SocialSignals] RSS -> fetching for destination: %s", dest_name)
    try:
        from backend.social_signals.providers.rss import fetch_rss_signals
        signals = fetch_rss_signals(
            destination_name=dest_name,
            weather_condition=weather_condition,
        )
        _PROVIDER_HEALTH["RSS"] = "AVAILABLE"
        logger.info("[SocialSignals] RSS -> success: %d signals", len(signals))
        return ProviderResult("RSS", signals)
    except Exception as e:
        health = _classify_error(str(e))
        _PROVIDER_HEALTH["RSS"] = health
        logger.warning("[SocialSignals] RSS -> FAILED [%s]: %s: %s", health, type(e).__name__, e)
        return ProviderResult("RSS", [], error=str(e))
