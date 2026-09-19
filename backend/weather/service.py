"""Weather/live conditions service - provider-backed, never fabricated.

Uses Open-Meteo (https://api.open-meteo.com) by default. No API key required,
but respects WEATHER_BASE_URL / WEATHER_API_KEY from env for future providers.
All helpers are defensive; missing/invalid fields yield None and skipped rows.
Provider failures raise WeatherProviderError (mapped to 502). Not configured raises 503.
"""
import time
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

class WeatherNotConfigured(Exception):
    pass

class WeatherProviderError(Exception):
    pass

# Simple TTL cache
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_S = 600  # 10 min

WMO_DESCR = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail",
}

def _wmo_to_text(code: Any) -> str:
    try:
        return WMO_DESCR.get(int(float(code)), f"Code {code}")
    except:
        return "Unknown"

def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except: return None

def clear_weather_cache():
    _cache.clear()

def _cache_key(lat: float, lon: float, days: int, date_str: Optional[str]) -> str:
    return f"{lat:.4f}|{lon:.4f}|{days}|{date_str or ''}"

def _httpx_get(url: str, params: Dict[str, Any], timeout_s: float) -> httpx.Response:
    return httpx.get(url, params=params, timeout=float(timeout_s))

def validate_date_str(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.strip()
    try:
        date.fromisoformat(s)
        return s
    except ValueError as e:
        raise ValueError("date must be YYYY-MM-DD") from e

def fetch_weather(
    latitude: float,
    longitude: float,
    base_url: str,
    timeout_s: float,
    days: int = 5,
    date_str: Optional[str] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    if not base_url.strip():
        raise WeatherNotConfigured("Weather provider is not configured")
    lat = _to_float(latitude)
    lon = _to_float(longitude)
    if lat is None or lon is None:
        raise ValueError("Valid latitude and longitude are required")
    if days < 1 or days > 16:
        raise ValueError("days must be between 1 and 16")
    key = _cache_key(lat, lon, days, date_str)
    now = time.time()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < CACHE_TTL_S:
            return data
    # Open-Meteo params
    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": days,
    }
    if date_str:
        # start_date/end_date overrides forecast_days if date provided
        # For simplicity, use date as start and compute end
        try:
            d = date.fromisoformat(date_str)
        except:
            raise
        # Open-Meteo allows start_date/end_date
        params["start_date"] = d.isoformat()
        # end date = start + days-1
        from datetime import timedelta
        end = d + timedelta(days=days-1)
        params["end_date"] = end.isoformat()
        params.pop("forecast_days", None)
    if api_key.strip():
        params["apikey"] = api_key.strip()  # some providers use apikey

    url = base_url.rstrip("/") + ("" if "forecast" in base_url else "/v1/forecast")
    # If base_url already contains /v1/forecast, don't append
    if base_url.rstrip("/").endswith("/forecast"):
        url = base_url.rstrip("/")

    try:
        resp = _httpx_get(url, params, timeout_s)
        resp.raise_for_status()
    except httpx.TimeoutException as e:
        raise WeatherProviderError("Weather provider timed out") from e
    except httpx.HTTPError as e:
        raise WeatherProviderError(f"Weather provider failed: {e}") from e
    except Exception as e:
        raise WeatherProviderError(f"Weather provider failed: {e}") from e
    try:
        payload = resp.json()
    except Exception as e:
        raise WeatherProviderError("Weather provider returned invalid response") from e
    if isinstance(payload, dict) and payload.get("error"):
        raise WeatherProviderError(f"Weather provider error: {payload.get('reason') or payload.get('error')}")

    # Parse current
    cur = payload.get("current") or {}
    current = {
        "temperature": _to_float(cur.get("temperature_2m")),
        "feels_like": _to_float(cur.get("apparent_temperature")),
        "condition": _wmo_to_text(cur.get("weather_code")) if cur.get("weather_code") is not None else None,
        "humidity": _to_float(cur.get("relative_humidity_2m")),
        "wind_speed": _to_float(cur.get("wind_speed_10m")),
    }
    # Daily forecast
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    tmins = daily.get("temperature_2m_min") or []
    tmaxs = daily.get("temperature_2m_max") or []
    wcodes = daily.get("weather_code") or []
    precs = daily.get("precipitation_probability_max") or []
    forecast: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        forecast.append({
            "date": str(d),
            "temperature_min": _to_float(tmins[i]) if i < len(tmins) else None,
            "temperature_max": _to_float(tmaxs[i]) if i < len(tmaxs) else None,
            "condition": _wmo_to_text(wcodes[i]) if i < len(wcodes) and wcodes[i] is not None else None,
            "precipitation_probability": _to_float(precs[i]) if i < len(precs) else None,
        })
        if len(forecast) >= days:
            break

    result = {
        "current": current,
        "forecast": forecast,
        "source": "open-meteo",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache[key] = (now, result)
    return result

def resolve_coordinates(
    destination: str,
    latitude: Optional[float],
    longitude: Optional[float],
    db,
    nominatim_url: str,
    timeout_s: float,
) -> Tuple[Optional[float], Optional[float], str]:
    """Resolve lat/lng: explicit > Destination model > Nominatim geocode."""
    if latitude is not None and longitude is not None:
        try:
            return float(latitude), float(longitude), destination
        except: pass
    # try Destination model
    if db is not None and destination:
        try:
            from backend.models.models import Destination
            row = db.query(Destination).filter(
                (Destination.name.ilike(destination.strip())) | (Destination.slug.ilike(destination.strip()))
            ).first()
            if row and row.latitude is not None and row.longitude is not None:
                return float(row.latitude), float(row.longitude), row.name
        except: pass
    # geocode fallback
    if destination:
        try:
            from backend.places.service import geocode_place
            g = geocode_place(destination, nominatim_url, timeout_s)
            if g:
                return g[0], g[1], g[2]
        except: pass
    return None, None, destination
