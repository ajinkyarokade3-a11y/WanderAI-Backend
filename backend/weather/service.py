"""Weather/live conditions service - provider-backed, never fabricated.

Uses Open-Meteo (https://api.open-meteo.com) by default. No API key required,
but respects WEATHER_BASE_URL / WEATHER_API_KEY from env for future providers.
All helpers are defensive; missing/invalid fields yield None and skipped rows.
Provider failures raise WeatherProviderError (mapped to 502). Not configured raises 503.
"""
import time
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class WeatherNotConfigured(Exception):
    pass


class WeatherProviderError(Exception):
    pass


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class WeatherLocation(BaseModel):
    latitude: float
    longitude: float


class WeatherCurrent(BaseModel):
    temperature: Optional[float] = None
    feels_like: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    condition: Optional[str] = None
    severity: Optional[str] = None


class WeatherForecastItem(BaseModel):
    timestamp: Optional[str] = None
    temperature: Optional[float] = None
    feels_like: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    condition: Optional[str] = None
    severity: Optional[str] = None


class WeatherData(BaseModel):
    available: bool = True
    location: Optional[WeatherLocation] = None
    current: Optional[WeatherCurrent] = None
    forecast: List[WeatherForecastItem] = Field(default_factory=list)
    observed_at: Optional[str] = None
    forecast_generated_at: Optional[str] = None


class WeatherError(BaseModel):
    code: str
    message: str


class WeatherUnavailable(BaseModel):
    available: bool = False
    error: WeatherError


# ---------------------------------------------------------------------------
# Severity Normalization
# ---------------------------------------------------------------------------

# WMO weather condition codes (subset relevant to severity)
_THUNDERSTORM_CODES = {95, 96, 99}
_HEAVY_SNOW_CODES = {75, 77, 85, 86}
_ICE_FOG_CODES = {45, 48}
_SNOW_CODES = {71, 73}
_HEAVY_RAIN_CODES = {65, 82}
_MODERATE_RAIN_CODES = {53, 63, 81}
_DRIZZLE_CODES = {51, 55}


def compute_severity(
    weather_code: Optional[float],
    precipitation: Optional[float],
    wind_speed: Optional[float],
    temperature: Optional[float],
    is_forecast: bool = False,
) -> str:
    """Deterministic severity from measurable provider data.

    Returns one of: "low", "moderate", "high", "severe", "unknown".

    Rules (evaluated in order, first match wins):
    - Thunderstorm / hail codes          -> severe
    - Heavy snow / ice / blizzard        -> severe
    - Ice fog                             -> severe
    - Heavy rain (>= 20 mm/day forecast, >= 7.6 mm/h current) -> high
    - Violent showers (code 82)           -> high
    - Moderate rain (>= 7.6 mm/day, >= 2.5 mm/h) -> moderate
    - Thunderstorm-adjacent (code 95/96/99) -> severe
    - High wind (>= 62 km/h)              -> high
    - Extreme heat (>= 40 C) or cold (<= -10 C) -> high
    - Snow codes 71/73                   -> moderate
    - Moderate wind (>= 40 km/h)          -> moderate
    - Drizzle / light rain                -> low
    - Insufficient data                   -> unknown
    """
    code_int = int(weather_code) if weather_code is not None else None

    # Thunderstorm / hail
    if code_int in _THUNDERSTORM_CODES:
        return "severe"

    # Heavy snow / blizzard
    if code_int in _HEAVY_SNOW_CODES:
        return "severe"

    # Ice fog
    if code_int in _ICE_FOG_CODES:
        return "severe"

    # Precipitation thresholds
    if precipitation is not None:
        if is_forecast:
            if precipitation >= 20.0:
                return "high"
            if precipitation >= 7.6:
                return "moderate"
        else:
            if precipitation >= 7.6:
                return "high"
            if precipitation >= 2.5:
                return "moderate"

    # Heavy rain codes
    if code_int in _HEAVY_RAIN_CODES:
        return "high"

    # Moderate rain codes
    if code_int in _MODERATE_RAIN_CODES:
        return "moderate"

    # Snow codes
    if code_int in _SNOW_CODES:
        return "moderate"

    # Wind speed (km/h)
    if wind_speed is not None:
        if wind_speed >= 62.0:
            return "high"
        if wind_speed >= 40.0:
            return "moderate"

    # Temperature extremes (Celsius)
    if temperature is not None:
        if temperature >= 40.0 or temperature <= -10.0:
            return "high"

    # Drizzle / light precipitation
    if code_int in _DRIZZLE_CODES:
        return "low"

    # If we have a weather code but no specific rule matched, use moderate
    if code_int is not None:
        return "moderate"

    return "unknown"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
DEFAULT_CACHE_TTL_S = 600


def _get_cache_ttl() -> int:
    try:
        from backend.database.config import settings
        return int(getattr(settings, "WEATHER_CACHE_TTL_S", DEFAULT_CACHE_TTL_S))
    except Exception:
        return DEFAULT_CACHE_TTL_S


def clear_weather_cache():
    _cache.clear()


def _cache_key(lat: float, lon: float, days: int, date_str: Optional[str]) -> str:
    return f"weather:{lat:.4f}|{lon:.4f}|{days}|{date_str or ''}"


def _redact_key(text: Any) -> str:
    """Strip credential query params from provider error text.

    httpx embeds the request URL (including ``apikey=...``) in HTTP
    errors; this content reaches API error responses and server logs, so
    the key value must never survive here.
    """
    return re.sub(r"(api_?key=)[^&\s]*", r"\1[REDACTED]", str(text), flags=re.IGNORECASE)


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


# ---------------------------------------------------------------------------
# WMO Weather Code Mapping
# ---------------------------------------------------------------------------

WMO_DESCR = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


def _wmo_to_text(code: Any) -> str:
    try:
        return WMO_DESCR.get(int(float(code)), f"Code {code}")
    except Exception:
        return "Unknown"


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fetch Weather (Open-Meteo)
# ---------------------------------------------------------------------------

def fetch_weather(
    latitude: float,
    longitude: float,
    base_url: str,
    timeout_s: float,
    days: int = 5,
    date_str: Optional[str] = None,
    api_key: str = "",
) -> Dict[str, Any]:
    """Fetch current weather + daily forecast from Open-Meteo.

    Returns a dict with normalized ``current`` and ``forecast`` fields.
    Raises WeatherNotConfigured if base_url is empty.
    Raises WeatherProviderError on any provider failure.
    Raises ValueError for invalid coordinates or days.
    """
    if not base_url.strip():
        raise WeatherNotConfigured("Weather provider is not configured")
    lat = _to_float(latitude)
    lon = _to_float(longitude)
    if lat is None or lon is None:
        raise ValueError("Valid latitude and longitude are required")
    if days < 1 or days > 16:
        raise ValueError("days must be between 1 and 16")

    cache_ttl = _get_cache_ttl()
    key = _cache_key(lat, lon, days, date_str)
    now = time.time()
    if key in _cache:
        ts, data = _cache[key]
        if now - ts < cache_ttl:
            return data

    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m,precipitation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,apparent_temperature_max,apparent_temperature_min",
        "timezone": "auto",
        "forecast_days": days,
    }
    if date_str:
        try:
            d = date.fromisoformat(date_str)
        except Exception:
            raise
        params["start_date"] = d.isoformat()
        end = d + timedelta(days=days - 1)
        params["end_date"] = end.isoformat()
        params.pop("forecast_days", None)
    if api_key.strip():
        params["apikey"] = api_key.strip()

    url = base_url.rstrip("/")
    if not url.endswith("/forecast"):
        url = url + "/v1/forecast"

    try:
        resp = _httpx_get(url, params, timeout_s)
        resp.raise_for_status()
    except httpx.TimeoutException as e:
        raise WeatherProviderError("Weather provider timed out") from e
    except httpx.HTTPError as e:
        raise WeatherProviderError(f"Weather provider failed: {_redact_key(e)}") from e
    except Exception as e:
        raise WeatherProviderError(f"Weather provider failed: {_redact_key(e)}") from e
    try:
        payload = resp.json()
    except Exception as e:
        raise WeatherProviderError("Weather provider returned invalid response") from e
    if isinstance(payload, dict) and payload.get("error"):
        raise WeatherProviderError(
            f"Weather provider error: {payload.get('reason') or payload.get('error')}"
        )

    # Parse current
    cur = payload.get("current") or {}
    cur_temp = _to_float(cur.get("temperature_2m"))
    cur_feels = _to_float(cur.get("apparent_temperature"))
    cur_precip = _to_float(cur.get("precipitation"))
    cur_wind = _to_float(cur.get("wind_speed_10m"))
    cur_code = _to_float(cur.get("weather_code"))
    cur_condition = _wmo_to_text(cur.get("weather_code")) if cur.get("weather_code") is not None else None

    current = {
        "temperature": cur_temp,
        "feels_like": cur_feels,
        "humidity": _to_float(cur.get("relative_humidity_2m")),
        "precipitation": cur_precip,
        "wind_speed": cur_wind,
        "condition": cur_condition,
        "severity": compute_severity(cur_code, cur_precip, cur_wind, cur_temp, is_forecast=False),
    }

    # Daily forecast
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    tmins = daily.get("temperature_2m_min") or []
    tmaxs = daily.get("temperature_2m_max") or []
    wcodes = daily.get("weather_code") or []
    precs = daily.get("precipitation_sum") or []
    winds = daily.get("wind_speed_10m_max") or []
    feels_max = daily.get("apparent_temperature_max") or []
    feels_min = daily.get("apparent_temperature_min") or []

    forecast: List[Dict[str, Any]] = []
    for i, d in enumerate(dates):
        fc_temp_max = _to_float(tmaxs[i]) if i < len(tmaxs) else None
        fc_temp_min = _to_float(tmins[i]) if i < len(tmins) else None
        fc_code = _to_float(wcodes[i]) if i < len(wcodes) else None
        fc_precip = _to_float(precs[i]) if i < len(precs) else None
        fc_wind = _to_float(winds[i]) if i < len(winds) else None
        fc_feels_max = _to_float(feels_max[i]) if i < len(feels_max) else None
        fc_feels_min = _to_float(feels_min[i]) if i < len(feels_min) else None

        # Use max temp as the representative temperature for the day
        fc_temp = fc_temp_max
        # Use average of feels-like min/max if available, else None
        fc_feels = None
        if fc_feels_max is not None and fc_feels_min is not None:
            fc_feels = round((fc_feels_max + fc_feels_min) / 2, 2)

        forecast.append({
            "date": str(d),
            "timestamp": str(d),
            "temperature": fc_temp,
            "temperature_min": fc_temp_min,
            "temperature_max": fc_temp_max,
            "feels_like": fc_feels,
            "precipitation": fc_precip,
            "precipitation_probability": _to_float(daily.get("precipitation_probability_max", [None] * len(dates))[i] if i < len(daily.get("precipitation_probability_max", [])) else None),
            "wind_speed": fc_wind,
            "condition": _wmo_to_text(wcodes[i]) if i < len(wcodes) and wcodes[i] is not None else None,
            "severity": compute_severity(fc_code, fc_precip, fc_wind, fc_temp, is_forecast=True),
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


# ---------------------------------------------------------------------------
# Weather Context for AI Pipeline
# ---------------------------------------------------------------------------

def build_weather_context(
    latitude: float,
    longitude: float,
    base_url: str,
    timeout_s: float,
    days: int = 5,
    api_key: str = "",
) -> Dict[str, Any]:
    """Build structured weather context for the AI/disruption pipeline.

    Returns:
        - On success: ``{"available": True, "location": {...}, "current": {...},
          "forecast": [...], "observed_at": "..."}``
        - On failure: ``{"available": False, "reason": "..."}``

    Never raises; all failures return an unavailable state.
    """
    try:
        data = fetch_weather(latitude, longitude, base_url, timeout_s, days=days, api_key=api_key)
    except WeatherNotConfigured:
        return {"available": False, "reason": "not_configured"}
    except WeatherProviderError:
        return {"available": False, "reason": "provider_unavailable"}
    except Exception:
        return {"available": False, "reason": "provider_unavailable"}

    current = data.get("current") or {}
    forecast = data.get("forecast") or []

    return {
        "available": True,
        "location": {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
        },
        "current": {
            "temperature": current.get("temperature"),
            "feels_like": current.get("feels_like"),
            "humidity": current.get("humidity"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed"),
            "condition": current.get("condition"),
            "severity": current.get("severity"),
        },
        "forecast": [
            {
                "timestamp": item.get("timestamp") or item.get("date"),
                "temperature": item.get("temperature"),
                "feels_like": item.get("feels_like"),
                "precipitation": item.get("precipitation"),
                "wind_speed": item.get("wind_speed"),
                "condition": item.get("condition"),
                "severity": item.get("severity"),
            }
            for item in forecast
        ],
        "observed_at": data.get("retrieved_at"),
    }


def get_weather_for_trip(
    db: Any,
    trip: Any,
    base_url: str,
    timeout_s: float,
    api_key: str = "",
) -> Dict[str, Any]:
    """Fetch weather for a trip's destination coordinates.

    Uses the trip's destination latitude/longitude. Returns unavailable
    state if the trip has no destination or the destination has no coordinates.
    """
    if not trip or not trip.destination:
        return {"available": False, "reason": "no_destination"}

    dest = trip.destination
    lat = getattr(dest, "latitude", None)
    lon = getattr(dest, "longitude", None)

    if lat is None or lon is None:
        return {"available": False, "reason": "no_coordinates"}

    days = max(1, min(getattr(trip, "duration_days", 5) or 5, 7))
    return build_weather_context(
        float(lat), float(lon), base_url, timeout_s, days=days, api_key=api_key
    )


# ---------------------------------------------------------------------------
# Coordinate Resolution
# ---------------------------------------------------------------------------

def resolve_coordinates(
    destination: str,
    latitude: Optional[float],
    longitude: Optional[float],
    db: Any,
    nominatim_url: str,
    timeout_s: float,
) -> Tuple[Optional[float], Optional[float], str]:
    """Resolve lat/lng: explicit > Destination model > Nominatim geocode."""
    if latitude is not None and longitude is not None:
        try:
            return float(latitude), float(longitude), destination
        except Exception:
            pass
    # try Destination model
    if db is not None and destination:
        try:
            from backend.models.models import Destination
            row = db.query(Destination).filter(
                (Destination.name.ilike(destination.strip())) | (Destination.slug.ilike(destination.strip()))
            ).first()
            if row and row.latitude is not None and row.longitude is not None:
                return float(row.latitude), float(row.longitude), row.name
        except Exception:
            pass
    # geocode fallback
    if destination:
        try:
            from backend.places.service import geocode_place
            g = geocode_place(destination, nominatim_url, timeout_s)
            if g:
                return g[0], g[1], g[2]
        except Exception:
            pass
    return None, None, destination
