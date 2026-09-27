"""Tests for weather service, API endpoint, and AI pipeline integration.

All tests mock the provider HTTP layer (_httpx_get) — no real API calls,
no API key required, no internet access needed.
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import (
    Activity, Alert, Destination, ItineraryItem, Trip, User,
)
from backend.weather import service as weather_service
from backend.weather.service import (
    WeatherNotConfigured,
    WeatherProviderError,
    build_weather_context,
    clear_weather_cache,
    compute_severity,
    fetch_weather,
    get_weather_for_trip,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Mock Open-Meteo response fixtures
# ---------------------------------------------------------------------------

MOCK_CURRENT = {
    "temperature_2m": 30.2,
    "relative_humidity_2m": 72,
    "apparent_temperature": 34.1,
    "weather_code": 63,
    "wind_speed_10m": 18.5,
    "precipitation": 2.4,
}

MOCK_DAILY = {
    "time": ["2026-09-27", "2026-09-28", "2026-09-29"],
    "temperature_2m_max": [31.0, 29.5, 28.0],
    "temperature_2m_min": [25.0, 24.0, 23.0],
    "weather_code": [63, 81, 0],
    "precipitation_sum": [4.2, 8.1, 0.0],
    "precipitation_probability_max": [80, 90, 10],
    "wind_speed_10m_max": [16.2, 21.4, 10.0],
    "apparent_temperature_max": [35.0, 32.0, 29.0],
    "apparent_temperature_min": [28.0, 27.0, 24.0],
}

MOCK_RESPONSE = {
    "current": MOCK_CURRENT,
    "daily": MOCK_DAILY,
}


def _mock_httpx_get(payload):
    """Return a mock _httpx_get that yields the given JSON payload."""
    def _fake_get(url, params, timeout):
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp
    return _fake_get


def _mock_httpx_get_error(exc):
    """Return a mock _httpx_get that raises the given exception."""
    def _fake_get(url, params, timeout):
        raise exc
    return _fake_get


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_weather_cache()
    yield
    clear_weather_cache()


# ---------------------------------------------------------------------------
# Severity computation tests
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_thunderstorm_is_severe(self):
        assert compute_severity(95, 0, 0, 25) == "severe"
        assert compute_severity(96, 0, 0, 25) == "severe"
        assert compute_severity(99, 0, 0, 25) == "severe"

    def test_heavy_snow_is_severe(self):
        assert compute_severity(75, 0, 0, -5) == "severe"
        assert compute_severity(86, 0, 0, -5) == "severe"

    def test_ice_fog_is_severe(self):
        assert compute_severity(45, 0, 0, -2) == "severe"
        assert compute_severity(48, 0, 0, -2) == "severe"

    def test_heavy_rain_current_is_high(self):
        assert compute_severity(65, 10.0, 0, 28) == "high"

    def test_heavy_rain_forecast_is_high(self):
        assert compute_severity(65, 25.0, 0, 28, is_forecast=True) == "high"

    def test_moderate_rain_current_is_moderate(self):
        assert compute_severity(63, 3.0, 0, 28) == "moderate"

    def test_moderate_rain_forecast_is_moderate(self):
        assert compute_severity(63, 10.0, 0, 28, is_forecast=True) == "moderate"

    def test_high_wind_is_high(self):
        assert compute_severity(0, 0, 70, 25) == "high"

    def test_moderate_wind_is_moderate(self):
        assert compute_severity(0, 0, 45, 25) == "moderate"

    def test_extreme_heat_is_high(self):
        assert compute_severity(0, 0, 10, 42) == "high"

    def test_extreme_cold_is_high(self):
        assert compute_severity(0, 0, 10, -15) == "high"

    def test_snow_is_moderate(self):
        assert compute_severity(71, 0, 10, -2) == "moderate"
        assert compute_severity(73, 0, 10, -2) == "moderate"

    def test_drizzle_is_low(self):
        assert compute_severity(51, 0.5, 5, 20) == "low"

    def test_insufficient_data_is_unknown(self):
        assert compute_severity(None, None, None, None) == "unknown"

    def test_violent_showers_is_high(self):
        assert compute_severity(82, 0, 0, 25) == "high"


# ---------------------------------------------------------------------------
# Weather service: successful response
# ---------------------------------------------------------------------------

class TestFetchWeatherSuccess:
    def test_successful_response(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        result = fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert result["source"] == "open-meteo"
        assert result["current"]["temperature"] == 30.2
        assert result["current"]["feels_like"] == 34.1
        assert result["current"]["humidity"] == 72
        assert result["current"]["precipitation"] == 2.4
        assert result["current"]["wind_speed"] == 18.5
        assert result["current"]["condition"] == "Moderate rain"
        assert result["current"]["severity"] == "moderate"

    def test_forecast_normalization(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        result = fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert len(result["forecast"]) == 3
        fc = result["forecast"][0]
        assert fc["date"] == "2026-09-27"
        assert fc["temperature"] == 31.0
        assert fc["temperature_min"] == 25.0
        assert fc["precipitation"] == 4.2
        assert fc["wind_speed"] == 16.2
        assert fc["condition"] == "Moderate rain"
        assert fc["severity"] == "moderate"

    def test_forecast_feels_like(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        result = fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        fc = result["forecast"][0]
        assert fc["feels_like"] == 31.5  # avg of 35.0 and 28.0

    def test_forecast_severity(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        result = fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert result["forecast"][0]["severity"] == "moderate"
        assert result["forecast"][1]["severity"] == "moderate"
        assert result["forecast"][2]["severity"] == "moderate"

    def test_timestamp_handling(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        result = fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        for fc in result["forecast"]:
            assert "timestamp" in fc
            assert "date" in fc


# ---------------------------------------------------------------------------
# Weather service: failure cases
# ---------------------------------------------------------------------------

class TestFetchWeatherFailures:
    def test_missing_base_url(self):
        with pytest.raises(WeatherNotConfigured):
            fetch_weather(19.076, 72.877, "", 8.0)

    def test_provider_timeout(self, monkeypatch):
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.TimeoutException("timeout")),
        )
        with pytest.raises(WeatherProviderError, match="timed out"):
            fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)

    def test_provider_http_error(self, monkeypatch):
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())),
        )
        with pytest.raises(WeatherProviderError):
            fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)

    def test_malformed_response(self, monkeypatch):
        def _fake_get(url, params, timeout):
            resp = MagicMock()
            resp.json.side_effect = ValueError("not json")
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(weather_service, "_httpx_get", _fake_get)
        with pytest.raises(WeatherProviderError, match="invalid response"):
            fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)

    def test_provider_error_payload(self, monkeypatch):
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get({"error": True, "reason": "Invalid latitude"}),
        )
        with pytest.raises(WeatherProviderError, match="Invalid latitude"):
            fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)

    def test_invalid_coordinates(self):
        with pytest.raises(ValueError):
            fetch_weather(None, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)

    def test_invalid_days(self):
        with pytest.raises(ValueError):
            fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0, days=0)


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------

class TestCaching:
    def test_first_request_calls_provider(self, monkeypatch):
        call_count = [0]
        def _fake_get(url, params, timeout):
            call_count[0] += 1
            resp = MagicMock()
            resp.json.return_value = MOCK_RESPONSE
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(weather_service, "_httpx_get", _fake_get)
        fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert call_count[0] == 1

    def test_repeated_request_uses_cache(self, monkeypatch):
        call_count = [0]
        def _fake_get(url, params, timeout):
            call_count[0] += 1
            resp = MagicMock()
            resp.json.return_value = MOCK_RESPONSE
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(weather_service, "_httpx_get", _fake_get)
        fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert call_count[0] == 1

    def test_expired_cache_calls_provider_again(self, monkeypatch):
        call_count = [0]
        def _fake_get(url, params, timeout):
            call_count[0] += 1
            resp = MagicMock()
            resp.json.return_value = MOCK_RESPONSE
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(weather_service, "_httpx_get", _fake_get)
        monkeypatch.setattr(weather_service, "_get_cache_ttl", lambda: 0)
        fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        fetch_weather(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# build_weather_context tests
# ---------------------------------------------------------------------------

class TestBuildWeatherContext:
    def test_success(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        ctx = build_weather_context(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert ctx["available"] is True
        assert ctx["location"]["latitude"] == 19.076
        assert ctx["location"]["longitude"] == 72.877
        assert ctx["current"]["temperature"] == 30.2
        assert ctx["current"]["severity"] == "moderate"
        assert len(ctx["forecast"]) == 3
        assert ctx["observed_at"] is not None

    def test_provider_unavailable(self, monkeypatch):
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.ConnectError("connection failed")),
        )
        ctx = build_weather_context(19.076, 72.877, "https://api.open-meteo.com/v1/forecast", 8.0)
        assert ctx["available"] is False
        assert ctx["reason"] == "provider_unavailable"

    def test_not_configured(self):
        ctx = build_weather_context(19.076, 72.877, "", 8.0)
        assert ctx["available"] is False
        assert ctx["reason"] == "not_configured"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestWeatherEndpoint:
    def test_valid_coordinates(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        r = client.get("/api/weather", params={"latitude": 19.076, "longitude": 72.877, "destination": "Mumbai"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["current"]["temperature"] == 30.2
        assert body["current"]["severity"] == "moderate"
        assert len(body["forecast"]) == 3

    def test_invalid_latitude(self):
        r = client.get("/api/weather", params={"latitude": 91, "longitude": 72.877, "destination": "Mumbai"})
        assert r.status_code == 422

    def test_invalid_longitude(self):
        r = client.get("/api/weather", params={"latitude": 19.076, "longitude": -181, "destination": "Mumbai"})
        assert r.status_code == 422

    def test_provider_failure_returns_502(self, monkeypatch):
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.ConnectError("connection failed")),
        )
        r = client.get("/api/weather", params={"latitude": 19.076, "longitude": 72.877, "destination": "Mumbai"})
        assert r.status_code == 502

    def test_missing_destination(self):
        r = client.get("/api/weather", params={"latitude": 19.076, "longitude": 72.877})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Trip-specific weather endpoint tests
# ---------------------------------------------------------------------------

class TestTripWeatherEndpoint:
    def _create_trip_with_destination(self):
        db = SessionLocal()
        try:
            user = db.query(User).first()
            destination = db.query(Destination).filter(Destination.slug == "manali").first()
            assert user is not None
            assert destination is not None
            trip = Trip(
                user_id=user.id,
                destination_id=destination.id,
                title="Weather Test Trip",
                traveler_count=2,
                duration_days=5,
                total_budget=50000.0,
                currency="INR",
            )
            db.add(trip)
            db.commit()
            return trip.id
        finally:
            db.close()

    def test_trip_weather_success(self, monkeypatch):
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        trip_id = self._create_trip_with_destination()
        r = client.get(f"/api/trips/{trip_id}/weather")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is True
        assert body["current"]["temperature"] == 30.2
        assert len(body["forecast"]) == 3

    def test_trip_weather_provider_failure(self, monkeypatch):
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.ConnectError("connection failed")),
        )
        trip_id = self._create_trip_with_destination()
        r = client.get(f"/api/trips/{trip_id}/weather")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["available"] is False
        assert body["reason"] == "provider_unavailable"

    def test_trip_not_found(self):
        r = client.get("/api/trips/non-existent-id/weather")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# AI / Disruption pipeline integration tests
# ---------------------------------------------------------------------------

class TestDisruptionPipelineWeather:
    def _create_trip_with_disruption(self):
        db = SessionLocal()
        try:
            user = db.query(User).first()
            destination = db.query(Destination).filter(Destination.slug == "manali").first()
            assert user is not None
            assert destination is not None

            activities = db.query(Activity).filter(
                Activity.destination_id == destination.id, Activity.is_active == True
            ).all()
            assert len(activities) >= 2

            trip = Trip(
                user_id=user.id,
                destination_id=destination.id,
                title="Weather Pipeline Test Trip",
                traveler_count=2,
                duration_days=5,
                total_budget=50000.0,
                currency="INR",
            )
            db.add(trip)
            db.flush()

            outdoor_activity = next(
                (a for a in activities if a.category == "adventure"),
                activities[0],
            )
            item = ItineraryItem(
                trip_id=trip.id,
                day_number=3,
                order_index=1,
                item_type="activity",
                title=outdoor_activity.title,
                description=outdoor_activity.description,
                activity_id=outdoor_activity.id,
                location=outdoor_activity.meeting_point or "Solang Valley",
                cost=outdoor_activity.price_per_person * 2,
                status="confirmed",
            )
            db.add(item)

            alert = Alert(
                trip_id=trip.id,
                alert_type="weather",
                severity="critical",
                title="Heavy snowfall at Solang Valley",
                description="Roads to Solang Valley temporarily blocked",
                is_resolved=False,
            )
            db.add(alert)
            db.commit()

            alternative = next((a for a in activities if a.id != outdoor_activity.id), None)
            return trip.id, outdoor_activity.id, outdoor_activity.title, alternative
        finally:
            db.close()

    def _mock_ai_response(self, monkeypatch, response_text):
        def _fake_generate(request, **kwargs):
            return SimpleNamespace(
                text=response_text,
                provider="gemini",
                model="gemini-2.5-flash",
            )
        monkeypatch.setattr("backend.ai.disruption_analysis.generate_with_failover", _fake_generate)

    def test_weather_context_included_in_disruption_request(self, monkeypatch):
        """Weather context is included in the Gemini prompt when available."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get(MOCK_RESPONSE))
        trip_id, item_activity_id, item_title, alternative = self._create_trip_with_disruption()

        captured_prompts = []
        def _fake_generate(request, **kwargs):
            captured_prompts.append(request.prompt)
            return SimpleNamespace(
                text=json.dumps({
                    "disruption_cause": "Heavy snowfall",
                    "affected_day": 3,
                    "affected_activity": item_title,
                    "severity": "critical",
                    "impact_summary": "Day 3 activity unavailable.",
                    "affected_bookings": [],
                    "affected_vendors": [],
                    "transport_changes": [],
                    "alternatives": [],
                    "risks_limitations": [],
                }),
                provider="gemini",
                model="gemini-2.5-flash",
            )
        monkeypatch.setattr("backend.ai.disruption_analysis.generate_with_failover", _fake_generate)

        r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
        assert r.status_code == 200, r.text

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "weather_context" in prompt
        assert "30.2" in prompt
        assert "Moderate rain" in prompt

    def test_weather_unavailable_state_passed_to_ai(self, monkeypatch):
        """When weather provider fails, unavailable state is passed to Gemini."""
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.ConnectError("connection failed")),
        )
        trip_id, item_activity_id, item_title, alternative = self._create_trip_with_disruption()

        captured_prompts = []
        def _fake_generate(request, **kwargs):
            captured_prompts.append(request.prompt)
            return SimpleNamespace(
                text=json.dumps({
                    "disruption_cause": "Heavy snowfall",
                    "affected_day": 3,
                    "affected_activity": item_title,
                    "severity": "critical",
                    "impact_summary": "Day 3 activity unavailable.",
                    "affected_bookings": [],
                    "affected_vendors": [],
                    "transport_changes": [],
                    "alternatives": [],
                    "risks_limitations": [],
                }),
                provider="gemini",
                model="gemini-2.5-flash",
            )
        monkeypatch.setattr("backend.ai.disruption_analysis.generate_with_failover", _fake_generate)

        r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
        assert r.status_code == 200, r.text

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "weather_context" in prompt
        assert "available" in prompt
        assert "false" in prompt

    def test_disruption_analysis_works_when_weather_unavailable(self, monkeypatch):
        """Existing disruption analysis still works when weather is unavailable."""
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(httpx.ConnectError("connection failed")),
        )
        trip_id, item_activity_id, item_title, alternative = self._create_trip_with_disruption()

        self._mock_ai_response(monkeypatch, json.dumps({
            "disruption_cause": "Heavy snowfall",
            "affected_day": 3,
            "affected_activity": item_title,
            "severity": "critical",
            "impact_summary": "Day 3 activity unavailable.",
            "affected_bookings": [],
            "affected_vendors": [],
            "transport_changes": [],
            "alternatives": [],
            "risks_limitations": [],
        }))

        r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["disruption_cause"] == "Heavy snowfall"
        assert body["status"] == "pending_approval"
