"""Tests for Digital Twin weather-driven trip simulation.

Tests the simulation service and POST /api/trips/{trip_id}/simulate endpoint
with mocked weather data — no real API calls.

Tests verify causal, activity-specific impact logic:
- Clear weather produces no impact
- Different activity types receive different impacts
- Indoor activities are generally unaffected
- Delay estimates are only shown when logically applicable
- Impact types and recommended actions are correct
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import joinedload

from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import (
    Activity, Destination, ItineraryItem, TransportOption, Trip, User,
)
from backend.weather import service as weather_service
from backend.simulation.service import simulate_trip_weather_impact

client = TestClient(app)


def _mock_httpx_get(payload):
    def _fake_get(url, params, timeout):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp
    return _fake_get


def _mock_httpx_get_error(exc):
    def _fake_get(url, params, timeout):
        raise exc
    return _fake_get


@pytest.fixture(autouse=True)
def _clear_cache():
    weather_service.clear_weather_cache()
    yield
    weather_service.clear_weather_cache()


def _create_simulation_trip():
    """Create a trip with flight, outdoor activity, indoor activity, and boat."""
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

        outdoor_activity = next(
            (a for a in activities if a.category == "adventure"),
            activities[0],
        )
        indoor_activity = next(
            (a for a in activities if a.category in ("culture", "culinary", "relaxation")),
            None,
        )
        # If no indoor activity found, create one
        if indoor_activity is None:
            indoor_activity = Activity(
                destination_id=destination.id,
                title="Indoor Cooking Class",
                category="culinary",
                duration_hours=2.0,
                price_per_person=1500.0,
                currency="INR",
                difficulty_level="easy",
                rating=4.5,
                is_active=True,
            )
            db.add(indoor_activity)
            db.flush()

        trip = Trip(
            user_id=user.id,
            destination_id=destination.id,
            title="Simulation Test Trip",
            traveler_count=2,
            duration_days=3,
            total_budget=50000.0,
            currency="INR",
        )
        db.add(trip)
        db.flush()

        # Create transport options for proper type resolution
        flight_transport = TransportOption(
            destination_id=destination.id,
            type="flight",
            name="Flight to Destination",
            route_from="Delhi",
            route_to="Manali",
            duration_hours=2.0,
            price=5000.0,
            currency="INR",
            is_active=True,
        )
        boat_transport = TransportOption(
            destination_id=destination.id,
            type="boat",
            name="Boat Ride",
            route_from="Dock",
            route_to="Lake",
            duration_hours=1.0,
            price=1000.0,
            currency="INR",
            is_active=True,
        )
        db.add_all([flight_transport, boat_transport])
        db.commit()

        # Day 1: flight transport -> outdoor activity -> indoor activity -> boat
        flight = ItineraryItem(
            trip_id=trip.id, day_number=1, order_index=1, item_type="transport",
            title="Flight to Destination", description="Domestic flight",
            start_time="08:00", end_time="10:00", cost=5000, status="confirmed",
            transport_id=flight_transport.id,
        )
        outdoor = ItineraryItem(
            trip_id=trip.id, day_number=1, order_index=2, item_type="activity",
            title=outdoor_activity.title, description="Outdoor activity",
            start_time="11:00", end_time="13:00", cost=2000, status="confirmed",
            activity_id=outdoor_activity.id,
        )
        indoor = ItineraryItem(
            trip_id=trip.id, day_number=1, order_index=3, item_type="activity",
            title=indoor_activity.title, description="Indoor activity",
            start_time="14:00", end_time="16:00", cost=1500, status="confirmed",
            activity_id=indoor_activity.id,
        )
        boat = ItineraryItem(
            trip_id=trip.id, day_number=1, order_index=4, item_type="transport",
            title="Boat Ride", description="Scenic boat ride",
            start_time="17:00", end_time="18:00", cost=1000, status="confirmed",
            transport_id=boat_transport.id,
        )

        db.add_all([flight, outdoor, indoor, boat])
        db.commit()

        # Refresh to ensure all relationships are loaded
        db.refresh(flight_transport)
        db.refresh(boat_transport)
        db.refresh(outdoor_activity)
        db.refresh(indoor_activity)

        return trip.id, {
            "flight_id": flight.id,
            "outdoor_id": outdoor.id,
            "indoor_id": indoor.id,
            "boat_id": boat.id,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Scenario A — Clear Weather (no impact)
# ---------------------------------------------------------------------------

class TestClearWeather:
    def test_clear_weather_no_impact(self, monkeypatch):
        """Clear sky produces zero affected items."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 25, "feels_like": 26, "humidity": 50,
                            "precipitation": 0, "wind_speed": 5,
                            "condition": "Clear sky", "severity": "low"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx)
            assert result["affected_items"] == []
            assert result["replanning_required"] is False
        finally:
            db.close()

    def test_clear_weather_all_items_no_impact(self, monkeypatch):
        """All items show 'No weather impact' under clear sky."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 25, "feels_like": 26, "humidity": 50,
                            "precipitation": 0, "wind_speed": 5,
                            "condition": "Clear sky", "severity": "low"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx)
            all_items = result.get("all_items", [])
            assert len(all_items) == 4
            for item in all_items:
                assert item["severity"] == "none"
                assert item["impact_type"] == "none"
                assert item["recommended_action"] == "no_change"
                assert item["reason"] == "No weather impact"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Scenario B — Heavy Rain
# ---------------------------------------------------------------------------

class TestHeavyRain:
    def test_heavy_rain_different_impacts(self, monkeypatch):
        """Heavy rain produces different impacts for different activity types."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 10, "relative_humidity_2m": 85,
                        "apparent_temperature": 8, "weather_code": 65,
                        "wind_speed_10m": 50, "precipitation": 15},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [12],
                      "temperature_2m_min": [8], "weather_code": [65],
                      "precipitation_sum": [20], "precipitation_probability_max": [90],
                      "wind_speed_10m_max": [55], "apparent_temperature_max": [10],
                      "apparent_temperature_min": [5]},
        }))
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = weather_service.build_weather_context(
                trip.destination.latitude, trip.destination.longitude,
                "https://api.open-meteo.com/v1/forecast", 8.0
            )
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="heavy_rain")
            affected = {i["item_id"]: i for i in result["affected_items"]}

            # Flight should be affected with delay
            assert ids["flight_id"] in affected
            assert affected[ids["flight_id"]]["impact_type"] == "delay"
            assert affected[ids["flight_id"]]["estimated_delay_minutes"] is not None

            # Outdoor activity should be affected
            assert ids["outdoor_id"] in affected
            assert affected[ids["outdoor_id"]]["impact_type"] in ("modified", "postponed")

            # Indoor activity should NOT be affected
            assert ids["indoor_id"] not in affected

            # Boat should be affected
            assert ids["boat_id"] in affected
        finally:
            db.close()

    def test_heavy_rain_no_generic_messages(self, monkeypatch):
        """No generic 'affected by' messages under heavy rain."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 10, "relative_humidity_2m": 85,
                        "apparent_temperature": 8, "weather_code": 65,
                        "wind_speed_10m": 50, "precipitation": 15},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [12],
                      "temperature_2m_min": [8], "weather_code": [65],
                      "precipitation_sum": [20], "precipitation_probability_max": [90],
                      "wind_speed_10m_max": [55], "apparent_temperature_max": [10],
                      "apparent_temperature_min": [5]},
        }))
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = weather_service.build_weather_context(
                trip.destination.latitude, trip.destination.longitude,
                "https://api.open-meteo.com/v1/forecast", 8.0
            )
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="heavy_rain")
            for item in result["affected_items"]:
                assert "affected by" not in item["reason"].lower()
                assert "may be affected" not in item["reason"].lower()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Scenario C — High Wind
# ---------------------------------------------------------------------------

class TestHighWind:
    def test_high_wind_boat_cancelled(self, monkeypatch):
        """High wind causes boat cancellation."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 20, "feels_like": 20, "humidity": 50,
                            "precipitation": 0, "wind_speed": 70,
                            "condition": "High Wind", "severity": "high"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="high_wind")
            affected = {i["item_id"]: i for i in result["affected_items"]}

            # Boat should be cancelled under high wind
            assert ids["boat_id"] in affected
            assert affected[ids["boat_id"]]["impact_type"] == "cancelled"
            assert affected[ids["boat_id"]]["recommended_action"] == "cancel"
            assert affected[ids["boat_id"]]["estimated_delay_minutes"] is None

            # Indoor activity should NOT be affected
            assert ids["indoor_id"] not in affected
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Scenario D — Severe Weather
# ---------------------------------------------------------------------------

class TestSevereWeather:
    def test_severe_weather_significant_disruption(self, monkeypatch):
        """Severe weather causes significant disruption for outdoor activities."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 5, "feels_like": 2, "humidity": 90,
                            "precipitation": 25, "wind_speed": 80,
                            "condition": "Thunderstorm", "severity": "severe"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="severe")
            affected = {i["item_id"]: i for i in result["affected_items"]}

            # Flight: high risk, delay
            assert ids["flight_id"] in affected
            assert affected[ids["flight_id"]]["severity"] == "high"
            assert affected[ids["flight_id"]]["estimated_delay_minutes"] == 120

            # Boat: cancelled
            assert ids["boat_id"] in affected
            assert affected[ids["boat_id"]]["impact_type"] == "cancelled"

            # Outdoor activity: postponed
            assert ids["outdoor_id"] in affected
            assert affected[ids["outdoor_id"]]["impact_type"] == "postponed"

            # Indoor activity: NOT affected
            assert ids["indoor_id"] not in affected

            # Replanning required
            assert result["replanning_required"] is True
        finally:
            db.close()

    def test_severe_weather_no_delay_for_cancelled(self, monkeypatch):
        """Cancelled items should not show arbitrary delays."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 5, "relative_humidity_2m": 90,
                        "apparent_temperature": 2, "weather_code": 95,
                        "wind_speed_10m": 80, "precipitation": 25},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [8],
                      "temperature_2m_min": [3], "weather_code": [95],
                      "precipitation_sum": [30], "precipitation_probability_max": [95],
                      "wind_speed_10m_max": [85], "apparent_temperature_max": [5],
                      "apparent_temperature_min": [0]},
        }))
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = weather_service.build_weather_context(
                trip.destination.latitude, trip.destination.longitude,
                "https://api.open-meteo.com/v1/forecast", 8.0
            )
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="severe")
            affected = {i["item_id"]: i for i in result["affected_items"]}

            # Boat is cancelled — no delay
            boat = affected.get(ids["boat_id"])
            if boat:
                assert boat["estimated_delay_minutes"] is None
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Scenario E — Unavailable
# ---------------------------------------------------------------------------

class TestUnavailable:
    def test_unavailable_returns_empty(self):
        """Unavailable scenario returns empty simulation."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            result = simulate_trip_weather_impact(
                db, trip,
                {"available": True, "current": {"severity": "severe"}},
                scenario="unavailable",
            )
            assert result["affected_items"] == []
            assert result["replanning_required"] is False
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Delay logic tests
# ---------------------------------------------------------------------------

class TestDelayLogic:
    def test_no_delay_for_indoor_activities(self, monkeypatch):
        """Indoor activities never show weather-related delays."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 5, "feels_like": 2, "humidity": 90,
                            "precipitation": 25, "wind_speed": 80,
                            "condition": "Thunderstorm", "severity": "severe"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="severe")
            all_items = {i["item_id"]: i for i in result.get("all_items", [])}

            # Indoor activity should have no delay
            indoor = all_items.get(ids["indoor_id"])
            if indoor:
                assert indoor["estimated_delay_minutes"] is None
                assert indoor["severity"] == "none"
        finally:
            db.close()

    def test_cancelled_items_no_delay(self, monkeypatch):
        """Cancelled items should not show delay estimates."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 5, "relative_humidity_2m": 90,
                        "apparent_temperature": 2, "weather_code": 95,
                        "wind_speed_10m": 80, "precipitation": 25},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [8],
                      "temperature_2m_min": [3], "weather_code": [95],
                      "precipitation_sum": [30], "precipitation_probability_max": [95],
                      "wind_speed_10m_max": [85], "apparent_temperature_max": [5],
                      "apparent_temperature_min": [0]},
        }))
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = weather_service.build_weather_context(
                trip.destination.latitude, trip.destination.longitude,
                "https://api.open-meteo.com/v1/forecast", 8.0
            )
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="severe")
            affected = {i["item_id"]: i for i in result["affected_items"]}

            # Boat is cancelled — no delay
            boat = affected.get(ids["boat_id"])
            if boat and boat["impact_type"] == "cancelled":
                assert boat["estimated_delay_minutes"] is None
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Affected count tests
# ---------------------------------------------------------------------------

class TestAffectedCount:
    def test_clear_weather_zero_affected(self, monkeypatch):
        """Clear weather: 0 items affected."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 25, "feels_like": 26, "humidity": 50,
                            "precipitation": 0, "wind_speed": 5,
                            "condition": "Clear sky", "severity": "low"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx)
            assert len(result["affected_items"]) == 0
        finally:
            db.close()

    def test_severe_weather_excludes_indoor(self, monkeypatch):
        """Severe weather: indoor activities not counted as affected."""
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            weather_ctx = {
                "available": True,
                "current": {"temperature": 5, "feels_like": 2, "humidity": 90,
                            "precipitation": 25, "wind_speed": 80,
                            "condition": "Thunderstorm", "severity": "severe"},
                "forecast": [],
                "observed_at": "2026-09-27T10:00:00+00:00",
            }
            result = simulate_trip_weather_impact(db, trip, weather_ctx, scenario="severe")
            affected_ids = {i["item_id"] for i in result["affected_items"]}
            # Indoor activity should NOT be in affected list
            assert ids["indoor_id"] not in affected_ids
            # But should be in all_items with no impact
            all_ids = {i["item_id"] for i in result.get("all_items", [])}
            assert ids["indoor_id"] in all_ids
        finally:
            db.close()


# ---------------------------------------------------------------------------
# No mutation test
# ---------------------------------------------------------------------------

class TestNoMutation:
    def test_simulation_does_not_mutate_itinerary(self, monkeypatch):
        """Simulation does not modify the real itinerary items."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 5, "relative_humidity_2m": 90,
                        "apparent_temperature": 2, "weather_code": 95,
                        "wind_speed_10m": 80, "precipitation": 25},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [8],
                      "temperature_2m_min": [3], "weather_code": [95],
                      "precipitation_sum": [30], "precipitation_probability_max": [95],
                      "wind_speed_10m_max": [85], "apparent_temperature_max": [5],
                      "apparent_temperature_min": [0]},
        }))
        trip_id, ids = _create_simulation_trip()
        db = SessionLocal()
        try:
            trip = db.query(Trip).options(
                joinedload(Trip.itinerary).joinedload(ItineraryItem.activity),
                joinedload(Trip.itinerary).joinedload(ItineraryItem.transport),
            ).filter(Trip.id == trip_id).first()
            original_statuses = {i.id: i.status for i in trip.itinerary}
            original_titles = {i.id: i.title for i in trip.itinerary}

            weather_ctx = weather_service.build_weather_context(
                trip.destination.latitude, trip.destination.longitude,
                "https://api.open-meteo.com/v1/forecast", 8.0
            )
            simulate_trip_weather_impact(db, trip, weather_ctx, scenario="severe")

            db.expire_all()
            trip2 = db.query(Trip).filter(Trip.id == trip_id).first()
            for item in trip2.itinerary:
                assert item.status == original_statuses[item.id]
                assert item.title == original_titles[item.id]
        finally:
            db.close()


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestSimulationEndpoint:
    def test_simulate_endpoint_success(self, monkeypatch):
        """POST /api/trips/{trip_id}/simulate returns structured simulation."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 5, "relative_humidity_2m": 90,
                        "apparent_temperature": 2, "weather_code": 95,
                        "wind_speed_10m": 80, "precipitation": 25},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [8],
                      "temperature_2m_min": [3], "weather_code": [95],
                      "precipitation_sum": [30], "precipitation_probability_max": [95],
                      "wind_speed_10m_max": [85], "apparent_temperature_max": [5],
                      "apparent_temperature_min": [0]},
        }))
        trip_id, _ = _create_simulation_trip()
        r = client.post(f"/api/trips/{trip_id}/simulate", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["trip_id"] == trip_id
        assert "weather_context" in body
        assert "affected_items" in body
        assert "all_items" in body
        assert "dependencies" in body
        assert "conflicts" in body
        assert "replanning_required" in body
        assert "simulation_timestamp" in body

    def test_simulate_endpoint_with_scenario(self, monkeypatch):
        """POST with scenario override returns simulated scenario."""
        monkeypatch.setattr(weather_service, "_httpx_get", _mock_httpx_get({
            "current": {"temperature_2m": 25, "relative_humidity_2m": 50,
                        "apparent_temperature": 26, "weather_code": 1,
                        "wind_speed_10m": 5, "precipitation": 0},
            "daily": {"time": ["2026-09-27"], "temperature_2m_max": [28],
                      "temperature_2m_min": [22], "weather_code": [1],
                      "precipitation_sum": [0], "precipitation_probability_max": [10],
                      "wind_speed_10m_max": [10], "apparent_temperature_max": [30],
                      "apparent_temperature_min": [24]},
        }))
        trip_id, _ = _create_simulation_trip()
        r = client.post(f"/api/trips/{trip_id}/simulate", json={"scenario": "severe"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scenario"] == "severe"
        assert len(body["affected_items"]) > 0
        # Verify new fields exist
        for item in body["affected_items"]:
            assert "impact_type" in item
            assert "recommended_action" in item

    def test_simulate_endpoint_trip_not_found(self):
        """POST with non-existent trip ID returns 404."""
        r = client.post("/api/trips/non-existent-id/simulate", json={})
        assert r.status_code == 404

    def test_simulate_endpoint_weather_unavailable(self, monkeypatch):
        """Weather unavailable returns valid simulation with no impact."""
        monkeypatch.setattr(
            weather_service, "_httpx_get",
            _mock_httpx_get_error(Exception("connection failed")),
        )
        trip_id, _ = _create_simulation_trip()
        r = client.post(f"/api/trips/{trip_id}/simulate", json={})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["affected_items"] == []
        assert body["replanning_required"] is False
