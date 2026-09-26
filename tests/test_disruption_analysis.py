"""Tests for disruption impact analysis and AI replan flow."""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import (
    Activity, Alert, ChangeHistory, Destination, ItineraryItem, Trip, User,
)

client = TestClient(app)


def _create_test_trip_with_disruption():
    """Create a trip with itinerary, an alert, and catalog activities.

    Returns (trip_id, item_activity_id, item_title, alert_id, activities_data).
    Each call creates a fresh trip so tests are independent.
    """
    db = SessionLocal()
    try:
        user = db.query(User).first()
        destination = db.query(Destination).filter(Destination.slug == "manali").first()
        assert user is not None, "No user in database"
        assert destination is not None, "No manali destination in database"

        # Check if catalog activities exist
        activities = db.query(Activity).filter(
            Activity.destination_id == destination.id, Activity.is_active == True  # noqa: E712
        ).all()
        assert len(activities) >= 2, f"Need at least 2 catalog activities, found {len(activities)}"

        # Create trip
        trip = Trip(
            user_id=user.id,
            destination_id=destination.id,
            title="Disruption Test Trip",
            traveler_count=2,
            duration_days=5,
            total_budget=50000.0,
            currency="INR",
        )
        db.add(trip)
        db.flush()

        # Create an itinerary item (outdoor activity on Day 3)
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

        # Create an alert
        alert = Alert(
            trip_id=trip.id,
            alert_type="weather",
            severity="critical",
            title="Heavy snowfall at Solang Valley",
            description="Roads to Solang Valley temporarily blocked due to heavy snowfall",
            is_resolved=False,
        )
        db.add(alert)
        db.commit()

        # Extract values before returning to avoid detached instance issues
        trip_id = trip.id
        item_activity_id = item.activity_id
        item_title = item.title
        alert_id = alert.id
        activities_data = [
            {"id": a.id, "title": a.title, "category": a.category,
             "price_per_person": a.price_per_person, "meeting_point": a.meeting_point}
            for a in activities
        ]
        return trip_id, item_activity_id, item_title, alert_id, activities_data
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _mock_ai_response(monkeypatch, response_text):
    """Mock the failover router to return a canned AI response."""
    def _fake_generate(request, **kwargs):
        return SimpleNamespace(
            text=response_text,
            provider="gemini",
            model="gemini-2.5-flash",
        )
    monkeypatch.setattr("backend.ai.disruption_analysis.generate_with_failover", _fake_generate)


def _mock_ai_failure(monkeypatch):
    """Mock the failover router to raise an AIProviderError."""
    from backend.ai.ai_types import AIErrorCategory, AIProviderError

    def _fake_generate(request, **kwargs):
        raise AIProviderError(
            "all providers failed",
            category=AIErrorCategory.PROVIDER_UNAVAILABLE,
            provider="",
            model="",
        )
    monkeypatch.setattr("backend.ai.disruption_analysis.generate_with_failover", _fake_generate)


def test_disruption_analysis_success(monkeypatch):
    """Successful analysis returns full disruption impact data."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    # Pick an alternative activity (not the one already in the itinerary)
    alternative = next((a for a in activities_data if a["id"] != item_activity_id), None)
    assert alternative is not None, "Need at least one alternative activity"

    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Heavy snowfall blocking roads to Solang Valley",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Day 3 outdoor activity at Solang Valley is unavailable due to heavy snowfall and road closures.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [{"description": "Route to Solang Valley is blocked", "required": True}],
        "alternatives": [
            {
                "activity_id": alternative["id"],
                "title": alternative["title"],
                "cost": alternative["price_per_person"] * 2,
                "rationale": "Indoor cultural activity available at the same destination",
                "estimated_time_impact": "No time impact",
                "estimated_cost_impact": 0,
                "risks": ["Weather dependent"],
            }
        ],
        "estimated_schedule_impact": "Day 3 activity needs replacement",
        "estimated_cost_impact": 0,
        "risks_limitations": ["Alternative is weather dependent", "Subject to availability"],
    }))

    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trip_id"] == trip_id
    assert body["disruption_cause"] == "Heavy snowfall blocking roads to Solang Valley"
    assert body["affected_day"] == 3
    assert body["severity"] == "critical"
    assert len(body["alternatives"]) == 1
    assert body["alternatives"][0]["activity_id"] == alternative["id"]
    assert body["status"] == "pending_approval"
    assert "source" in body


def test_disruption_analysis_no_alternatives(monkeypatch):
    """Analysis with no available alternatives returns empty alternatives array."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Heavy snowfall blocking roads to Solang Valley",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Day 3 outdoor activity is unavailable.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [],
        "alternatives": [],
        "risks_limitations": ["No suitable alternatives available"],
    }))

    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alternatives"] == []
    assert body["status"] == "pending_approval"


def test_disruption_analysis_validates_alternatives_against_catalog(monkeypatch):
    """AI-suggested alternatives that don't exist in the catalog are filtered out."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    # AI suggests a non-existent activity ID
    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Heavy snowfall",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Day 3 outdoor activity is unavailable.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [],
        "alternatives": [
            {
                "activity_id": "non-existent-activity-id",
                "title": "Fake Activity",
                "cost": 1000,
                "rationale": "This should be filtered out",
            }
        ],
        "risks_limitations": [],
    }))

    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alternatives"] == []


def test_disruption_analysis_ai_failure(monkeypatch):
    """AI/API failure returns 502 with user-safe message."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    _mock_ai_failure(monkeypatch)

    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 502, r.text
    assert "temporarily unavailable" in r.json()["detail"].lower()


def test_disruption_analysis_approve_applies_itinerary_change(monkeypatch):
    """Approving an alternative updates the itinerary and records audit log."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    # Pick an alternative activity
    alternative = next((a for a in activities_data if a["id"] != item_activity_id), None)
    assert alternative is not None

    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Heavy snowfall",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Day 3 outdoor activity is unavailable.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [],
        "alternatives": [
            {
                "activity_id": alternative["id"],
                "title": alternative["title"],
                "cost": alternative["price_per_person"] * 2,
                "rationale": "Indoor cultural activity",
            }
        ],
        "risks_limitations": [],
    }))

    # Generate analysis
    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 200, r.text
    analysis = r.json()
    assert len(analysis["alternatives"]) == 1

    # Apply the alternative
    apply_r = client.post(f"/api/trips/{trip_id}/apply-replan", json={
        "alternative_id": alternative["id"],
        "notes": "Operator approved AI-suggested alternative",
    })
    assert apply_r.status_code == 200, apply_r.text
    body = apply_r.json()
    assert body["success"] is True

    # Verify itinerary was updated
    db = SessionLocal()
    try:
        updated_item = db.query(ItineraryItem).filter(
            ItineraryItem.trip_id == trip_id, ItineraryItem.day_number == 3
        ).first()
        assert updated_item is not None
        assert updated_item.activity_id == alternative["id"]
        assert updated_item.title == alternative["title"]

        # Verify alert was resolved
        resolved_alert = db.query(Alert).filter(Alert.trip_id == trip_id).first()
        assert resolved_alert is not None
        assert resolved_alert.is_resolved is True

        # Verify audit log entries exist
        history = db.query(ChangeHistory).filter(
            ChangeHistory.trip_id == trip_id
        ).all()
        actions = {h.action for h in history}
        assert "disruption_analysis_generated" in actions
        assert "replan_applied" in actions
    finally:
        db.close()


def test_disruption_analysis_reject_leaves_itinerary_unchanged(monkeypatch):
    """Rejecting/dismissing a disruption does not change the itinerary."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Heavy snowfall",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Day 3 outdoor activity is unavailable.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [],
        "alternatives": [],
        "risks_limitations": [],
    }))

    # Generate analysis
    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 200, r.text

    # Dismiss the disruption
    dismiss_r = client.post(f"/api/trips/{trip_id}/dismiss-disruption", json={
        "reason": "Operator decided to keep current itinerary",
    })
    assert dismiss_r.status_code == 200, dismiss_r.text
    assert dismiss_r.json()["success"] is True

    # Verify itinerary is unchanged
    db = SessionLocal()
    try:
        unchanged_item = db.query(ItineraryItem).filter(
            ItineraryItem.trip_id == trip_id, ItineraryItem.day_number == 3
        ).first()
        assert unchanged_item is not None
        assert unchanged_item.activity_id == item_activity_id

        # Verify alert was resolved
        resolved_alert = db.query(Alert).filter(Alert.trip_id == trip_id).first()
        assert resolved_alert is not None
        assert resolved_alert.is_resolved is True

        # Verify dismissal was recorded in audit log
        history = db.query(ChangeHistory).filter(
            ChangeHistory.trip_id == trip_id, ChangeHistory.action == "disruption_dismissed"
        ).all()
        assert len(history) >= 1
    finally:
        db.close()


def test_disruption_analysis_audit_log_records_analysis(monkeypatch):
    """Generating an analysis records an audit log entry."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Heavy snowfall",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Day 3 outdoor activity is unavailable.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [],
        "alternatives": [],
        "risks_limitations": [],
    }))

    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={})
    assert r.status_code == 200, r.text

    db = SessionLocal()
    try:
        history = db.query(ChangeHistory).filter(
            ChangeHistory.trip_id == trip_id,
            ChangeHistory.action == "disruption_analysis_generated",
        ).all()
        assert len(history) >= 1
        assert history[0].changed_by == "ai"
    finally:
        db.close()


def test_disruption_analysis_with_explicit_disruption_payload(monkeypatch):
    """Analysis uses the provided disruption data instead of the alert."""
    trip_id, item_activity_id, item_title, alert_id, activities_data = _create_test_trip_with_disruption()

    _mock_ai_response(monkeypatch, json.dumps({
        "disruption_cause": "Road closure due to landslide",
        "affected_day": 3,
        "affected_activity": item_title,
        "severity": "critical",
        "impact_summary": "Road to Solang Valley blocked by landslide.",
        "affected_bookings": [],
        "affected_vendors": [],
        "transport_changes": [],
        "alternatives": [],
        "risks_limitations": [],
    }))

    r = client.post(f"/api/trips/{trip_id}/disruption-analysis", json={
        "disruption": {
            "type": "road_closure",
            "severity": "critical",
            "title": "Landslide blocking road",
            "description": "Road to Solang Valley blocked by landslide",
        }
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["disruption_cause"] == "Road closure due to landslide"
    assert body["severity"] == "critical"
