"""Live transportation research tests — post-requirements flow.

Transportation is NOT part of onboarding/checklist: after the traveler
confirms Origin + Destination + Dates + Travelers, the backend researches
real transfer options, Gemini analyzes them, and the itinerary generator
includes the best transfer (switchable later via change-transport).

No live network is used: geocoding falls back to a default distance and
Gemini analysis is bypassed (gemini_service=None) so the deterministic
distance ranking applies.
"""

import uuid

from fastapi.testclient import TestClient

from backend.database.connection import SessionLocal
from backend.main import app
from backend.models.models import Activity, Destination, Hotel, TransportOption
from backend.transportation.live_research import research_live_transport_options

client = TestClient(app)


def _make_destination(with_catalog_transport=False, origin="Mumbai"):
    db = SessionLocal()
    tag = uuid.uuid4().hex[:8]
    dest = Destination(id=f"dest-trn-{tag}", name=f"Research Town {tag}",
                       slug=f"research-town-{tag}", country="India",
                       state_region="Test Region",
                       description="Test destination for transport research.",
                       latitude=26.1, longitude=91.7)
    db.add(dest)
    db.flush()
    db.add(Hotel(destination_id=dest.id, name=f"Research Stay {tag}",
                 category="mid-range", price_per_night=3000.0, currency="INR",
                 rating=4.0, is_active=True))
    for index in range(2):
        db.add(Activity(destination_id=dest.id, title=f"Research Stop {tag}-{index}",
                        category="culture", duration_hours=2.0,
                        price_per_person=500.0, currency="INR", is_active=True))
    if with_catalog_transport:
        db.add(TransportOption(
            id=f"trn-cat-{tag}", destination_id=dest.id, vendor_id=None,
            type="train", name="Catalog Express",
            route_from=f"{origin} Central", route_to=dest.name,
            duration_hours=8.0, price=2000.0, currency="INR", capacity=72,
            features=[], is_active=True))
    db.commit()
    dest_id = dest.id
    db.close()
    return dest_id


def _delete_destination(dest_id):
    from backend.models.models import Trip
    db = SessionLocal()
    try:
        for trip in db.query(Trip).filter_by(destination_id=dest_id).all():
            db.delete(trip)
        db.query(TransportOption).filter(TransportOption.destination_id == dest_id).delete()
        db.query(Activity).filter(Activity.destination_id == dest_id).delete()
        db.query(Hotel).filter(Hotel.destination_id == dest_id).delete()
        db.query(Destination).filter(Destination.id == dest_id).delete()
        db.commit()
    finally:
        db.close()


def test_research_persists_live_options_when_catalog_has_no_route():
    dest_id = _make_destination(with_catalog_transport=False)
    try:
        db = SessionLocal()
        try:
            dest = db.query(Destination).filter(Destination.id == dest_id).one()
            result = research_live_transport_options(
                db, dest, "Mumbai", traveler_count=2, currency="INR",
                duration_days=3, total_budget=50000.0, gemini_service=None)
            assert result["researched"] >= 2
            assert result["source"] == "distance_fallback"
            assert result["error"] is None
            rows = db.query(TransportOption).filter(
                TransportOption.destination_id == dest_id).all()
            assert all(r.inventory_source == "live" for r in rows)
            assert all(r.verification_status == "live_researched" for r in rows)
            assert all("Mumbai" in (r.route_from or "") for r in rows)
        finally:
            db.close()
    finally:
        _delete_destination(dest_id)


def test_research_is_idempotent():
    dest_id = _make_destination(with_catalog_transport=False)
    try:
        db = SessionLocal()
        try:
            dest = db.query(Destination).filter(Destination.id == dest_id).one()
            first = research_live_transport_options(
                db, dest, "Pune", traveler_count=2, currency="INR", gemini_service=None)
            second = research_live_transport_options(
                db, dest, "Pune", traveler_count=2, currency="INR", gemini_service=None)
            assert first["researched"] >= 2
            names = sorted(r.name for r in db.query(TransportOption).filter(
                TransportOption.destination_id == dest_id).all())
            assert len(names) == len(set(names))
            assert second["researched"] >= 0
        finally:
            db.close()
    finally:
        _delete_destination(dest_id)


def test_catalog_route_short_circuits_research():
    dest_id = _make_destination(with_catalog_transport=True, origin="Mumbai")
    try:
        db = SessionLocal()
        try:
            dest = db.query(Destination).filter(Destination.id == dest_id).one()
            result = research_live_transport_options(
                db, dest, "Mumbai", traveler_count=2, currency="INR", gemini_service=None)
            assert result["source"] == "catalog"
            assert result["researched"] == 0
            assert len(result["options"]) == 1
        finally:
            db.close()
    finally:
        _delete_destination(dest_id)


def test_create_trip_with_origin_includes_researched_transfer():
    """Full product flow: requirements confirmed -> research -> itinerary
    carries a Day-1 transfer for the origin route (switchable later)."""
    dest_id = _make_destination(with_catalog_transport=False)
    try:
        r = client.post("/api/trips", json={
            "title": "Origin Research Trip", "destination_id": dest_id,
            "origin": "Mumbai", "duration_days": 3, "traveler_count": 2,
            "total_budget": 60000.0, "currency": "INR",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["origin"] == "Mumbai"
        transfers = [i for i in body["itinerary"] if i["item_type"] == "transport"]
        assert len(transfers) >= 1
        assert "Mumbai" in (transfers[0]["description"] or transfers[0]["title"])
        assert body["cost_breakdown"]["transport"] > 0
    finally:
        _delete_destination(dest_id)


def test_create_trip_without_origin_carries_no_transfer():
    """No origin -> no research -> itinerary simply has no transfer items."""
    dest_id = _make_destination(with_catalog_transport=False)
    try:
        r = client.post("/api/trips", json={
            "title": "No Origin Trip", "destination_id": dest_id,
            "duration_days": 3, "traveler_count": 2,
            "total_budget": 60000.0, "currency": "INR",
        })
        assert r.status_code == 200, r.text
        assert all(i["item_type"] != "transport" for i in r.json()["itinerary"])
    finally:
        _delete_destination(dest_id)
