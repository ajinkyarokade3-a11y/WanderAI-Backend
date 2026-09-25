"""Transport switch contract + details tests.

Covers the reported bug ( switcher shows options but change-transport 400s):
the generic GET /transport listing resolves destinations by NAME, which is
ambiguous with duplicate-name rows — the trip-scoped
GET /trips/{id}/transport-options endpoint fixes the contract so every
displayed ID is selectable. Also covers international feasibility
(Mumbai -> Singapore = flight only), invalid-ID rejection, booking link /
detail preservation, and duration-based timing on switch.

Provider/geocode network is monkeypatched; no live calls are made.
"""

import uuid

from fastapi.testclient import TestClient

from backend.database.connection import SessionLocal
from backend.main import app
from backend.models.models import Activity, Destination, Hotel, TransportOption, Trip

client = TestClient(app)


def _make_dest(name, slug_suffix, country="India", lat=26.1, lng=91.7,
               with_detailed_transport=False):
    db = SessionLocal()
    tag = uuid.uuid4().hex[:8]
    dest = Destination(id=f"dest-sw-{tag}", name=name, slug=f"{slug_suffix}-{tag}",
                       country=country, state_region="Test Region",
                       description="Switch test destination.",
                       latitude=lat, longitude=lng)
    db.add(dest)
    db.flush()
    db.add(Hotel(destination_id=dest.id, name=f"Stay {tag}", category="mid-range",
                 price_per_night=3000.0, currency="INR", is_active=True))
    for i in range(2):
        db.add(Activity(destination_id=dest.id, title=f"Stop {tag}-{i}", category="culture",
                        duration_hours=2.0, price_per_person=500.0, currency="INR",
                        is_active=True))
    if with_detailed_transport:
        db.add(TransportOption(
            destination_id=dest.id, type="flight", name="Test Airways TX-101",
            route_from="Mumbai", route_to=name, duration_hours=2.5,
            price=8500.0, currency="INR", capacity=180,
            features=["Meal included"], is_active=True,
            service_number="TX-101", operator_name="Test Airways",
            departure_time="06:40", arrival_time="09:10",
            stops=["Non-stop"], travel_class="Economy",
            availability_status="available",
            source_url="https://example.com/book/tx-101"))
    db.commit()
    did = dest.id
    db.close()
    return did


def _cleanup(*dest_ids):
    db = SessionLocal()
    try:
        for did in dest_ids:
            for trip in db.query(Trip).filter_by(destination_id=did).all():
                db.delete(trip)
            db.query(TransportOption).filter(TransportOption.destination_id == did).delete()
            db.query(Activity).filter(Activity.destination_id == did).delete()
            db.query(Hotel).filter(Hotel.destination_id == did).delete()
            db.query(Destination).filter(Destination.id == did).delete()
        db.commit()
    finally:
        db.close()


def _india_geo(name, *args, **kwargs):
    return (19.0760, 72.8777, f"{name}, Maharashtra, India")


def test_trip_scoped_options_match_switch_validation_with_duplicate_names():
    """Regression: duplicate destination names made GET /transport return a
    sibling row's IDs (400 on switch). The trip-scoped endpoint only ever
    returns the trip's own row, so every ID switches cleanly; a sibling
    row's ID is still correctly rejected."""
    d1 = _make_dest("Switchville", "switchville", with_detailed_transport=True)
    d2 = _make_dest("Switchville", "switchville", with_detailed_transport=True)
    try:
        r = client.post("/api/trips", json={"title": "T", "destination_id": d2,
                                            "origin": "Mumbai", "duration_days": 3,
                                            "traveler_count": 2, "total_budget": 60000.0,
                                            "currency": "INR"})
        assert r.status_code == 200, r.text
        trip_id = r.json()["id"]

        opts = client.get(f"/api/trips/{trip_id}/transport-options")
        assert opts.status_code == 200, opts.text
        own_ids = {o["id"] for o in opts.json()}
        assert own_ids, "trip-scoped listing must not be empty"

        db = SessionLocal()
        try:
            sibling = db.query(TransportOption).filter(
                TransportOption.destination_id == d1).first()
            sibling_id = sibling.id
        finally:
            db.close()
        assert sibling_id not in own_ids

        # Sibling-row ID: correctly rejected (validation NOT bypassed).
        bad = client.post(f"/api/trips/{trip_id}/change-transport",
                          json={"transport_id": sibling_id})
        assert bad.status_code == 400

        # Own-row ID: switches and persists with details + booking link.
        good = client.post(f"/api/trips/{trip_id}/change-transport",
                           json={"transport_id": sorted(own_ids)[0]})
        assert good.status_code == 200, good.text
        transfers = [i for i in good.json()["itinerary"] if i["item_type"] == "transport"]
        assert len(transfers) == 1
        details = transfers[0]["transport_details"]
        assert details["transport_id"] == sorted(own_ids)[0]
        assert details["operator"] == "Test Airways"
        assert details["service_number"] == "TX-101"
        assert details["booking_url"] == "https://example.com/book/tx-101"
    finally:
        _cleanup(d1, d2)


def test_switch_updates_cost_timing_and_preserves_rest():
    d = _make_dest("Costville", "costville", with_detailed_transport=True)
    try:
        r = client.post("/api/trips", json={"title": "T", "destination_id": d,
                                            "duration_days": 3, "traveler_count": 2,
                                            "total_budget": 60000.0, "currency": "INR"})
        assert r.status_code == 200, r.text
        trip_id, before = r.json()["id"], r.json()
        non_transport_before = sorted(
            (i["id"], i["title"], i["cost"]) for i in before["itinerary"]
            if i["item_type"] != "transport")

        db = SessionLocal()
        try:
            target = db.query(TransportOption).filter(
                TransportOption.destination_id == d).first()
            target_id, target_price = target.id, float(target.price)
        finally:
            db.close()
        s = client.post(f"/api/trips/{trip_id}/change-transport",
                        json={"transport_id": target_id})
        assert s.status_code == 200, s.text
        after = s.json()
        # Cost follows the new option's price.
        assert after["cost_breakdown"]["transport"] == target_price
        # Timing follows the 2.5h duration on the standard window.
        transfer = next(i for i in after["itinerary"] if i["item_type"] == "transport")
        assert transfer["start_time"] == "09:30 AM"
        assert transfer["end_time"] == "12:00 PM"
        assert transfer["status"] == "confirmed"
        # Everything else preserved byte-for-byte.
        non_transport_after = sorted(
            (i["id"], i["title"], i["cost"]) for i in after["itinerary"]
            if i["item_type"] != "transport")
        assert non_transport_after == non_transport_before
    finally:
        _cleanup(d)


def test_invalid_transport_id_rejected():
    d = _make_dest("Rejectville", "rejectville")
    try:
        r = client.post("/api/trips", json={"title": "T", "destination_id": d,
                                            "duration_days": 3, "traveler_count": 2,
                                            "total_budget": 60000.0, "currency": "INR"})
        assert r.status_code == 200, r.text
        s = client.post(f"/api/trips/{r.json()['id']}/change-transport",
                        json={"transport_id": f"nope-{uuid.uuid4().hex[:8]}"})
        assert s.status_code == 400
        assert "not found" in s.json()["detail"].lower()
    finally:
        _cleanup(d)


def test_singapore_lists_flight_only(monkeypatch):
    """Mumbai -> Singapore: train/road must never be offered."""
    import backend.transportation.live_research as live_research
    import backend.places.service as places_service

    monkeypatch.setattr(places_service, "geocode_place",
                        lambda name, *a, **k: (19.0760, 72.8777,
                                              f"{name}, Maharashtra, India"))
    d = _make_dest("Singapore", "singapore-test", country="Singapore",
                   lat=1.3521, lng=103.8198)
    try:
        r = client.post("/api/trips", json={"title": "SG", "destination_id": d,
                                            "origin": "Mumbai", "duration_days": 3,
                                            "traveler_count": 2, "total_budget": 90000.0,
                                            "currency": "INR"})
        assert r.status_code == 200, r.text
        trip_id = r.json()["id"]
        opts = client.get(f"/api/trips/{trip_id}/transport-options")
        assert opts.status_code == 200, opts.text
        modes = {o["type"] for o in opts.json()}
        assert modes, "expected researched flight option"
        assert modes == {"flight"}, f"infeasible modes offered: {modes}"
        transfers = [i for i in r.json()["itinerary"] if i["item_type"] == "transport"]
        assert transfers, "itinerary must include the flight transfer"
    finally:
        _cleanup(d)
    _ = live_research  # module import anchored for monkeypatch target clarity


def test_research_unit_international_flight_only(monkeypatch):
    import backend.transportation.live_research as live_research

    monkeypatch.setattr(live_research, "_geocode_origin",
                        lambda origin: (19.0760, 72.8777, "Mumbai, Maharashtra, India", "india"))
    db = SessionLocal()
    tag = uuid.uuid4().hex[:8]
    dest = Destination(id=f"dest-int-{tag}", name="Singapore", slug=f"singapore-{tag}",
                       country="Singapore", state_region="Singapore",
                       description="t", latitude=1.3521, longitude=103.8198)
    db.add(dest)
    db.commit()
    try:
        result = live_research.research_live_transport_options(
            db, dest, "Mumbai", traveler_count=2, currency="INR", gemini_service=None)
        assert result["international"] is True
        assert {o.type for o in result["options"]} == {"flight"}
    finally:
        _cleanup(dest.id)


def test_research_unit_domestic_keeps_surface_modes(monkeypatch):
    import backend.transportation.live_research as live_research

    monkeypatch.setattr(live_research, "_geocode_origin",
                        lambda origin: (19.0760, 72.8777, "Mumbai, Maharashtra, India", "india"))
    db = SessionLocal()
    tag = uuid.uuid4().hex[:8]
    dest = Destination(id=f"dest-dom-{tag}", name="Udaipur", slug=f"udaipur-{tag}",
                       country="India", state_region="Rajasthan",
                       description="t", latitude=24.58, longitude=73.71)
    db.add(dest)
    db.commit()
    try:
        result = live_research.research_live_transport_options(
            db, dest, "Mumbai", traveler_count=2, currency="INR", gemini_service=None)
        assert result["international"] is False
        assert {o.type for o in result["options"]} >= {"train", "private_cab"}
    finally:
        _cleanup(dest.id)
