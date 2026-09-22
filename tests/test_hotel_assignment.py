"""Day-wise, proximity-based hotel assignment tests.

Covers the generic coordinate-driven algorithm (no per-region rules):
retention, new-stay creation, per-day independence, multi-night stays,
missing location data, budgets, and API validation. Fixture coordinates
are real-world points used only as test geometry.
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import SessionLocal
from backend.itinerary.generator import ItineraryGenerator
from backend.itinerary.hotel_assignment import (
    HOTEL_ASSIGNMENT_CONFIG,
    DayAnchor,
    centroid,
    estimate_travel_minutes,
    haversine_km,
    plan_overnight_stays,
)
from backend.models.models import (
    Activity,
    Destination,
    Hotel,
    ItineraryItem,
    TransportOption,
    Trip,
    TripPreference,
    User,
)
from backend.recommendation.engine import RecommendationEngine

client = TestClient(app)

# Real-world test geometry (algorithm itself is location-independent).
AMRITSAR = (31.6340, 74.8723)
LUDHIANA = (30.9010, 75.8573)  # ~124 km straight-line from Amritsar
# Synthetic middle-band point ~50 km north of Amritsar (between the 40 km
# preferred and 60 km significant thresholds).
MIDTOWN = (32.0836, 74.8723)


def _stub_hotel(name, lat, lng, price=2000.0):
    return SimpleNamespace(
        id=f"stub-{name}",
        name=name,
        latitude=lat,
        longitude=lng,
        price_per_night=price,
        description=f"{name} description",
        address=name,
    )


# ---------------------------------------------------------------------------
# Pure unit tests (no database)
# ---------------------------------------------------------------------------

def test_haversine_known_distance():
    dist = haversine_km(*AMRITSAR, *LUDHIANA)
    assert dist is not None
    assert 110.0 < dist < 135.0


def test_haversine_missing_coords_returns_none():
    assert haversine_km(None, 1.0, 2.0, 3.0) is None
    assert haversine_km(1.0, None, 2.0, 3.0) is None
    assert haversine_km("bad", 1.0, 2.0, 3.0) is None


def test_estimate_travel_minutes_documents_fallback():
    assert estimate_travel_minutes(100.0, 50.0) == pytest.approx(120.0)
    assert estimate_travel_minutes(None) is None
    assert estimate_travel_minutes(10.0, 0) is None


def test_centroid_skips_bad_points():
    assert centroid([(1.0, 2.0), (None, 2.0), (3.0, 4.0)]) == (2.0, 3.0)
    assert centroid([]) is None
    assert centroid([(None, None)]) is None


def test_config_thresholds_present_and_ordered():
    cfg = HOTEL_ASSIGNMENT_CONFIG
    assert cfg["max_preferred_distance_km"] < cfg["significant_distance_km"]
    assert (
        cfg["max_preferred_travel_time_minutes"]
        < cfg["significant_travel_time_minutes"]
    )


def _anchors(*places):
    return {
        night: DayAnchor(latitude=lat, longitude=lng, label=f"Day {night + 1}")
        for night, (lat, lng) in enumerate(places, start=1)
    }


def test_close_anchors_retain_single_hotel():
    hotels = [_stub_hotel("H1", *AMRITSAR), _stub_hotel("H2", *LUDHIANA)]
    stays, spent = plan_overnight_stays(
        nights=3,
        anchors=_anchors(AMRITSAR, AMRITSAR, AMRITSAR),
        hotels=hotels,
    )
    assert [s.hotel.id for s in stays] == ["stub-H1"] * 3
    assert all(s.retained or s.night == 1 for s in stays)
    assert spent == pytest.approx(2000.0 * 3)


def test_opening_stay_nearby_keeps_near_wording():
    hotels = [_stub_hotel("H1", *AMRITSAR)]
    stays, _ = plan_overnight_stays(
        nights=1,
        anchors={1: DayAnchor(latitude=AMRITSAR[0], longitude=AMRITSAR[1], label="Spot")},
        hotels=hotels,
    )
    assert stays[0].reason == "Opening stay near Spot."
    assert stays[0].distance_km == pytest.approx(0.0)


def test_opening_stay_far_avoids_near_wording():
    kochi = (9.9658, 76.2421)
    kovalam = (8.4004, 76.9787)
    hotels = [_stub_hotel("H1", *kochi)]
    stays, _ = plan_overnight_stays(
        nights=1,
        anchors={1: DayAnchor(latitude=kovalam[0], longitude=kovalam[1], label="Day 2")},
        hotels=hotels,
    )
    assert "near" not in stays[0].reason
    assert "about 192 km" in stays[0].reason
    assert "Day 2" in stays[0].reason
    assert stays[0].distance_km == pytest.approx(192, abs=2.0)


def test_opening_stay_missing_hotel_coords_honest():
    hotels = [_stub_hotel("H1", None, None)]
    stays, _ = plan_overnight_stays(
        nights=1,
        anchors={1: DayAnchor(latitude=10.0, longitude=20.0, label="Spot")},
        hotels=hotels,
    )
    assert stays[0].reason == "Opening stay for Spot."
    assert stays[0].distance_km is None
    assert "near" not in stays[0].reason


def test_opening_stay_missing_anchor_honest():
    hotels = [_stub_hotel("H1", *AMRITSAR)]
    stays, _ = plan_overnight_stays(nights=1, anchors={}, hotels=hotels)
    assert stays[0].reason == "Opening stay for the trip area."
    assert stays[0].distance_km is None


def test_later_night_reasons_unchanged():
    hotels = [_stub_hotel("H-Amritsar", *AMRITSAR), _stub_hotel("H-Ludhiana", *LUDHIANA)]
    stays, _ = plan_overnight_stays(
        nights=3,
        anchors=_anchors(AMRITSAR, LUDHIANA, LUDHIANA),
        hotels=hotels,
    )
    assert stays[0].hotel.id == "stub-H-Amritsar"
    assert stays[1].hotel.id == "stub-H-Ludhiana"
    assert not stays[1].retained
    assert stays[2].hotel.id == "stub-H-Ludhiana"
    assert stays[2].retained
    assert "practical day trip" in stays[2].reason


def test_far_anchor_creates_new_stay_with_reason():
    hotels = [_stub_hotel("H-Amritsar", *AMRITSAR), _stub_hotel("H-Ludhiana", *LUDHIANA)]
    stays, _ = plan_overnight_stays(
        nights=2,
        anchors=_anchors(AMRITSAR, LUDHIANA),
        hotels=hotels,
    )
    assert stays[0].hotel.id == "stub-H-Amritsar"
    assert stays[1].hotel.id == "stub-H-Ludhiana"
    assert not stays[1].retained
    assert "Ludhiana" not in stays[1].reason  # label is generic Day-N text
    assert "Day 3" in stays[1].reason  # night 2 serves the next morning
    assert "about" in stays[1].reason  # estimates never exact


def test_middle_band_retains_without_churn():
    hotels = [_stub_hotel("H-Amritsar", *AMRITSAR), _stub_hotel("H-Midtown", *MIDTOWN)]
    stays, _ = plan_overnight_stays(
        nights=2,
        anchors=_anchors(AMRITSAR, MIDTOWN),
        hotels=hotels,
    )
    assert stays[1].hotel.id == "stub-H-Amritsar"
    assert stays[1].retained
    assert "comfortable reach" in stays[1].reason


def test_missing_anchor_coords_retain():
    hotels = [_stub_hotel("H1", *AMRITSAR)]
    stays, _ = plan_overnight_stays(nights=2, anchors={}, hotels=hotels)
    assert [s.hotel.id for s in stays] == ["stub-H1"] * 2
    assert "location detail" in stays[1].reason


def test_missing_hotel_coords_retain():
    hotels = [_stub_hotel("H1", None, None), _stub_hotel("H2", *LUDHIANA)]
    stays, _ = plan_overnight_stays(
        nights=2, anchors=_anchors(AMRITSAR, LUDHIANA), hotels=hotels
    )
    # First night takes ranked-first H1; night 2 cannot compare -> retains.
    assert stays[1].hotel.id == "stub-H1"
    assert stays[1].retained


def test_pinned_nights_never_reassigned_and_cost_free():
    hotels = [_stub_hotel("H-Amritsar", *AMRITSAR), _stub_hotel("H-Ludhiana", *LUDHIANA)]
    pinned = {2: hotels[0]}
    stays, spent = plan_overnight_stays(
        nights=2,
        anchors=_anchors(LUDHIANA, LUDHIANA),
        hotels=hotels,
        pinned_hotels=pinned,
        total_pot=2000.0,
    )
    assert stays[1].hotel.id == "stub-H-Amritsar"
    assert "chosen" in stays[1].reason
    assert spent == pytest.approx(2000.0)  # only night 1 spent


def test_unaffordable_night_returns_none_hotel():
    hotels = [_stub_hotel("Expensive", *AMRITSAR, price=999999.0)]
    stays, _ = plan_overnight_stays(
        nights=1, anchors=_anchors(AMRITSAR), hotels=hotels, total_pot=100.0
    )
    assert stays[0].hotel is None


# ---------------------------------------------------------------------------
# Integration tests through the generator (dev PostgreSQL, isolated fixtures)
# ---------------------------------------------------------------------------

def _unique(prefix):
    return f"{prefix} {uuid.uuid4().hex[:8]}"


@pytest.fixture()
def punjab_trip():
    """4-day trip: days 1-2 around Amritsar, days 3-4 around Ludhiana."""
    suffix = uuid.uuid4().hex[:8]
    db = SessionLocal()
    try:
        user = User(email=f"stay-{suffix}@example.test", full_name="Stay Tester")
        db.add(user)
        db.flush()
        dest = Destination(
            name=f"Test Punjab {suffix}",
            slug=f"test-punjab-{suffix}",
            country="India",
            state_region="Punjab",
            description="Day-wise stay test destination.",
            latitude=31.0,
            longitude=75.0,
            inventory_source="catalog",
            verification_status="catalog_verified",
            is_featured=False,
        )
        db.add(dest)
        db.flush()
        hotels = [
            Hotel(
                destination_id=dest.id, name=f"Amritsar Stay {suffix}",
                category="mid-range", price_per_night=2000.0, currency="INR",
                rating=4.2, address="Amritsar", latitude=AMRITSAR[0],
                longitude=AMRITSAR[1], inventory_source="catalog",
                verification_status="catalog_verified", is_active=True,
            ),
            Hotel(
                destination_id=dest.id, name=f"Ludhiana Stay {suffix}",
                category="mid-range", price_per_night=2200.0, currency="INR",
                rating=4.3, address="Ludhiana", latitude=LUDHIANA[0],
                longitude=LUDHIANA[1], inventory_source="catalog",
                verification_status="catalog_verified", is_active=True,
            ),
        ]
        db.add_all(hotels)
        coords = [AMRITSAR, AMRITSAR, LUDHIANA, LUDHIANA]
        activities = [
            Activity(
                destination_id=dest.id, title=f"Spot {idx} {suffix}",
                category="culture", duration_hours=2.0, price_per_person=0.0,
                currency="INR", difficulty_level="easy", rating=4.5,
                meeting_point=f"Point {idx}", latitude=lat, longitude=lng,
                inventory_source="catalog", verification_status="catalog_verified",
                is_active=True,
            )
            for idx, (lat, lng) in enumerate(coords)
        ]
        db.add_all(activities)
        transport = TransportOption(
            destination_id=dest.id, type="private_cab", name=f"Cab {suffix}",
            route_from="Origin", route_to=dest.name, duration_hours=4.0,
            price=1000.0, currency="INR", capacity=4,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        db.add(transport)
        trip = Trip(
            user_id=user.id, destination_id=dest.id,
            title=f"Stay Trip {suffix}", status="planning", duration_days=4,
            total_budget=100000.0, currency="INR", traveler_count=2, pace="balanced",
        )
        db.add(trip)
        db.commit()
        ids = {
            "trip_id": trip.id,
            "hotel_ids": [h.id for h in hotels],
            "activity_ids": [a.id for a in activities],
            "transport_id": transport.id,
        }
        yield ids
    finally:
        try:
            trip_id = ids["trip_id"]
            db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip_id).delete(
                synchronize_session=False
            )
            db.query(TripPreference).filter(TripPreference.trip_id == trip_id).delete(
                synchronize_session=False
            )
            db.query(Trip).filter(Trip.id == trip_id).delete(synchronize_session=False)
            for model, key in (
                (Hotel, "hotel_ids"),
                (Activity, "activity_ids"),
                (TransportOption, "transport_id"),
            ):
                wanted = ids[key] if isinstance(ids[key], list) else [ids[key]]
                for row_id in wanted:
                    db.query(model).filter(model.id == row_id).delete(
                        synchronize_session=False
                    )
            db.query(Destination).filter(
                Destination.slug == f"test-punjab-{suffix}"
            ).delete(synchronize_session=False)
            db.query(User).filter(User.email == f"stay-{suffix}@example.test").delete(
                synchronize_session=False
            )
            db.commit()
        finally:
            db.close()


def _patch_ranking(monkeypatch, ids):
    def fake_recommendations(self, destination_id, preferences, discovery_session_id=None):
        return {
            "recommended_hotels": [{"id": i} for i in ids["hotel_ids"]],
            "recommended_activities": [{"id": i} for i in ids["activity_ids"]],
            "recommended_transport": [{"id": ids["transport_id"]}],
        }

    monkeypatch.setattr(
        RecommendationEngine, "get_recommendations", fake_recommendations
    )


def _hotel_items(db, trip_id):
    return (
        db.query(ItineraryItem)
        .filter(ItineraryItem.trip_id == trip_id, ItineraryItem.item_type == "hotel")
        .order_by(ItineraryItem.day_number)
        .all()
    )


def test_generator_assigns_far_nights_to_nearby_stays(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        items = ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
        hotel_items = [i for i in items if i.item_type == "hotel"]
        by_night = {i.day_number: i.hotel_id for i in hotel_items}
        # Nights 1..3 all assigned; Ludhiana nights move to the Ludhiana stay.
        assert set(by_night) == {1, 2, 3}
        amritsar_id, ludhiana_id = punjab_trip["hotel_ids"]
        assert by_night[1] == amritsar_id
        assert by_night[2] == ludhiana_id
        assert by_night[3] == ludhiana_id
        for item in hotel_items:
            reason = (item.meta_data or {}).get("ui", {}).get("hotel_assignment_reason")
            assert reason, "each stay must carry a user-facing reason"
        # Per-night cost kept (no multiplied totals on day items).
        assert all(float(i.cost or 0) > 0 for i in hotel_items)
    finally:
        db.close()


def test_close_together_trip_keeps_one_hotel(punjab_trip, monkeypatch):
    db = SessionLocal()
    try:
        for activity_id in punjab_trip["activity_ids"]:
            row = db.query(Activity).filter(Activity.id == activity_id).first()
            row.latitude, row.longitude = AMRITSAR
        db.commit()
    finally:
        db.close()
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        items = ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
        hotel_ids = {i.hotel_id for i in items if i.item_type == "hotel"}
        assert len(hotel_ids) == 1
    finally:
        db.close()


def test_change_day_hotel_leaves_other_days_untouched(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
    finally:
        db.close()
    amritsar_id, ludhiana_id = punjab_trip["hotel_ids"]
    res = client.post(
        f"/api/trips/{punjab_trip['trip_id']}/change-day-accommodation",
        json={"day_number": 2, "accommodation_id": amritsar_id},
    )
    assert res.status_code == 200
    daily = {
        d["day_number"]: d["hotel"]["id"]
        for d in res.json()["daily_accommodations"]
    }
    assert daily[2] == amritsar_id
    assert daily[1] == amritsar_id  # night-1 base, untouched
    assert daily[3] == ludhiana_id  # untouched


def test_global_change_updates_every_stay(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
    finally:
        db.close()
    amritsar_id, _ = punjab_trip["hotel_ids"]
    res = client.post(
        f"/api/trips/{punjab_trip['trip_id']}/change-accommodation",
        json={"accommodation_id": amritsar_id},
    )
    assert res.status_code == 200
    daily = [d["hotel"]["id"] for d in res.json()["daily_accommodations"]]
    assert daily and all(hotel_id == amritsar_id for hotel_id in daily)
    assert res.json()["selected_accommodation"]["id"] == amritsar_id


def test_change_first_and_last_day(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
    finally:
        db.close()
    amritsar_id, ludhiana_id = punjab_trip["hotel_ids"]
    first = client.post(
        f"/api/trips/{punjab_trip['trip_id']}/change-day-accommodation",
        json={"day_number": 1, "accommodation_id": ludhiana_id},
    )
    assert first.status_code == 200
    daily = {
        d["day_number"]: d["hotel"]["id"]
        for d in first.json()["daily_accommodations"]
    }
    assert daily[1] == ludhiana_id and daily[2] == ludhiana_id and daily[3] == ludhiana_id
    last = client.post(
        f"/api/trips/{punjab_trip['trip_id']}/change-day-accommodation",
        json={"day_number": 3, "accommodation_id": amritsar_id},
    )
    assert last.status_code == 200
    daily = {
        d["day_number"]: d["hotel"]["id"]
        for d in last.json()["daily_accommodations"]
    }
    assert daily[3] == amritsar_id and daily[1] == ludhiana_id


def test_regenerate_preserves_confirmed_day(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
    finally:
        db.close()
    amritsar_id, ludhiana_id = punjab_trip["hotel_ids"]
    # Night 2 is Ludhiana by proximity; traveler confirms Amritsar instead.
    res = client.post(
        f"/api/trips/{punjab_trip['trip_id']}/change-day-accommodation",
        json={"day_number": 2, "accommodation_id": amritsar_id},
    )
    assert res.status_code == 200
    db = SessionLocal()
    try:
        optimized = ItineraryGenerator(db).optimize_for_trip(punjab_trip["trip_id"])
        hotel_days = {
            i.day_number: i.hotel_id
            for i in _hotel_items(db, punjab_trip["trip_id"])
            if i.day_number in (1, 2, 3)
        }
        assert hotel_days[2] == amritsar_id  # confirmed choice survives regen
        assert optimized, "regen must still produce items"
    finally:
        db.close()


def test_failed_change_keeps_itinerary_intact(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        before = ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
        before_ids = sorted(i.id for i in before)
    finally:
        db.close()
    res = client.post(
        f"/api/trips/{punjab_trip['trip_id']}/change-day-accommodation",
        json={"day_number": 2, "accommodation_id": "htl-nope-missing"},
    )
    assert res.status_code == 400
    db = SessionLocal()
    try:
        after_ids = sorted(
            i.id
            for i in db.query(ItineraryItem)
            .filter(ItineraryItem.trip_id == punjab_trip["trip_id"])
            .all()
        )
        assert after_ids == before_ids
    finally:
        db.close()


def test_hotel_option_and_daily_contract_include_coords(punjab_trip, monkeypatch):
    _patch_ranking(monkeypatch, punjab_trip)
    db = SessionLocal()
    try:
        ItineraryGenerator(db).generate_for_trip(punjab_trip["trip_id"])
    finally:
        db.close()
    res = client.get(f"/api/trips/{punjab_trip['trip_id']}")
    assert res.status_code == 200
    body = res.json()
    assert body["selected_accommodation"]["latitude"] is not None
    for entry in body["daily_accommodations"]:
        assert entry["hotel"]["latitude"] is not None
        assert entry["hotel"]["longitude"] is not None
    reasons = [
        (row.get("meta_data") or {}).get("ui", {}).get("hotel_assignment_reason")
        for row in body["itinerary"]
        if row["item_type"] == "hotel"
    ]
    assert reasons and all(reasons)
