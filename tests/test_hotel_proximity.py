"""Proximity scoring in hotel ranking (generic, coordinate-driven).

No destination names influence the implementation; fixtures use synthetic
points. All unit tests are database-free.
"""

from types import SimpleNamespace

import pytest

from backend.recommendation.engine import RecommendationEngine

ENGINE = RecommendationEngine.__new__(RecommendationEngine)


def _hotel(name, lat, lng, price=2000.0, rating=4.5):
    return SimpleNamespace(
        id=f"htl-{name}", name=name, category="mid-range",
        description=f"{name} stay", amenities=[], address=name,
        price_per_night=price, rating=rating, currency="INR",
        destination_id="dest-test", vendor_id=None,
        latitude=lat, longitude=lng, images=[], is_active=True,
        created_at=None, inventory_source="catalog",
        verification_status="catalog_verified", discovery_session_id=None,
    )


def _rank(hotels, activities=None):
    return ENGINE._rank_hotels(hotels, {}, activities)


def test_nearer_hotel_wins_ceteris_paribus():
    near = _hotel("near", 10.0, 20.0)
    far = _hotel("far", 12.0, 22.0)
    anchors = [(10.0, 20.0), (10.05, 20.05)]
    ranked = _rank([far, near], anchors)
    assert [h.id for h, _, _ in ranked] == ["htl-near", "htl-far"]
    scores = {h.id: s for h, s, _ in ranked}
    assert scores["htl-near"] > scores["htl-far"]
    proximity = {h.id: k for h, _, k in ranked}
    assert proximity["htl-near"] is not None and proximity["htl-far"] is not None
    assert proximity["htl-near"] < 20.0 < proximity["htl-far"]


def test_missing_hotel_coords_neutral_no_crash():
    known = _hotel("known", 10.0, 20.0)
    unknown = _hotel("unknown", None, None)
    ranked = _rank([known, unknown], [(10.0, 20.0)])
    by_id = {h.id: (s, k) for h, s, k in ranked}
    assert by_id["htl-unknown"][1] is None
    legacy_base = ENGINE._rank_hotels([unknown], {})[0][1]
    assert by_id["htl-unknown"][0] == pytest.approx(
        legacy_base + RecommendationEngine.PROXIMITY_NEUTRAL
    )


def test_no_activity_coords_matches_legacy():
    hotels = [_hotel("a", 10.0, 20.0), _hotel("b", 12.0, 22.0)]
    legacy = ENGINE._rank_hotels(hotels, {})
    assert legacy is not None
    assert all(len(entry) == 3 for entry in legacy)
    for (_, legacy_score, legacy_km), (_, new_score, new_km) in zip(
        legacy, _rank(hotels, [])
    ):
        assert new_score == legacy_score
        assert new_km is None
    assert _rank(hotels, None) == _rank(hotels, [])


def test_near_reasonable_beats_far_top_rated():
    far = _hotel("far", 15.0, 25.0, price=2000.0, rating=5.0)
    near = _hotel("near", 10.0, 20.0, price=2000.0, rating=4.0)
    ranked = _rank([far, near], [(10.0, 20.0)])
    assert ranked[0][0].id == "htl-near"


def test_invalid_coordinates_never_crash():
    hotels = [
        _hotel("bad1", "north", "east"),
        _hotel("bad2", 999.0, 20.0),
        _hotel("bad3", 10.0, None),
        _hotel("ok", 10.0, 20.0),
    ]
    ranked = _rank(hotels, [(10.0, 20.0), ("x", "y"), (95.0, 0.0)])
    assert {h.id for h, _, _ in ranked} == {
        "htl-bad1", "htl-bad2", "htl-bad3", "htl-ok"}
    assert ranked[0][0].id == "htl-ok"


def test_hotel_response_carries_proximity():
    hotel = _hotel("h", 10.0, 20.0)
    full = ENGINE._hotel_response(hotel, 80.0, 12.5)
    assert full["proximity_km"] == 12.5
    assert full["match_score"] == 80.0
    assert full["id"] == "htl-h"
    legacy = ENGINE._hotel_response(hotel, 80.0)
    assert legacy["proximity_km"] is None


# ---------------------------------------------------------------------------
# End-to-end: proximity ranking flows into generated stays (dev PostgreSQL)
# ---------------------------------------------------------------------------

def test_generator_prefers_near_reasonable_hotel_over_far_top_rated(monkeypatch):
    """Far 5-star vs near 4-star at equal price: generation must base the
    trip on the nearby stay. Uses the REAL recommendation engine (not a
    stub) so ranking influence is proven, with isolated fixture rows."""
    import uuid

    from backend.database.connection import SessionLocal
    from backend.itinerary.generator import ItineraryGenerator
    from backend.models.models import (
        Activity, Destination, Hotel, ItineraryItem, TransportOption, Trip,
    )

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest = Destination(
            id=f"dst-prox-{tag}", name=f"Proxville {tag}",
            slug=f"proxville-{tag}", country="India",
            state_region=f"Proxville {tag}", description="Proximity test.",
            latitude=10.0, longitude=20.0, inventory_source="catalog",
            verification_status="catalog_verified",
        )
        db.add(dest)
        db.flush()
        far = Hotel(
            id=f"htl-prox-far-{tag}", destination_id=dest.id,
            name=f"Far Grand {tag}", category="mid-range",
            price_per_night=2000.0, currency="INR", rating=5.0,
            address="Far Town", latitude=15.0, longitude=25.0,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        near = Hotel(
            id=f"htl-prox-near-{tag}", destination_id=dest.id,
            name=f"Near Stay {tag}", category="mid-range",
            price_per_night=2000.0, currency="INR", rating=4.0,
            address="Near Town", latitude=10.0, longitude=20.0,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        db.add_all([far, near])
        for idx in range(3):
            db.add(Activity(
                id=f"act-prox-{tag}-{idx}", destination_id=dest.id,
                title=f"Prox Spot {tag}-{idx}", category="culture",
                duration_hours=2.0, price_per_person=0.0, currency="INR",
                rating=4.5, meeting_point="Near Town",
                latitude=10.0 + idx * 0.01, longitude=20.0,
                inventory_source="catalog", verification_status="catalog_verified",
                is_active=True,
            ))
        db.add(TransportOption(
            id=f"trn-prox-{tag}", destination_id=dest.id, type="private_cab",
            name=f"Prox Cab {tag}", route_from="Origin", route_to=dest.name,
            duration_hours=2.0, price=500.0, currency="INR", capacity=4,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        ))
        db.commit()
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Prox {tag}", duration_days=3, total_budget=100000.0,
            currency="INR", traveler_count=2, pace="balanced",
        )
        db.add(trip)
        db.commit()
        items = ItineraryGenerator(db).generate_for_trip(trip.id)
        stays = sorted(
            [i for i in items if i.item_type == "hotel"],
            key=lambda i: i.day_number,
        )
        assert stays, "expected per-night stay items"
        assert all(s.hotel_id == near.id for s in stays)
        reasons = [
            (s.meta_data or {}).get("ui", {}).get("hotel_assignment_reason")
            for s in stays
        ]
        assert all(reasons)
    finally:
        if trip is not None:
            db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip.id).delete()
            db.query(Trip).filter(Trip.id == trip.id).delete()
            db.commit()
        if dest is not None:
            db.query(Activity).filter(Activity.destination_id == dest.id).delete()
            db.query(Hotel).filter(Hotel.destination_id == dest.id).delete()
            db.query(TransportOption).filter(TransportOption.destination_id == dest.id).delete()
            db.query(Destination).filter(Destination.id == dest.id).delete()
            db.commit()
        db.close()
