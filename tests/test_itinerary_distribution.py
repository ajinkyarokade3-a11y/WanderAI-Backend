"""Daily allocation: verified activities must spread across sightseeing days.

Regression tests for empty later days on multi-day trips. Previously the
generator packed activities two-per-day starting at day 1, so a trip with the
minimum required activities (e.g. 3 for 4 days, 5 for 6 days) left days 3+ with
zero activities. Allocation is now round-robin across the trip duration.
"""
import uuid

import pytest
from alembic import command
from alembic.config import Config

from backend.database.connection import SessionLocal
from backend.itinerary.generator import ItineraryGenerator
from backend.models.models import (
    Activity,
    Destination,
    Hotel,
    ItineraryItem,
    TransportOption,
    Trip,
)


@pytest.fixture(scope="session", autouse=True)
def _migrated():
    command.upgrade(Config("alembic.ini"), "head")


def _build_inventory(db, tag, activity_count):
    dest = Destination(
        id=f"dst-dist-{tag}",
        name=f"Distville {tag}",
        slug=f"distville-{tag}",
        country="India",
        state_region=f"Distville {tag}",
        description="Distribution test destination.",
        inventory_source="catalog",
        verification_status="catalog_verified",
    )
    db.add(dest)
    db.flush()
    hotel = Hotel(
        id=f"htl-dist-{tag}",
        destination_id=dest.id,
        name=f"Distville Stay {tag}",
        price_per_night=1000.0,
        currency="INR",
        rating=4.5,
        inventory_source="catalog",
        verification_status="catalog_verified",
        is_active=True,
    )
    transport = TransportOption(
        id=f"trn-dist-{tag}",
        destination_id=dest.id,
        type="private_cab",
        name=f"Distville Cab {tag}",
        route_from="Origin",
        route_to=f"Distville {tag}",
        price=1000.0,
        currency="INR",
        capacity=6,
        inventory_source="catalog",
        verification_status="catalog_verified",
        is_active=True,
    )
    db.add_all([hotel, transport])
    activity_ids = []
    for idx in range(activity_count):
        activity = Activity(
            id=f"act-dist-{tag}-{idx}",
            destination_id=dest.id,
            title=f"Distville Spot {tag}-{idx}",
            category="culture",
            duration_hours=2.0,
            price_per_person=100.0,
            currency="INR",
            rating=4.5,
            meeting_point=f"Distville {tag}",
            inventory_source="catalog",
            verification_status="catalog_verified",
            is_active=True,
        )
        db.add(activity)
        activity_ids.append(activity.id)
    db.commit()
    return dest, hotel, transport, activity_ids


def _stub_engine(monkeypatch, hotel_id, activity_ids, transport_id):
    import backend.itinerary.generator as itinerary_generator

    class StubEngine:
        def __init__(self, db):
            pass

        def get_recommendations(self, destination_id, preferences, discovery_session_id=None):
            return {
                "ai_insights": None,
                "recommended_hotels": [{"id": hotel_id}],
                "recommended_activities": [{"id": aid} for aid in activity_ids],
                "recommended_transport": [{"id": transport_id}],
            }

    monkeypatch.setattr(itinerary_generator, "RecommendationEngine", StubEngine)


def _generate(monkeypatch, tag, duration_days, activity_count):
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest, hotel, transport, activity_ids = _build_inventory(db, tag, activity_count)
        _stub_engine(monkeypatch, hotel.id, activity_ids, transport.id)
        trip = Trip(
            user_id="usr-alex-morgan-001",
            destination_id=dest.id,
            title=f"Distribution {tag}",
            duration_days=duration_days,
            total_budget=500000.0,
            currency="INR",
            traveler_count=2,
            pace="balanced",
        )
        db.add(trip)
        db.commit()
        items = ItineraryGenerator(db).generate_for_trip(trip.id)
        return [
            {"day_number": i.day_number, "order_index": i.order_index,
             "item_type": i.item_type, "activity_id": i.activity_id}
            for i in items
        ], activity_ids
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


def _activity_days(items):
    days = {}
    for item in items:
        if item["item_type"] == "activity":
            days.setdefault(item["day_number"], []).append(item["activity_id"])
    return days


def test_six_day_trip_covers_days_1_to_6(monkeypatch):
    items, activity_ids = _generate(monkeypatch, uuid.uuid4().hex[:8], 6, 6)
    days = _activity_days(items)
    assert sorted(days) == [1, 2, 3, 4, 5, 6]
    placed = [aid for aids in days.values() for aid in aids]
    assert sorted(placed) == sorted(activity_ids)
    assert len(set(placed)) == len(placed)


def test_six_day_trip_with_minimum_five_covers_days_1_to_5(monkeypatch):
    items, activity_ids = _generate(monkeypatch, uuid.uuid4().hex[:8], 6, 5)
    days = _activity_days(items)
    assert sorted(days) == [1, 2, 3, 4, 5]
    placed = [aid for aids in days.values() for aid in aids]
    assert sorted(placed) == sorted(activity_ids)
    assert len(set(placed)) == len(placed)


def test_four_day_trip_places_one_activity_on_days_1_2_3(monkeypatch):
    items, activity_ids = _generate(monkeypatch, uuid.uuid4().hex[:8], 4, 3)
    days = _activity_days(items)
    assert sorted(days) == [1, 2, 3]
    assert all(len(aids) == 1 for aids in days.values())
    placed = [aid for aids in days.values() for aid in aids]
    assert sorted(placed) == sorted(activity_ids)


def test_empty_last_day_gets_derived_departure_note(monkeypatch):
    items, _ = _generate(monkeypatch, uuid.uuid4().hex[:8], 6, 5)
    notes = [i for i in items if i["item_type"] == "note" and i["day_number"] == 6]
    assert len(notes) == 1
    # Note links no inventory and day 6 still has no activity: derived content.
    assert notes[0]["activity_id"] is None
    assert all(not (i["item_type"] == "activity" and i["day_number"] == 6) for i in items)


def test_second_round_fills_mornings_without_duplicates(monkeypatch):
    items, activity_ids = _generate(monkeypatch, uuid.uuid4().hex[:8], 3, 5)
    days = _activity_days(items)
    assert sorted(days) == [1, 2, 3]
    counts = sorted(len(aids) for aids in days.values())
    assert counts == [1, 2, 2]
    placed = [aid for aids in days.values() for aid in aids]
    assert sorted(placed) == sorted(activity_ids)
    assert len(set(placed)) == len(placed)
    for day in days:
        orders = sorted(
            item["order_index"] for item in items
            if item["day_number"] == day
        )
        assert len(set(orders)) == len(orders)
