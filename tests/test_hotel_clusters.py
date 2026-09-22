"""Geographic (cluster-based) hotel assignment tests.

Hotels follow activity geography, not the calendar: one travel region
reuses one verified stay; a genuinely new region triggers a fresh
proximity evaluation; thin inventory retains honestly. All fixtures use
synthetic coordinates only.
"""

import uuid
from types import SimpleNamespace

import pytest

from backend.itinerary.hotel_assignment import (
    DayAnchor,
    cluster_anchor_points,
    haversine_km,
    plan_overnight_stays,
)

# Synthetic geometry: north cluster (two stops ~70 km apart), south
# cluster (~400 km from the north one).
NORTH_A = (15.33, 76.46)
NORTH_B = (15.91, 75.68)
SOUTH_A = (12.30, 76.65)
SOUTH_B = (12.42, 75.73)


def _hotel(name, lat, lng, price=2000.0):
    return SimpleNamespace(
        id=f"stub-{name}", name=name, latitude=lat, longitude=lng,
        price_per_night=price, description=f"{name} description",
        address=name,
    )


def _anchor(lat, lng, label="Area"):
    return DayAnchor(latitude=lat, longitude=lng, label=label)


def test_cluster_linking_groups_nearby_stops():
    regions = cluster_anchor_points([
        ("vijayapura", (16.82, 75.71)),
        ("hampi", NORTH_A),
        ("badami", NORTH_B),
        ("mysuru", SOUTH_A),
        ("madikeri", SOUTH_B),
        ("jog", (14.22, 74.81)),
    ])
    north = {regions["vijayapura"], regions["hampi"], regions["badami"]}
    south = {regions["mysuru"], regions["madikeri"]}
    assert len(north) == 1, "north stops chain into one region"
    assert len(south) == 1, "south stops chain into one region"
    assert north != south, "north and south stay separate regions"
    assert regions["jog"] not in north | south, "outlier stays separate"


def test_cluster_linking_deterministic_and_safe():
    pts = [("a", NORTH_A), ("b", NORTH_B), ("c", SOUTH_A)]
    assert cluster_anchor_points(pts) == cluster_anchor_points(pts)
    # Invalid points are skipped, never invented.
    assert cluster_anchor_points(
        [("a", NORTH_A), ("bad", (999.0, 0.0)), ("none", (None, None))]) == {"a": 0}


def test_1_same_cluster_reuses_hotel_without_switching():
    """Test 1 — same cluster: one hotel reused, no churning."""
    north_hotel = _hotel("North", *NORTH_A)
    south_hotel = _hotel("South", *SOUTH_A)
    stays, _spent = plan_overnight_stays(
        nights=3,
        anchors={1: _anchor(*NORTH_A, "Day 2"),
                 2: _anchor(*NORTH_B, "Day 3"),
                 3: _anchor(*NORTH_A, "Day 4")},
        hotels=[north_hotel, south_hotel],
        night_regions={1: 0, 2: 0, 3: 0},
    )
    assert [s.hotel.id for s in stays] == ["stub-North"] * 3
    assert all(s.retained or s.night == 1 for s in stays)


def test_2_distant_cluster_with_verified_hotel_switches():
    """Test 2 — new distant cluster + verified hotel: switch, shorter trip."""
    north_hotel = _hotel("North", *NORTH_A)
    south_hotel = _hotel("South", *SOUTH_A)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: _anchor(*NORTH_A, "Day 2"),
                 2: _anchor(*SOUTH_A, "Day 3")},
        hotels=[north_hotel, south_hotel],
        night_regions={1: 0, 2: 1},
    )
    assert stays[0].hotel.id == "stub-North"
    assert stays[1].hotel.id == "stub-South"
    assert not stays[1].retained
    # distance_km records the previous stay's burden (the reason to move);
    # the newly selected stay itself must sit near the new cluster.
    assert stays[1].distance_km is not None and stays[1].distance_km > 300.0
    new_dist = haversine_km(
        south_hotel.latitude, south_hotel.longitude, *SOUTH_A)
    assert new_dist is not None and new_dist < 40.0


def test_3_distant_cluster_without_verified_hotel_retained_honestly():
    """Test 3 — new distant cluster, thin inventory: retain + warn, no fake."""
    only = _hotel("Only", *NORTH_A)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: _anchor(*NORTH_A, "Day 2"),
                 2: _anchor(*SOUTH_A, "Day 3")},
        hotels=[only],
        night_regions={1: 0, 2: 1},
    )
    assert [s.hotel.id for s in stays] == ["stub-Only"] * 2
    assert stays[1].retained
    assert "limited" in stays[1].reason
    assert "longer travel" in stays[1].reason
    assert stays[1].distance_km is not None and stays[1].distance_km > 300.0


def test_4_hotel_change_does_not_break_budget():
    """Test 4 — budget authority: spend stays within pot, verified prices."""
    north_hotel = _hotel("North", *NORTH_A, price=2000.0)
    south_hotel = _hotel("South", *SOUTH_A, price=2500.0)
    stays, spent = plan_overnight_stays(
        nights=2,
        anchors={1: _anchor(*NORTH_A, "Day 2"),
                 2: _anchor(*SOUTH_A, "Day 3")},
        hotels=[north_hotel, south_hotel],
        night_regions={1: 0, 2: 1},
        total_pot=4500.0,
    )
    assert spent == pytest.approx(4500.0)
    assert all(s.hotel is not None for s in stays)
    # A pot too small for any stay yields explicit None hotels, never debt.
    poor, poor_spent = plan_overnight_stays(
        nights=1, anchors={1: _anchor(*NORTH_A, "Day 2")},
        hotels=[north_hotel], total_pot=100.0,
    )
    assert poor[0].hotel is None and poor_spent == pytest.approx(0.0)


def test_5_arrival_day_respects_arrival_location():
    """Test 5 — opening stay near arrival, never an impossible transfer."""
    arrival_hotel = _hotel("Arrival", *SOUTH_A)
    cluster_hotel = _hotel("Cluster", *NORTH_A)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: _anchor(*NORTH_A, "Day 2"),
                 2: _anchor(*NORTH_A, "Day 3")},
        hotels=[cluster_hotel, arrival_hotel],
        night_regions={1: 0, 2: 0},
        arrival_point=SOUTH_A,
    )
    first_dist = haversine_km(
        SOUTH_A[0], SOUTH_A[1],
        stays[0].hotel.latitude, stays[0].hotel.longitude)
    assert first_dist is not None and first_dist <= 60.0, (
        "night 1 must be reachable from the arrival area")
    assert stays[0].hotel.id == "stub-Arrival"
    # Later nights still follow the activity cluster.
    assert stays[1].hotel.id == "stub-Cluster"


def test_6_departure_day_reuses_previous_night():
    """Test 6 — final night: previous stay kept, no new hotel invented."""
    only = _hotel("Only", *NORTH_A)
    stays, _spent = plan_overnight_stays(
        nights=3,
        anchors={1: _anchor(*NORTH_A, "Day 2"),
                 2: _anchor(*NORTH_A, "Day 3"),
                 3: _anchor(*NORTH_A, "Day 4")},
        hotels=[only],
        night_regions={1: 0, 2: 0, 3: 0},
    )
    assert stays[2].hotel.id == stays[1].hotel.id == "stub-Only"
    assert len({s.hotel.id for s in stays}) == 1


def _stub_trip_rows(db, tag):
    from backend.models.models import (
        Activity, Destination, Hotel, TransportOption,
    )
    dest = Destination(
        id=f"dst-clu-{tag}", name=f"Clusterville {tag}",
        slug=f"clusterville-{tag}", country="India",
        state_region=f"Clusterville {tag}",
        description="Cluster hotel regression destination.",
        latitude=NORTH_A[0], longitude=NORTH_A[1],
        inventory_source="catalog", verification_status="catalog_verified",
    )
    db.add(dest)
    db.flush()
    north = Hotel(
        id=f"htl-clu-n-{tag}", destination_id=dest.id,
        name=f"North Stay {tag}", price_per_night=2000.0, currency="INR",
        rating=4.5, latitude=NORTH_A[0], longitude=NORTH_A[1],
        inventory_source="catalog", verification_status="catalog_verified",
        is_active=True,
    )
    south = Hotel(
        id=f"htl-clu-s-{tag}", destination_id=dest.id,
        name=f"South Stay {tag}", price_per_night=2000.0, currency="INR",
        rating=4.5, latitude=SOUTH_A[0], longitude=SOUTH_A[1],
        inventory_source="catalog", verification_status="catalog_verified",
        is_active=True,
    )
    transport = TransportOption(
        id=f"trn-clu-{tag}", destination_id=dest.id, type="private_cab",
        name=f"Cluster Cab {tag}", route_from="Origin", route_to=dest.name,
        price=1000.0, currency="INR", capacity=6,
        inventory_source="catalog", verification_status="catalog_verified",
        is_active=True,
    )
    db.add_all([north, south, transport])
    acts = []
    for idx, (lat, lng) in enumerate(
            [NORTH_A, NORTH_B, SOUTH_A, SOUTH_B]):
        activity = Activity(
            id=f"act-clu-{tag}-{idx}", destination_id=dest.id,
            title=f"Cluster Spot {tag}-{idx}", category="culture",
            duration_hours=2.0, price_per_person=100.0, currency="INR",
            rating=4.5, meeting_point="Clusterville",
            latitude=lat, longitude=lng, inventory_source="catalog",
            verification_status="catalog_verified", is_active=True,
        )
        db.add(activity)
        acts.append(activity.id)
    db.commit()
    return dest, north, south, transport, acts


def _generate_cluster_trip(monkeypatch, duration_days=4):
    import backend.itinerary.generator as itinerary_generator
    from backend.database.connection import SessionLocal
    from backend.itinerary.generator import ItineraryGenerator
    from backend.models.models import ItineraryItem, Trip

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest, north, south, transport, acts = _stub_trip_rows(db, tag)

        class StubEngine:
            def __init__(self, db):
                pass

            def get_recommendations(self, destination_id, preferences,
                                    discovery_session_id=None):
                return {
                    "ai_insights": None,
                    "recommended_hotels": [{"id": north.id},
                                           {"id": south.id}],
                    "recommended_activities": [{"id": aid} for aid in acts],
                    "recommended_transport": [{"id": transport.id}],
                }

        monkeypatch.setattr(
            itinerary_generator, "RecommendationEngine", StubEngine)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Cluster {tag}", duration_days=duration_days,
            total_budget=500000.0, currency="INR", traveler_count=2,
            pace="balanced",
        )
        db.add(trip)
        db.commit()
        return list(ItineraryGenerator(db).generate_for_trip(trip.id))
    finally:
        if trip is not None:
            from backend.models.models import Trip as TripModel
            db.query(ItineraryItem).filter(
                ItineraryItem.trip_id == trip.id).delete()
            db.query(TripModel).filter(TripModel.id == trip.id).delete()
            db.commit()
        if dest is not None:
            from backend.models.models import (
                Activity, Destination, Hotel, TransportOption,
            )
            db.query(Activity).filter(
                Activity.destination_id == dest.id).delete()
            db.query(Hotel).filter(
                Hotel.destination_id == dest.id).delete()
            db.query(TransportOption).filter(
                TransportOption.destination_id == dest.id).delete()
            db.query(Destination).filter(Destination.id == dest.id).delete()
            db.commit()
        db.close()


def test_7_no_activity_overlap_after_hotel_assignment(monkeypatch):
    """Test 7 — hotel assignment never disturbs activity timing."""
    from backend.itinerary.route_plan import detect_overlaps, parse_clock_time

    items = _generate_cluster_trip(monkeypatch)
    acts = [i for i in items if i.item_type == "activity"]
    assert len(acts) == 4, "no selected activity lost"
    by_day = {}
    for item in acts:
        by_day.setdefault(item.day_number, []).append(item)
    for day, stops in by_day.items():
        ordered = sorted(stops, key=lambda i: parse_clock_time(i.start_time))
        for first, second in zip(ordered, ordered[1:]):
            assert parse_clock_time(second.start_time) >= parse_clock_time(
                first.end_time), f"overlap on day {day}"
        assert detect_overlaps(
            [(i.id, i.start_time, i.end_time) for i in stops]) == {}


def test_8_hotel_is_not_sightseeing(monkeypatch):
    """Test 8 — accommodation flagged, sorted last, outside activity slots."""
    from backend.api.routes import _route_day_summaries

    items = _generate_cluster_trip(monkeypatch)
    hotels = [i for i in items if i.item_type == "hotel"]
    assert hotels, "expected overnight stays"
    for hotel in hotels:
        ui = (hotel.meta_data or {}).get("ui", {})
        assert ui.get("is_accommodation") is True
        assert ui.get("hotel_assignment_reason")
    by_day = {}
    for item in items:
        by_day.setdefault(item.day_number, []).append(item)
    for day_items in by_day.values():
        ordered = sorted(day_items, key=lambda i: i.order_index)
        hotel_pos = [n for n, i in enumerate(ordered)
                     if i.item_type == "hotel"]
        act_pos = [n for n, i in enumerate(ordered)
                   if i.item_type == "activity"]
        if hotel_pos and act_pos:
            assert min(hotel_pos) > min(act_pos)
    trip = SimpleNamespace(start_date=None, duration_days=4)
    rows = [{"id": i.id, "day_number": i.day_number,
             "item_type": i.item_type, "title": i.title,
             "meta_data": i.meta_data or {}}
            for i in items]
    for row in rows:
        row.update((row.pop("meta_data") or {}).get("ui", {}))
    summaries = _route_day_summaries(trip, rows)
    counted = sum(s["activity_count"] for s in summaries)
    assert counted == len([i for i in items if i.item_type == "activity"])
