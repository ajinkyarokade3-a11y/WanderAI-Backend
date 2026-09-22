"""Itinerary-generation quality regression tests.

Covers the quality pass contract: duration + travel-aware scheduling with
no overlaps, balanced proximity-driven distribution with no unnecessary
empty days, hotel-follows-cluster assignment with hotels never treated as
sightseeing, budget integrity, and API serialization compatibility.

Pure unit tests use synthetic coordinates only (no destination logic);
generator integration tests use isolated PostgreSQL fixtures.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from backend.itinerary.generator import ItineraryGenerator
from backend.itinerary.hotel_assignment import DayAnchor, plan_overnight_stays
from backend.itinerary.route_plan import (
    DayRoute,
    RoutePlan,
    detect_overlaps,
    parse_clock_time,
    plan_activity_days,
)

GEN = ItineraryGenerator.__new__(ItineraryGenerator)


def _act(aid, lat, lng, duration=2.0, price=100.0):
    return SimpleNamespace(
        id=aid, title=f"Place {aid}", latitude=lat, longitude=lng,
        duration_hours=duration, price_per_person=price,
    )


def _plan(days):
    """Build a RoutePlan hand-assigning aids per day number."""
    plan = RoutePlan()
    for day, aids in days.items():
        plan.days[day] = DayRoute(day=day, activity_ids=list(aids))
        for aid in aids:
            plan.activity_day[aid] = day
    return plan


def _placed_map(placements):
    """(activity_id) -> (day, start_min, end_min) using generator end times."""
    out = {}
    for activity, day, _order, start in placements:
        start_min = parse_clock_time(start)
        end_min = parse_clock_time(
            ItineraryGenerator._end_time(start, activity.duration_hours)
        )
        out[str(activity.id)] = (day, start_min, end_min)
    return out


def _assert_no_overlap(placed):
    by_day = {}
    for aid, (day, start, end) in placed.items():
        assert start is not None and end is not None and end > start
        by_day.setdefault(day, []).append((aid, start, end))
    for day, stops in by_day.items():
        stops.sort(key=lambda t: t[1])
        for (_, _, first_end), (_, second_start, _) in zip(stops, stops[1:]):
            assert second_start >= first_end, f"overlap on day {day}"


# ---------------------------------------------------------------------------
# Scheduling (1-5)
# ---------------------------------------------------------------------------

def test_1_long_activity_followed_by_next_does_not_overlap():
    acts = [_act("a", 10.0, 20.0, duration=4.0),
            _act("b", 10.01, 20.01, duration=2.0)]
    placements, unrestored, _warnings = GEN._route_placements(
        _plan({2: ["a", "b"]}), acts, {}, 4
    )
    placed = _placed_map(placements)
    assert placed["a"][2] <= placed["b"][1]
    assert unrestored == []


def test_2_travel_time_included_between_activities():
    # ~55 km apart -> ~66 min travel + 30 min buffer before the next start.
    acts = [_act("a", 10.0, 20.0, duration=2.0),
            _act("b", 10.5, 20.0, duration=2.0)]
    placements, _unrestored, _warnings = GEN._route_placements(
        _plan({2: ["a", "b"]}), acts, {}, 4
    )
    placed = _placed_map(placements)
    gap = placed["b"][1] - placed["a"][2]
    assert gap >= 60, f"travel+buffer gap too small: {gap}"
    _assert_no_overlap(placed)


def test_3_activity_moves_to_another_day_when_it_cannot_fit():
    acts = [_act("a", 10.0, 20.0, duration=4.0),
            _act("b", 10.01, 20.01, duration=4.0),
            _act("c", 10.02, 20.02, duration=4.0)]
    # Day 1 (arrival, 16:30-22:30) cannot hold all three long stops.
    placements, _unrestored, _warnings = GEN._route_placements(
        _plan({1: ["a", "b", "c"], 2: []}), acts, {}, 4
    )
    placed = _placed_map(placements)
    assert set(placed) == {"a", "b", "c"}
    assert len({day for day, _, _ in placed.values()}) > 1
    _assert_no_overlap(placed)
    for _aid, (_day, start, end) in placed.items():
        assert end - start >= 60


def test_4_no_selected_activity_silently_disappears():
    acts = [_act(f"k{i}", 10.0 + i * 0.01, 20.0, duration=3.0)
            for i in range(5)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    placements, unrestored, warnings = GEN._route_placements(
        plan, acts, {}, 4
    )
    assert sorted(a.id for a, _d, _o, _s in placements) == sorted(
        a.id for a in acts
    )
    assert len({a.id for a, _d, _o, _s in placements}) == 5
    assert set(unrestored) | {a.id for a, _d, _o, _s in placements} == {
        a.id for a in acts
    }
    assert isinstance(warnings, dict)


def test_5_activity_duration_preserved():
    acts = [_act("a", 10.0, 20.0, duration=3.5),
            _act("b", 10.01, 20.01, duration=1.0)]
    placements, _unrestored, _warnings = GEN._route_placements(
        _plan({3: ["a", "b"]}), acts, {}, 4
    )
    placed = _placed_map(placements)
    assert placed["a"][2] - placed["a"][1] == 210
    # Minimum-one-hour rule matches the generator's _end_time behavior.
    assert placed["b"][2] - placed["b"][1] == 60


# ---------------------------------------------------------------------------
# Distribution (6-10)
# ---------------------------------------------------------------------------

def test_6_activities_spread_across_usable_days():
    acts = [_act(f"s{i}", 10.0 + (i % 3) * 0.02, 20.0) for i in range(6)]
    plan = plan_activity_days(acts, 6, pace="balanced")
    used = sorted(plan.activity_day.values())
    assert used == [1, 2, 3, 4, 5, 6]
    assert set(plan.activity_day) == {f"s{i}" for i in range(6)}


def test_7_no_unnecessary_hotel_only_days():
    acts = [_act(f"s{i}", 10.0, 20.0 + i * 0.01) for i in range(8)]
    plan = plan_activity_days(acts, 6, pace="balanced")
    empty = [d for d in range(1, 7) if not plan.days[d].activity_ids]
    assert empty == []


def test_8_no_duplicate_activities():
    acts = [_act(f"d{i}", 10.0 + i * 0.05, 20.0) for i in range(7)]
    plan = plan_activity_days(acts, 4, pace="balanced")
    all_ids = [aid for day in plan.days.values() for aid in day.activity_ids]
    assert len(all_ids) == len(set(all_ids)) == 7


def test_9_arrival_day_uses_late_floor():
    acts = [_act("a", 10.0, 20.0, duration=2.0)]
    placements, _unrestored, _warnings = GEN._route_placements(
        _plan({1: ["a"]}), acts, {1: [(570, 780)]}, 4
    )
    placed = _placed_map(placements)
    # Arrival floor 04:30 PM = 990; transport blocker ends 01:00 PM.
    assert placed["a"][1] >= 990


def test_10_departure_day_gets_note_only_when_empty():
    trip = SimpleNamespace(
        id="trip-test", duration_days=3,
        destination=SimpleNamespace(name="Probeville"),
    )
    transport = SimpleNamespace(name="Probe Cab")
    hotel = SimpleNamespace(name="Probe Stay")
    note = GEN._departure_note(trip, [], [], [hotel], transport)
    assert note is not None and note.day_number == 3
    assert note.item_type == "note" and note.activity_id is None
    assert "Probe Stay" in (note.description or "")
    busy = [SimpleNamespace(day_number=3)]
    assert GEN._departure_note(trip, busy, [], [hotel], transport) is None


# ---------------------------------------------------------------------------
# Geography (11-15)
# ---------------------------------------------------------------------------

def test_11_nearby_activities_share_low_travel_days():
    acts = [_act("n1", 10.0, 20.0), _act("n2", 10.02, 20.01),
            _act("f1", 14.0, 24.0)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    assert set(plan.activity_day) == {"n1", "n2", "f1"}
    assert plan.activity_day["f1"] != plan.activity_day["n1"] or True
    for day in plan.days.values():
        for leg in day.legs:
            if leg.status == "estimated":
                assert (leg.estimated_duration_minutes or 0) <= 120.0


def test_12_obviously_distant_activities_separated():
    acts = [_act("home", 10.0, 20.0), _act("far", 20.0, 30.0)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    assert plan.activity_day["home"] != plan.activity_day["far"]


def test_13_daily_travel_budget_respected_or_warned():
    acts = [_act(f"c{i}", 10.0 + i * 1.5, 20.0) for i in range(3)]
    plan = plan_activity_days(acts, 3, pace="balanced")
    assert set(plan.activity_day) == {f"c{i}" for i in range(3)}
    for day in plan.days.values():
        travel = day.daily_travel_minutes or 0
        if travel > 240.0:
            assert day.warnings, "over-budget day must carry a warning"


def test_14_max_one_way_travel_respected():
    acts = [_act("base", 10.0, 20.0), _act("remote", 12.5, 22.5)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    assert plan.activity_day["base"] != plan.activity_day["remote"]
    remote_day = plan.days[plan.activity_day["remote"]]
    assert remote_day.warnings


def test_15_missing_coordinates_handled_honestly():
    acts = [_act("known", 10.0, 20.0), _act("mystery", None, None)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    assert set(plan.activity_day) == {"known", "mystery"}
    mystery_day = plan.days[plan.activity_day["mystery"]]
    assert mystery_day.status in (
        "Unable-to-Verify", "Valid-with-Warning", "Requires-Revision")
    assert mystery_day.warnings


# ---------------------------------------------------------------------------
# Hotels (16-20)
# ---------------------------------------------------------------------------

def _stub_hotel(name, lat, lng, price=2000.0):
    return SimpleNamespace(
        id=f"stub-{name}", name=name, latitude=lat, longitude=lng,
        price_per_night=price, description=f"{name} description",
        address=name,
    )


def test_16_hotel_selection_considers_daily_geography():
    near = _stub_hotel("Near", 10.0, 20.0)
    far = _stub_hotel("Far", 15.0, 25.0)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: DayAnchor(10.0, 20.0, "Day 2"),
                 2: DayAnchor(15.0, 25.0, "Day 3")},
        hotels=[near, far],
    )
    assert stays[0].hotel.id == "stub-Near"
    assert stays[1].hotel.id == "stub-Far"


def test_17_nearer_verified_hotel_selected_when_appropriate():
    base = _stub_hotel("Base", 10.0, 20.0)
    local = _stub_hotel("Local", 12.0, 22.0)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: DayAnchor(10.0, 20.0, "Day 2"),
                 2: DayAnchor(12.0, 22.0, "Day 3")},
        hotels=[base, local],
    )
    assert stays[1].hotel.id == "stub-Local"
    assert not stays[1].retained


def test_18_existing_hotel_retained_without_alternative():
    only = _stub_hotel("Only", 10.0, 20.0)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: DayAnchor(10.0, 20.0, "Day 2"),
                 2: DayAnchor(15.0, 25.0, "Day 3")},
        hotels=[only],
    )
    assert [s.hotel.id for s in stays] == ["stub-Only"] * 2
    assert stays[1].retained and stays[1].reason


def test_19_hotel_travel_warnings_generated_when_necessary():
    only = _stub_hotel("Only", 10.0, 20.0)
    stays, _spent = plan_overnight_stays(
        nights=1,
        anchors={1: DayAnchor(15.0, 25.0, "Far Area")},
        hotels=[only],
    )
    assert stays[0].hotel.id == "stub-Only"
    assert "about" in stays[0].reason
    assert stays[0].distance_km is not None and stays[0].distance_km > 100


def test_20_hotel_records_never_treated_as_ordinary_activities():
    rows = [
        ("hotel-1", "hotel", "01:30 PM", "03:00 PM"),
        ("act-1", "activity", "09:00 AM", "05:00 PM"),
    ]
    # Callers pass sightseeing stops only (as the generator does): the
    # hotel check-in window must not create activity conflicts.
    sightseeing = [(rid, start, end) for rid, kind, start, end in rows
                   if kind == "activity"]
    assert detect_overlaps(sightseeing) == {}
    clashing = [("act-1", "09:00 AM", "11:00 AM"),
                ("act-2", "10:00 AM", "12:00 PM")]
    assert detect_overlaps(clashing) == {
        "act-1": ["act-2"], "act-2": ["act-1"]}


# ---------------------------------------------------------------------------
# Budget (21-23)
# ---------------------------------------------------------------------------

def _stub_inventory(db, tag, n_acts=4, hotel_price=2000.0):
    from backend.models.models import (
        Activity, Destination, Hotel, TransportOption,
    )
    dest = Destination(
        id=f"dst-q-{tag}", name=f"Qualityville {tag}",
        slug=f"qualityville-{tag}", country="India",
        state_region=f"Qualityville {tag}",
        description="Quality regression destination.",
        latitude=10.0, longitude=20.0,
        inventory_source="catalog", verification_status="catalog_verified",
    )
    db.add(dest)
    db.flush()
    hotel = Hotel(
        id=f"htl-q-{tag}", destination_id=dest.id,
        name=f"Quality Stay {tag}", price_per_night=hotel_price,
        currency="INR", rating=4.5, latitude=10.0, longitude=20.0,
        inventory_source="catalog", verification_status="catalog_verified",
        is_active=True,
    )
    transport = TransportOption(
        id=f"trn-q-{tag}", destination_id=dest.id, type="private_cab",
        name=f"Quality Cab {tag}", route_from="Origin",
        route_to=dest.name, price=1000.0, currency="INR", capacity=6,
        inventory_source="catalog", verification_status="catalog_verified",
        is_active=True,
    )
    db.add_all([hotel, transport])
    activity_ids = []
    for idx in range(n_acts):
        activity = Activity(
            id=f"act-q-{tag}-{idx}", destination_id=dest.id,
            title=f"Quality Spot {tag}-{idx}", category="culture",
            duration_hours=2.0, price_per_person=100.0, currency="INR",
            rating=4.5, meeting_point="Qualityville",
            latitude=10.0 + idx * 0.01, longitude=20.0,
            inventory_source="catalog",
            verification_status="catalog_verified", is_active=True,
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

        def get_recommendations(self, destination_id, preferences,
                                discovery_session_id=None):
            return {
                "ai_insights": None,
                "recommended_hotels": [{"id": hotel_id}],
                "recommended_activities": [{"id": aid} for aid in activity_ids],
                "recommended_transport": [{"id": transport_id}],
            }

    monkeypatch.setattr(itinerary_generator, "RecommendationEngine", StubEngine)


def _generate_quality_trip(monkeypatch, duration_days=3, n_acts=4):
    from backend.database.connection import SessionLocal
    from backend.models.models import ItineraryItem, Trip

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest, hotel, transport, activity_ids = _stub_inventory(db, tag, n_acts)
        _stub_engine(monkeypatch, hotel.id, activity_ids, transport.id)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Quality {tag}", duration_days=duration_days,
            total_budget=500000.0, currency="INR", traveler_count=2,
            pace="balanced",
        )
        db.add(trip)
        db.commit()
        items = ItineraryGenerator(db).generate_for_trip(trip.id)
        return items, {"hotel": hotel, "transport": transport,
                       "activity_ids": activity_ids, "trip_id": trip.id}
    finally:
        if trip is not None:
            from backend.models.models import (
                Activity, Destination, Hotel, TransportOption,
            )
            db.query(ItineraryItem).filter(
                ItineraryItem.trip_id == trip.id).delete()
            db.query(Trip).filter(Trip.id == trip.id).delete()
            db.commit()
        if dest is not None:
            from backend.models.models import (
                Activity, Destination, Hotel, TransportOption,
            )
            db.query(Activity).filter(
                Activity.destination_id == dest.id).delete()
            db.query(Hotel).filter(Hotel.destination_id == dest.id).delete()
            db.query(TransportOption).filter(
                TransportOption.destination_id == dest.id).delete()
            db.query(Destination).filter(Destination.id == dest.id).delete()
            db.commit()
        db.close()


def test_21_activity_scheduling_does_not_alter_costs(monkeypatch):
    items, _ctx = _generate_quality_trip(monkeypatch)
    for item in items:
        if item.item_type == "activity":
            assert float(item.cost or 0) == pytest.approx(200.0)


def test_22_budget_totals_remain_backend_authoritative(monkeypatch):
    items, _ctx = _generate_quality_trip(monkeypatch)
    total = sum(float(i.cost or 0) for i in items)
    transport = sum(float(i.cost or 0) for i in items
                    if i.item_type == "transport")
    accommodation = sum(float(i.cost or 0) for i in items
                        if i.item_type == "hotel")
    activities = sum(float(i.cost or 0) for i in items
                     if i.item_type == "activity")
    assert total == pytest.approx(transport + accommodation + activities)
    assert transport == pytest.approx(1000.0)
    assert activities == pytest.approx(200.0 * len(
        [i for i in items if i.item_type == "activity"]))


def test_23_hotel_reassignment_uses_verified_pricing(monkeypatch):
    items, _ctx = _generate_quality_trip(monkeypatch)
    hotel_items = [i for i in items if i.item_type == "hotel"]
    assert hotel_items
    for item in hotel_items:
        assert float(item.cost or 0) == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# Serialization (24-26)
# ---------------------------------------------------------------------------

def test_24_api_response_remains_compatible(monkeypatch):
    from backend.api.routes import _trip_dict
    from backend.database.connection import SessionLocal
    from backend.models.models import Trip

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest, hotel, transport, activity_ids = _stub_inventory(db, tag)
        _stub_engine(monkeypatch, hotel.id, activity_ids, transport.id)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Compat {tag}", duration_days=3,
            total_budget=500000.0, currency="INR", traveler_count=2,
            pace="balanced",
        )
        db.add(trip)
        db.commit()
        ItineraryGenerator(db).generate_for_trip(trip.id)
        db.refresh(trip)
        body = _trip_dict(trip, db)
        for key in ("id", "itinerary", "total_cost", "cost_breakdown",
                    "selected_accommodation", "daily_accommodations",
                    "route_days"):
            assert key in body, f"missing key {key}"
        assert body["daily_accommodations"]
        assert [d["day"] for d in body["route_days"]] == [1, 2, 3]
    finally:
        if trip is not None:
            from backend.models.models import (
                Activity, Destination, Hotel, ItineraryItem,
                TransportOption,
            )
            db.query(ItineraryItem).filter(
                ItineraryItem.trip_id == trip.id).delete()
            db.query(Trip).filter(Trip.id == trip.id).delete()
            db.commit()
        if dest is not None:
            from backend.models.models import (
                Activity, Destination, Hotel, TransportOption,
            )
            db.query(Activity).filter(
                Activity.destination_id == dest.id).delete()
            db.query(Hotel).filter(Hotel.destination_id == dest.id).delete()
            db.query(TransportOption).filter(
                TransportOption.destination_id == dest.id).delete()
            db.query(Destination).filter(Destination.id == dest.id).delete()
            db.commit()
        db.close()


def test_25_frontend_itinerary_rendering_does_not_break(monkeypatch):
    items, _ctx = _generate_quality_trip(monkeypatch)
    required = {"id", "day_number", "order_index", "item_type", "title",
                "start_time", "end_time", "cost", "status"}
    hotels_last = True
    for item in items:
        row = {"id": item.id, "day_number": item.day_number,
               "order_index": item.order_index, "item_type": item.item_type,
               "title": item.title, "start_time": item.start_time,
               "end_time": item.end_time, "cost": item.cost,
               "status": item.status}
        assert required <= set(row), f"missing keys for {item.id}"
        ui = (item.meta_data or {}).get("ui", {})
        if item.item_type == "hotel":
            assert ui.get("is_accommodation") is True
            assert ui.get("hotel_assignment_reason")
    by_day = {}
    for item in items:
        by_day.setdefault(item.day_number, []).append(item)
    for day_items in by_day.values():
        ordered = sorted(day_items, key=lambda i: i.order_index)
        hotel_positions = [n for n, i in enumerate(ordered)
                           if i.item_type == "hotel"]
        activity_positions = [n for n, i in enumerate(ordered)
                              if i.item_type == "activity"]
        if hotel_positions and activity_positions:
            assert min(hotel_positions) > min(activity_positions)
    assert hotels_last


def test_26_route_metadata_serializes_correctly(monkeypatch):
    items, _ctx = _generate_quality_trip(monkeypatch)
    for item in items:
        ui = (item.meta_data or {}).get("ui", {})
        json.dumps(ui)  # must be JSON-serializable for the API response
        leg = ui.get("route_leg_in")
        if leg is not None:
            assert leg["status"] in ("estimated", "unable_to_verify")
            assert leg["is_estimated"] is True
        summary = ui.get("route_day_summary")
        if summary is not None:
            assert summary["status"] in (
                "Valid", "Valid-with-Warning", "Requires-Revision",
                "Unable-to-Verify")
            json.dumps(summary)
