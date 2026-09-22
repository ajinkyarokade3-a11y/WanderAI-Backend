"""Pipeline-consistency regression tests (generic, destination-independent).

Covers the end-to-end contract with mocked deterministic coordinates and
isolated database fixtures (no live APIs):
separation, distribution, region-following stays, stay-plan/summary
agreement, cost reconciliation, accommodation-vs-activity separation,
overlap freedom, payload consistency, honest impossible-travel handling,
and no fabricated data.
"""

import uuid
from types import SimpleNamespace

import pytest

from backend.api.routes import _trip_dict
from backend.itinerary.generator import ItineraryGenerator
from backend.itinerary.route_plan import (
    DayRoute,
    RoutePlan,
    detect_overlaps,
    parse_clock_time,
    plan_activity_days,
)

GEN = ItineraryGenerator.__new__(ItineraryGenerator)

NORTH = (10.0, 20.0)
SOUTH = (13.0, 23.0)


def _act(aid, lat, lng, duration=2.0, price=100.0):
    return SimpleNamespace(
        id=aid, title=f"Place {aid}", latitude=lat, longitude=lng,
        duration_hours=duration, price_per_person=price,
    )


def test_1_geographically_separated_activities_not_forced_same_day():
    home = _act("home", 10.0, 20.0)
    far = _act("far", 20.0, 30.0)  # ~1500 km away
    plan = plan_activity_days([home, far], 2, pace="balanced")
    assert plan.activity_day["home"] != plan.activity_day["far"]
    assert set(plan.activity_day) == {"home", "far"}


def test_2_activities_distributed_across_feasible_days():
    acts = [_act(f"s{i}", 10.0 + i * 0.01, 20.0) for i in range(6)]
    plan = plan_activity_days(acts, 6, pace="balanced")
    used = sorted(plan.activity_day.values())
    assert used == [1, 2, 3, 4, 5, 6]


def test_12_impossible_travel_warned_not_hidden():
    home = _act("home", 10.0, 20.0)
    far = _act("far", 20.0, 30.0)
    plan = plan_activity_days([home, far], 2, pace="balanced")
    far_day = plan.days[plan.activity_day["far"]]
    assert far_day.warnings, "impossible travel must surface a warning"
    assert far_day.status in ("Valid-with-Warning", "Requires-Revision")


def test_19_unrestored_past_midnight_stays_honest():
    # Two 7h stops, one arrival day (360 min window): nothing fits, so both
    # stay unrestored with explicit overtime notes instead of overlaps.
    acts = [_act("a", 10.0, 20.0, duration=7.0),
            _act("b", 10.01, 20.01, duration=7.0)]
    plan = RoutePlan(days={1: DayRoute(day=1, activity_ids=["a", "b"])})
    plan.activity_day = {"a": 1, "b": 1}
    placements, unrestored, warnings = GEN._route_placements(plan, acts, {}, 4)
    assert sorted(unrestored) == ["a", "b"]
    assert warnings, "unrestored stops must carry warnings"
    for activity, _day, _order, start in placements:
        assert parse_clock_time(start) is not None, (
            "persisted start times must stay representable")
    assert any("after midnight" in w
               for ws in warnings.values() for w in ws)


# ---------------------------------------------------------------------------
# Multi-hotel generator fixture (two clusters, one hotel each).
# ---------------------------------------------------------------------------

def _stub_two_cluster_trip(monkeypatch, duration_days=4):
    import backend.itinerary.generator as itinerary_generator
    from backend.database.connection import SessionLocal
    from backend.models.models import (
        Activity, Destination, Hotel, ItineraryItem, TransportOption, Trip,
    )

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest = Destination(
            id=f"dst-pc-{tag}", name=f"Proveville {tag}",
            slug=f"proveville-{tag}", country="India",
            state_region=f"Proveville {tag}",
            description="Pipeline consistency destination.",
            latitude=10.0, longitude=20.0,
            inventory_source="catalog", verification_status="catalog_verified",
        )
        db.add(dest)
        db.flush()
        north = Hotel(
            id=f"htl-pc-n-{tag}", destination_id=dest.id,
            name=f"North Stay {tag}", category="mid-range",
            price_per_night=2000.0, currency="INR", rating=4.5,
            address="North Town", latitude=NORTH[0], longitude=NORTH[1],
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        south = Hotel(
            id=f"htl-pc-s-{tag}", destination_id=dest.id,
            name=f"South Stay {tag}", category="mid-range",
            price_per_night=3000.0, currency="INR", rating=4.5,
            address="South Town", latitude=SOUTH[0], longitude=SOUTH[1],
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        transport = TransportOption(
            id=f"trn-pc-{tag}", destination_id=dest.id, type="private_cab",
            name=f"Prove Cab {tag}", route_from="Origin",
            route_to=dest.name, price=1000.0, currency="INR", capacity=6,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        db.add_all([north, south, transport])
        coords = [NORTH, (10.05, 20.05), SOUTH, (13.05, 23.05)]
        activity_ids = []
        for idx, (lat, lng) in enumerate(coords):
            activity = Activity(
                id=f"act-pc-{tag}-{idx}", destination_id=dest.id,
                title=f"Prove Spot {tag}-{idx}", category="culture",
                duration_hours=2.0, price_per_person=100.0, currency="INR",
                rating=4.5, meeting_point="Proveville",
                latitude=lat, longitude=lng, inventory_source="catalog",
                verification_status="catalog_verified", is_active=True,
            )
            db.add(activity)
            activity_ids.append(activity.id)
        db.commit()

        class StubEngine:
            def __init__(self, db):
                pass

            def get_recommendations(self, destination_id, preferences,
                                    discovery_session_id=None):
                return {
                    "ai_insights": None,
                    "recommended_hotels": [{"id": north.id},
                                           {"id": south.id}],
                    "recommended_activities": [{"id": aid}
                                               for aid in activity_ids],
                    "recommended_transport": [{"id": transport.id}],
                }

        monkeypatch.setattr(itinerary_generator, "RecommendationEngine",
                            StubEngine)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Prove {tag}", duration_days=duration_days,
            total_budget=500000.0, currency="INR", traveler_count=2,
            pace="balanced",
        )
        db.add(trip)
        db.commit()
        ItineraryGenerator(db).generate_for_trip(trip.id)
        db.refresh(trip)
        body = _trip_dict(trip, db)
        # Capture plain values before the session closes: ORM attribute
        # access on detached rows would raise DetachedInstanceError.
        ctx = {"north": {"id": north.id, "name": north.name,
                            "coords": (north.latitude, north.longitude)},
               "south": {"id": south.id, "name": south.name,
                         "coords": (south.latitude, south.longitude)},
               "activity_ids": list(activity_ids)}
        return body, ctx
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


def test_3_4_5_hotel_follows_region_with_reuse(monkeypatch):
    """Stays follow day regions; same-region nights reuse the incumbent."""
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    by_night = {d["day_number"]: d["hotel"]["id"]
                for d in body["daily_accommodations"]}
    assert by_night[1] == ctx["north"]["id"]
    assert by_night[2] == ctx["south"]["id"]
    assert by_night[3] == ctx["south"]["id"]


def test_6_summary_matches_authoritative_stay_plan(monkeypatch):
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    sel = body["selected_accommodation"]
    assert sel["id"] == ctx["north"]["id"]
    # North covers exactly night 1 of 3 required: scoped honestly.
    assert sel["nights"] == 1
    assert sel["total_price"] == pytest.approx(sel["price_per_night"] * 1)
    assert sel["covers_all_nights"] is False
    stay_sum = round(sum(d["stay_cost"] for d in body["daily_accommodations"]), 2)
    assert stay_sum == pytest.approx(body["cost_breakdown"]["accommodation"])
    # route_days hotels agree with the stay plan (evening stay per day,
    # last day falls back to the stay the traveler woke up in).
    nightly = {d["day_number"]: d["hotel"]["id"]
               for d in body["daily_accommodations"]}
    by_day = {rd["day"]: rd["hotel"] for rd in body["route_days"]}
    assert by_day[1] == ctx["north"]["name"]
    assert by_day[2] == ctx["south"]["name"]
    assert by_day[3] == ctx["south"]["name"]
    assert by_day[4] == ctx["south"]["name"]
    assert nightly[1] == ctx["north"]["id"]


def test_7_stay_costs_reconcile(monkeypatch):
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    hotel_items = [r for r in body["itinerary"] if r["item_type"] == "hotel"]
    north_sum = sum(float(r["cost"] or 0) for r in hotel_items
                    if r["hotel_id"] == ctx["north"]["id"])
    south_sum = sum(float(r["cost"] or 0) for r in hotel_items
                    if r["hotel_id"] == ctx["south"]["id"])
    assert north_sum == pytest.approx(2000.0 * 1)
    assert south_sum == pytest.approx(3000.0 * 2)
    assert north_sum + south_sum == pytest.approx(
        body["cost_breakdown"]["accommodation"])
    cb = body["cost_breakdown"]
    assert cb["total"] == pytest.approx(
        cb["transport"] + cb["accommodation"] + cb["activities"])
    assert cb["remaining_budget"] == pytest.approx(
        cb["target_budget"] - cb["total"])


def test_8_9_accommodation_never_sightseeing(monkeypatch):
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    for row in body["itinerary"]:
        if row["item_type"] == "hotel":
            assert row.get("is_accommodation") is True
    for rd in body["route_days"]:
        day_rows = [r for r in body["itinerary"]
                    if r["day_number"] == rd["day"]]
        acts = [r for r in day_rows if r["item_type"] == "activity"]
        assert rd["activity_count"] == len(acts)
        assert "hotel" not in [r["item_type"] for r in acts]


def test_10_no_activity_overlaps(monkeypatch):
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    by_day = {}
    for row in body["itinerary"]:
        if row["item_type"] == "activity":
            by_day.setdefault(row["day_number"], []).append(row)
    assert by_day, "expected sightseeing on usable days"
    for day, stops in by_day.items():
        assert detect_overlaps(
            [(s["id"], s["start_time"], s["end_time"]) for s in stops]) == {}


def test_11_final_payload_internally_consistent(monkeypatch):
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    for key in ("itinerary", "route_days", "selected_accommodation",
                "daily_accommodations", "cost_breakdown", "total_cost"):
        assert key in body
    assert [d["day"] for d in body["route_days"]] == [1, 2, 3, 4]
    # Every activity row links to recommended inventory; every hotel row
    # links to a real hotel row.
    act_ids = set(ctx["activity_ids"])
    hotel_ids = {ctx["north"]["id"], ctx["south"]["id"]}
    for row in body["itinerary"]:
        if row["item_type"] == "activity":
            assert row["activity_id"] in act_ids
        if row["item_type"] == "hotel":
            assert row["hotel_id"] in hotel_ids
    # Warnings survive to the payload; statuses are honest.
    statuses = {rd["status"] for rd in body["route_days"]}
    assert statuses <= {"Valid", "Valid-with-Warning", "Requires-Revision",
                        "Unable-to-Verify", "Empty", None}


def test_13_no_fabricated_hotels_or_coordinates(monkeypatch):
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    catalog = {ctx["north"]["id"]: ctx["north"]["coords"],
               ctx["south"]["id"]: ctx["south"]["coords"]}
    for row in body["itinerary"]:
        if row["item_type"] != "hotel":
            continue
        assert row["hotel_id"] in catalog
        lat, lng = catalog[row["hotel_id"]]
        assert row.get("latitude") == lat
        assert row.get("longitude") == lng


def test_16_leg_origin_is_previous_night_stay(monkeypatch):
    """Day N>1 legs start from night N-1's hotel, not tonight's."""
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    legs = {}
    for row in body["itinerary"]:
        leg = row.get("route_leg_in")
        if isinstance(leg, dict) and row["item_type"] == "activity":
            legs.setdefault(row["day_number"], leg)
    # Day 2 morning: traveler wakes in night-1 (north) hotel.
    assert legs[2]["origin"]["item_type"] == "hotel"
    assert legs[2]["origin"]["item_id"] == ctx["north"]["id"]
    # Day 3 morning: night-2 (south) hotel.
    assert legs[3]["origin"]["item_id"] == ctx["south"]["id"]


def test_17_duplicate_hotel_night_flagged():
    hotel = SimpleNamespace(id="h1", name="Stay One", latitude=10.0,
                            longitude=20.0)
    a1 = SimpleNamespace(id="a1", title="Spot", latitude=10.0, longitude=20.0,
                         duration_hours=2.0)
    mk = lambda iid, day, order, kind, **kw: SimpleNamespace(
        id=iid, trip_id="t", day_number=day, order_index=order,
        item_type=kind, title=kw.get("title", kind),
        description=None, start_time=kw.get("start_time", "09:00 AM"),
        end_time=kw.get("end_time", "11:00 AM"), cost=0.0, status="proposed",
        hotel_id=kw.get("hotel_id"), activity_id=kw.get("activity_id"),
        transport_id=None, location="X", meta_data={"ui": {}},
    )
    new_items = [
        mk("n-h1", 2, 50, "hotel", title="Stay One", hotel_id="h1",
           start_time="01:30 PM", end_time="03:00 PM"),
        mk("n-h2", 2, 51, "hotel", title="Stay Two", hotel_id="h2",
           start_time="01:30 PM", end_time="03:00 PM"),
        mk("n-a1", 2, 1, "activity", title="Spot", activity_id="a1"),
    ]
    stays = [SimpleNamespace(night=2, hotel=hotel, reason="r",
                             distance_km=1.0, travel_minutes=2.0,
                             retained=True)]
    trip = SimpleNamespace(destination=SimpleNamespace(name="P"))
    plan = RoutePlan(days={2: DayRoute(day=2, activity_ids=["a1"])})
    GEN._attach_route_metadata(
        new_items, [(a1, 2, 1, "09:00 AM")], stays, [], trip, plan, {}, [])
    ui = new_items[2].meta_data["ui"]
    assert any("2 stays" in w for w in ui.get("route_warnings", [])), ui


def test_18_route_days_last_day_hotel_fallback(monkeypatch):
    body, ctx = _stub_two_cluster_trip(monkeypatch)
    by_day = {rd["day"]: rd for rd in body["route_days"]}
    # Day 4 has sightseeing but no night-4 item: previous stay still shown.
    assert by_day[4]["activity_count"] == 1
    assert by_day[4]["hotel"] == ctx["south"]["name"]


# ---------------------------------------------------------------------------
# Final-pass failure modes: coherence gate, timeless stays, region exposure,
# evening legs, repair targeting, warning channels.
# ---------------------------------------------------------------------------

def _priced(acts, traveler_count=2):
    return [(a, float(a.price_per_person or 0) * traveler_count) for a in acts]


def test_20_coherence_gate_prefers_compatible_subset():
    near_a = _act("na", 10.0, 20.0)
    near_b = _act("nb", 10.05, 20.05)
    mid = _act("mid", 12.0, 22.0)  # ~300 km away, own region
    far = _act("far", 20.0, 30.0)  # ~1500 km away, own region
    priced = _priced([near_a, near_b, mid, far])
    kept, notes = GEN._coherent_subset(priced, 1, 2)
    kept_ids = [a.id for a, _ in kept]
    # Two usable days admit two regions; the farthest low-ranked region drops.
    assert kept_ids == ["na", "nb", "mid"]
    assert notes and any("not selected" in n for n in notes)


def test_21_coherence_gate_backfills_minimum_honestly():
    near_a = _act("na", 10.0, 20.0)
    far = _act("far", 20.0, 30.0)
    priced = _priced([near_a, far])
    kept, notes = GEN._coherent_subset(priced, 2, 1)
    assert sorted(a.id for a, _ in kept) == ["far", "na"]
    assert any("minimum" in n for n in notes)


def test_22_coherence_gate_exempts_unlocatable():
    known = _act("known", 10.0, 20.0)
    mystery = _act("mystery", None, None)
    kept, _notes = GEN._coherent_subset(_priced([known, mystery]), 1, 1)
    assert sorted(a.id for a, _ in kept) == ["known", "mystery"]


def test_23_spilled_stop_repaired_to_near_day_not_far_day():
    # Day 1 (arrival window) overflows one 4h stop; day 2 is nearby and
    # fits, day 3 is distant: repair must choose day 2.
    a = _act("a", 10.0, 20.0, duration=4.0)
    b = _act("b", 10.01, 20.01, duration=4.0)
    c = _act("c", 10.02, 20.02, duration=2.0)
    d = _act("d", 20.0, 30.0, duration=2.0)
    plan = RoutePlan(days={1: DayRoute(day=1, activity_ids=["a", "b"]),
                           2: DayRoute(day=2, activity_ids=["c"]),
                           3: DayRoute(day=3, activity_ids=["d"])})
    plan.activity_day = {"a": 1, "b": 1, "c": 2, "d": 3}
    placements, unrestored, _warnings = GEN._route_placements(
        plan, [a, b, c, d], {}, 4)
    by_id = {act.id: day for act, day, _o, _s in placements for act in [act]}
    assert unrestored == []
    assert by_id["b"] == 2
    assert by_id["d"] == 3


def test_24_evening_leg_to_hotel_validated():
    hotel = SimpleNamespace(id="h1", name="Far Stay", latitude=20.0,
                            longitude=30.0)
    a1 = SimpleNamespace(id="a1", title="Spot", latitude=10.0, longitude=20.0,
                         duration_hours=2.0)
    mk = lambda iid, day, order, kind, **kw: SimpleNamespace(
        id=iid, trip_id="t", day_number=day, order_index=order,
        item_type=kind, title=kw.get("title", kind),
        description=None, start_time=kw.get("start_time", "09:00 AM"),
        end_time=kw.get("end_time", "11:00 AM"), cost=0.0, status="proposed",
        hotel_id=kw.get("hotel_id"), activity_id=kw.get("activity_id"),
        transport_id=None, location="X", meta_data={"ui": {}},
    )
    new_items = [
        mk("n-a1", 2, 1, "activity", title="Spot", activity_id="a1"),
    ]
    stays = [SimpleNamespace(night=1, hotel=hotel, reason="r",
                             distance_km=1.0, travel_minutes=2.0,
                             retained=True),
             SimpleNamespace(night=2, hotel=hotel, reason="r",
                             distance_km=1.0, travel_minutes=2.0,
                             retained=True)]
    trip = SimpleNamespace(destination=SimpleNamespace(name="P"))
    plan = RoutePlan(days={2: DayRoute(day=2, activity_ids=["a1"])})
    GEN._attach_route_metadata(
        new_items, [(a1, 2, 1, "09:00 AM")], stays, [], trip, plan, {}, [])
    ui = new_items[0].meta_data["ui"]
    assert any("ends about" in w for w in ui.get("route_warnings", [])), ui
    assert ui["route_day_summary"]["status"] == "Valid-with-Warning"


def _stub_three_cluster_trip(monkeypatch):
    """Two-day trip, three far-apart clusters: east is dropped + recorded."""
    import backend.itinerary.generator as itinerary_generator
    from backend.database.connection import SessionLocal
    from backend.models.models import (
        Activity, Destination, Hotel, ItineraryItem, TransportOption, Trip,
    )

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest = Destination(
            id=f"dst-tc-{tag}", name=f"Tripleville {tag}",
            slug=f"tripleville-{tag}", country="India",
            state_region=f"Tripleville {tag}",
            description="Coherence-drop destination.",
            latitude=10.0, longitude=20.0,
            inventory_source="catalog", verification_status="catalog_verified",
        )
        db.add(dest)
        db.flush()
        north = Hotel(
            id=f"htl-tc-n-{tag}", destination_id=dest.id,
            name=f"North Stay {tag}", category="mid-range",
            price_per_night=2000.0, currency="INR", rating=4.5,
            address="North Town", latitude=NORTH[0], longitude=NORTH[1],
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        south = Hotel(
            id=f"htl-tc-s-{tag}", destination_id=dest.id,
            name=f"South Stay {tag}", category="mid-range",
            price_per_night=2000.0, currency="INR", rating=4.5,
            address="South Town", latitude=SOUTH[0], longitude=SOUTH[1],
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        transport = TransportOption(
            id=f"trn-tc-{tag}", destination_id=dest.id, type="private_cab",
            name=f"Triple Cab {tag}", route_from="Origin",
            route_to=dest.name, price=1000.0, currency="INR", capacity=6,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        db.add_all([north, south, transport])
        spots = [("n1",) + NORTH, ("s1",) + SOUTH, ("e1", 20.0, 30.0)]
        activity_ids = []
        for title, lat, lng in spots:
            activity = Activity(
                id=f"act-tc-{tag}-{title}", destination_id=dest.id,
                title=f"Triple {title} {tag}", category="culture",
                duration_hours=2.0, price_per_person=100.0, currency="INR",
                rating=4.5, meeting_point="Tripleville",
                latitude=lat, longitude=lng, inventory_source="catalog",
                verification_status="catalog_verified", is_active=True,
            )
            db.add(activity)
            activity_ids.append(activity.id)
        db.commit()
        north_id, south_id = north.id, south.id

        class StubEngine:
            def __init__(self, db):
                pass

            def get_recommendations(self, destination_id, preferences,
                                    discovery_session_id=None):
                return {
                    "ai_insights": None,
                    "recommended_hotels": [{"id": north_id},
                                           {"id": south_id}],
                    "recommended_activities": [{"id": aid}
                                               for aid in activity_ids],
                    "recommended_transport": [{"id": transport.id}],
                }

        monkeypatch.setattr(itinerary_generator, "RecommendationEngine",
                            StubEngine)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Triple {tag}", duration_days=2,
            total_budget=500000.0, currency="INR", traveler_count=2,
            pace="balanced",
        )
        db.add(trip)
        db.commit()
        ItineraryGenerator(db).generate_for_trip(trip.id)
        db.refresh(trip)
        body = _trip_dict(trip, db)
        return body
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


def test_25_distant_cluster_dropped_and_recorded(monkeypatch):
    body = _stub_three_cluster_trip(monkeypatch)
    titles = [r["title"] for r in body["itinerary"]
              if r["item_type"] == "activity"]
    assert not any("Triple e1" in t for t in titles)
    assert body["itinerary_warnings"], "exclusion must be recorded"
    assert any("not selected" in w for w in body["itinerary_warnings"])


def test_26_no_fake_hotel_durations_in_payload(monkeypatch):
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    for row in body["itinerary"]:
        if row["item_type"] == "hotel":
            assert row["start_time"] is None and row["end_time"] is None, row
            assert row.get("is_accommodation") is True


def test_27_overnight_region_exposed_consistently(monkeypatch):
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    for entry in body["daily_accommodations"]:
        assert entry["hotel"].get("overnight_region") is not None
    regions = {rd["day"]: rd["overnight_region"]
               for rd in body["route_days"]}
    assert all(v is None or isinstance(v, int)
               for v in regions.values())
    # Day-level region matches the nightly entry region.
    for entry in body["daily_accommodations"]:
        day = entry["day_number"]
        assert regions[day] == entry["hotel"].get("overnight_region")


def test_28_itinerary_warnings_channel_defaults_empty(monkeypatch):
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    assert isinstance(body.get("itinerary_warnings"), list)


# ---------------------------------------------------------------------------
# Final-pass failure modes (§13): evaluations, repair loop, chain exposure.
# ---------------------------------------------------------------------------

def _no_live_key(monkeypatch):
    from backend.database.config import settings
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "")


def test_29_distant_hotel_rejected_with_concrete_reason():
    from backend.itinerary.hotel_assignment import DayAnchor, plan_overnight_stays

    near = SimpleNamespace(id="h-near", name="Near Stay", latitude=10.0,
                           longitude=20.0, price_per_night=2000.0)
    mid = SimpleNamespace(id="h-mid", name="Mid Stay", latitude=10.1,
                          longitude=20.1, price_per_night=2000.0)
    far = SimpleNamespace(id="h-far", name="Far Stay", latitude=15.0,
                          longitude=25.0, price_per_night=2000.0)
    stays, _spent = plan_overnight_stays(
        nights=1,
        anchors={1: DayAnchor(latitude=10.0, longitude=20.0, label="Base")},
        hotels=[far, mid, near],
        arrival_point=(10.0, 20.0),
    )
    assert stays[0].hotel.id == "h-near"
    by_id = {e["hotel_id"]: e for e in stays[0].evaluations}
    assert set(by_id) == {"h-far", "h-mid"}
    assert by_id["h-mid"]["decision"] == "rejected_rank_order"
    assert by_id["h-far"]["decision"] == "rejected_farther"
    assert by_id["h-far"]["distance_km"] is not None
    assert by_id["h-far"]["distance_km"] > 100


def test_30_evaluation_reasons_use_known_vocabulary():
    from backend.itinerary.hotel_assignment import DayAnchor, plan_overnight_stays

    known = {"pinned", "unaffordable", "budget_forced", "opening",
             "retained_no_detail", "retained_no_coords", "retained_preferred",
             "moved", "retained_no_near_option", "retained_comfortable",
             "rejected_unaffordable", "rejected_no_coords",
             "rejected_rank_order", "rejected_farther",
             "rejected_outside_preferred"}
    pricey = SimpleNamespace(id="h-pricey", name="Pricey", latitude=10.0,
                             longitude=20.0, price_per_night=999999.0)
    cheap = SimpleNamespace(id="h-cheap", name="Cheap", latitude=15.0,
                            longitude=25.0, price_per_night=100.0)
    stays, _spent = plan_overnight_stays(
        nights=1,
        anchors={1: DayAnchor(latitude=10.0, longitude=20.0, label="Base")},
        hotels=[pricey, cheap],
        total_pot=500.0,
    )
    assert stays[0].hotel.id == "h-cheap"
    assert stays[0].decision in known
    for entry in stays[0].evaluations:
        assert entry["decision"] in known, entry


def test_31_nearer_unaffordable_hotel_keeps_current_honestly():
    from backend.itinerary.hotel_assignment import DayAnchor, plan_overnight_stays

    base = SimpleNamespace(id="h-base", name="Base", latitude=10.0,
                           longitude=20.0, price_per_night=2000.0)
    luxe_near = SimpleNamespace(id="h-luxe", name="Luxe Near",
                                latitude=10.02, longitude=20.02,
                                price_per_night=50000.0)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: DayAnchor(latitude=10.0, longitude=20.0, label="D2"),
                 2: DayAnchor(latitude=10.02, longitude=20.02, label="D3")},
        hotels=[base, luxe_near],
        total_pot=9000.0,
    )
    assert [s.hotel.id for s in stays] == ["h-base", "h-base"]
    luxe_evals = [e for s in stays for e in s.evaluations
                  if e["hotel_id"] == "h-luxe"]
    assert luxe_evals and all(e["decision"] == "rejected_unaffordable"
                              for e in luxe_evals)


def test_32_evening_warning_names_tonights_hotel():
    hotel_morning = SimpleNamespace(id="h-am", name="Morning Inn",
                                    latitude=10.0, longitude=20.0)
    hotel_night = SimpleNamespace(id="h-pm", name="Evening Lodge",
                                  latitude=20.0, longitude=30.0)
    a1 = SimpleNamespace(id="a1", title="Spot", latitude=10.0, longitude=20.0,
                         duration_hours=2.0)
    mk = lambda iid, day, order, kind, **kw: SimpleNamespace(
        id=iid, trip_id="t", day_number=day, order_index=order,
        item_type=kind, title=kw.get("title", kind),
        description=None, start_time=kw.get("start_time", "09:00 AM"),
        end_time=kw.get("end_time", "11:00 AM"), cost=0.0, status="proposed",
        hotel_id=kw.get("hotel_id"), activity_id=kw.get("activity_id"),
        transport_id=None, location="X", meta_data={"ui": {}},
    )
    new_items = [mk("n-a1", 2, 1, "activity", title="Spot", activity_id="a1")]
    stays = [SimpleNamespace(night=1, hotel=hotel_morning, reason="r",
                             distance_km=0.0, travel_minutes=0.0,
                             retained=True),
             SimpleNamespace(night=2, hotel=hotel_night, reason="r",
                             distance_km=0.0, travel_minutes=0.0,
                             retained=True)]
    trip = SimpleNamespace(destination=SimpleNamespace(name="P"))
    plan = RoutePlan(days={2: DayRoute(day=2, activity_ids=["a1"])})
    GEN._attach_route_metadata(
        new_items, [(a1, 2, 1, "09:00 AM")], stays, [], trip, plan, {}, [])
    ui = new_items[0].meta_data["ui"]
    evening = [w for w in ui.get("route_warnings", []) if "ends about" in w]
    assert evening and any("Evening Lodge" in w for w in evening), ui
    morning = [w for w in ui.get("route_warnings", []) if "starts about" in w]
    assert not morning, ui  # morning origin (Morning Inn) is nearby: no warning


def test_33_stay_count_matches_trip_dates(monkeypatch):
    _no_live_key(monkeypatch)
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    assert len(body["daily_accommodations"]) == 3  # duration 4 → 3 nights
    nights = sorted(d["day_number"] for d in body["daily_accommodations"])
    assert nights == [1, 2, 3]


def test_34_daily_chain_dates_consecutive(monkeypatch):
    import datetime as _dt

    from backend.database.connection import SessionLocal
    from backend.models.models import Trip

    _no_live_key(monkeypatch)
    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        import backend.itinerary.generator as itinerary_generator
        from backend.models.models import (
            Activity, Destination, Hotel, ItineraryItem, TransportOption,
        )
        from backend.itinerary.generator import ItineraryGenerator

        dest = Destination(
            id=f"dst-dc-{tag}", name=f"Dateville {tag}",
            slug=f"dateville-{tag}", country="India",
            state_region=f"Dateville {tag}", description="Date test.",
            latitude=10.0, longitude=20.0,
            inventory_source="catalog", verification_status="catalog_verified",
        )
        db.add(dest)
        db.flush()
        hotel = Hotel(
            id=f"htl-dc-{tag}", destination_id=dest.id,
            name=f"Date Stay {tag}", price_per_night=2000.0, currency="INR",
            rating=4.5, latitude=10.0, longitude=20.0,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        transport = TransportOption(
            id=f"trn-dc-{tag}", destination_id=dest.id, type="private_cab",
            name=f"Date Cab {tag}", route_from="Origin", route_to=dest.name,
            price=1000.0, currency="INR", capacity=6,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        db.add_all([hotel, transport])
        aids = []
        for idx in range(3):
            act = Activity(
                id=f"act-dc-{tag}-{idx}", destination_id=dest.id,
                title=f"Date Spot {tag}-{idx}", category="culture",
                duration_hours=2.0, price_per_person=100.0, currency="INR",
                rating=4.5, meeting_point="Dateville",
                latitude=10.0 + idx * 0.01, longitude=20.0,
                inventory_source="catalog",
                verification_status="catalog_verified", is_active=True,
            )
            db.add(act)
            aids.append(act.id)
        db.commit()

        class StubEngine:
            def __init__(self, db):
                pass

            def get_recommendations(self, destination_id, preferences,
                                    discovery_session_id=None):
                return {
                    "ai_insights": None,
                    "recommended_hotels": [{"id": hotel.id}],
                    "recommended_activities": [{"id": aid} for aid in aids],
                    "recommended_transport": [{"id": transport.id}],
                }

        monkeypatch.setattr(itinerary_generator, "RecommendationEngine",
                            StubEngine)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Dates {tag}", duration_days=3,
            total_budget=500000.0, currency="INR", traveler_count=2,
            pace="balanced",
            start_date=_dt.datetime(2026, 10, 5),
            end_date=_dt.datetime(2026, 10, 7),
        )
        db.add(trip)
        db.commit()
        ItineraryGenerator(db).generate_for_trip(trip.id)
        db.refresh(trip)
        body = _trip_dict(trip, db)
        checkins = [d["hotel"]["check_in_date"]
                    for d in body["daily_accommodations"]]
        checkouts = [d["hotel"]["check_out_date"]
                     for d in body["daily_accommodations"]]
        assert checkins == ["2026-10-05", "2026-10-06"]
        assert checkouts == ["2026-10-06", "2026-10-07"]
        assert body["route_days"][0]["date"] == "2026-10-05"
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


def test_35_daily_entries_carry_evidence_status(monkeypatch):
    _no_live_key(monkeypatch)
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    for entry in body["daily_accommodations"]:
        assert entry["hotel"]["inventory_source"] == "catalog"
        assert entry["hotel"]["verification_status"] == "catalog_verified"


def test_36_warning_ids_all_resolve_in_payload(monkeypatch):
    _no_live_key(monkeypatch)
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    act_ids = {r["activity_id"] for r in body["itinerary"]
               if r["item_type"] == "activity" and r["activity_id"]}
    hotel_ids = {r["hotel_id"] for r in body["itinerary"]
                 if r["item_type"] == "hotel" and r["hotel_id"]}
    texts = list(body.get("itinerary_warnings", []))
    for row in body["itinerary"]:
        texts.extend(row.get("route_warnings", []))
    for rd in body["route_days"]:
        texts.extend(rd.get("warnings", []))
    for text in texts:
        for token in str(text).split():
            token = token.strip(",.;:()")
            if token.startswith("dyn-act-") or token.startswith("act-"):
                assert token in act_ids, f"stale activity id in warning: {text}"
            if token.startswith("htl-"):
                assert token in hotel_ids, f"stale hotel id in warning: {text}"


def test_37_dropped_activity_leaves_no_route_warnings(monkeypatch):
    _no_live_key(monkeypatch)
    body = _stub_three_cluster_trip(monkeypatch)
    for row in body["itinerary"]:
        for warning in row.get("route_warnings", []):
            assert "Triple e1" not in warning
    for rd in body["route_days"]:
        for warning in rd.get("warnings", []):
            assert "Triple e1" not in warning


def test_38_no_silent_extreme_hotel_legs(monkeypatch):
    """Every estimated hotel leg above the limit has a matching warning."""
    _no_live_key(monkeypatch)
    body, _ctx = _stub_two_cluster_trip(monkeypatch)
    for row in body["itinerary"]:
        leg = row.get("route_leg_in")
        if not isinstance(leg, dict):
            continue
        minutes = leg.get("estimated_duration_minutes") or 0
        if leg.get("status") == "estimated" and minutes > 120.0:
            day = row["day_number"]
            day_warnings = []
            for other in body["itinerary"]:
                if other["day_number"] == day:
                    day_warnings.extend(other.get("route_warnings", []))
            assert any("closer stay" in w or "min from" in w
                       for w in day_warnings), (row, day_warnings)


def test_39_repair_moves_act_and_records_audit(monkeypatch):
    """Global repair relocates the extreme-placed stop and audits it."""
    _no_live_key(monkeypatch)
    n1 = _act("n1", 10.0, 20.0, duration=2.0)
    n2 = _act("n2", 10.05, 20.05, duration=2.0)
    # ~155 km / ~187 min from the northern pair: extreme with company,
    # clean alone on an empty day whose stay is next door.
    s1 = _act("s1", 11.0, 21.0, duration=2.0)
    north = SimpleNamespace(id="h-north", name="North Hotel", latitude=10.0,
                            longitude=20.0, price_per_night=2000.0)
    south = SimpleNamespace(id="h-south", name="South Hotel", latitude=11.0,
                            longitude=21.0, price_per_night=2000.0)
    plan = RoutePlan(days={2: DayRoute(day=2, activity_ids=["n1", "s1"]),
                           3: DayRoute(day=3, activity_ids=["n2"]),
                           4: DayRoute(day=4, activity_ids=[])})
    plan.activity_day = {"n1": 2, "s1": 2, "n2": 3}
    placements, _unrestored, warnings = GEN._route_placements(
        plan, [n1, n2, s1], {}, 4)
    stays = [
        SimpleNamespace(night=2, hotel=north, reason="r", distance_km=1.0,
                        travel_minutes=1.0, retained=True,
                        decision="opening", evaluations=[]),
        SimpleNamespace(night=3, hotel=south, reason="r", distance_km=1.0,
                        travel_minutes=1.0, retained=True,
                        decision="opening", evaluations=[]),
    ]
    trip = SimpleNamespace(destination=SimpleNamespace(name="P"))
    rerun_state = {"calls": 0}

    def _fake_plan_stays(*args, **kwargs):
        rerun_state["calls"] += 1
        return list(stays), 4000.0, {2: 0, 3: 1}

    monkeypatch.setattr(ItineraryGenerator, "_plan_stays",
                        lambda self, *a, **k: _fake_plan_stays())
    out = GEN._repair_global_route(
        placements, stays, {2: 0, 3: 1}, plan, [n1, n2, s1], {}, 4,
        trip, [north, south], 2, north, 4000.0, None, [],
        [], warnings, 4000.0,
    )
    new_placements, new_stays, _regions, _unrestored, _warn, _spent = out
    by_day = {}
    for act, day, _o, _s in new_placements:
        by_day.setdefault(act.id, day)
    # s1 leaves the northern day for the empty day with a southern stay.
    assert by_day["s1"] == 4
    assert rerun_state["calls"] >= 1
    notes = getattr(trip, "_itinerary_warnings", [])
    assert any("Repaired" in n and "s1" in n for n in notes), notes


def test_40_repair_rejects_worse_trade_and_records(monkeypatch):
    """A move that merely swaps one extreme for another is rejected."""
    _no_live_key(monkeypatch)
    n1 = _act("n1", 10.0, 20.0, duration=2.0)
    s1 = _act("s1", 11.0, 21.0, duration=2.0)
    north = SimpleNamespace(id="h-north", name="North Hotel", latitude=10.0,
                            longitude=20.0, price_per_night=2000.0)
    plan = RoutePlan(days={2: DayRoute(day=2, activity_ids=["n1", "s1"]),
                           3: DayRoute(day=3, activity_ids=[])})
    plan.activity_day = {"n1": 2, "s1": 2}
    placements, _unrestored, warnings = GEN._route_placements(
        plan, [n1, s1], {}, 4)
    stays = [
        SimpleNamespace(night=2, hotel=north, reason="r", distance_km=1.0,
                        travel_minutes=1.0, retained=True,
                        decision="opening", evaluations=[]),
        SimpleNamespace(night=3, hotel=north, reason="r", distance_km=1.0,
                        travel_minutes=1.0, retained=True,
                        decision="opening", evaluations=[]),
    ]
    trip = SimpleNamespace(destination=SimpleNamespace(name="P"))
    monkeypatch.setattr(
        ItineraryGenerator, "_plan_stays",
        lambda self, *a, **k: (list(stays), 4000.0, {2: 0, 3: 0}))
    out = GEN._repair_global_route(
        placements, stays, {2: 0, 3: 0}, plan, [n1, s1], {}, 4,
        trip, [north], 2, north, 4000.0, None, [],
        [], warnings, 4000.0,
    )
    new_placements = out[0]
    by_day = {}
    for act, day, _o, _s in new_placements:
        by_day.setdefault(act.id, day)
    # Day 3 is empty and fits, but both its stays are the northern hotel:
    # moving s1 there keeps extreme legs, so the move is rejected.
    assert by_day["s1"] == 2
    notes = getattr(trip, "_itinerary_warnings", [])
    assert any("Kept" in n for n in notes), notes
