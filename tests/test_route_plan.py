"""Generalized route-planning tests (Stages 1-4).

All fixtures use synthetic coordinate grids so no destination, state,
city, or attraction name can influence the algorithm. Kerala/Rajasthan
shapes appear only as coordinate spreads in two regression cases.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from backend.itinerary import hotel_assignment as hotel_engine
from backend.itinerary import route_plan as planner
from backend.itinerary.route_plan import (
    ROUTE_PLAN_CONFIG,
    DayRoute,
    RouteEndpoint,
    RouteLeg,
    build_leg,
    detect_overlaps,
    parse_clock_time,
    plan_activity_days,
    resolve_pace,
    summarize_legs,
)


def _act(aid, lat, lng, duration=2.0):
    return SimpleNamespace(id=aid, title=f"Place {aid}", latitude=lat,
                           longitude=lng, duration_hours=duration)


def _ep(aid, kind="activity", name=None):
    return RouteEndpoint(item_id=aid, item_type=kind, name=name or aid,
                         coordinates_available=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_route_config_defaults_documented():
    cfg = ROUTE_PLAN_CONFIG
    for key in ("max_one_way_travel_minutes", "max_daily_travel_minutes",
                "activity_travel_buffer_minutes", "max_activities_per_day",
                "default_travel_speed_kmh", "pace_profiles"):
        assert key in cfg, f"missing documented setting {key}"
    assert cfg["default_travel_speed_kmh"] == pytest.approx(
        hotel_engine.HOTEL_ASSIGNMENT_CONFIG["assumed_speed_kmh"]
    )


def test_pace_profiles_resolve():
    assert resolve_pace("relaxed")["max_activities_per_day"] == 2
    assert resolve_pace("balanced")["max_activities_per_day"] == 4
    assert resolve_pace("packed")["max_activities_per_day"] == 6
    assert resolve_pace("nonsense") == resolve_pace("balanced")
    assert resolve_pace(None) == resolve_pace("balanced")


def test_shared_estimation_functions_reused():
    assert planner.haversine_km is hotel_engine.haversine_km
    assert planner.estimate_travel_minutes is hotel_engine.estimate_travel_minutes


# ---------------------------------------------------------------------------
# Leg estimation and labeling
# ---------------------------------------------------------------------------

def test_estimated_leg_labeling():
    leg = build_leg(_ep("a"), _ep("b"), (10.0, 20.0), (10.1, 20.1), 50.0)
    assert leg.status == "estimated"
    assert leg.is_estimated is True
    assert leg.estimation_method == "haversine_speed_estimate"
    assert leg.distance_km is not None and leg.distance_km > 0
    assert leg.estimated_duration_minutes is not None
    assert leg.coordinates_available is True
    assert leg.uncertainty_reason is None


def test_missing_coordinates_never_become_zero():
    leg = build_leg(_ep("a"), _ep("b"), None, (10.1, 20.1), 50.0)
    assert leg.status == "unable_to_verify"
    assert leg.distance_km is None
    assert leg.estimated_duration_minutes is None
    assert leg.uncertainty_reason
    leg2 = build_leg(_ep("a"), _ep("b"), (10.0, 20.0), None, 50.0)
    assert leg2.status == "unable_to_verify"
    assert leg2.distance_km is None


def test_zero_distance_with_coords_is_estimated_not_missing():
    leg = build_leg(_ep("a"), _ep("b"), (10.0, 20.0), (10.0, 20.0), 50.0)
    assert leg.status == "estimated"
    assert leg.distance_km == pytest.approx(0.0)


def test_leg_dict_json_round_trip():
    leg = build_leg(_ep("a"), _ep("b"), (10.0, 20.0), (10.2, 20.0), 50.0)
    assert json.loads(json.dumps(leg.to_dict()))["status"] == "estimated"
    day = DayRoute(day=1, activity_ids=["a", "b"], legs=[leg])
    assert json.loads(json.dumps(day.to_dict()))["day"] == 1


def test_summarize_legs_handles_unverifiable():
    total, longest = summarize_legs([
        build_leg(_ep("a"), _ep("b"), (1.0, 1.0), (1.1, 1.1), 50.0),
        build_leg(_ep("b"), _ep("c"), None, (1.1, 1.1), 50.0),
    ])
    assert total is not None and longest is not None
    assert summarize_legs([]) == (None, None)


# ---------------------------------------------------------------------------
# Day assignment: grouping, budgets, determinism
# ---------------------------------------------------------------------------

def test_two_clusters_separate_days():
    acts = [_act("n1", 10.0, 20.0), _act("n2", 10.05, 20.05),
            _act("f1", 13.0, 23.0), _act("f2", 13.05, 23.05)]
    plan = plan_activity_days(acts, 4, pace="balanced")
    # Load is balanced across all usable days (no hotel-only gaps), while
    # no single day mixes the far-apart clusters.
    assert sorted(plan.activity_day) == ["f1", "f2", "n1", "n2"]
    assert sorted(plan.activity_day.values()) == [1, 2, 3, 4]
    near_days = {plan.activity_day["n1"], plan.activity_day["n2"]}
    far_days = {plan.activity_day["f1"], plan.activity_day["f2"]}
    assert not near_days & far_days


def test_distant_outlier_flagged_not_dropped():
    acts = [_act("home", 10.0, 20.0), _act("far", 16.0, 26.0)]  # ~900 km
    plan = plan_activity_days(acts, 2, pace="balanced")
    assert set(plan.activity_day) == {"home", "far"}
    assert plan.activity_day["home"] != plan.activity_day["far"]
    far_day = plan.days[plan.activity_day["far"]]
    assert far_day.status == "Valid-with-Warning"
    assert any("one-way" in w for w in far_day.warnings)
    assert plan.activity_day["home"] == 1


def test_daily_budget_spreads_chain():
    acts = [_act(f"c{i}", 10.0 + i * 1.0, 20.0) for i in range(4)]  # ~111 km apart
    plan = plan_activity_days(acts, 4, pace="balanced")
    assert set(plan.activity_day) == {f"c{i}" for i in range(4)}
    used = {plan.activity_day[f"c{i}"] for i in range(4)}
    assert len(used) > 1  # chain cannot fit one day budget


def test_count_cap_respected():
    acts = [_act(f"k{i}", 10.0 + i * 0.01, 20.0) for i in range(6)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    counts = sorted(len(plan.days[d].activity_ids) for d in (1, 2))
    assert counts == [2, 4]
    assert set(plan.activity_day) == {f"k{i}" for i in range(6)}


def test_deterministic_output():
    acts = [_act("b", 10.2, 20.1), _act("a", 10.0, 20.0), _act("c", 12.5, 22.5),
            _act("d", 10.1, 20.05)]
    first = plan_activity_days(acts, 3, pace="balanced")
    second = plan_activity_days(acts, 3, pace="balanced")
    assert first.activity_day == second.activity_day
    assert {d: r.activity_ids for d, r in first.days.items()} == \
           {d: r.activity_ids for d, r in second.days.items()}


def test_no_coordinates_keeps_round_robin_days():
    acts = [_act(f"m{i}", None, None) for i in range(5)]
    plan = plan_activity_days(acts, 6, pace="balanced")
    assert sorted(plan.activity_day.values()) == [1, 2, 3, 4, 5]
    for day in range(1, 6):
        assert plan.days[day].status == "Unable-to-Verify"


def test_empty_days_allowed_no_forced_clusters():
    acts = [_act("x", 10.0, 20.0), _act("y", 10.05, 20.05)]
    plan = plan_activity_days(acts, 5, pace="balanced")
    assert set(plan.activity_day.values()) <= {1, 2, 3, 4, 5}
    empty = [d for d in range(1, 6) if not plan.days[d].activity_ids]
    assert empty  # days emerge from data; never force-filled


def test_empty_input_no_crash():
    plan = plan_activity_days([], 4, pace="balanced")
    assert plan.activity_day == {}
    assert all(not r.activity_ids for r in plan.days.values())


def test_invalid_coordinates_treated_as_missing():
    acts = [_act("ok", 10.0, 20.0), _act("bad", 999.0, 20.0),
            _act("alsobad", 10.0, "east")]
    plan = plan_activity_days(acts, 3, pace="balanced")
    assert set(plan.activity_day) == {"ok", "bad", "alsobad"}
    assert plan.days[plan.activity_day["bad"]].status in (
        "Unable-to-Verify", "Valid-with-Warning", "Requires-Revision")


def test_durations_recorded_and_spread():
    acts = [_act("short", 10.0, 20.0, duration=1.0),
            _act("long", 14.0, 24.0, duration=6.0)]
    plan = plan_activity_days(acts, 2, pace="balanced")
    minutes = sorted(
        r.activity_minutes for r in plan.days.values() if r.activity_ids
    )
    assert minutes == [60.0, 360.0]
    assert set(plan.activity_day) == {"short", "long"}


def test_more_activities_than_limit_all_kept_with_warnings():
    acts = [_act(f"o{i}", 10.0 + i * 0.001, 20.0) for i in range(5)]
    plan = plan_activity_days(acts, 1, pace="relaxed")  # cap 2/day
    assert set(plan.activity_day) == {f"o{i}" for i in range(5)}
    assert plan.days[1].status == "Requires-Revision"
    assert plan.days[1].warnings


def test_hotel_rows_never_confused_with_activities():
    hotels = [_act("htl-1", 10.0, 20.0), _act("htl-2", 13.0, 23.0)]
    plan = plan_activity_days(hotels, 2, pace="balanced")
    # Planner is id-agnostic: ids pass through verbatim, nothing invented.
    assert sorted(plan.activity_day) == ["htl-1", "htl-2"]


# ---------------------------------------------------------------------------
# Regression geometries (coordinate spreads only, no name logic)
# ---------------------------------------------------------------------------

def test_south_spread_geometry_separates_coast_and_hills():
    # Coastal cluster, hill cluster ~125 km / ~150 min away (beyond the
    # one-way limit, mirroring a plains-to-hills run), plus a far-southern
    # outlier.
    acts = [_act("coast1", 9.97, 76.24), _act("coast2", 9.93, 76.27),
            _act("hill1", 10.85, 77.05), _act("hill2", 10.90, 77.10),
            _act("south", 8.40, 76.98)]
    plan = plan_activity_days(acts, 4, pace="balanced")
    assert set(plan.activity_day) == {"coast1", "coast2", "hill1", "hill2", "south"}
    # No day mixes the coast cluster with the hill cluster or the south
    # outlier; the outlier is kept with an honest warning, never dropped.
    coast_days = {plan.activity_day["coast1"], plan.activity_day["coast2"]}
    hill_days = {plan.activity_day["hill1"], plan.activity_day["hill2"]}
    assert not coast_days & hill_days
    assert plan.activity_day["south"] not in coast_days
    south_day = plan.days[plan.activity_day["south"]]
    assert south_day.status in ("Valid-with-Warning", "Requires-Revision")
    assert south_day.warnings


def test_west_spread_geometry_separates_desert_and_hills():
    # Base cluster, far-west outlier ~550 km, hill outlier ~250 km.
    acts = [_act("base1", 26.9, 75.8), _act("base2", 26.95, 75.85),
            _act("west", 26.9, 70.9), _act("hill", 27.6, 76.6)]
    plan = plan_activity_days(acts, 4, pace="balanced")
    assert set(plan.activity_day) == {"base1", "base2", "west", "hill"}
    # Neither outlier shares a day with the base cluster; all are kept.
    base_days = {plan.activity_day["base1"], plan.activity_day["base2"]}
    assert plan.activity_day["west"] not in base_days
    assert plan.activity_day["hill"] not in base_days


def test_unnamed_triangle_plus_outlier():
    acts = [_act("p1", 0.0, 0.0), _act("p2", 0.2, 0.1), _act("p3", 0.1, 0.25),
            _act("remote", 4.0, 4.0)]
    plan = plan_activity_days(acts, 3, pace="balanced")
    assert set(plan.activity_day) == {"p1", "p2", "p3", "remote"}
    # The remote outlier never shares a day with the triangle; every
    # usable day carries sightseeing instead of clustering early.
    triangle_days = {plan.activity_day["p1"], plan.activity_day["p2"],
                     plan.activity_day["p3"]}
    assert plan.activity_day["remote"] not in triangle_days
    assert triangle_days | {plan.activity_day["remote"]} == {1, 2, 3}


# ---------------------------------------------------------------------------
# Clock parsing and overlap detection (pure, no database)
# ---------------------------------------------------------------------------

def test_parse_clock_time():
    assert parse_clock_time("09:00 AM") == 540
    assert parse_clock_time("12:00 PM") == 720
    assert parse_clock_time("12:00 AM") == 0
    assert parse_clock_time("1:30 PM") == 810
    assert parse_clock_time("13:00") == 780
    assert parse_clock_time("not a time") is None
    assert parse_clock_time(None) is None
    assert parse_clock_time("25:00") is None
    assert parse_clock_time("09:60 AM") is None


def test_detect_overlaps_finds_clashes_both_ways():
    hits = detect_overlaps([
        ("a", "04:30 PM", "08:30 PM"),
        ("b", "06:00 PM", "09:30 PM"),
        ("c", "09:00 AM", "10:00 AM"),
    ])
    assert hits == {"a": ["b"], "b": ["a"]}


def test_detect_overlaps_adjacent_is_fine():
    hits = detect_overlaps([
        ("a", "09:00 AM", "10:00 AM"),
        ("b", "10:00 AM", "11:00 AM"),
    ])
    assert hits == {}


def test_detect_overlaps_skips_bad_times():
    hits = detect_overlaps([
        ("a", "09:00 AM", "10:00 AM"),
        ("b", None, "11:00 AM"),
        ("c", "10:00 PM", "02:00 AM"),
    ])
    assert hits == {}


# ---------------------------------------------------------------------------
# Serialization through the existing API response (dev PostgreSQL)
# ---------------------------------------------------------------------------

def test_route_metadata_flows_through_trip_dict(monkeypatch):
    from backend.api.routes import _trip_dict
    from backend.database.connection import SessionLocal
    from backend.itinerary.generator import ItineraryGenerator
    from backend.models.models import (
        Activity, Destination, Hotel, ItineraryItem, TransportOption, Trip,
    )
    from backend.recommendation.engine import RecommendationEngine

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest = Destination(
            id=f"dst-route-{tag}", name=f"Routeburg {tag}",
            slug=f"routeburg-{tag}", country="India",
            state_region=f"Routeburg {tag}", description="Route test.",
            latitude=10.0, longitude=20.0, inventory_source="catalog",
            verification_status="catalog_verified",
        )
        db.add(dest)
        db.flush()
        hotel = Hotel(
            id=f"htl-route-{tag}", destination_id=dest.id,
            name=f"Routeburg Stay {tag}", price_per_night=1000.0,
            currency="INR", rating=4.5, latitude=10.0, longitude=20.0,
            inventory_source="catalog", verification_status="catalog_verified",
            is_active=True,
        )
        transport = TransportOption(
            id=f"trn-route-{tag}", destination_id=dest.id, type="private_cab",
            name=f"Routeburg Cab {tag}", route_from="Origin",
            route_to=dest.name, duration_hours=2.0, price=500.0,
            currency="INR", capacity=6, inventory_source="catalog",
            verification_status="catalog_verified", is_active=True,
        )
        db.add_all([hotel, transport])
        activity_ids = []
        for idx, (lat, lng) in enumerate([(10.0, 20.0), (10.05, 20.05)]):
            activity = Activity(
                id=f"act-route-{tag}-{idx}", destination_id=dest.id,
                title=f"Routeburg Spot {tag}-{idx}", category="culture",
                duration_hours=2.0, price_per_person=100.0, currency="INR",
                rating=4.5, meeting_point=f"Routeburg {tag}",
                latitude=lat, longitude=lng, inventory_source="catalog",
                verification_status="catalog_verified", is_active=True,
            )
            db.add(activity)
            activity_ids.append(activity.id)
        db.commit()

        import backend.itinerary.generator as itinerary_generator

        class StubEngine:
            def __init__(self, db):
                pass

            def get_recommendations(self, destination_id, preferences,
                                    discovery_session_id=None):
                return {
                    "ai_insights": None,
                    "recommended_hotels": [{"id": hotel.id}],
                    "recommended_activities": [{"id": aid} for aid in activity_ids],
                    "recommended_transport": [{"id": transport.id}],
                }

        monkeypatch.setattr(itinerary_generator, "RecommendationEngine", StubEngine)
        assert RecommendationEngine  # engine import stays referenced
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Route {tag}", duration_days=3, total_budget=50000.0,
            currency="INR", traveler_count=2, pace="balanced",
        )
        db.add(trip)
        db.commit()
        ItineraryGenerator(db).generate_for_trip(trip.id)
        body = _trip_dict(trip, db)

        hotel_rows = [r for r in body["itinerary"] if r["item_type"] == "hotel"]
        activity_rows = [r for r in body["itinerary"] if r["item_type"] == "activity"]
        assert hotel_rows, "hotel records stay separate from activities"
        assert len(activity_rows) == 2 and len({r["id"] for r in activity_rows}) == 2
        for row in activity_rows:
            leg = row.get("route_leg_in")
            assert isinstance(leg, dict)
            assert leg["status"] in ("estimated", "unable_to_verify")
            assert leg["is_estimated"] is True
            if leg["status"] == "estimated":
                assert leg["estimation_method"] == "haversine_speed_estimate"
        summaries = [
            r.get("route_day_summary") for r in body["itinerary"]
            if r.get("route_day_summary")
        ]
        assert summaries
        assert all(s["status"] in (
            "Valid", "Valid-with-Warning", "Requires-Revision", "Unable-to-Verify"
        ) for s in summaries)
        assert body["selected_accommodation"] is not None
        assert body["daily_accommodations"]
        assert "cost_breakdown" in body
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
