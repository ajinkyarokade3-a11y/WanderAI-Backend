"""Overnight-anchor, clustering-verdict, and stay-validation tests.

All fixtures use synthetic coordinates only — no destination, city, or
property names influence the implementation. Geometries mirror real
failure shapes (a far-flung outlier ~500 km from a base cluster) without
hardcoding any place.
"""

from types import SimpleNamespace

from backend.itinerary.generator import ItineraryGenerator
from backend.itinerary.hotel_assignment import (
    DayAnchor,
    HOTEL_ASSIGNMENT_CONFIG,
    cluster_anchor_points,
    estimate_travel_minutes,
    haversine_km,
    plan_overnight_stays,
    validate_stay_assignment,
)
from backend.itinerary.route_plan import plan_activity_days

GEN = ItineraryGenerator.__new__(ItineraryGenerator)


def _act(aid, title, lat, lng, duration=2.0):
    return SimpleNamespace(id=aid, title=title, latitude=lat,
                           longitude=lng, duration_hours=duration)


def _hotel(hid, name, lat, lng, price=2000.0):
    return SimpleNamespace(id=hid, name=name, latitude=lat, longitude=lng,
                           price_per_night=price)


def test_1_close_activities_practical_anchor_no_extreme_warning():
    acts = [_act("a1", "Base East", 26.90, 75.80),
            _act("a2", "Base West", 26.95, 75.85)]
    anchor, diagnostics = GEN._overnight_anchor(acts, [], "Area", (None, None), 1)
    assert anchor is not None
    assert diagnostics["method"] == "centroid"
    assert diagnostics["max_spread_km"] is not None
    assert diagnostics["max_spread_km"] < 40.0
    hotel = _hotel("h1", "Base Stay", 26.91, 75.81)
    stays, spent = plan_overnight_stays(
        nights=1,
        anchors={1: anchor},
        hotels=[hotel],
    )
    assert stays[0].hotel.id == "h1"
    assert stays[0].distance_km is not None and stays[0].distance_km < 40.0
    assert stays[0].decision == "opening"
    assert "near Base East" in stays[0].reason
    assert "120" not in stays[0].reason
    assert spent == 2000.0


def test_2_scattered_activities_use_subgroup_fallback_deterministically():
    north_a = _act("na", "North A", 15.33, 76.46)
    north_b = _act("nb", "North B", 15.91, 75.68)
    south = _act("so", "South", 12.30, 76.65)
    first, diag_first = GEN._overnight_anchor(
        [north_a, north_b, south], [], "Area", (None, None), 1)
    second, diag_second = GEN._overnight_anchor(
        [north_a, north_b, south], [], "Area", (None, None), 1)
    assert diag_first["method"] == "subgroup-fallback"
    assert diag_first["max_spread_km"] is not None
    assert diag_first["max_spread_km"] > HOTEL_ASSIGNMENT_CONFIG[
        "significant_distance_km"]
    # Same inputs always produce the same anchor.
    assert (first.latitude, first.longitude) == (
        second.latitude, second.longitude)
    # The anchor sits on the largest subgroup (the northern pair), not on
    # the artificial midpoint between the regions.
    assert abs(first.latitude - 15.62) < 0.01
    assert abs(first.longitude - 76.07) < 0.01
    # A hotel near the subgroup is genuinely near the anchor.
    hotel = _hotel("h1", "North Stay", 15.33, 76.46)
    stays, _spent = plan_overnight_stays(
        nights=1, anchors={1: first}, hotels=[hotel])
    assert stays[0].hotel.id == "h1"
    assert stays[0].distance_km is not None
    assert stays[0].distance_km < 60.0


def test_3_single_linkage_chain_documented():
    # A-B and B-C each within range while A-C exceeds it: one chained
    # region (documented behavior, verified safe for stay planning).
    regions = cluster_anchor_points([
        ("a", (0.0, 0.0)),
        ("b", (0.0, 0.9)),   # ~100 km from A
        ("c", (0.0, 1.8)),   # ~100 km from B, ~200 km from A
    ])
    assert regions["a"] == regions["b"] == regions["c"]
    # ...yet hotel selection still follows real distances: an anchor at C
    # moves to the C hotel despite the shared region.
    hotel_a = _hotel("ha", "Stay A", 0.0, 0.0)
    hotel_c = _hotel("hc", "Stay C", 0.0, 1.8)
    stays, _spent = plan_overnight_stays(
        nights=2,
        anchors={1: DayAnchor(0.0, 0.0, "A"),
                 2: DayAnchor(0.0, 1.8, "C")},
        hotels=[hotel_a, hotel_c],
        night_regions={1: 0, 2: 0},
    )
    assert stays[0].hotel.id == "ha"
    assert stays[1].hotel.id == "hc"
    assert stays[1].retained is False


def test_3b_close_and_distant_and_single_and_same_city():
    close = cluster_anchor_points([
        ("a", (10.0, 20.0)), ("b", (10.05, 20.05))])
    assert close["a"] == close["b"]
    distant = cluster_anchor_points([
        ("a", (10.0, 20.0)), ("b", (30.0, 40.0))])
    assert distant["a"] != distant["b"]
    single = cluster_anchor_points([("only", (10.0, 20.0))])
    assert single == {"only": 0}
    same_city = cluster_anchor_points([
        ("a", (10.0, 20.0)), ("b", (10.0, 20.0)), ("c", (10.01, 20.0))])
    assert len(set(same_city.values())) == 1
    broken = cluster_anchor_points([
        ("ok", (10.0, 20.0)),
        ("none", (None, None)),
        ("bad", ("x", "y")),
        ("oor", (999.0, 20.0)),
    ])
    assert broken == {"ok": 0}


def test_4_missing_coordinates_skip_safely_with_dest_fallback():
    ghost = _act("g", "Ghost", None, None)
    anchor, diagnostics = GEN._overnight_anchor(
        [ghost], [], "Fallback Area", (27.0, 74.2), 2)
    assert anchor is not None
    assert (anchor.latitude, anchor.longitude) == (27.0, 74.2)
    assert diagnostics["method"] == "destination-fallback"
    anchor2, diagnostics2 = GEN._overnight_anchor(
        [], [], "Fallback Area", (None, None), 2)
    assert anchor2 is None
    assert diagnostics2["method"] == "none"


def test_5_pinned_hotel_retained_but_travel_validated():
    pinned = _hotel("h-pin", "Pinned Stay", 10.0, 20.0, price=9000.0)
    anchor = DayAnchor(latitude=15.0, longitude=25.0, label="Far Area")
    stays, spent = plan_overnight_stays(
        nights=1,
        anchors={1: anchor},
        hotels=[pinned],
        pinned_hotels={1: pinned},
        total_pot=100.0,  # pinned nights cost nothing extra upstream
    )
    assert stays[0].hotel.id == "h-pin"
    assert stays[0].retained is True
    assert spent == 0.0
    report = validate_stay_assignment(stays[0], anchor, [pinned])
    assert report["distance_km"] is not None and report["distance_km"] > 300.0
    assert report["retention_consistent"] is True  # pinned: nothing to compare
    assert report["issues"] == []


def test_6_no_nearby_hotel_fallback_preserved_with_explanation():
    only = _hotel("h-only", "Only Stay", 10.0, 20.0)
    anchor = DayAnchor(latitude=15.0, longitude=25.0, label="Far Area")
    stays, _spent = plan_overnight_stays(
        nights=2, anchors={1: anchor, 2: anchor}, hotels=[only])
    assert [s.hotel.id for s in stays] == ["h-only", "h-only"]
    assert stays[1].retained is True
    assert "limited" in stays[1].reason or "closest verified" in stays[1].reason
    report = validate_stay_assignment(stays[1], anchor, [only])
    assert report["retention_consistent"] is True
    assert report["within_preferred"] is False


def test_7_remote_outlier_warning_preserved_with_correct_math():
    # Mirrors the reported failure geometry with synthetic names: a base
    # pair plus an outlier ~500 km away, thin single-hotel inventory.
    base_a = _act("base-a", "Base Alpha", 26.90, 75.80)
    base_b = _act("base-b", "Base Beta", 26.95, 75.85)
    outlier = _act("out", "Far Outlier", 26.81, 70.51)
    plan = plan_activity_days([base_a, base_b, outlier], 3, pace="balanced")
    assert set(plan.activity_day) == {"base-a", "base-b", "out"}
    out_day = plan.days[plan.activity_day["out"]]
    assert any("one-way" in w for w in out_day.warnings)
    assert out_day.status in ("Valid-with-Warning", "Requires-Revision")
    # The warned minutes match the real nearest-stop distance.
    expected_min = estimate_travel_minutes(
        min(haversine_km(26.81, 70.51, 26.90, 75.80),
            haversine_km(26.81, 70.51, 26.95, 75.85)), 50.0)
    assert any(f"{expected_min:.0f}" in w for w in out_day.warnings)
    # Hotel layer retains honestly: no nearer hotel exists.
    base_hotel = _hotel("h-base", "Base Stay", 26.91, 75.81)
    anchor, _diagnostics = GEN._overnight_anchor(
        [outlier], [], "Area", (None, None), 3)
    stays, spent = plan_overnight_stays(
        nights=2, anchors={1: anchor, 2: anchor}, hotels=[base_hotel],
        total_pot=100000.0)
    assert [s.hotel.id for s in stays] == ["h-base", "h-base"]
    assert stays[1].retained is True
    assert stays[1].distance_km is not None and stays[1].distance_km > 400.0
    report = validate_stay_assignment(
        stays[1], anchor, [base_hotel], affordable_ids=["h-base"])
    assert report["retention_consistent"] is True
    assert spent == 4000.0
