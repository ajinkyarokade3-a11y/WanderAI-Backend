"""Regional live hotel discovery tests (provider mocked, no network).

Covers: multi-region survival, one empty region, cross-region dedup,
missing coordinates, verification gates, session isolation. All fixtures
use synthetic coordinates only.
"""

import uuid
from types import SimpleNamespace

import pytest

from backend.database.connection import SessionLocal
from backend.live_fill.regional_hotels import discover_regional_hotels
from backend.models.models import Destination, Hotel

MYSURU = (12.30, 76.65)
HAMPI = (15.33, 76.46)
JOG = (14.22, 74.81)


def _region(lat, lng, label, minutes=180.0):
    return {"latitude": lat, "longitude": lng, "label": label,
            "activity_minutes": minutes}


def _hotel_row(name, lat, lng, price=2000.0):
    return SimpleNamespace(id=f"row-{name}", name=name, latitude=lat,
                           longitude=lng, price_per_night=price)


def _search_factory(results_by_query, calls=None):
    def _search(query):
        if calls is not None:
            calls.append(query)
        return [dict(r) for r in results_by_query.get(query, [])]
    return _search


def _reverse_factory(mapping):
    def _reverse(lat, lng, *args):
        for (plat, plng), place in mapping.items():
            if abs(lat - plat) < 0.5 and abs(lng - plng) < 0.5:
                return dict(place)
        return None
    return _reverse


def _live(name, lat, lng, price=2500.0):
    return {"name": name, "price_per_night": price, "rating": 4.2,
            "image_url": None, "amenities": [], "description": None,
            "latitude": lat, "longitude": lng, "hotel_class": 3,
            "location": "Test Town"}


def _make_destination(db, tag):
    dest = Destination(
        id=f"dst-reg-{tag}", name=f"Regionville {tag}",
        slug=f"regionville-{tag}", country="India",
        state_region=f"Regionville {tag}", description="Regional test.",
        latitude=13.0, longitude=76.0, inventory_source="catalog",
        verification_status="catalog_verified",
    )
    db.add(dest)
    db.commit()
    return dest


def _cleanup(db, dest):
    dest_id = getattr(dest, "id", dest)
    if not isinstance(dest_id, str):
        dest_id = str(dest_id)
    try:
        db.query(Hotel).filter(Hotel.destination_id == dest_id).delete()
        db.query(Destination).filter(Destination.id == dest_id).delete()
        db.commit()
    finally:
        db.close()


def test_a_multiple_regions_all_survive():
    """Test A — Mysuru/Hampi/Jog Falls regions each yield candidates."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        search = _search_factory({
            "Mysuru Town": [_live("Mysuru Stay", 12.31, 76.66)],
            "Hampi Town": [_live("Hampi Stay", 15.34, 76.46),
                           _live("Hampi Riverside", 15.35, 76.47)],
            "Sagara Town": [_live("Jog Stay", 14.20, 74.85)],
        })
        reverse = _reverse_factory({
            MYSURU: {"locality": "Mysuru Town", "broader": ""},
            HAMPI: {"locality": "Hampi Town", "broader": ""},
            JOG: {"locality": "Sagara Town", "broader": ""},
        })
        report = discover_regional_hotels(
            db, destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[
                _region(*MYSURU, "Mysuru area"),
                _region(*HAMPI, "Hampi area"),
                _region(*JOG, "Jog area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse)
        assert len(report["added"]) == 4
        rows = db.query(Hotel).filter(
            Hotel.destination_id == dest.id).all()
        assert len(rows) == 4
        assert all(h.inventory_source == "live" for h in rows)
        assert all(h.latitude is not None and h.longitude is not None
                   for h in rows)
        assert all(float(h.price_per_night or 0) > 0 for h in rows)
        assert report["per_region"]["Hampi area"]["kept"] == 2
    finally:
        _cleanup(db, dest)


def test_b_one_empty_region_does_not_fail_others():
    """Test B — a region with zero results never sinks the batch."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        search = _search_factory({
            "Mysuru Town": [_live("Mysuru Stay", 12.31, 76.66)],
            "Nowhere Town": [],
        })
        reverse = _reverse_factory({
            MYSURU: {"locality": "Mysuru Town", "broader": "Nowhere Town"},
            JOG: {"locality": "Nowhere Town", "broader": ""},
        })
        report = discover_regional_hotels(
            db, destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[_region(*MYSURU, "Mysuru area"),
                              _region(*JOG, "Jog area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse)
        assert len(report["added"]) == 1
        assert report["per_region"]["Mysuru area"]["kept"] == 1
    finally:
        _cleanup(db, dest)


def test_c_duplicate_property_across_regions_kept_once():
    """Test C — same property from two searches persists once."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        search = _search_factory({
            "Mysuru Town": [_live("Grand Chain Hotel", 13.15, 76.83)],
            "Far Town": [_live("Grand Chain Hotel", 13.15, 76.83)],
        })
        reverse = _reverse_factory({
            MYSURU: {"locality": "Mysuru Town", "broader": ""},
            (14.00, 77.00): {"locality": "Far Town", "broader": ""},
        })
        report = discover_regional_hotels(
            db, destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[_region(*MYSURU, "Mysuru area"),
                              _region(14.00, 77.00, "Far area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse)
        rows = db.query(Hotel).filter(
            Hotel.destination_id == dest.id).all()
        assert len(rows) == 1
        total_dups = sum(v["duplicates"]
                         for v in report["per_region"].values())
        assert total_dups == 1
    finally:
        _cleanup(db, dest)


def test_d_missing_coordinates_never_invented():
    """Test D — unresolvable coordinates are skipped, never fabricated."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        search = _search_factory({
            "Hampi Town": [
                _live("Ghost Hotel", None, None),
                _live("Hampi Stay", 15.34, 76.46)],
        })
        reverse = _reverse_factory({
            HAMPI: {"locality": "Hampi Town", "broader": ""},
        })
        report = discover_regional_hotels(
            db, destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[_region(*HAMPI, "Hampi area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse,
            geocode_fn=lambda *args: None)
        rows = db.query(Hotel).filter(
            Hotel.destination_id == dest.id).all()
        assert [h.name for h in rows] == ["Hampi Stay"]
        assert report["per_region"]["Hampi area"]["no_coords"] == 1
    finally:
        _cleanup(db, dest)


def test_e_verification_gates_hold():
    """Test E — nameless/priceless results never reach the catalog."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        search = _search_factory({
            "Hampi Town": [
                {"name": "", "price_per_night": 2000.0,
                 "latitude": 15.34, "longitude": 76.46},
                {"name": "Priceless Inn", "price_per_night": None,
                 "latitude": 15.34, "longitude": 76.46},
                {"name": "Free Stay", "price_per_night": 0,
                 "latitude": 15.34, "longitude": 76.46},
                _live("Hampi Stay", 15.34, 76.46),
            ],
        })
        reverse = _reverse_factory({
            HAMPI: {"locality": "Hampi Town", "broader": ""},
        })
        report = discover_regional_hotels(
            db, destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[_region(*HAMPI, "Hampi area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse)
        rows = db.query(Hotel).filter(
            Hotel.destination_id == dest.id).all()
        assert [h.name for h in rows] == ["Hampi Stay"]
        assert report["per_region"]["Hampi area"]["no_price"] == 2
    finally:
        _cleanup(db, dest)


def test_f_session_isolation_between_destinations():
    """Test F — Karnataka discoveries never leak into another destination."""
    db = SessionLocal()
    tag = uuid.uuid4().hex[:8]
    dest_a = _make_destination(db, f"a-{tag}")
    dest_b = _make_destination(db, f"b-{tag}")
    dest_a_id, dest_b_id = dest_a.id, dest_b.id
    try:
        search = _search_factory({
            "Hampi Town": [_live("Hampi Stay", 15.34, 76.46)],
        })
        reverse = _reverse_factory({
            HAMPI: {"locality": "Hampi Town", "broader": ""},
        })
        discover_regional_hotels(
            db, destination=dest_a, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[_region(*HAMPI, "Hampi area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse)
        assert db.query(Hotel).filter(
            Hotel.destination_id == dest_a.id).count() == 1
        assert db.query(Hotel).filter(
            Hotel.destination_id == dest_b.id).count() == 0
    finally:
        db.query(Hotel).filter(
            Hotel.destination_id.in_([dest_a_id, dest_b_id])).delete(
                synchronize_session=False)
        db.query(Destination).filter(
            Destination.id.in_([dest_a_id, dest_b_id])).delete(
                synchronize_session=False)
        db.commit()
        db.close()


def test_g_far_away_filler_results_rejected():
    """Provider filler from another state is rejected, never persisted."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        search = _search_factory({
            "Hampi Town": [
                _live("Goa Beach Villa", 15.45, 73.81),
                _live("Island Villa", 6.83, 79.96),
                _live("Hampi Stay", 15.34, 76.46)],
        })
        reverse = _reverse_factory({
            HAMPI: {"locality": "Hampi Town", "broader": ""},
        })
        report = discover_regional_hotels(
            db, destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            region_centroids=[_region(*HAMPI, "Hampi area")],
            existing_hotels=[],
            search_fn=search, reverse_fn=reverse)
        rows = db.query(Hotel).filter(
            Hotel.destination_id == dest.id).all()
        assert [h.name for h in rows] == ["Hampi Stay"]
        assert report["per_region"]["Hampi area"]["too_far"] == 2
    finally:
        _cleanup(db, dest)


def test_search_regions_derive_per_day_centroids_from_activities():
    """Search anchors come from planned days' real activity coordinates."""
    from types import SimpleNamespace

    from backend.itinerary.generator import ItineraryGenerator
    from backend.itinerary.route_plan import DayRoute, RoutePlan

    acts = [SimpleNamespace(id="a1", title="Hampi Ruins", latitude=15.33,
                            longitude=76.46, duration_hours=4.0),
            SimpleNamespace(id="a2", title="Mysore Palace", latitude=12.30,
                            longitude=76.65, duration_hours=3.0)]
    plan = RoutePlan(days={1: DayRoute(day=1, activity_ids=["a1"]),
                           2: DayRoute(day=2, activity_ids=["a2"])})
    regions = ItineraryGenerator._search_regions(plan, acts)
    assert len(regions) == 2
    by_label = {r["label"]: r for r in regions}
    assert by_label["Hampi Ruins area"]["latitude"] == pytest.approx(15.33)
    assert by_label["Hampi Ruins area"]["activity_minutes"] == pytest.approx(240.0)
    assert by_label["Mysore Palace area"]["activity_minutes"] == pytest.approx(180.0)


def test_covered_regions_skip_search_and_reruns_add_nothing():
    """Covered clusters cost no provider call; reruns are idempotent."""
    db = SessionLocal()
    dest = _make_destination(db, uuid.uuid4().hex[:8])
    try:
        calls: list = []
        search = _search_factory({
            "Hampi Town": [_live("Hampi Stay", 15.34, 76.46)],
        }, calls=calls)
        reverse = _reverse_factory({
            MYSURU: {"locality": "Mysuru Town", "broader": ""},
            HAMPI: {"locality": "Hampi Town", "broader": ""},
        })
        kwargs = dict(destination=dest, currency="INR", traveler_count=2,
                      check_in="2026-10-20", check_out="2026-10-22",
                      existing_hotels=[_hotel_row("Base", *MYSURU)],
                      search_fn=search, reverse_fn=reverse)
        first = discover_regional_hotels(
            db, region_centroids=[_region(*MYSURU, "Mysuru area"),
                                  _region(*HAMPI, "Hampi area")], **kwargs)
        assert first["skipped_covered"] == ["Mysuru area"]
        assert len(first["added"]) == 1 and calls == ["Hampi Town"]
        second = discover_regional_hotels(
            db, region_centroids=[_region(*MYSURU, "Mysuru area"),
                                  _region(*HAMPI, "Hampi area")],
            existing_hotels=[_hotel_row("Base", *MYSURU),
                             _hotel_row("Hampi Stay", 15.34, 76.46)],
            destination=dest, currency="INR", traveler_count=2,
            check_in="2026-10-20", check_out="2026-10-22",
            search_fn=search, reverse_fn=reverse)
        assert second["added"] == []
        assert db.query(Hotel).filter(
            Hotel.destination_id == dest.id).count() == 1
    finally:
        _cleanup(db, dest)
