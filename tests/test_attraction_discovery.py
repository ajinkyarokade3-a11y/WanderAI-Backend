"""SerpApi attraction discovery + generator enrichment (offline, mocked provider).

The live SerpApi quota is an account-level constraint, so these tests stub
the HTTP boundary: they prove query discipline, normalization, dedupe,
destination association, never-raise fallbacks, caching, and the generator
integration (transient candidates with SerpApi provenance + image_url in
``meta_data.ui``) without consuming quota or inventing places.
"""
import uuid

import pytest
from alembic import command
from alembic.config import Config

from backend.attractions import service as attractions
from backend.database.connection import SessionLocal
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


@pytest.fixture(autouse=True)
def _clean_caches():
    attractions.clear_attraction_cache()
    yield
    attractions.clear_attraction_cache()


def _local_result(name="City Palace Viewpoint", address="Lake Road, Udaipur",
                  place_id="ChIJ123", lat=24.58, lng=73.68):
    return {
        "title": name,
        "place_id": place_id,
        "address": address,
        "rating": 4.6,
        "gps_coordinates": {"latitude": lat, "longitude": lng},
        "thumbnail": "https://example-cdn.test/photos/palace.jpg",
        "website": "https://example.test/palace",
        "types": ["tourist_attraction"],
    }


# --- query + normalization -------------------------------------------------

def test_build_attraction_query_names_category_and_destination():
    query = attractions.build_attraction_query("Udaipur", "museums")
    assert "museums" in query and "Udaipur" in query


def test_build_attraction_query_rejects_blank():
    with pytest.raises(ValueError):
        attractions.build_attraction_query("", "museums")


def test_normalize_keeps_real_fields_only():
    item = attractions.normalize_attraction_result(_local_result())
    assert item is not None
    assert item["name"] == "City Palace Viewpoint"
    assert item["place_id"] == "ChIJ123"
    assert item["latitude"] == 24.58 and item["longitude"] == 73.68
    assert item["image_url"] == "https://example-cdn.test/photos/palace.jpg"
    assert item["source"] == "serpapi"


def test_normalize_skips_nameless_and_non_http():
    assert attractions.normalize_attraction_result({"title": "   "}) is None
    assert attractions.normalize_attraction_result({"title": "X", "thumbnail": "ftp://x/y"})["image_url"] is None


# --- association gate -------------------------------------------------------

def test_associated_by_address_token():
    item = {"address": "Lake Pichola, Udaipur, Rajasthan"}
    assert attractions.is_associated_with_destination(item, "Udaipur") is True


def test_associated_by_coordinates():
    item = {"latitude": 24.59, "longitude": 73.69}
    assert attractions.is_associated_with_destination(item, "Udaipur", 24.58, 73.68) is True


def test_rejects_far_coordinates():
    item = {"address": "Somewhere Else", "latitude": 48.85, "longitude": 2.35}
    assert attractions.is_associated_with_destination(item, "Udaipur", 24.58, 73.68) is False


def test_rejects_unverifiable():
    item = {"name": "Mystery Spot"}
    assert attractions.is_associated_with_destination(item, "Distville Xyz") is False


# --- aggregator behavior (mocked HTTP) ---------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _stub_http(monkeypatch, calls, payloads):
    def fake_get(url, params, timeout_s):
        calls.append(params.get("q"))
        payload = payloads.get(params.get("q"), {"local_results": []})
        return _FakeResponse(payload)
    monkeypatch.setattr(attractions, "_http_get", fake_get)


def test_discover_dedupes_and_stops_early(monkeypatch):
    calls: list = []
    shared = {"local_results": [
        _local_result(name="Lake Palace", place_id="A1"),
        _local_result(name="Lake Palace", place_id="A1"),
        _local_result(name="Old Fort", place_id="B2", address="Fort Road, Udaipur"),
    ]}
    _stub_http(monkeypatch, calls, {
        "tourist attractions in Udaipur": shared,
        "museums in Udaipur": shared,
    })
    found = attractions.discover_live_attractions(
        "Udaipur", api_key="key", base_url="https://serpapi.test", max_total=2,
    )
    assert [p["name"] for p in found] == ["Lake Palace", "Old Fort"]
    assert len(calls) == 1  # early stop: second category never queried


def test_discover_exclude_names_do_not_count_toward_target(monkeypatch):
    calls: list = []
    shared = {"local_results": [
        _local_result(name="Lake Palace", place_id="A1"),
        _local_result(name="Old Fort", place_id="B2", address="Fort Road, Udaipur"),
    ]}
    _stub_http(monkeypatch, calls, {
        "tourist attractions in Udaipur": shared,
        "museums in Udaipur": {"local_results": [
            _local_result(name="City Museum", place_id="C3", address="Museum Road, Udaipur")]},
    })
    found = attractions.discover_live_attractions(
        "Udaipur", api_key="key", base_url="https://serpapi.test", max_total=2,
        exclude_names=["Lake Palace"],
    )
    assert [p["name"] for p in found] == ["Old Fort", "City Museum"]
    assert len(calls) == 2


def test_discover_never_raises_and_skips_unassociated(monkeypatch):
    def boom(url, params, timeout_s):
        raise RuntimeError("provider down")
    monkeypatch.setattr(attractions, "_http_get", boom)
    assert attractions.discover_live_attractions("Udaipur", api_key="k") == []
    assert attractions.discover_live_attractions("", api_key="k") == []
    assert attractions.discover_live_attractions("Udaipur", api_key="") == []


def test_discover_caches_per_query(monkeypatch):
    calls: list = []
    _stub_http(monkeypatch, calls, {
        "museums in Udaipur": {"local_results": [_local_result()]},
    })
    first = attractions.discover_live_attractions(
        "Udaipur", api_key="k", categories=["museums"], max_total=5)
    second = attractions.discover_live_attractions(
        "Udaipur", api_key="k", categories=["museums"], max_total=5)
    assert len(first) == 1 and first == second
    assert len(calls) == 1


# --- generator integration (stubbed discovery + images) ----------------------

def _build_trip_with_catalog(db, tag):
    dest = Destination(
        id=f"dst-enrich-{tag}", name="Enrichville", slug=f"enrichville-{tag}",
        country="India", state_region="Enrichville",
        description="Enrichment test destination.",
        inventory_source="catalog", verification_status="catalog_verified",
    )
    db.add(dest)
    db.flush()
    hotel = Hotel(
        id=f"htl-enrich-{tag}", destination_id=dest.id, name=f"Enrich Stay {tag}",
        price_per_night=1000.0, currency="INR", rating=4.5,
        inventory_source="catalog", verification_status="catalog_verified", is_active=True,
    )
    transport = TransportOption(
        id=f"trn-enrich-{tag}", destination_id=dest.id, type="private_cab",
        name=f"Enrich Cab {tag}", route_from="Origin", route_to="Enrichville",
        price=1000.0, currency="INR", capacity=6,
        inventory_source="catalog", verification_status="catalog_verified", is_active=True,
    )
    db.add_all([hotel, transport])
    activity = Activity(
        id=f"act-enrich-{tag}", destination_id=dest.id, title="Enrich Museum",
        category="culture", duration_hours=2.0, price_per_person=100.0,
        currency="INR", rating=4.5, meeting_point="Enrichville",
        inventory_source="catalog", verification_status="catalog_verified", is_active=True,
    )
    db.add(activity)
    db.commit()
    return dest, hotel, transport, activity


def test_generator_top_up_uses_transient_live_places_with_provenance(monkeypatch):
    import backend.itinerary.generator as itinerary_generator

    live_places = [
        {"id": "serpapi-attraction-P1", "place_id": "P1", "name": "Enrich Fort",
         "address": "Fort Road, Enrichville", "rating": 4.7,
         "latitude": 12.9716, "longitude": 77.5946, "website": None,
         "image_url": None, "maps_url": "https://www.google.com/maps/search/?api=1&query=Enrich+Fort",
         "types": [], "source": "serpapi"},
        {"id": "serpapi-attraction-P2", "place_id": "P2", "name": "Enrich Museum",
         "address": "Museum Road, Enrichville", "rating": 4.5,
         "latitude": 12.9717, "longitude": 77.5947, "website": None,
         "image_url": None, "maps_url": "https://www.google.com/maps/search/?api=1&query=Enrich+Museum",
         "types": [], "source": "serpapi"},
    ]
    monkeypatch.setattr(
        itinerary_generator, "discover_live_attractions", lambda *a, **k: list(live_places))
    monkeypatch.setattr(
        itinerary_generator, "get_real_image_for_location",
        lambda title, dest, *a, **k: f"https://img.test/{title.replace(' ', '_')}.jpg")

    class TripEngine:
        def __init__(self, db=None):
            pass

        def get_recommendations(self, destination_id, preferences, discovery_session_id=None):
            raise AssertionError("patched per test")

    tag = uuid.uuid4().hex[:8]
    db = SessionLocal()
    trip = None
    dest = None
    try:
        dest, hotel, transport, activity = _build_trip_with_catalog(db, tag)
        trip_engine = TripEngine()

        def _recommendations(destination_id, preferences, discovery_session_id=None):
            return {
                "ai_insights": None,
                "recommended_hotels": [{"id": hotel.id}],
                "recommended_activities": [{"id": activity.id}],
                "recommended_transport": [{"id": transport.id}],
            }

        trip_engine.get_recommendations = _recommendations  # type: ignore[method-assign]
        monkeypatch.setattr(itinerary_generator, "RecommendationEngine", lambda db: trip_engine)
        trip = Trip(
            user_id="usr-alex-morgan-001", destination_id=dest.id,
            title=f"Enrich {tag}", duration_days=3, total_budget=500000.0,
            currency="INR", traveler_count=2, pace="balanced",
        )
        db.add(trip)
        db.commit()
        items = itinerary_generator.ItineraryGenerator(db).generate_for_trip(trip.id)
        acts = [i for i in items if i.item_type == "activity"]
        by_title = {i.title: i for i in acts}
        # Catalog activity intact; duplicate live name deduplicated away.
        assert "Enrich Museum" in by_title
        assert by_title["Enrich Museum"].activity_id == activity.id
        # Live place added as transient with provenance + image.
        assert "Enrich Fort" in by_title
        fort = by_title["Enrich Fort"]
        assert fort.activity_id is None
        ui = fort.meta_data.get("ui") or {}
        assert ui.get("image_url") == "https://img.test/Enrich_Fort.jpg"
        assert any("SerpApi" in str(e.get("label", "")) for e in (ui.get("evidence") or []))
        assert ui.get("latitude") == 12.9716
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
