"""Live inventory fill tests — SerpApi/OSM providers complete catalog gaps.

Provider calls are mocked; no live network is used. Covers:
- destination with transport but no hotels/activities -> trip creates (200)
  with rows marked inventory_source="live"
- destination with nothing (transport missing) -> still 422, honest error
- no SerpApi key -> still 422 (hotels cannot be filled)
- fill is idempotent (no duplicate rows on retry)
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.database.config import settings
from backend.database.connection import SessionLocal
from backend.main import app
from backend.models.models import Activity, Destination, Hotel, TransportOption

client = TestClient(app)


def _fake_hotels(*args, **kwargs):
    return [
        {"name": "Live Grand Ziro", "price_per_night": 6000.0, "rating": 4.3,
         "image_url": "https://example.com/h1.jpg", "amenities": ["WiFi"],
         "description": "Live listed hotel", "latitude": 27.6, "longitude": 93.8,
         "hotel_class": 4, "location": "Ziro"},
        {"name": "Live Valley Inn", "price_per_night": 3500.0, "rating": 4.0,
         "image_url": None, "amenities": [], "description": None,
         "latitude": 27.61, "longitude": 93.81, "hotel_class": 3, "location": "Ziro"},
    ]


def _fake_places(*args, **kwargs):
    return {"destination": "Ziro Test", "latitude": 27.6, "longitude": 93.8,
            "places": [
                {"name": "Talley Valley Viewpoint", "latitude": 27.6,
                 "longitude": 93.8, "kind": "viewpoint",
                 "image_url": "https://example.com/p1.jpg"},
                {"name": "Ziro Heritage Museum", "latitude": 27.61,
                 "longitude": 93.81, "kind": "museum", "image_url": None},
                {"name": "Old Ziro Market Walk", "latitude": 27.62,
                 "longitude": 93.82, "kind": "market", "image_url": None},
                {"name": "Meghna Cave Trail", "latitude": 27.63,
                 "longitude": 93.83, "kind": "trekking", "image_url": None},
            ], "source": "overpass+commons"}


@pytest.fixture()
def _live_providers(monkeypatch):
    import backend.hotels.service as hotels_service
    import backend.places.service as places_service
    monkeypatch.setattr(hotels_service, "search_serpapi_hotels", _fake_hotels)
    monkeypatch.setattr(places_service, "get_live_places", _fake_places)
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "test-key")


def _make_bare_destination(with_transport=True):
    db = SessionLocal()
    tag = uuid.uuid4().hex[:8]
    dest = Destination(id=f"dest-test-{tag}", name=f"Ziro Test {tag}",
                       slug=f"ziro-test-{tag}", country="India",
                       state_region="Arunachal Pradesh",
                       description="Test destination for live fill.",
                       latitude=27.6, longitude=93.8)
    db.add(dest)
    db.flush()
    if with_transport:
        db.add(TransportOption(
            id=f"trn-test-{tag}", destination_id=dest.id, vendor_id=None,
            type="private_cab", name="Test Valley Cab",
            route_from="Test Airport", route_to="Test Town",
            duration_hours=2.0, price=3000.0, currency="INR", capacity=4,
            features=[], is_active=True))
    db.commit()
    dest_id = dest.id
    db.close()
    return dest_id


def _delete_destination(dest_id):
    db = SessionLocal()
    try:
        for trip in db.query(__import__("backend.models.models", fromlist=["Trip"]).Trip).filter_by(destination_id=dest_id).all():
            db.delete(trip)
        db.query(TransportOption).filter(TransportOption.destination_id == dest_id).delete()
        db.query(Destination).filter(Destination.id == dest_id).delete()
        db.commit()
    finally:
        db.close()


def test_trip_creates_from_live_fill(_live_providers):
    dest_id = _make_bare_destination(with_transport=True)
    try:
        r = client.post("/api/trips", json={
            "title": "Ziro Live Trip", "destination_id": dest_id,
            "duration_days": 3, "traveler_count": 2,
            "total_budget": 50000.0, "currency": "INR",
        })
        assert r.status_code == 200, r.text
        assert len(r.json()["itinerary"]) > 0
        db = SessionLocal()
        try:
            hotels = db.query(Hotel).filter(Hotel.destination_id == dest_id).all()
            acts = db.query(Activity).filter(Activity.destination_id == dest_id).all()
            assert len(hotels) == 2 and all(h.inventory_source == "live" for h in hotels)
            assert len(acts) >= 2 and all(a.inventory_source == "live" for a in acts)
            assert all(a.verification_status == "live_provider" for a in acts)
            assert any(h.images for h in hotels)  # real photo kept
        finally:
            db.close()
    finally:
        _delete_destination(dest_id)


def test_trip_creates_without_transport_when_none_exists(_live_providers):
    """Transport has no live provider and is traveler-arranged: its absence
    no longer blocks creation; the itinerary just has no transfer items."""
    dest_id = _make_bare_destination(with_transport=False)
    try:
        r = client.post("/api/trips", json={
            "title": "No Transfer Trip", "destination_id": dest_id,
            "duration_days": 3, "traveler_count": 2,
            "total_budget": 50000.0, "currency": "INR",
        })
        assert r.status_code == 200, r.text
        items = r.json()["itinerary"]
        assert len(items) > 0
        assert all(i["item_type"] != "transport" for i in items)
    finally:
        _delete_destination(dest_id)


def test_unknown_place_builds_from_live_when_gemini_down(_live_providers, monkeypatch):
    """Unknown-place replay (Assam scenario): Gemini discovery raises,
    Nominatim + SerpApi/OSM build a real trip anyway. Uses a unique name so
    repeated runs never collide with real destinations. No invented data."""
    import uuid as _uuid
    from backend.dynamic_destination.service import DynamicDestinationDiscoveryService
    import backend.places.service as places_service

    place = f"Ziro Test {_uuid.uuid4().hex[:6]}"

    def _gemini_down(*args, **kwargs):
        raise RuntimeError("Gemini destination inventory research failed")

    monkeypatch.setattr(DynamicDestinationDiscoveryService, "discover_and_persist", _gemini_down)
    monkeypatch.setattr(places_service, "geocode_place",
                        lambda name, *a, **k: (27.6, 93.8, f"{name}, Arunachal Pradesh, India"))
    r = client.post("/api/trips", json={
        "title": "Unknown Place Getaway",
        "destination_name": place,
        "duration_days": 5,
        "traveler_count": 4,
        "total_budget": 80000.0,
        "currency": "INR",
        "start_date": "2026-10-16T00:00:00",
        "end_date": "2026-10-20T00:00:00",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["destination"]["inventory_source"] == "live"
    assert body["duration_days"] == 5
    assert len(body["itinerary"]) > 0
    db = SessionLocal()
    try:
        dest = db.query(Destination).filter(Destination.id == body["destination"]["id"]).one()
        assert dest.country == "India"  # from geocoder display name, not invented
    finally:
        db.close()
    _delete_destination(body["destination"]["id"])


def test_no_serpapi_key_still_fails_honestly(_live_providers, monkeypatch):
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "")
    dest_id = _make_bare_destination(with_transport=True)
    try:
        r = client.post("/api/trips", json={
            "title": "Keyless Trip", "destination_id": dest_id,
            "duration_days": 3, "traveler_count": 2,
            "total_budget": 50000.0, "currency": "INR",
        })
        assert r.status_code == 422
        assert "hotel" in r.json()["detail"].lower()
    finally:
        _delete_destination(dest_id)


def test_hotel_search_caches_successes(monkeypatch):
    """Repeat searches share quota: one HTTP call serves both."""
    import backend.hotels.service as hotels_service

    calls = []

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"properties": [{
                "name": "Cache Hotel", "property_token": "tok1",
                "gps_coordinates": {"latitude": 1.0, "longitude": 2.0},
                "rate_per_night": {"extracted_lowest": 5000},
                "images": [{"original_image": "https://example.com/c.jpg"}],
                "amenities": ["WiFi"], "overall_rating": 4.2, "hotel_class": 4}]}

    def _fake_get(url, params, timeout):
        calls.append(params)
        return _Resp()

    monkeypatch.setattr(hotels_service, "_http_get", _fake_get)
    from backend.live_fill.service import _cached_hotel_search
    kwargs = dict(api_key="k", destination="Cacheville",
                  check_in="2026-11-01", check_out="2026-11-04",
                  adults=2, currency="INR")
    first = _cached_hotel_search(**kwargs)
    second = _cached_hotel_search(**kwargs)
    assert len(calls) == 1
    assert first == second and first[0]["name"] == "Cache Hotel"


def test_quota_exhaustion_hint(_live_providers, monkeypatch):
    """SerpApi 429 surfaces as a quota hint, not a bare inventory error."""
    import backend.hotels.service as hotels_service
    from backend.hotels.service import SerpApiError

    def _quota(*args, **kwargs):
        raise SerpApiError("Hotel search failed: Client error '429 Too Many Requests'")

    monkeypatch.setattr(hotels_service, "search_serpapi_hotels", _quota)
    dest_id = _make_bare_destination(with_transport=True)
    try:
        r = client.post("/api/trips", json={
            "title": "Quota Trip", "destination_id": dest_id,
            "duration_days": 3, "traveler_count": 2,
            "total_budget": 50000.0, "currency": "INR",
        })
        assert r.status_code == 422
        assert "quota" in r.json()["detail"].lower()
    finally:
        _delete_destination(dest_id)


def test_places_retry_tight_radius(monkeypatch):
    """Wide-query timeout (empty) retries once tight and recovers."""
    import backend.places.service as places_service

    calls = []

    def _flaky(lat, lng, url, timeout_s, radius_m, limit):
        calls.append(radius_m)
        if len(calls) == 1:
            return []
        return [{"name": "Tight Spot", "latitude": lat, "longitude": lng, "kind": "viewpoint"}]

    monkeypatch.setattr(places_service, "fetch_attractions", _flaky)
    monkeypatch.setattr(places_service, "fetch_place_images", lambda *a, **k: [])
    out = places_service.get_live_places("Delhi", 28.6, 77.2, 5,
                                         "http://x", "http://x", "http://x", 25, 30000)
    assert [p["name"] for p in out["places"]] == ["Tight Spot"]
    assert calls == [30000, 10000]


def test_empty_places_signals_provider_issue(monkeypatch):
    """Zero attractions everywhere surfaces as a fill error (drives hint)."""
    import backend.places.service as places_service
    from backend.database.config import settings
    from backend.live_fill.service import fill_destination_inventory

    monkeypatch.setattr(places_service, "fetch_attractions", lambda *a, **k: [])
    monkeypatch.setattr(settings, "SERPAPI_API_KEY", "")
    dest_id = _make_bare_destination(with_transport=True)
    try:
        db = SessionLocal()
        try:
            from backend.models.models import Destination
            dest = db.query(Destination).filter(Destination.id == dest_id).one()
            out = fill_destination_inventory(
                db, dest, currency="INR", traveler_count=2, duration_days=3)
            assert out["activities"] == 0
            assert out["activity_error"] is not None
        finally:
            db.close()
    finally:
        _delete_destination(dest_id)


def test_fill_is_idempotent(_live_providers):
    from backend.live_fill.service import fill_destination_inventory
    dest_id = _make_bare_destination(with_transport=True)
    try:
        db = SessionLocal()
        try:
            dest = db.query(Destination).filter(Destination.id == dest_id).one()
            first = fill_destination_inventory(
                db, dest, currency="INR", traveler_count=2, duration_days=3)
            second = fill_destination_inventory(
                db, dest, currency="INR", traveler_count=2, duration_days=3)
            assert first["hotels"] == 2 and first["activities"] == 4
            assert (second["hotels"], second["activities"]) == (0, 0)
            assert second["hotel_error"] is None and second["activity_error"] is None
            assert db.query(Hotel).filter(Hotel.destination_id == dest_id).count() == 2
        finally:
            db.close()
    finally:
        _delete_destination(dest_id)
