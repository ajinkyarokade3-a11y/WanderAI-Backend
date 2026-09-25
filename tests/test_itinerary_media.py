"""Regression tests for the itinerary media pipeline.

Covers the full chain for every itinerary item type:
imageless place items gain one relevant provider photo at creation
(SerpApi "<place>, <destination>", cached, persisted on meta_data.ui);
traveler mutations keep photos relevant; reads fall back to the
destination-level photo only as a last resort; transport/notes stay
imageless by design; broken seed URLs never return.
"""
from datetime import datetime
from types import SimpleNamespace

from backend.api.routes import _trip_dict
from backend.database.connection import SessionLocal
from backend.images.service import backfill_missing_place_images, clear_image_cache
from backend.models.models import Destination, ItineraryItem, Trip
import backend.images.service as images_service


HERO = "https://example.test/udaipur-hero.jpg"
PLACE_PHOTO = "https://example.test/city-palace.jpg"


def _place_item(item_type, title, image_url=None):
    ui = {"image_url": image_url} if image_url else {}
    return SimpleNamespace(item_type=item_type, title=title,
                           meta_data={"ui": ui} if ui else {"ui": {}})


def _mock_provider(monkeypatch, calls, url=PLACE_PHOTO):
    def fake(location, destination=None, api_key="", base_url="",
             timeout_s=10.0, count=1):
        calls.append((location, destination))
        return [url]
    monkeypatch.setattr(images_service, "get_real_images_for_location", fake)


def test_backfill_fills_each_place_type_once_per_unique_title(monkeypatch):
    clear_image_cache()
    calls: list = []
    _mock_provider(monkeypatch, calls)
    items = [
        _place_item("hotel", "Lake Pichola Palace Hotel"),
        _place_item("activity", "City Palace Tour"),
        _place_item("meal", "Mewari Thali House"),
        _place_item("activity", "City Palace Tour"),  # duplicate title
        _place_item("transport", "Airport Cab"),  # not a place: skipped
        _place_item("note", "Departure note"),  # not a place: skipped
        _place_item("hotel", "Has Photo Already", image_url="https://example.test/keep.jpg"),
    ]
    filled = backfill_missing_place_images(items, "Udaipur", api_key="key",
                                           base_url="https://serpapi.com")
    assert filled == 4  # 3 unique titles + duplicate title item
    assert len(calls) == 3  # one provider call per unique place
    assert items[0].meta_data["ui"]["image_url"] == PLACE_PHOTO
    assert items[1].meta_data["ui"]["image_url"] == PLACE_PHOTO
    assert items[2].meta_data["ui"]["image_url"] == PLACE_PHOTO
    assert items[3].meta_data["ui"]["image_url"] == PLACE_PHOTO
    assert "image_url" not in items[4].meta_data["ui"]
    assert "image_url" not in items[5].meta_data["ui"]
    assert items[6].meta_data["ui"]["image_url"] == "https://example.test/keep.jpg"
    assert all("Udaipur" in (query[1] or "") for query in calls)


def test_backfill_never_raises_and_needs_no_key(monkeypatch):
    assert backfill_missing_place_images(None, "Udaipur", api_key="k") == 0
    assert backfill_missing_place_images([], "Udaipur", api_key="k") == 0
    calls: list = []
    _mock_provider(monkeypatch, calls)
    items = [_place_item("hotel", "Some Hotel")]
    assert backfill_missing_place_images(items, "Udaipur", api_key="") == 0
    assert calls == []
    assert "image_url" not in items[0].meta_data["ui"]

    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(images_service, "get_real_images_for_location", boom)
    assert backfill_missing_place_images(items, "Udaipur", api_key="k") == 0
    assert "image_url" not in items[0].meta_data["ui"]


def _media_trip():
    now = datetime(2026, 1, 1, 12, 0, 0)
    dest = Destination(id="dest-media-test", name="Udaipur", slug="udaipur",
                       country="India", state_region="Rajasthan",
                       description="City of Lakes", hero_image_url=HERO,
                       best_time_to_visit="Oct-Mar", tags=[],
                       latitude=24.58, longitude=73.71, source_url=None,
                       evidence=[], inventory_source="catalog",
                       verification_status="catalog_verified",
                       discovery_session_id=None, is_featured=False,
                       created_at=now)
    trip = Trip(id="trip-media-test", user_id=None, destination_id=dest.id,
                title="Udaipur trip", status="planning", start_date=None,
                end_date=None, duration_days=2, total_budget=50000.0,
                currency="INR", traveler_count=2, pace="balanced",
                origin="Mumbai", discovery_session_id=None,
                confirmed_at=None, confirmed_by=None,
                created_at=now, updated_at=now)
    trip.destination = dest
    trip.bookings = []
    trip.preferences = None
    trip.alerts = []
    trip.itinerary = [
        ItineraryItem(id="mi-hotel", trip_id=trip.id, day_number=1,
                      order_index=50, item_type="hotel",
                      title="Imageless Hotel", description="stay",
                      start_time=None, end_time=None, cost=8000.0,
                      status="proposed", hotel_id=None, activity_id=None,
                      transport_id=None, location="Udaipur",
                      meta_data={"ui": {}}),
        ItineraryItem(id="mi-meal", trip_id=trip.id, day_number=1,
                      order_index=20, item_type="meal",
                      title="Imageless Restaurant", description="lunch",
                      start_time=None, end_time=None, cost=900.0,
                      status="confirmed", hotel_id=None, activity_id=None,
                      transport_id=None, location="Udaipur",
                      meta_data={"ui": {}}),
        ItineraryItem(id="mi-photo", trip_id=trip.id, day_number=1,
                      order_index=10, item_type="activity",
                      title="Photo Activity", description="sightseeing",
                      start_time="10:00 AM", end_time="12:00 PM", cost=500.0,
                      status="proposed", hotel_id=None, activity_id=None,
                      transport_id=None, location="Udaipur",
                      meta_data={"ui": {"image_url": "https://example.test/keep.jpg"}}),
        ItineraryItem(id="mi-cab", trip_id=trip.id, day_number=1,
                      order_index=1, item_type="transport",
                      title="Airport Cab", description="transfer",
                      start_time="09:30 AM", end_time="10:30 AM", cost=2400.0,
                      status="proposed", hotel_id=None, activity_id=None,
                      transport_id=None, location="Udaipur",
                      meta_data={"ui": {}}),
    ]
    return trip


def test_trip_dict_hero_fallback_applies_to_places_only():
    db = SessionLocal()
    try:
        body = _trip_dict(_media_trip(), db)
    finally:
        db.close()
    by_id = {row["id"]: row for row in body["itinerary"]}
    # Imageless places show the destination-level photo (last fallback).
    assert by_id["mi-hotel"]["image_url"] == HERO
    assert by_id["mi-meal"]["image_url"] == HERO
    # A real place photo is never overridden by the generic fallback.
    assert by_id["mi-photo"]["image_url"] == "https://example.test/keep.jpg"
    # Transfers stay imageless by design (detail card, not a place).
    assert by_id["mi-cab"].get("image_url") in (None, "")


def test_seed_has_no_dead_photo_urls():
    import os

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "database", "seed_data", "seed.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert "photo-1568495286058-9c3e0b8b0e0e" not in source
    assert "htl-uda-001" in source and "act-uda-001" in source
