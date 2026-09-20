"""Regression test for the reported 'Crafting Goa Getaway' failure.

Goa shipped with a hotel but zero activities / transport options, so the
trip validator honestly refused to build Goa trips ("missing transport
options, at least 2 distinct activities"). The seed now carries real Goa
inventory; a 3-day solo Goa trip must create successfully.
"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_goa_3day_solo_trip_creates_successfully():
    goa_id = client.get("/api/destinations/goa").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Goa Getaway",
        "destination_id": goa_id,
        "duration_days": 3,
        "traveler_count": 1,
        "total_budget": 40000.0,
        "currency": "INR",
        "pace": "relaxed",
        "preferences": {"interests": ["beaches"], "travel_companions": "solo"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["destination"]["slug"] == "goa"
    assert len(body["itinerary"]) > 0
    # Every hotel/activity stop must carry a renderable photo URL.
    stops = [i for i in body["itinerary"] if i["item_type"] in ("hotel", "activity")]
    assert stops, "expected hotel/activity stops"
    for stop in stops:
        assert (stop.get("image_url") or "").startswith("http"), stop
    # cleanup so the suite stays repeatable
    assert client.delete(f"/api/trips/{body['id']}").status_code == 200


def test_kashmir_6day_couple_trip_creates_successfully():
    """Exact replay of the reported 'Crafting Kashmir Getaway' failure:
    Kashmir, 6 days, couple, 16-21 Oct 2026."""
    kas_id = client.get("/api/destinations/kashmir").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Kashmir Getaway",
        "destination_id": kas_id,
        "duration_days": 6,
        "traveler_count": 2,
        "total_budget": 90000.0,
        "currency": "INR",
        "pace": "balanced",
        "start_date": "2026-10-16T00:00:00",
        "end_date": "2026-10-21T00:00:00",
        "preferences": {"interests": ["heritage", "culture"],
                        "travel_companions": "couple"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["destination"]["slug"] == "kashmir"
    assert len(body["itinerary"]) > 0
    assert client.delete(f"/api/trips/{body['id']}").status_code == 200


def test_every_catalog_place_builds_a_trip():
    """Whatever catalog place the user picks must produce a trip."""
    import pytest as _pytest
    for slug in ["manali", "goa", "kashmir", "kerala", "rajasthan", "udaipur"]:
        dest_id = client.get(f"/api/destinations/{slug}").json()["id"]
        r = client.post("/api/trips", json={
            "title": f"{slug.title()} Check",
            "destination_id": dest_id,
            "duration_days": 3,
            "traveler_count": 2,
            "total_budget": 60000.0,
            "currency": "INR",
        })
        assert r.status_code == 200, f"{slug}: {r.text}"
        assert client.delete(f"/api/trips/{r.json()['id']}").status_code == 200


def test_duration_is_derived_from_dates():
    """A 16-21 Oct range is 6 days even if the caller sends duration_days=3
    (the exact mismatch from the reported Kerala screenshot)."""
    ker_id = client.get("/api/destinations/kerala").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Kerala Duration Check",
        "destination_id": ker_id,
        "duration_days": 3,
        "traveler_count": 2,
        "total_budget": 75000.0,
        "currency": "INR",
        "start_date": "2026-10-16T00:00:00",
        "end_date": "2026-10-21T00:00:00",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duration_days"] == 6
    days = sorted({i["day_number"] for i in body["itinerary"]})
    assert days == [1, 2, 3, 4, 5, 6]
    assert client.delete(f"/api/trips/{body['id']}").status_code == 200


def test_end_before_start_rejected():
    ker_id = client.get("/api/destinations/kerala").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Backwards Trip",
        "destination_id": ker_id,
        "start_date": "2026-10-21T00:00:00",
        "end_date": "2026-10-16T00:00:00",
    })
    assert r.status_code == 422


def test_long_trips_spread_across_all_days():
    """6-day generation covers every day with no repeated activities."""
    ker_id = client.get("/api/destinations/kerala").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Kerala Spread Check",
        "destination_id": ker_id,
        "duration_days": 6,
        "traveler_count": 2,
        "total_budget": 75000.0,
        "currency": "INR",
        "start_date": "2026-10-16T00:00:00",
        "end_date": "2026-10-21T00:00:00",
    })
    assert r.status_code == 200, r.text
    items = r.json()["itinerary"]
    by_day = {}
    for item in items:
        by_day.setdefault(item["day_number"], []).append(item)
    assert sorted(by_day) == [1, 2, 3, 4, 5, 6]
    act_ids = [i["activity_id"] for i in items if i["activity_id"]]
    assert len(act_ids) == len(set(act_ids)) >= 5
    assert client.delete(f"/api/trips/{r.json()['id']}").status_code == 200


def test_rich_trips_fill_two_stops_per_day():
    """8 activities over 5 days -> most days carry morning + afternoon stops,
    no repeats, no order collisions within a day."""
    kas_id = client.get("/api/destinations/kashmir").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Kashmir Full Days",
        "destination_id": kas_id,
        "duration_days": 5,
        "traveler_count": 2,
        "total_budget": 120000.0,
        "currency": "INR",
    })
    assert r.status_code == 200, r.text
    items = r.json()["itinerary"]
    by_day = {}
    for item in items:
        by_day.setdefault(item["day_number"], []).append(item)
    assert sorted(by_day) == [1, 2, 3, 4, 5]
    acts = [i for i in items if i["activity_id"]]
    assert len(acts) >= 7
    assert len({i["activity_id"] for i in acts}) == len(acts)
    assert sum(1 for d in by_day.values() if len([i for i in d if i["activity_id"]]) >= 2) >= 2
    for day, stops in by_day.items():
        assert len({s["order_index"] for s in stops}) == len(stops)
    assert client.delete(f"/api/trips/{r.json()['id']}").status_code == 200


def test_generation_stays_within_budget():
    """The reported blowout: 50k/3 travelers/5 days planned 76,950.
    Hotel must leave room for transfers + required activities, and forced
    minimum picks must be the cheapest ones."""
    manali_id = client.get("/api/destinations/manali").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Budget Honesty Check",
        "destination_id": manali_id,
        "duration_days": 5,
        "traveler_count": 3,
        "total_budget": 50000.0,
        "currency": "INR",
        "pace": "balanced",
    })
    assert r.status_code == 200, r.text
    items = r.json()["itinerary"]
    total = sum(float(i.get("cost") or 0) for i in items)
    assert total <= 50000.0, f"planned {total} over 50000 budget"
    by_day = {}
    for item in items:
        by_day.setdefault(item["day_number"], []).append(item)
    assert sorted(by_day) == [1, 2, 3, 4, 5]
    assert client.delete(f"/api/trips/{r.json()['id']}").status_code == 200


def test_legacy_items_resolve_images_from_catalog():
    """Trips generated before image metadata existed still render photos
    via the catalog fallback in the trip serializer."""
    trip_id = "trp-manali-alpine-demo-001"
    body = client.get(f"/api/trips/{trip_id}").json()
    stops = [i for i in body["itinerary"] if i["item_type"] in ("hotel", "activity")]
    assert stops, "expected hotel/activity stops"
    for stop in stops:
        assert (stop.get("image_url") or "").startswith("http"), stop
