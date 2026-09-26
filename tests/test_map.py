"""Map endpoint tests: plotted stops, center, unmapped honesty, 404.

Regression coverage for manually added activities:
- catalog-backed adds resolve coordinates and plot on the correct day
- explicit lat/lng adds are preserved
- edits/moves keep the pin on the right day; deletes remove it
- unresolvable names stay honest (has_coordinates=false) without 500s
"""
import backend.places.service as places_service
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _goa_trip():
    goa_id = client.get("/api/destinations/goa").json()["id"]
    r = client.post("/api/trips", json={
        "title": "Map Trip", "destination_id": goa_id,
        "duration_days": 3, "traveler_count": 2,
        "total_budget": 60000.0, "currency": "INR"})
    assert r.status_code == 200, r.text
    return r.json()


def _day_map(trip_id, day_number):
    m = client.get(f"/api/trips/{trip_id}/map").json()
    for day in m["days"]:
        if day["day_number"] == day_number:
            return m, day["stops"]
    return m, []


def test_trip_map_plots_real_coordinates():
    trip = _goa_trip()
    try:
        m = client.get(f"/api/trips/{trip['id']}/map").json()
        assert m["trip_id"] == trip["id"]
        assert m["destination"]["name"] == "Goa"
        assert m["center"] and -90 <= m["center"]["latitude"] <= 90
        plotted = [s for d in m["days"] for s in d["stops"] if s["has_coordinates"]]
        assert len(plotted) >= 5
        assert all(s["latitude"] and s["longitude"] for s in plotted)
        days = sorted(d["day_number"] for d in m["days"])
        assert days == [1, 2, 3]
    finally:
        client.delete(f"/api/trips/{trip['id']}")


def test_trip_map_404():
    assert client.get("/api/trips/no-such-trip/map").status_code == 404


def test_custom_stop_without_coordinates_is_honest(monkeypatch):
    trip = _goa_trip()
    try:
        # Force the coordinate provider to fail so the custom name is
        # genuinely unresolvable: the stop must persist as an honest gap.
        monkeypatch.setattr(places_service, "geocode_place", lambda *a, **k: None)
        client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": "Made Up Custom Stop", "day_number": 1})
        m = client.get(f"/api/trips/{trip['id']}/map").json()
        flat = [s for d in m["days"] for s in d["stops"]]
        custom = [s for s in flat if s["title"] == "Made Up Custom Stop"]
        assert len(custom) == 1 and custom[0]["has_coordinates"] is False
        assert custom[0]["latitude"] is None and custom[0]["longitude"] is None
        assert m["unmapped_count"] >= 1
    finally:
        client.delete(f"/api/trips/{trip['id']}")

def test_added_catalog_activity_appears_on_map_with_coordinates(monkeypatch):
    """Add -> map must include the new Day-1 stop with real coordinates."""
    trip = _goa_trip()
    try:
        before = client.get(f"/api/trips/{trip['id']}/map").json()
        before_titles = {s["title"] for d in before["days"] for s in d["stops"]}
        monkeypatch.setattr(
            places_service, "geocode_place",
            lambda name, *a, **k: (19.0760, 72.8777, f"{name}, Mumbai, India"))
        catalog_title = next(iter(before_titles))
        added = client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": catalog_title, "day_number": 1}).json()
        added_ids = {row["id"] for row in added["itinerary"]
                     if row["title"] == catalog_title and row["day_number"] == 1}
        assert added_ids, "added activity must persist on Day 1"
        m, day1 = _day_map(trip["id"], 1)
        match = [s for s in day1 if s["title"] == catalog_title
                 and s["item_id"] in added_ids]
        assert len(match) >= 1, "new activity must be returned for Day 1"
        assert match[0]["has_coordinates"] is True
        assert match[0]["latitude"] is not None and match[0]["longitude"] is not None
        existing = [s["order_index"] for s in day1 if s["item_id"] not in added_ids]
        orders = [s["order_index"] for s in day1 if s["item_id"] in added_ids]
        if existing:
            assert min(orders) > max(existing)
        for title in before_titles:
            assert any(s["title"] == title for d in m["days"] for s in d["stops"])
    finally:
        client.delete(f"/api/trips/{trip['id']}")


def test_added_activity_with_explicit_coordinates_is_preserved(monkeypatch):
    trip = _goa_trip()
    try:
        def _boom(*a, **k):
            raise Exception("no net")
        monkeypatch.setattr(places_service, "geocode_place", _boom)
        r = client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": "Explicit Coord Stop", "day_number": 2,
            "latitude": 15.5, "longitude": 74.1})
        assert r.status_code == 200, r.text
        m, day2 = _day_map(trip["id"], 2)
        match = [s for s in day2 if s["title"] == "Explicit Coord Stop"]
        assert len(match) == 1
        assert match[0]["has_coordinates"] is True
        assert match[0]["latitude"] == 15.5 and match[0]["longitude"] == 74.1
        for day in m["days"]:
            if day["day_number"] != 2:
                assert all(s["title"] != "Explicit Coord Stop" for s in day["stops"])
    finally:
        client.delete(f"/api/trips/{trip['id']}")


def test_added_activity_by_catalog_id_links_and_plots():
    dest_id = client.get("/api/destinations/goa").json()["id"]
    catalog = client.get(f"/api/activities?destination_id={dest_id}").json()
    assert catalog, "Goa catalog must have activities"
    trip = _goa_trip()
    try:
        pick = catalog[0]
        r = client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": pick["title"] + " (encore)", "day_number": 1,
            "activity_id": pick["id"]})
        assert r.status_code == 200, r.text
        m, day1 = _day_map(trip["id"], 1)
        match = [s for s in day1 if s["title"] == pick["title"] + " (encore)"]
        assert len(match) == 1
        assert match[0]["activity_id"] == pick["id"]
        assert match[0]["has_coordinates"] is True
        assert match[0]["latitude"] == pick["latitude"]
        assert match[0]["longitude"] == pick["longitude"]
    finally:
        client.delete(f"/api/trips/{trip['id']}")


def test_edit_move_and_delete_reflected_on_map(monkeypatch):
    trip = _goa_trip()
    try:
        monkeypatch.setattr(
            places_service, "geocode_place",
            lambda name, *a, **k: (19.0760, 72.8777, f"{name}, Mumbai, India"))
        added = client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": "Movable Stop", "day_number": 1}).json()
        item_id = next(row["id"] for row in added["itinerary"]
                       if row["title"] == "Movable Stop")
        edit = client.post(f"/api/trips/{trip['id']}/edit-activity", json={
            "item_id": item_id, "day_number": 2})
        assert edit.status_code == 200, edit.text
        m = client.get(f"/api/trips/{trip['id']}/map").json()
        assert all(s["title"] != "Movable Stop"
                   for d in m["days"] if d["day_number"] == 1 for s in d["stops"])
        day2 = next(d["stops"] for d in m["days"] if d["day_number"] == 2)
        assert any(s["title"] == "Movable Stop" and s["has_coordinates"] for s in day2)
        deleted = client.post(f"/api/trips/{trip['id']}/delete-activity", json={
            "item_id": item_id})
        assert deleted.status_code == 200, deleted.text
        m = client.get(f"/api/trips/{trip['id']}/map").json()
        assert all(s["title"] != "Movable Stop" for d in m["days"] for s in d["stops"])
    finally:
        client.delete(f"/api/trips/{trip['id']}")


def test_unresolvable_activity_does_not_break_map(monkeypatch):
    trip = _goa_trip()
    try:
        monkeypatch.setattr(places_service, "geocode_place", lambda *a, **k: None)
        before = client.get(f"/api/trips/{trip['id']}/map")
        assert before.status_code == 200
        r = client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": "Xyzzy Nowhere Place 12345", "day_number": 1})
        assert r.status_code == 200, r.text
        m = client.get(f"/api/trips/{trip['id']}/map").json()
        flat = [s for d in m["days"] for s in d["stops"]]
        bad = [s for s in flat if s["title"] == "Xyzzy Nowhere Place 12345"]
        assert len(bad) == 1 and bad[0]["has_coordinates"] is False
        assert any(s["has_coordinates"] for s in flat), "mapped stops must survive"
    finally:
        client.delete(f"/api/trips/{trip['id']}")
