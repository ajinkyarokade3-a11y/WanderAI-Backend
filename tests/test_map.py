"""Map endpoint tests: plotted stops, center, unmapped honesty, 404."""
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


def test_custom_stop_without_coordinates_is_honest():
    trip = _goa_trip()
    try:
        client.post(f"/api/trips/{trip['id']}/add-activity", json={
            "title": "Made Up Custom Stop", "day_number": 1})
        m = client.get(f"/api/trips/{trip['id']}/map").json()
        flat = [s for d in m["days"] for s in d["stops"]]
        custom = [s for s in flat if s["title"] == "Made Up Custom Stop"]
        assert len(custom) == 1 and custom[0]["has_coordinates"] is False
        assert m["unmapped_count"] >= 1
    finally:
        client.delete(f"/api/trips/{trip['id']}")
