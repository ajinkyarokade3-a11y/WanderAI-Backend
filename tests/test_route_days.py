"""Additive route_days[] summaries (no selection/budget/ranking changes)."""
from datetime import datetime
from types import SimpleNamespace

from backend.api.routes import _route_day_summaries


def _trip(start=None, duration=4):
    return SimpleNamespace(start_date=start, duration_days=duration)


def _row(day, item_type, title="Stop", ui=None, **extra):
    row = {"id": f"{day}-{item_type}-{title[:4]}", "day_number": day,
           "item_type": item_type, "title": title}
    if ui is not None:
        row.update(ui)
        row["meta_data"] = {"ui": ui}
    row.update(extra)
    return row


def test_empty_days_honest_not_valid():
    rows = [_row(1, "activity", "Alpha"),
            _row(1, "hotel", "Stay",
                 {"route_day_summary": {"status": "Valid"}}),
            _row(2, "hotel", "Stay")]
    out = _route_day_summaries(_trip(datetime(2026, 9, 23), 4), rows)
    assert [d["day"] for d in out] == [1, 2, 3, 4]
    assert out[0]["status"] == "Valid"
    assert out[0]["date"] == "2026-09-23"
    assert out[1]["date"] == "2026-09-24"
    assert out[1]["status"] == "Empty"
    assert out[1]["hotel"] == "Stay"
    assert out[1]["explanation"] == (
        "Overnight stay at Stay — no sightseeing planned for this day.")
    for empty in out[2:]:
        assert empty["activity_count"] == 0
        assert empty["status"] == "Empty"
        assert empty["explanation"] == "No activities planned for this day."
        assert empty["daily_travel_minutes"] is None
    assert out[0]["hotel"] == "Stay"


def test_warnings_merged_deduped_and_kept():
    rows = [_row(1, "activity", "A", {"route_warnings": ["W1", "W2"]}),
            _row(1, "activity", "B",
                 {"route_warnings": ["W2", "W3"],
                  "route_day_summary": {"status": "Requires-Revision",
                                        "daily_travel_minutes": 300.0,
                                        "max_one_way_minutes": 200.0}})]
    out = _route_day_summaries(_trip(None, 1), rows)
    assert out[0]["warnings"] == ["W1", "W2", "W3"]
    assert out[0]["status"] == "Requires-Revision"
    assert out[0]["daily_travel_minutes"] == 300.0
    assert out[0]["max_one_way_minutes"] == 200.0
    assert out[0]["activity_count"] == 2


def test_legacy_rows_without_metadata_stay_neutral():
    rows = [_row(1, "activity", "Old"), _row(2, "activity", "Older")]
    out = _route_day_summaries(_trip(None, 2), rows)
    assert out[0]["status"] is None
    assert out[0]["warnings"] == []
    assert out[0]["date"] is None


def test_duration_shorter_than_items_still_covers_all_days():
    rows = [_row(1, "activity", "A"), _row(5, "activity", "E")]
    out = _route_day_summaries(_trip(None, 2), rows)
    assert [d["day"] for d in out] == [1, 2, 3, 4, 5]


def test_live_response_contains_route_days_and_legacy_keys():
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    manali_id = client.get("/api/destinations/manali").json()["id"]
    created = client.post("/api/trips", json={
        "title": "Route Days Probe", "destination_id": manali_id,
        "duration_days": 3, "total_budget": 60000.0, "currency": "INR",
        "traveler_count": 2, "pace": "balanced",
    })
    assert created.status_code == 200, created.text
    body = created.json()
    try:
        # Legacy contract intact plus the additive field.
        for key in ("id", "title", "itinerary", "bookings", "total_cost",
                    "cost_breakdown", "selected_accommodation",
                    "accommodation_alternatives", "daily_accommodations",
                    "preferences", "route_days"):
            assert key in body, f"missing key {key}"
        days = body["route_days"]
        assert [d["day"] for d in days] == [1, 2, 3]
        for day in days:
            assert set(day) == {"day", "date", "activity_count", "hotel",
                                "overnight_region",
                                "daily_travel_minutes", "max_one_way_minutes",
                                "status", "explanation", "warnings"}
            assert isinstance(day["warnings"], list)
    finally:
        client.delete(f"/api/trips/{body['id']}")
    assert client.get(f"/api/trips/{body['id']}").status_code == 404
