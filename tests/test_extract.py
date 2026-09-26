"""Voice/extract-preferences contract + heuristic richness tests."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

VOICE_MSG = ("Plan me a 6-day trip to Udaipur for 2 people under Rs 75000. "
             "I love heritage, local food and slow lake evenings.")

def test_voice_text_alias_accepted():
    r = client.post("/api/ai/extract-preferences", json={"text": VOICE_MSG})
    assert r.status_code == 200, r.text
    assert r.json()["detected_destination"] == "Udaipur"


def test_missing_text_rejected():
    assert client.post("/api/ai/extract-preferences", json={}).status_code == 422
    assert client.post("/api/ai/extract-preferences", json={"context": {}}).status_code == 422


def test_heuristic_extracts_voice_example():
    r = client.post("/api/ai/extract-preferences", json={"text_prompt": VOICE_MSG})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["detected_destination"] == "Udaipur"
    assert body["duration_days"] == 6
    assert body["traveler_count"] == 2
    assert body["budget_amount"] == 75000
    assert body["budget_currency"] == "INR"
    assert body["pace"] == "relaxed"
    assert len(body["interests"]) >= 2


def test_gemini_failure_still_returns_heuristic(monkeypatch):
    """Blocked key must not yield emptier results than no key."""
    from backend.ai.gemini_service import gemini_service

    def _boom(*args, **kwargs):
        raise RuntimeError("denied")

    monkeypatch.setattr(gemini_service, "_client",
                        SimpleNamespace(models=SimpleNamespace(generate_content=_boom)))
    r = client.post("/api/ai/extract-preferences", json={"text_prompt": VOICE_MSG})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "fallback_on_error"
    assert body["detected_destination"] == "Udaipur"
    assert body["duration_days"] == 6
    assert body["traveler_count"] == 2


def _extract_heuristic(monkeypatch, text):
    """Force the deterministic fallback path (no network, no Gemini)."""
    from backend.ai.gemini_service import GeminiService

    monkeypatch.setattr(GeminiService, "is_available", lambda self: False)
    r = client.post("/api/ai/extract-preferences", json={"text_prompt": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_duration_survives_without_dates_goa_from_mumbai(monkeypatch):
    body = _extract_heuristic(monkeypatch, "5 day trip to Goa from Mumbai")
    assert body["duration_days"] == 5
    assert body["detected_destination"] == "Goa"
    assert body["detected_origin"] == "Mumbai"
    assert body["start_date"] is None
    assert body["end_date"] is None


def test_duration_survives_without_dates_goa_only(monkeypatch):
    body = _extract_heuristic(monkeypatch, "5 day trip to Goa")
    assert body["duration_days"] == 5
    assert body["start_date"] is None
    assert body["end_date"] is None


def test_no_hallucinated_origin_travelers_or_budget(monkeypatch):
    """'5 days trip to kerala' must not invent mumbai / 2 / couple."""
    body = _extract_heuristic(monkeypatch, "5 days trip to kerala")
    assert body["detected_destination"] == "Kerala"
    assert body["duration_days"] == 5
    assert body["detected_origin"] is None
    assert body["traveler_count"] is None
    assert body["travel_companions"] is None
    assert body["budget_tier"] is None
    assert body["start_date"] is None
    assert body["end_date"] is None


def test_extract_with_start_context_computes_end_date(monkeypatch):
    """Review screen can re-call extract with picked START to prefill END."""
    from backend.ai.gemini_service import GeminiService

    monkeypatch.setattr(GeminiService, "is_available", lambda self: False)
    r = client.post("/api/ai/extract-preferences", json={
        "text_prompt": "5 days trip to kerala",
        "context": {"start_date": "2026-09-23"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duration_days"] == 5
    assert body["end_date"] == "2026-09-27"


def test_duration_phrase_variants(monkeypatch):
    for text in ("5 days in Goa", "Goa for 5 days", "Goa 5-day trip",
                 "trip for five days in Goa"):
        body = _extract_heuristic(monkeypatch, text)
        assert body["duration_days"] == 5, text
        assert body["start_date"] is None and body["end_date"] is None, text


def test_review_dates_computes_end_inclusive():
    r = client.post("/api/trips/review-dates",
                    json={"start_date": "2026-09-27", "duration_days": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["start_date"] == "2026-09-27"
    assert body["end_date"] == "2026-10-01"
    assert body["computed_end_date"] == "2026-10-01"
    assert body["duration_days"] == 5
    assert body["end_overridden"] is False


def test_review_dates_accepts_ddmmyyyy_start():
    r = client.post("/api/trips/review-dates",
                    json={"start_date": "27-09-2026", "duration_days": 5})
    assert r.status_code == 200, r.text
    assert r.json()["end_date"] == "2026-10-01"


def test_review_dates_manual_override_accepted():
    r = client.post("/api/trips/review-dates",
                    json={"start_date": "2026-09-27", "duration_days": 5,
                          "end_date": "2026-10-04"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["end_date"] == "2026-10-04"
    assert body["duration_days"] == 8
    assert body["end_overridden"] is True


def test_review_dates_rejects_bad_range():
    assert client.post(
        "/api/trips/review-dates",
        json={"start_date": "2026-09-27", "duration_days": 5,
              "end_date": "2026-09-20"}).status_code == 422
    assert client.post(
        "/api/trips/review-dates",
        json={"duration_days": 5}).status_code == 422
    assert client.post(
        "/api/trips/review-dates",
        json={"start_date": "2026-09-27", "duration_days": 0}).status_code == 422
    assert client.post(
        "/api/trips/review-dates",
        json={"start_date": "not-a-date", "duration_days": 5}).status_code == 422


def test_trip_creation_accepts_review_dates_and_generates_itinerary(monkeypatch):
    import backend.images.service as images_service

    monkeypatch.setattr(images_service, "get_real_images_for_location",
                        lambda *a, **k: [])
    goa_id = client.get("/api/destinations/goa").json()["id"]
    trip_id = None
    try:
        review = client.post(
            "/api/trips/review-dates",
            json={"start_date": "2026-09-27", "duration_days": 5}).json()
        r = client.post("/api/trips", json={
            "title": "Review Dates Trip", "destination_id": goa_id,
            "duration_days": 5, "traveler_count": 2,
            "total_budget": 60000.0, "currency": "INR",
            "start_date": f"{review['start_date']}T00:00:00",
            "end_date": f"{review['end_date']}T00:00:00",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        trip_id = body["id"]
        assert body["duration_days"] == 5
        assert sorted({i["day_number"] for i in body["itinerary"]}) == [1, 2, 3, 4, 5]
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_manual_dates_still_work_and_backwards_rejected():
    goa_id = client.get("/api/destinations/goa").json()["id"]
    trip_id = None
    try:
        r = client.post("/api/trips", json={
            "title": "Manual Dates Trip", "destination_id": goa_id,
            "traveler_count": 2, "total_budget": 60000.0, "currency": "INR",
            "start_date": "2026-11-10T09:00:00",
            "end_date": "2026-11-12T18:00:00",
        })
        assert r.status_code == 200, r.text
        trip_id = r.json()["id"]
        assert r.json()["duration_days"] == 3
        bad = client.post("/api/trips", json={
            "title": "Backwards", "destination_id": goa_id,
            "start_date": "2026-10-21T00:00:00",
            "end_date": "2026-10-16T00:00:00",
        })
        assert bad.status_code == 422
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")
