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
    assert {"food", "heritage", "lake"} <= set(body["interests"])


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
