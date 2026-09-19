"""Regression tests for SerpApi image-proxy transient handling.

A single throttled SerpApi response (429/5xx) must be retried within the proxy
instead of blanking the photo card; permanent errors still fail fast to the
200-empty contract, and retries stay bounded.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

import backend.images.service as images_service
from backend.database.config import settings as _settings
from backend.main import app

client = TestClient(app)

GOOD_PAYLOAD = {"images_results": [{
    "title": "Amber Fort",
    "original": "https://example.test/amber.jpg",
    "thumbnail": "https://example.test/amber-thumb.jpg",
}]}


def _resp(status, payload=None):
    return httpx.Response(
        status,
        json=payload if payload is not None else {},
        request=httpx.Request("GET", "https://serpapi.com/search"),
    )


@pytest.fixture()
def _setup(monkeypatch):
    images_service.clear_image_cache()
    monkeypatch.setattr(_settings, "SERPAPI_API_KEY", "test-serpapi-key")
    sleeps = []
    monkeypatch.setattr(images_service.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_rate_limit_retried_then_succeeds(monkeypatch, _setup):
    sleeps = _setup
    calls = {"count": 0}
    script = [_resp(429), _resp(429), _resp(200, GOOD_PAYLOAD)]

    def fake_get(url, params, timeout):
        calls["count"] += 1
        return script[calls["count"] - 1]

    monkeypatch.setattr(images_service, "_http_get", fake_get)
    response = client.get("/api/places/image",
                          params={"location": "Amber Fort", "destination": "Rajasthan"})
    assert response.status_code == 200
    assert response.json()["image_url"] == "https://example.test/amber.jpg"
    assert calls["count"] == 3
    assert sleeps == [0.5, 1.0]


def test_persistent_transient_failure_gives_up_bounded(monkeypatch, _setup):
    calls = {"count": 0}

    def always_503(url, params, timeout):
        calls["count"] += 1
        return _resp(503)

    monkeypatch.setattr(images_service, "_http_get", always_503)
    response = client.get("/api/places/image", params={"location": "Amber Fort"})
    assert response.status_code == 200
    assert response.json()["image_url"] is None
    assert response.json()["source"] == "none"
    assert calls["count"] == 3


def test_permanent_error_fails_fast_without_retry(monkeypatch, _setup):
    sleeps = _setup
    calls = {"count": 0}

    def unauthorized(url, params, timeout):
        calls["count"] += 1
        return _resp(401, {"error": "Invalid API key"})

    monkeypatch.setattr(images_service, "_http_get", unauthorized)
    response = client.get("/api/places/image", params={"location": "Amber Fort"})
    assert response.status_code == 200
    assert response.json()["image_url"] is None
    assert calls["count"] == 1
    assert sleeps == []
