"""Tests for Social Signals service and API endpoint.

Tests mock the provider HTTP layer — no real API calls, no internet needed.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import (
    Activity, Destination, ItineraryItem, Trip, User,
)
from backend.social_signals import service as social_service
from backend.social_signals.schemas import (
    Confidence,
    Sentiment,
    SignalType,
    SocialData,
    SocialSignal,
    SourceType,
    WeatherRelation,
)
from backend.social_signals.service import (
    SocialSignalsNotConfigured,
    SocialSignalsProviderError,
    build_social_context,
    clear_social_signals_cache,
    fetch_social_signals,
    get_social_signals_for_trip,
)

client = TestClient(app)


def _mock_httpx_get(payload):
    def _fake_get(url, params, timeout):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp
    return _fake_get


def _mock_httpx_get_error(exc):
    def _fake_get(url, params, timeout):
        raise exc
    return _fake_get


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_social_signals_cache()
    yield
    clear_social_signals_cache()


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------

class TestSocialSignalsService:
    def test_missing_base_url(self):
        with pytest.raises(SocialSignalsNotConfigured):
            fetch_social_signals(19.0, 72.8, "", 8.0)

    def test_provider_timeout(self, monkeypatch):
        import httpx
        mock_httpx = type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.TimeoutException("timeout"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        })
        monkeypatch.setattr(social_service, "httpx", mock_httpx)
        with pytest.raises(SocialSignalsProviderError, match="timed out"):
            fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)

    def test_provider_http_error(self, monkeypatch):
        import httpx
        mock_httpx = type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.ConnectError("fail"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        })
        monkeypatch.setattr(social_service, "httpx", mock_httpx)
        with pytest.raises(SocialSignalsProviderError):
            fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)

    def test_successful_response(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([
                {"title": "Crowd report", "summary": "High activity", "source": "test", "category": "crowd", "severity": "medium"},
                {"title": "Event", "summary": "Festival", "source": "test", "category": "event", "severity": "low"},
            ])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        assert result["available"] is True
        assert len(result["signals"]) == 2
        assert result["overall_risk"] == "medium"

    def test_empty_signals(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        assert result["available"] is True
        assert result["signals"] == []
        assert result["overall_risk"] == "low"

    def test_overall_risk_high(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([
                {"title": "Closure", "summary": "Road closed", "severity": "high", "category": "closure"},
            ])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        assert result["overall_risk"] == "high"

    def test_caching(self, monkeypatch):
        call_count = [0]
        def _fake_get(url, params, timeout):
            call_count[0] += 1
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.json.return_value = [{"title": "Test", "severity": "low"}]
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        assert call_count[0] == 1

    def test_cache_expiry(self, monkeypatch):
        call_count = [0]
        def _fake_get(url, params, timeout):
            call_count[0] += 1
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.json.return_value = [{"title": "Test", "severity": "low"}]
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        monkeypatch.setattr(social_service, "_get_cache_ttl", lambda: 0)
        fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        fetch_social_signals(19.0, 72.8, "https://provider.example.com", 8.0)
        assert call_count[0] == 2


class TestBuildSocialContext:
    def test_success(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([{"title": "Test", "severity": "medium"}])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        ctx = build_social_context(19.0, 72.8, "https://provider.example.com", 8.0)
        assert ctx["available"] is True
        assert len(ctx["signals"]) == 1

    def test_not_configured(self):
        ctx = build_social_context(19.0, 72.8, "", 8.0)
        assert ctx["available"] is False
        assert ctx["reason"] == "not_configured"

    def test_provider_unavailable(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(Exception("fail"))),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        ctx = build_social_context(19.0, 72.8, "https://provider.example.com", 8.0)
        assert ctx["available"] is False
        assert ctx["reason"] == "provider_unavailable"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestSocialSignalsEndpoint:
    def _create_trip(self):
        db = SessionLocal()
        try:
            user = db.query(User).first()
            destination = db.query(Destination).filter(Destination.slug == "manali").first()
            assert user is not None
            assert destination is not None
            trip = Trip(
                user_id=user.id,
                destination_id=destination.id,
                title="Social Test Trip",
                traveler_count=2,
                duration_days=3,
                total_budget=50000.0,
                currency="INR",
            )
            db.add(trip)
            db.commit()
            return trip.id
        finally:
            db.close()

    def test_endpoint_all_providers_fail(self, monkeypatch):
        """When all providers fail, returns unavailable state."""
        from backend.social_signals.providers import gdelt, bluesky, rss
        import httpx
        mock_httpx = type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.TimeoutException("timeout"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        })
        monkeypatch.setattr(gdelt, "httpx", mock_httpx)
        monkeypatch.setattr(bluesky, "httpx", mock_httpx)
        monkeypatch.setattr(rss, "httpx", mock_httpx)
        trip_id = self._create_trip()
        r = client.get(f"/api/trips/{trip_id}/social-signals")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is False
        assert body["reason"] == "all_providers_failed"
        assert body["status"] == "unavailable"

    def test_endpoint_trip_not_found(self):
        r = client.get("/api/trips/non-existent-id/social-signals")
        assert r.status_code == 404

    def test_endpoint_success(self, monkeypatch):
        """GDELT provider returns signals through the endpoint."""
        from backend.social_signals.providers import gdelt, rss
        mock_httpx = type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "articles": [
                    {
                        "url": "https://example.com/article/1",
                        "title": "Heavy rain in Manali",
                        "domain": "example.com",
                        "seendate": "20260927120000",
                    },
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        })
        monkeypatch.setattr(gdelt, "httpx", mock_httpx)
        monkeypatch.setattr(rss, "httpx", mock_httpx)
        trip_id = self._create_trip()
        r = client.get(f"/api/trips/{trip_id}/social-signals")
        assert r.status_code == 200
        body = r.json()
        assert body["available"] is True
        assert len(body["signals"]) == 1
        assert body["sources"] == ["GDELT"]


# ---------------------------------------------------------------------------
# Schema tests — new normalized fields and enums
# ---------------------------------------------------------------------------


class TestSocialSignalSchema:
    def test_new_fields_serialize(self):
        sig = SocialSignal(
            title="Test",
            source="Bluesky",
            source_type=SourceType.SOCIAL,
            source_url="https://example.com/post/123",
            published_at="2026-09-27T10:00:00Z",
            location="Goa",
            relevance_score=0.85,
            signal_type=SignalType.TRAVELER_REPORT,
            sentiment=Sentiment.NEGATIVE,
            weather_relation=WeatherRelation.DIRECT,
            confidence=Confidence.MEDIUM,
        )
        d = sig.model_dump()
        assert d["source_type"] == "SOCIAL"
        assert d["source_url"] == "https://example.com/post/123"
        assert d["published_at"] == "2026-09-27T10:00:00Z"
        assert d["location"] == "Goa"
        assert d["relevance_score"] == 0.85
        assert d["signal_type"] == "TRAVELER_REPORT"
        assert d["sentiment"] == "NEGATIVE"
        assert d["weather_relation"] == "DIRECT"
        assert d["confidence"] == "MEDIUM"

    def test_optional_fields_default_none(self):
        sig = SocialSignal()
        d = sig.model_dump()
        assert d["source_type"] is None
        assert d["source_url"] is None
        assert d["published_at"] is None
        assert d["location"] is None
        assert d["relevance_score"] is None
        assert d["signal_type"] is None
        assert d["sentiment"] is None
        assert d["weather_relation"] is None
        assert d["confidence"] is None

    def test_existing_fields_preserved(self):
        sig = SocialSignal(
            title="Test",
            summary="Summary",
            source="test",
            category="crowd",
            severity="medium",
            observed_at="2026-09-27T10:00:00Z",
            relevance="high",
        )
        d = sig.model_dump()
        assert d["title"] == "Test"
        assert d["summary"] == "Summary"
        assert d["source"] == "test"
        assert d["category"] == "crowd"
        assert d["severity"] == "medium"
        assert d["observed_at"] == "2026-09-27T10:00:00Z"
        assert d["relevance"] == "high"

    def test_invalid_signal_type_rejected(self):
        with pytest.raises(Exception):
            SocialSignal(signal_type="INVALID_TYPE")

    def test_invalid_source_type_rejected(self):
        with pytest.raises(Exception):
            SocialSignal(source_type="INVALID_SOURCE")

    def test_invalid_sentiment_rejected(self):
        with pytest.raises(Exception):
            SocialSignal(sentiment="INVALID_SENTIMENT")

    def test_invalid_weather_relation_rejected(self):
        with pytest.raises(Exception):
            SocialSignal(weather_relation="INVALID_RELATION")

    def test_invalid_confidence_rejected(self):
        with pytest.raises(Exception):
            SocialSignal(confidence="INVALID_CONFIDENCE")

    def test_all_signal_type_values(self):
        for st in SignalType:
            sig = SocialSignal(signal_type=st)
            assert sig.signal_type == st

    def test_all_source_type_values(self):
        for st in SourceType:
            sig = SocialSignal(source_type=st)
            assert sig.source_type == st

    def test_all_sentiment_values(self):
        for s in Sentiment:
            sig = SocialSignal(sentiment=s)
            assert sig.sentiment == s

    def test_all_weather_relation_values(self):
        for wr in WeatherRelation:
            sig = SocialSignal(weather_relation=wr)
            assert sig.weather_relation == wr

    def test_all_confidence_values(self):
        for c in Confidence:
            sig = SocialSignal(confidence=c)
            assert sig.confidence == c


class TestSocialDataSchema:
    def test_new_response_fields(self):
        data = SocialData(
            available=True,
            signals=[SocialSignal(title="Test")],
            overall_risk="medium",
            observed_at="2026-09-27T10:00:00Z",
            sources=["GDELT", "Bluesky"],
            generated_at="2026-09-27T10:00:01Z",
            status="success",
            total=1,
        )
        d = data.model_dump()
        assert d["sources"] == ["GDELT", "Bluesky"]
        assert d["generated_at"] == "2026-09-27T10:00:01Z"
        assert d["status"] == "success"
        assert d["total"] == 1

    def test_new_response_fields_optional(self):
        data = SocialData(available=True, signals=[])
        d = data.model_dump()
        assert d["sources"] is None
        assert d["generated_at"] is None
        assert d["status"] is None
        assert d["total"] is None

    def test_backward_compatible_response(self):
        data = SocialData(
            available=True,
            signals=[SocialSignal(title="Test", severity="low")],
            overall_risk="low",
            observed_at="2026-09-27T10:00:00Z",
        )
        d = data.model_dump()
        assert d["available"] is True
        assert len(d["signals"]) == 1
        assert d["overall_risk"] == "low"
        assert d["observed_at"] == "2026-09-27T10:00:00Z"


class TestNormalizationNewFields:
    def test_normalize_includes_new_fields(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([{
                "title": "Flooding in Goa",
                "summary": "Heavy rain causing flooding",
                "source": "Bluesky",
                "source_type": "SOCIAL",
                "source_url": "https://bsky.app/post/123",
                "published_at": "2026-09-27T08:00:00Z",
                "location": "Goa",
                "relevance_score": 0.91,
                "signal_type": "TRAVELER_REPORT",
                "sentiment": "NEGATIVE",
                "weather_relation": "DIRECT",
                "confidence": "MEDIUM",
            }])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_social_signals(15.29, 73.82, "https://provider.example.com", 8.0)
        sig = result["signals"][0]
        assert sig["source_type"] == "SOCIAL"
        assert sig["source_url"] == "https://bsky.app/post/123"
        assert sig["published_at"] == "2026-09-27T08:00:00Z"
        assert sig["location"] == "Goa"
        assert sig["relevance_score"] == 0.91
        assert sig["signal_type"] == "TRAVELER_REPORT"
        assert sig["sentiment"] == "NEGATIVE"
        assert sig["weather_relation"] == "DIRECT"
        assert sig["confidence"] == "MEDIUM"

    def test_normalize_new_fields_default_none(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([{"title": "Test"}])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_social_signals(15.29, 73.82, "https://provider.example.com", 8.0)
        sig = result["signals"][0]
        assert sig["source_type"] is None
        assert sig["source_url"] is None
        assert sig["published_at"] is None
        assert sig["location"] is None
        assert sig["relevance_score"] is None
        assert sig["signal_type"] is None
        assert sig["sentiment"] is None
        assert sig["weather_relation"] is None
        assert sig["confidence"] is None

    def test_response_includes_sources_and_status(self, monkeypatch):
        monkeypatch.setattr(social_service, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([{"title": "Test", "severity": "low"}])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_social_signals(15.29, 73.82, "https://provider.example.com", 8.0)
        assert result["sources"] == ["external"]
        assert result["status"] == "success"
        assert result["total"] == 1
        assert "generated_at" in result
