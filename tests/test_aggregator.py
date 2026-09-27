"""Tests for the Social Signal Aggregator.

All tests mock providers — no real API calls.
"""
import pytest

from backend.social_signals.aggregator import (
    ProviderResult,
    aggregate_signals,
    get_aggregated_signals,
)
from backend.social_signals import service as social_service


def _signal(**overrides):
    sig = {
        "title": "Test Signal",
        "summary": "Test summary",
        "source": "test",
        "source_type": "NEWS",
        "source_url": "https://example.com/article/1",
        "published_at": "2026-09-27T12:00:00+00:00",
        "location": "Goa",
        "signal_type": "WEATHER_REPORT",
        "weather_relation": "DIRECT",
        "sentiment": "UNKNOWN",
        "confidence": "LOW",
        "relevance_score": 0.5,
    }
    sig.update(overrides)
    return sig


class TestAggregateSignals:
    def test_both_providers_succeed(self):
        gdelt = ProviderResult("GDELT", [_signal(source="GDELT", source_url="https://a.com/1")])
        bluesky = ProviderResult("Bluesky", [_signal(source="Bluesky", source_url="https://b.com/1")])
        result = aggregate_signals([gdelt, bluesky])
        assert result["available"] is True
        assert result["status"] == "success"
        assert result["sources"] == ["GDELT", "Bluesky"]
        assert result["total"] == 2
        assert len(result["signals"]) == 2

    def test_gdelt_fails_bluesky_succeeds(self):
        gdelt = ProviderResult("GDELT", [], error="timeout")
        bluesky = ProviderResult("Bluesky", [_signal(source="Bluesky")])
        result = aggregate_signals([gdelt, bluesky])
        assert result["available"] is True
        assert result["status"] == "partial"
        assert result["sources"] == ["Bluesky"]
        assert result["total"] == 1

    def test_bluesky_fails_gdelt_succeeds(self):
        gdelt = ProviderResult("GDELT", [_signal(source="GDELT")])
        bluesky = ProviderResult("Bluesky", [], error="rate_limited")
        result = aggregate_signals([gdelt, bluesky])
        assert result["available"] is True
        assert result["status"] == "partial"
        assert result["sources"] == ["GDELT"]
        assert result["total"] == 1

    def test_both_providers_fail(self):
        gdelt = ProviderResult("GDELT", [], error="timeout")
        bluesky = ProviderResult("Bluesky", [], error="http_error")
        result = aggregate_signals([gdelt, bluesky])
        assert result["available"] is False
        assert result["reason"] == "all_providers_failed"
        assert result["status"] == "unavailable"

    def test_empty_provider_results(self):
        gdelt = ProviderResult("GDELT", [])
        bluesky = ProviderResult("Bluesky", [])
        result = aggregate_signals([gdelt, bluesky])
        assert result["available"] is True
        assert result["status"] == "success"
        assert result["total"] == 0
        assert result["signals"] == []

    def test_duplicate_urls_removed(self):
        sig1 = _signal(source="GDELT", source_url="https://same.com/1")
        sig2 = _signal(source="Bluesky", source_url="https://same.com/1")
        gdelt = ProviderResult("GDELT", [sig1])
        bluesky = ProviderResult("Bluesky", [sig2])
        result = aggregate_signals([gdelt, bluesky])
        assert result["total"] == 1
        assert len(result["signals"]) == 1

    def test_different_providers_same_event_different_urls(self):
        sig1 = _signal(source="GDELT", source_url="https://a.com/1", title="Heavy rain in Goa")
        sig2 = _signal(source="Bluesky", source_url="https://b.com/1", title="Heavy rain in Goa")
        gdelt = ProviderResult("GDELT", [sig1])
        bluesky = ProviderResult("Bluesky", [sig2])
        result = aggregate_signals([gdelt, bluesky])
        assert result["total"] == 2
        assert len(result["signals"]) == 2

    def test_correct_source_list_order(self):
        gdelt = ProviderResult("GDELT", [_signal()])
        bluesky = ProviderResult("Bluesky", [_signal()])
        result = aggregate_signals([gdelt, bluesky])
        assert result["sources"] == ["GDELT", "Bluesky"]

    def test_sorted_by_relevance_score(self):
        sig_low = _signal(source_url="https://a.com/1", relevance_score=0.3)
        sig_high = _signal(source_url="https://a.com/2", relevance_score=0.9)
        sig_med = _signal(source_url="https://a.com/3", relevance_score=0.6)
        gdelt = ProviderResult("GDELT", [sig_low, sig_high, sig_med])
        result = aggregate_signals([gdelt])
        scores = [s["relevance_score"] for s in result["signals"]]
        assert scores == [0.9, 0.6, 0.3]

    def test_sorted_by_recency_when_scores_equal(self):
        sig_old = _signal(source_url="https://a.com/1", published_at="2026-09-26T12:00:00+00:00")
        sig_new = _signal(source_url="https://a.com/2", published_at="2026-09-27T12:00:00+00:00")
        gdelt = ProviderResult("GDELT", [sig_old, sig_new])
        result = aggregate_signals([gdelt])
        assert result["signals"][0]["published_at"] == "2026-09-27T12:00:00+00:00"

    def test_provider_result_success_property(self):
        p = ProviderResult("GDELT", [])
        assert p.success is True
        p2 = ProviderResult("GDELT", [], error="fail")
        assert p2.success is False


class TestGetAggregatedSignals:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        social_service.clear_social_signals_cache()
        yield
        social_service.clear_social_signals_cache()

    def _create_trip(self, db, dest_name="Goa"):
        from backend.models.models import Destination, Trip, User
        user = db.query(User).first()
        dest = db.query(Destination).filter(Destination.name == dest_name).first()
        if not dest:
            dest = Destination(
                name=dest_name,
                slug=dest_name.lower().replace(" ", "-"),
                country="India",
                latitude=15.2993,
                longitude=74.1240,
            )
            db.add(dest)
            db.flush()
        trip = Trip(
            user_id=user.id,
            destination_id=dest.id,
            title="Aggregator Test Trip",
            traveler_count=2,
            duration_days=3,
            total_budget=50000.0,
            currency="INR",
        )
        db.add(trip)
        db.commit()
        return trip

    def test_no_destination(self):
        from backend.database.connection import SessionLocal
        db = SessionLocal()
        try:
            result = get_aggregated_signals(db, None)
            assert result["available"] is False
            assert result["reason"] == "no_destination"
        finally:
            db.close()

    def test_cache_hit_avoids_provider_calls(self, monkeypatch):
        from backend.database.connection import SessionLocal
        call_count = {"gdelt": 0, "bluesky": 0}

        def mock_gdelt(dest_name, weather_condition=None):
            call_count["gdelt"] += 1
            return [_signal(source="GDELT")]

        def mock_bluesky(dest_name, weather_condition=None):
            call_count["bluesky"] += 1
            return [_signal(source="Bluesky")]

        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_gdelt",
            lambda d, w=None: ProviderResult("GDELT", mock_gdelt(d, w)),
        )
        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_bluesky",
            lambda d, w=None: ProviderResult("Bluesky", mock_bluesky(d, w)),
        )

        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            result1 = get_aggregated_signals(db, trip)
            result2 = get_aggregated_signals(db, trip)
            assert result1["total"] == result2["total"]
            assert call_count["gdelt"] == 1
            assert call_count["bluesky"] == 1
        finally:
            db.close()

    def test_partial_status_when_one_provider_fails(self, monkeypatch):
        from backend.database.connection import SessionLocal

        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_gdelt",
            lambda d, w=None: ProviderResult("GDELT", [], error="timeout"),
        )
        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_bluesky",
            lambda d, w=None: ProviderResult("Bluesky", [_signal(source="Bluesky")]),
        )
        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_rss",
            lambda d, w=None: ProviderResult("RSS", [], error="timeout"),
        )

        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            result = get_aggregated_signals(db, trip)
            assert result["available"] is True
            assert result["status"] == "partial"
            assert result["sources"] == ["Bluesky"]
        finally:
            db.close()

    def test_unavailable_when_all_providers_fail(self, monkeypatch):
        from backend.database.connection import SessionLocal

        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_gdelt",
            lambda d, w=None: ProviderResult("GDELT", [], error="timeout"),
        )
        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_bluesky",
            lambda d, w=None: ProviderResult("Bluesky", [], error="http_error"),
        )
        monkeypatch.setattr(
            "backend.social_signals.aggregator._fetch_rss",
            lambda d, w=None: ProviderResult("RSS", [], error="timeout"),
        )

        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            result = get_aggregated_signals(db, trip)
            assert result["available"] is False
            assert result["status"] == "unavailable"
        finally:
            db.close()
