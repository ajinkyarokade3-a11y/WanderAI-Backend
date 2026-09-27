"""Tests for Social Signals scoring: relevance, confidence, corroboration."""

import pytest
from datetime import datetime, timezone, timedelta

from backend.social_signals.scoring import (
    classify_signal_type,
    compute_confidence,
    compute_location_relevance,
    compute_recency_score,
    compute_relevance_score,
    compute_travel_relevance,
    compute_weather_relevance,
    detect_corroboration,
    score_signals,
)


def _signal(**overrides):
    sig = {
        "title": "Heavy rain in Goa",
        "summary": "",
        "source": "test",
        "source_type": "NEWS",
        "source_url": "https://example.com/1",
        "published_at": "2026-09-27T12:00:00+00:00",
        "location": "Goa",
        "signal_type": "WEATHER_REPORT",
        "weather_relation": "DIRECT",
        "sentiment": "UNKNOWN",
        "confidence": "LOW",
        "relevance_score": 0.0,
    }
    sig.update(overrides)
    return sig


# ---------------------------------------------------------------------------
# Location relevance
# ---------------------------------------------------------------------------

class TestLocationRelevance:
    def test_exact_destination_match(self):
        sig = _signal(location="Goa")
        assert compute_location_relevance(sig, "Goa") == 1.0

    def test_destination_in_title(self):
        sig = _signal(location="", title="Heavy rain in Goa")
        assert compute_location_relevance(sig, "Goa") == 0.7

    def test_destination_in_summary(self):
        sig = _signal(location="", title="", summary="Heavy rain in Goa")
        assert compute_location_relevance(sig, "Goa") == 0.7

    def test_unrelated_location(self):
        sig = _signal(location="Mumbai", title="Heavy rain in Mumbai")
        assert compute_location_relevance(sig, "Goa") == 0.0

    def test_partial_match(self):
        sig = _signal(location="Goa Airport", title="Flight delayed")
        assert compute_location_relevance(sig, "Goa") == 1.0

    def test_empty_destination(self):
        sig = _signal(location="Goa")
        assert compute_location_relevance(sig, "") == 0.0

    def test_case_insensitive(self):
        sig = _signal(location="goa")
        assert compute_location_relevance(sig, "Goa") == 1.0


# ---------------------------------------------------------------------------
# Weather relevance
# ---------------------------------------------------------------------------

class TestWeatherRelevance:
    def test_direct_relation(self):
        sig = _signal(weather_relation="DIRECT")
        assert compute_weather_relevance(sig, "Heavy Rain") == 1.0

    def test_indirect_relation(self):
        sig = _signal(weather_relation="INDIRECT")
        assert compute_weather_relevance(sig, "Heavy Rain") == 0.6

    def test_none_relation(self):
        sig = _signal(weather_relation="NONE")
        assert compute_weather_relevance(sig, "Heavy Rain") == 0.0

    def test_unknown_relation_with_weather_keyword(self):
        sig = _signal(weather_relation="UNKNOWN", title="Heavy rain in Goa")
        assert compute_weather_relevance(sig, "Heavy Rain") == 0.8

    def test_unknown_relation_no_weather_keyword(self):
        sig = _signal(weather_relation="UNKNOWN", title="Restaurant review")
        assert compute_weather_relevance(sig, "Heavy Rain") == 0.3

    def test_unknown_relation_with_general_weather_term(self):
        sig = _signal(weather_relation="UNKNOWN", title="Storm approaching")
        assert compute_weather_relevance(sig, "Heavy Rain") == 0.5


# ---------------------------------------------------------------------------
# Travel relevance
# ---------------------------------------------------------------------------

class TestTravelRelevance:
    def test_high_relevance_flight(self):
        sig = _signal(title="Flight cancelled at Goa airport")
        assert compute_travel_relevance(sig) == 1.0

    def test_high_relevance_road(self):
        sig = _signal(title="Road blocked in Goa")
        assert compute_travel_relevance(sig) == 1.0

    def test_medium_relevance_tourist(self):
        sig = _signal(title="Tourists stranded in Goa")
        assert compute_travel_relevance(sig) == 0.6

    def test_medium_relevance_hotel(self):
        sig = _signal(title="Hotel bookings down in Goa")
        assert compute_travel_relevance(sig) == 0.6

    def test_low_relevance_travel(self):
        sig = _signal(title="Travel tips for Goa")
        assert compute_travel_relevance(sig) == 0.6

    def test_no_relevance(self):
        sig = _signal(title="Restaurant review in Goa")
        assert compute_travel_relevance(sig) == 0.0


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------

class TestRecency:
    def test_recent_within_24h(self):
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(hours=12)).isoformat()
        assert compute_recency_score(ts, 72) == 1.0

    def test_within_time_window(self):
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(hours=48)).isoformat()
        assert compute_recency_score(ts, 72) == 0.7

    def test_within_double_window(self):
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(hours=100)).isoformat()
        assert compute_recency_score(ts, 72) == 0.4

    def test_old(self):
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(days=10)).isoformat()
        assert compute_recency_score(ts, 72) == 0.1

    def test_missing_timestamp(self):
        assert compute_recency_score(None, 72) == 0.5

    def test_empty_timestamp(self):
        assert compute_recency_score("", 72) == 0.5

    def test_future_timestamp(self):
        now = datetime.now(timezone.utc)
        ts = (now + timedelta(hours=12)).isoformat()
        assert compute_recency_score(ts, 72) == 0.5

    def test_malformed_timestamp(self):
        assert compute_recency_score("not-a-date", 72) == 0.5


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

class TestClassifySignalType:
    def test_flight_disruption(self):
        assert classify_signal_type("Flight cancelled at Goa airport") == "FLIGHT_DISRUPTION"

    def test_road_condition(self):
        assert classify_signal_type("Road blocked by landslide") == "ROAD_CONDITION"

    def test_transport_report(self):
        assert classify_signal_type("Train delayed due to rain") == "TRANSPORT_REPORT"

    def test_traveler_report(self):
        assert classify_signal_type("Tourists stranded in Goa") == "TRAVELER_REPORT"

    def test_weather_report(self):
        assert classify_signal_type("Heavy rain in Goa") == "WEATHER_REPORT"

    def test_trend(self):
        assert classify_signal_type("Goa weather trending on social media") == "TREND"

    def test_emerging_condition(self):
        assert classify_signal_type("New cafe opens in Goa") == "EMERGING_CONDITION"

    def test_empty_text(self):
        assert classify_signal_type("") == "EMERGING_CONDITION"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_low_confidence_social(self):
        sig = _signal(source_type="SOCIAL", source="Bluesky")
        assert compute_confidence(sig, [sig]) == "LOW"

    def test_medium_confidence_news(self):
        sig = _signal(source_type="NEWS", source="GDELT")
        assert compute_confidence(sig, [sig]) == "MEDIUM"

    def test_medium_confidence_official(self):
        sig = _signal(source_type="OFFICIAL", source="IMD")
        assert compute_confidence(sig, [sig]) == "MEDIUM"

    def test_high_confidence_corroborated(self):
        sig1 = _signal(source_type="NEWS", source="GDELT", title="Heavy rain in Goa")
        sig2 = _signal(source_type="SOCIAL", source="Bluesky", title="Heavy rain in Goa")
        assert compute_confidence(sig1, [sig1, sig2]) == "HIGH"
        assert compute_confidence(sig2, [sig1, sig2]) == "MEDIUM"

    def test_same_source_no_upgrade(self):
        sig1 = _signal(source_type="SOCIAL", source="Bluesky", title="Heavy rain in Goa")
        sig2 = _signal(source_type="SOCIAL", source="Bluesky", title="Heavy rain in Goa")
        assert compute_confidence(sig1, [sig1, sig2]) == "LOW"

    def test_old_signal_downgrade(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        sig = _signal(source_type="NEWS", source="GDELT", published_at=old_ts)
        assert compute_confidence(sig, [sig]) == "LOW"

    def test_multiple_same_source_upgrade(self):
        sig1 = _signal(source_type="SOCIAL", source="Bluesky", title="Rain in Goa")
        sig2 = _signal(source_type="SOCIAL", source="Bluesky", title="Flood in Goa")
        sig3 = _signal(source_type="SOCIAL", source="Bluesky", title="Storm in Goa")
        assert compute_confidence(sig1, [sig1, sig2, sig3]) == "MEDIUM"


# ---------------------------------------------------------------------------
# Corroboration
# ---------------------------------------------------------------------------

class TestCorroboration:
    def test_corroboration_detected(self):
        sig1 = _signal(source="GDELT", title="Heavy rain causing flooding in Goa")
        sig2 = _signal(source="Bluesky", title="Heavy rain causing flooding in Goa")
        result = detect_corroboration(sig1, [sig1, sig2])
        assert len(result) == 1
        assert result[0]["source"] == "Bluesky"

    def test_no_corroboration_same_source(self):
        sig1 = _signal(source="GDELT", title="Heavy rain in Goa")
        sig2 = _signal(source="GDELT", title="Heavy rain in Goa")
        result = detect_corroboration(sig1, [sig1, sig2])
        assert len(result) == 0

    def test_no_corroboration_different_topics(self):
        sig1 = _signal(source="GDELT", title="Heavy rain in Goa")
        sig2 = _signal(source="Bluesky", title="Restaurant review in Mumbai")
        result = detect_corroboration(sig1, [sig1, sig2])
        assert len(result) == 0

    def test_corroboration_different_sources_same_event(self):
        sig1 = _signal(source="GDELT", title="Flooding in Goa after heavy rain")
        sig2 = _signal(source="Bluesky", title="Goa flooding due to heavy rain")
        result = detect_corroboration(sig1, [sig1, sig2])
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Full scoring pipeline
# ---------------------------------------------------------------------------

class TestScoreSignals:
    def test_relevance_score_high(self):
        sig = _signal(
            title="Heavy rain causing flooding in Goa airport",
            location="Goa",
            weather_relation="DIRECT",
            published_at=datetime.now(timezone.utc).isoformat(),
        )
        score = compute_relevance_score(sig, "Goa", "Heavy Rain", 72)
        assert score >= 0.8

    def test_relevance_score_low(self):
        sig = _signal(
            title="Restaurant recommendations in Goa",
            location="Goa",
            weather_relation="NONE",
            published_at=datetime.now(timezone.utc).isoformat(),
        )
        score = compute_relevance_score(sig, "Goa", "Heavy Rain", 72)
        assert score < 0.6

    def test_score_signals_populates_fields(self):
        sig = _signal(title="Heavy rain in Goa", location="Goa")
        result = score_signals([sig], "Goa", "Heavy Rain", 72)
        assert result[0]["relevance_score"] > 0.0
        assert result[0]["confidence"] in ("LOW", "MEDIUM", "HIGH")
        assert result[0]["signal_type"] != ""

    def test_ranking_by_relevance(self):
        sig_high = _signal(
            title="Heavy rain causing flooding in Goa airport",
            location="Goa",
            weather_relation="DIRECT",
            source_url="https://a.com/1",
        )
        sig_low = _signal(
            title="Restaurant review in Goa",
            location="Goa",
            weather_relation="NONE",
            source_url="https://a.com/2",
        )
        result = score_signals([sig_low, sig_high], "Goa", "Heavy Rain", 72)
        result.sort(key=lambda s: s.get("relevance_score") or 0.0, reverse=True)
        assert result[0]["source_url"] == "https://a.com/1"
        assert result[1]["source_url"] == "https://a.com/2"

    def test_ranking_with_recency(self):
        now = datetime.now(timezone.utc)
        sig_recent = _signal(
            title="Heavy rain in Goa",
            location="Goa",
            weather_relation="DIRECT",
            source_url="https://a.com/1",
            published_at=now.isoformat(),
        )
        sig_old = _signal(
            title="Heavy rain in Goa",
            location="Goa",
            weather_relation="DIRECT",
            source_url="https://a.com/2",
            published_at=(now - timedelta(days=5)).isoformat(),
        )
        result = score_signals([sig_old, sig_recent], "Goa", "Heavy Rain", 72)
        result.sort(key=lambda s: s.get("relevance_score") or 0.0, reverse=True)
        assert result[0]["source_url"] == "https://a.com/1"
        assert result[1]["source_url"] == "https://a.com/2"

    def test_confidence_not_equal_to_relevance(self):
        sig = _signal(
            source_type="SOCIAL",
            source="Bluesky",
            title="Heavy rain causing severe flooding in Goa airport",
            location="Goa",
            weather_relation="DIRECT",
            published_at=datetime.now(timezone.utc).isoformat(),
        )
        result = score_signals([sig], "Goa", "Heavy Rain", 72)
        assert result[0]["relevance_score"] >= 0.8
        assert result[0]["confidence"] == "LOW"

    def test_cross_source_corroboration_increases_confidence(self):
        sig1 = _signal(
            source_type="NEWS",
            source="GDELT",
            title="Heavy rain causing flooding in Goa",
            location="Goa",
            weather_relation="DIRECT",
        )
        sig2 = _signal(
            source_type="SOCIAL",
            source="Bluesky",
            title="Heavy rain causing flooding in Goa",
            location="Goa",
            weather_relation="DIRECT",
        )
        result = score_signals([sig1, sig2], "Goa", "Heavy Rain", 72)
        assert result[0]["confidence"] == "HIGH"
        assert result[1]["confidence"] == "MEDIUM"

    def test_duplicate_source_claims_no_increase(self):
        sig1 = _signal(
            source_type="SOCIAL",
            source="Bluesky",
            title="Heavy rain in Goa",
            location="Goa",
        )
        sig2 = _signal(
            source_type="SOCIAL",
            source="Bluesky",
            title="Heavy rain in Goa",
            location="Goa",
        )
        result = score_signals([sig1, sig2], "Goa", "Heavy Rain", 72)
        assert result[0]["confidence"] == "LOW"
        assert result[1]["confidence"] == "LOW"
