"""Tests for the GDELT Social Signals provider.

All tests mock the HTTP layer — no real GDELT calls.
"""
import pytest

from backend.social_signals.providers import gdelt
from backend.social_signals.providers.gdelt import (
    GDELTError,
    GDELTHTTPError,
    GDELTParseError,
    GDELTTimeout,
    build_query,
    determine_signal_type,
    determine_weather_relation,
    fetch_gdelt_signals,
    is_recent,
    mentions_destination,
    mentions_weather,
    normalize_article,
    parse_seendate,
)
from backend.social_signals import service as social_service


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


def _gdelt_article(**overrides):
    article = {
        "url": "https://example.com/article/1",
        "title": "Heavy rain causes flooding in Goa",
        "domain": "example.com",
        "language": "english",
        "seendate": "20260927120000",
    }
    article.update(overrides)
    return article


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

class TestBuildQuery:
    def test_destination_only(self):
        q = build_query("Goa")
        assert '"Goa"' in q
        assert "rain" in q or "flood" in q

    def test_with_heavy_rain(self):
        q = build_query("Goa", "Heavy Rain")
        assert '"Goa"' in q
        assert "rain" in q.lower()

    def test_with_wind(self):
        q = build_query("Manali", "High Wind")
        assert '"Manali"' in q
        assert "wind" in q.lower()

    def test_with_snow(self):
        q = build_query("Shimla", "Snow")
        assert '"Shimla"' in q
        assert "snow" in q.lower()

    def test_with_unknown_weather(self):
        q = build_query("Goa", "Sunny")
        assert '"Goa"' in q
        assert '"Sunny"' in q


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

class TestParseSeendate:
    def test_valid_seendate(self):
        result = parse_seendate("20260927120000")
        assert result is not None
        assert "2026-09-27" in result
        assert "12:00" in result

    def test_empty_seendate(self):
        assert parse_seendate("") is None

    def test_none_seendate(self):
        assert parse_seendate(None) is None

    def test_invalid_seendate(self):
        assert parse_seendate("not-a-date") is None


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

class TestMentionsDestination:
    def test_title_mentions_destination(self):
        assert mentions_destination("Heavy rain in Goa", "Goa") is True

    def test_title_does_not_mention(self):
        assert mentions_destination("Heavy rain in Mumbai", "Goa") is False

    def test_case_insensitive(self):
        assert mentions_destination("Heavy rain in goa", "Goa") is True

    def test_empty_text(self):
        assert mentions_destination("", "Goa") is False

    def test_none_text(self):
        assert mentions_destination(None, "Goa") is False


class TestMentionsWeather:
    def test_rain(self):
        assert mentions_weather("Heavy rain in Goa") is True

    def test_flood(self):
        assert mentions_weather("Flooding reported") is True

    def test_storm(self):
        assert mentions_weather("Storm approaching") is True

    def test_no_weather(self):
        assert mentions_weather("Restaurant review") is False

    def test_specific_condition(self):
        assert mentions_weather("Heavy rain", "Heavy Rain") is True

    def test_empty_text(self):
        assert mentions_weather("") is False


class TestIsRecent:
    def test_recent_date(self):
        assert is_recent("2026-09-27T12:00:00+00:00", 72) is True

    def test_old_date(self):
        assert is_recent("2020-01-01T00:00:00+00:00", 72) is False

    def test_none_date(self):
        assert is_recent(None, 72) is True

    def test_invalid_date(self):
        assert is_recent("not-a-date", 72) is True


# ---------------------------------------------------------------------------
# Signal type and weather relation
# ---------------------------------------------------------------------------

class TestDetermineSignalType:
    def test_flight(self):
        assert determine_signal_type("Flight cancelled at Goa airport") == "FLIGHT_DISRUPTION"

    def test_road(self):
        assert determine_signal_type("Road blocked by landslide in Goa") == "ROAD_CONDITION"

    def test_train(self):
        assert determine_signal_type("Train delayed due to rain") == "TRANSPORT_REPORT"

    def test_tourist(self):
        assert determine_signal_type("Tourists stranded in Goa") == "TRAVELER_REPORT"

    def test_emerging(self):
        assert determine_signal_type("Heavy rain continues in Goa") == "WEATHER_REPORT"


class TestDetermineWeatherRelation:
    def test_direct_rain(self):
        assert determine_weather_relation("Heavy rain in Goa") == "DIRECT"

    def test_direct_flood(self):
        assert determine_weather_relation("Flooding in Goa") == "DIRECT"

    def test_indirect_travel(self):
        assert determine_weather_relation("Travel disrupted in Goa") == "INDIRECT"

    def test_unknown(self):
        assert determine_weather_relation("Goa restaurant review") == "UNKNOWN"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalizeArticle:
    def test_full_normalization(self):
        article = _gdelt_article()
        result = normalize_article(article, "Goa")
        assert result["title"] == "Heavy rain causes flooding in Goa"
        assert result["source"] == "example.com"
        assert result["source_type"] == "NEWS"
        assert result["source_url"] == "https://example.com/article/1"
        assert result["published_at"] is not None
        assert "2026-09-27" in result["published_at"]
        assert result["location"] == "Goa"
        assert result["signal_type"] == "WEATHER_REPORT"
        assert result["weather_relation"] == "DIRECT"
        assert result["sentiment"] == "UNKNOWN"
        assert result["confidence"] == "LOW"

    def test_missing_title(self):
        article = _gdelt_article(title=None)
        assert normalize_article(article, "Goa") is None

    def test_missing_url(self):
        article = _gdelt_article(url=None)
        assert normalize_article(article, "Goa") is None

    def test_flight_signal_type(self):
        article = _gdelt_article(title="Flight cancelled at Goa airport")
        result = normalize_article(article, "Goa")
        assert result["signal_type"] == "FLIGHT_DISRUPTION"

    def test_road_signal_type(self):
        article = _gdelt_article(title="Road blocked in Goa")
        result = normalize_article(article, "Goa")
        assert result["signal_type"] == "ROAD_CONDITION"


# ---------------------------------------------------------------------------
# Fetch with mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchGDELTSignals:
    def test_success_multiple_results(self, monkeypatch):
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "articles": [
                    _gdelt_article(url="https://a.com/1", title="Heavy rain in Goa"),
                    _gdelt_article(url="https://a.com/2", title="Flooding in Goa"),
                    _gdelt_article(url="https://a.com/3", title="Goa airport delay"),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_gdelt_signals("Goa")
        assert len(result) == 3
        assert all(s["source_type"] == "NEWS" for s in result)
        assert all(s["location"] == "Goa" for s in result)

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({"articles": []})),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_gdelt_signals("Goa")
        assert result == []

    def test_no_articles_key(self, monkeypatch):
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({})),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_gdelt_signals("Goa")
        assert result == []

    def test_timeout(self, monkeypatch):
        import httpx
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.TimeoutException("timeout"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        with pytest.raises(GDELTTimeout):
            fetch_gdelt_signals("Goa")

    def test_http_error(self, monkeypatch):
        import httpx
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.ConnectError("fail"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        with pytest.raises(GDELTHTTPError):
            fetch_gdelt_signals("Goa")

    def test_malformed_json(self, monkeypatch):
        def _fake_get(url, params, timeout):
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.json.side_effect = ValueError("bad json")
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        with pytest.raises(GDELTParseError):
            fetch_gdelt_signals("Goa")

    def test_non_dict_response(self, monkeypatch):
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([1, 2, 3])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        with pytest.raises(GDELTParseError):
            fetch_gdelt_signals("Goa")

    def test_filters_non_destination_articles(self, monkeypatch):
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "articles": [
                    _gdelt_article(url="https://a.com/1", title="Heavy rain in Goa"),
                    _gdelt_article(url="https://a.com/2", title="Heavy rain in Mumbai"),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_gdelt_signals("Goa")
        assert len(result) == 1
        assert "Goa" in result[0]["title"]

    def test_filters_old_articles(self, monkeypatch):
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "articles": [
                    _gdelt_article(url="https://a.com/1", title="Heavy rain in Goa", seendate="20200101000000"),
                    _gdelt_article(url="https://a.com/2", title="Flooding in Goa", seendate="20260927120000"),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_gdelt_signals("Goa")
        assert len(result) == 1
        assert "Flooding" in result[0]["title"]

    def test_no_api_key_required(self, monkeypatch):
        captured_params = {}
        def _fake_get(url, params, timeout):
            captured_params.update(params)
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.json.return_value = {"articles": [_gdelt_article()]}
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        fetch_gdelt_signals("Goa")
        assert "apikey" not in captured_params
        assert "key" not in captured_params

    def test_with_weather_condition(self, monkeypatch):
        captured_params = {}
        def _fake_get(url, params, timeout):
            captured_params.update(params)
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.json.return_value = {"articles": [_gdelt_article()]}
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        fetch_gdelt_signals("Goa", weather_condition="Heavy Rain")
        assert "rain" in captured_params["query"].lower()


# ---------------------------------------------------------------------------
# Service-level GDELT integration
# ---------------------------------------------------------------------------

class TestServiceGDELTIntegration:
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
            title="GDELT Test Trip",
            traveler_count=2,
            duration_days=3,
            total_budget=50000.0,
            currency="INR",
        )
        db.add(trip)
        db.commit()
        return trip

    def test_gdelt_service_success(self, monkeypatch):
        from backend.database.connection import SessionLocal
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "articles": [
                    _gdelt_article(url="https://a.com/1", title="Heavy rain in Goa"),
                    _gdelt_article(url="https://a.com/2", title="Flooding in Goa"),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            result = social_service.get_social_signals_for_trip_gdelt(db, trip)
            assert result["available"] is True
            assert len(result["signals"]) == 2
            assert result["sources"] == ["GDELT"]
            assert result["status"] == "success"
            assert result["total"] == 2
        finally:
            db.close()

    def test_gdelt_service_empty(self, monkeypatch):
        from backend.database.connection import SessionLocal
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({"articles": []})),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            result = social_service.get_social_signals_for_trip_gdelt(db, trip)
            assert result["available"] is True
            assert result["signals"] == []
            assert result["sources"] == ["GDELT"]
        finally:
            db.close()

    def test_gdelt_service_provider_failure(self, monkeypatch):
        from backend.database.connection import SessionLocal
        import httpx
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.TimeoutException("timeout"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            result = social_service.get_social_signals_for_trip_gdelt(db, trip)
            assert result["available"] is False
            assert result["reason"] == "provider_unavailable"
        finally:
            db.close()

    def test_gdelt_service_no_destination(self):
        from backend.database.connection import SessionLocal
        db = SessionLocal()
        try:
            result = social_service.get_social_signals_for_trip_gdelt(db, None)
            assert result["available"] is False
            assert result["reason"] == "no_destination"
        finally:
            db.close()

    def test_gdelt_service_caching(self, monkeypatch):
        from backend.database.connection import SessionLocal
        call_count = [0]
        def _fake_get(url, params, timeout):
            call_count[0] += 1
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.json.return_value = {"articles": [_gdelt_article()]}
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(gdelt, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        db = SessionLocal()
        try:
            trip = self._create_trip(db)
            social_service.get_social_signals_for_trip_gdelt(db, trip)
            social_service.get_social_signals_for_trip_gdelt(db, trip)
            assert call_count[0] == 1
        finally:
            db.close()
