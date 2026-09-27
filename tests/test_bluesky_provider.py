"""Tests for the Bluesky Social Signals provider.

All tests mock the HTTP layer — no real Bluesky calls.
"""
import pytest

from backend.social_signals.providers import bluesky
from backend.social_signals.providers.bluesky import (
    BlueskyError,
    BlueskyHTTPError,
    BlueskyParseError,
    BlueskyRateLimit,
    BlueskyTimeout,
    build_query,
    construct_post_url,
    determine_signal_type,
    determine_weather_relation,
    fetch_bluesky_signals,
    is_recent,
    mentions_destination,
    mentions_weather,
    normalize_post,
    parse_timestamp,
)


def _mock_httpx_get(payload, status_code=200):
    def _fake_get(url, params, timeout):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        return resp
    return _fake_get


def _mock_httpx_get_error(exc):
    def _fake_get(url, params, timeout):
        raise exc
    return _fake_get


def _bluesky_post(**overrides):
    post = {
        "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz789",
        "cid": "bafyabc123",
        "author": {
            "did": "did:plc:abc123",
            "handle": "traveler.bsky.social",
            "displayName": "Traveler",
        },
        "record": {
            "text": "Heavy rain in Goa right now, roads are flooded",
            "createdAt": "2026-09-27T12:00:00.000Z",
            "langs": ["en"],
        },
        "indexedAt": "2026-09-27T12:00:00.000Z",
    }
    post.update(overrides)
    return post


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
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_valid_z_timestamp(self):
        result = parse_timestamp("2026-09-27T12:00:00.000Z")
        assert result is not None
        assert "2026-09-27" in result

    def test_valid_offset_timestamp(self):
        result = parse_timestamp("2026-09-27T12:00:00+00:00")
        assert result is not None
        assert "2026-09-27" in result

    def test_empty_timestamp(self):
        assert parse_timestamp("") is None

    def test_none_timestamp(self):
        assert parse_timestamp(None) is None

    def test_invalid_timestamp(self):
        assert parse_timestamp("not-a-date") is None


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

class TestConstructPostUrl:
    def test_valid_uri_and_handle(self):
        url = construct_post_url(
            "at://did:plc:abc123/app.bsky.feed.post/xyz789",
            "traveler.bsky.social",
        )
        assert url == "https://bsky.app/profile/traveler.bsky.social/post/xyz789"

    def test_empty_uri(self):
        assert construct_post_url("", "handle") == ""

    def test_empty_handle(self):
        assert construct_post_url("at://did:plc:abc/app.bsky.feed.post/xyz", "") == ""

    def test_none_uri(self):
        assert construct_post_url(None, "handle") == ""


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

class TestMentionsDestination:
    def test_text_mentions_destination(self):
        assert mentions_destination("Heavy rain in Goa", "Goa") is True

    def test_text_does_not_mention(self):
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

class TestNormalizePost:
    def test_full_normalization(self):
        post = _bluesky_post()
        result = normalize_post(post, "Goa")
        assert result["title"] == "Heavy rain in Goa right now, roads are flooded"
        assert result["summary"] == "Heavy rain in Goa right now, roads are flooded"
        assert result["source"] == "Bluesky"
        assert result["source_type"] == "SOCIAL"
        assert result["source_url"] == "https://bsky.app/profile/traveler.bsky.social/post/xyz789"
        assert result["published_at"] is not None
        assert "2026-09-27" in result["published_at"]
        assert result["location"] == "Goa"
        assert result["signal_type"] == "ROAD_CONDITION"
        assert result["weather_relation"] == "DIRECT"
        assert result["sentiment"] == "UNKNOWN"
        assert result["confidence"] == "LOW"

    def test_long_text_truncated_title(self):
        long_text = "A" * 150
        post = _bluesky_post()
        post["record"] = {"text": long_text, "createdAt": "2026-09-27T12:00:00.000Z"}
        result = normalize_post(post, "Goa")
        assert len(result["title"]) == 103
        assert result["title"].endswith("...")

    def test_missing_text(self):
        post = _bluesky_post()
        post["record"] = {"text": "", "createdAt": "2026-09-27T12:00:00.000Z"}
        assert normalize_post(post, "Goa") is None

    def test_missing_created_at(self):
        post = _bluesky_post()
        post["record"] = {"text": "Heavy rain in Goa", "createdAt": ""}
        assert normalize_post(post, "Goa") is None

    def test_flight_signal_type(self):
        post = _bluesky_post()
        post["record"] = {"text": "Flight cancelled at Goa airport", "createdAt": "2026-09-27T12:00:00.000Z"}
        result = normalize_post(post, "Goa")
        assert result["signal_type"] == "FLIGHT_DISRUPTION"

    def test_tourist_signal_type(self):
        post = _bluesky_post()
        post["record"] = {"text": "Tourists stranded in Goa due to rain", "createdAt": "2026-09-27T12:00:00.000Z"}
        result = normalize_post(post, "Goa")
        assert result["signal_type"] == "TRAVELER_REPORT"


# ---------------------------------------------------------------------------
# Fetch with mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchBlueskySignals:
    def test_success_multiple_results(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "posts": [
                    _bluesky_post(),
                    _bluesky_post(
                        uri="at://did:plc:def456/app.bsky.feed.post/abc123",
                        author={"did": "did:plc:def456", "handle": "news.bsky.social", "displayName": "News"},
                        record={"text": "Flooding in Goa after heavy rain", "createdAt": "2026-09-27T11:00:00.000Z"},
                    ),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_bluesky_signals("Goa")
        assert len(result) == 2
        assert all(s["source_type"] == "SOCIAL" for s in result)
        assert all(s["location"] == "Goa" for s in result)
        assert all(s["confidence"] == "LOW" for s in result)

    def test_empty_response(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({"posts": []})),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_bluesky_signals("Goa")
        assert result == []

    def test_no_posts_key(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({})),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_bluesky_signals("Goa")
        assert result == []

    def test_timeout(self, monkeypatch):
        import httpx
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.TimeoutException("timeout"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        with pytest.raises(BlueskyTimeout):
            fetch_bluesky_signals("Goa")

    def test_http_error(self, monkeypatch):
        import httpx
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.ConnectError("fail"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        with pytest.raises(BlueskyHTTPError):
            fetch_bluesky_signals("Goa")

    def test_rate_limit(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({}, status_code=429)),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        with pytest.raises(BlueskyRateLimit):
            fetch_bluesky_signals("Goa")

    def test_http_500(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({}, status_code=500)),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        with pytest.raises(BlueskyHTTPError):
            fetch_bluesky_signals("Goa")

    def test_malformed_json(self, monkeypatch):
        def _fake_get(url, params, timeout):
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.status_code = 200
            resp.json.side_effect = ValueError("bad json")
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        with pytest.raises(BlueskyParseError):
            fetch_bluesky_signals("Goa")

    def test_non_dict_response(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get([1, 2, 3])),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        with pytest.raises(BlueskyParseError):
            fetch_bluesky_signals("Goa")

    def test_filters_non_destination_posts(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "posts": [
                    _bluesky_post(),
                    _bluesky_post(
                        uri="at://did:plc:def456/app.bsky.feed.post/abc123",
                        author={"did": "did:plc:def456", "handle": "news.bsky.social", "displayName": "News"},
                        record={"text": "Heavy rain in Mumbai", "createdAt": "2026-09-27T11:00:00.000Z"},
                    ),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_bluesky_signals("Goa")
        assert len(result) == 1
        assert "Goa" in result[0]["summary"]

    def test_filters_old_posts(self, monkeypatch):
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get({
                "posts": [
                    _bluesky_post(),
                    _bluesky_post(
                        uri="at://did:plc:def456/app.bsky.feed.post/abc123",
                        author={"did": "did:plc:def456", "handle": "news.bsky.social", "displayName": "News"},
                        record={"text": "Heavy rain in Goa", "createdAt": "2020-01-01T00:00:00.000Z"},
                    ),
                ],
            })),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        result = fetch_bluesky_signals("Goa")
        assert len(result) == 1
        assert "2026" in result[0]["published_at"]

    def test_with_weather_condition(self, monkeypatch):
        captured_params = {}
        def _fake_get(url, params, timeout):
            captured_params.update(params)
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"posts": [_bluesky_post()]}
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        fetch_bluesky_signals("Goa", weather_condition="Heavy Rain")
        assert "rain" in captured_params["q"].lower()

    def test_no_auth_required(self, monkeypatch):
        captured_headers = {}
        def _fake_get(url, params, timeout):
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"posts": [_bluesky_post()]}
            resp.raise_for_status.return_value = None
            return resp
        monkeypatch.setattr(bluesky, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_fake_get),
            "TimeoutException": Exception,
            "HTTPError": Exception,
        }))
        fetch_bluesky_signals("Goa")
        assert "Authorization" not in captured_headers
