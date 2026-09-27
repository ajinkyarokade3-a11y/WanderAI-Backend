"""Tests for the Google News RSS Social Signals provider."""

import pytest

from backend.social_signals.providers import rss
from backend.social_signals.providers.rss import (
    RSSHTTPError,
    RSSParseError,
    RSSRateLimit,
    RSSTimeout,
    build_query,
    determine_signal_type,
    determine_weather_relation,
    fetch_rss_signals,
    is_recent,
    mentions_destination,
    normalize_item,
    parse_rss_date,
)


def _mock_httpx_get(payload, status_code=200):
    def _fake_get(url, params, headers=None, timeout=None):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = payload
        resp.headers = {"content-type": "application/xml"}
        return resp
    return _fake_get


def _mock_httpx_get_error(exc):
    def _fake_get(url, params, headers=None, timeout=None):
        raise exc
    return _fake_get


_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Heavy rain in Manali causes flooding</title>
      <link>https://example.com/article/1</link>
      <pubDate>Mon, 27 Sep 2026 12:00:00 GMT</pubDate>
      <description>Heavy rain causing flooding in Manali</description>
    </item>
    <item>
      <title>Manali airport closed due to storm</title>
      <link>https://example.com/article/2</link>
      <pubDate>Mon, 27 Sep 2026 11:00:00 GMT</pubDate>
      <description>Airport closed</description>
    </item>
    <item>
      <title>Restaurant review in Manali</title>
      <link>https://example.com/article/3</link>
      <pubDate>Mon, 27 Sep 2026 10:00:00 GMT</pubDate>
      <description>Great food</description>
    </item>
  </channel>
</rss>
"""


class TestBuildQuery:
    def test_destination_only(self):
        q = build_query("Manali")
        assert '"Manali"' in q

    def test_with_weather(self):
        q = build_query("Manali", "Heavy Rain")
        assert '"Manali"' in q
        assert "rain" in q.lower()


class TestParseRssDate:
    def test_valid_rfc_date(self):
        result = parse_rss_date("Mon, 27 Sep 2026 12:00:00 GMT")
        assert result is not None
        assert "2026-09-27" in result

    def test_empty_date(self):
        assert parse_rss_date("") is None

    def test_none_date(self):
        assert parse_rss_date(None) is None

    def test_invalid_date(self):
        assert parse_rss_date("not-a-date") is None


class TestMentionsDestination:
    def test_mentions(self):
        assert mentions_destination("Heavy rain in Manali", "Manali") is True

    def test_not_mentions(self):
        assert mentions_destination("Heavy rain in Goa", "Manali") is False


class TestDetermineSignalType:
    def test_weather(self):
        assert determine_signal_type("Heavy rain in Manali") == "WEATHER_REPORT"

    def test_flight(self):
        assert determine_signal_type("Flight cancelled at Manali airport") == "FLIGHT_DISRUPTION"


class TestDetermineWeatherRelation:
    def test_direct(self):
        assert determine_weather_relation("Heavy rain in Manali") == "DIRECT"

    def test_unknown(self):
        assert determine_weather_relation("Restaurant review") == "UNKNOWN"


class TestNormalizeItem:
    def test_full_normalization(self):
        item = {
            "title": "Heavy rain in Manali",
            "link": "https://example.com/1",
            "pubDate": "Mon, 27 Sep 2026 12:00:00 GMT",
            "description": "Flooding reported",
        }
        result = normalize_item(item, "Manali")
        assert result["title"] == "Heavy rain in Manali"
        assert result["source"] == "Google News"
        assert result["source_type"] == "NEWS"
        assert result["source_url"] == "https://example.com/1"
        assert result["location"] == "Manali"
        assert result["signal_type"] == "WEATHER_REPORT"
        assert result["weather_relation"] == "DIRECT"
        assert result["confidence"] == "LOW"

    def test_missing_title(self):
        item = {"link": "https://example.com/1", "pubDate": "Mon, 27 Sep 2026 12:00:00 GMT"}
        assert normalize_item(item, "Manali") is None

    def test_missing_link(self):
        item = {"title": "Test", "pubDate": "Mon, 27 Sep 2026 12:00:00 GMT"}
        assert normalize_item(item, "Manali") is None


class TestFetchRSSSignals:
    def test_success(self, monkeypatch):
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get(_RSS_XML)),
        }))
        result = fetch_rss_signals("Manali")
        assert len(result) >= 1
        assert all(s["source_type"] == "NEWS" for s in result)

    def test_empty_response(self, monkeypatch):
        empty_xml = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get(empty_xml)),
        }))
        result = fetch_rss_signals("Manali")
        assert result == []

    def test_timeout(self, monkeypatch):
        import httpx
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.TimeoutException("timeout"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        with pytest.raises(RSSTimeout):
            fetch_rss_signals("Manali")

    def test_http_error(self, monkeypatch):
        import httpx
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get_error(httpx.ConnectError("fail"))),
            "TimeoutException": httpx.TimeoutException,
            "HTTPError": httpx.HTTPError,
        }))
        with pytest.raises(RSSHTTPError):
            fetch_rss_signals("Manali")

    def test_rate_limit(self, monkeypatch):
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get("", status_code=429)),
        }))
        with pytest.raises(RSSRateLimit):
            fetch_rss_signals("Manali")

    def test_malformed_xml(self, monkeypatch):
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get("not xml at all")),
        }))
        with pytest.raises(RSSParseError):
            fetch_rss_signals("Manali")

    def test_filters_non_destination(self, monkeypatch):
        xml_with_goa = """<?xml version="1.0"?>
        <rss version="2.0"><channel>
        <item>
          <title>Heavy rain in Goa</title>
          <link>https://example.com/goa</link>
          <pubDate>Mon, 27 Sep 2026 12:00:00 GMT</pubDate>
          <description>Goa rain</description>
        </item>
        </channel></rss>"""
        monkeypatch.setattr(rss, "httpx", type("MockHttpx", (), {
            "get": staticmethod(_mock_httpx_get(xml_with_goa)),
        }))
        result = fetch_rss_signals("Manali")
        assert len(result) == 0
