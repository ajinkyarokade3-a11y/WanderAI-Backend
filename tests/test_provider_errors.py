"""Provider error hygiene: credential values must never survive in errors.

httpx embeds the request URL (including api_key=...) in HTTP errors, and
these messages reach API responses and server logs. No test here touches
the network: 429 responses are fabricated locally with fake keys.
"""

import httpx
import pytest

from backend.hotels.service import SerpApiError, search_serpapi_hotels
from backend.restaurants.service import SerpApiRestaurantError, search_serpapi_restaurants
from backend.images.service import SerpApiImageError, search_serpapi_images
from backend.weather.service import WeatherProviderError, fetch_weather

FAKE_KEY = "FAKEKEY-SECRET-12345"


def _rate_limited(url, params, timeout_s=None):
    request = httpx.Request("GET", url, params=params)
    return httpx.Response(429, request=request)


def test_hotel_429_error_redacts_key_but_keeps_status(monkeypatch):
    import backend.hotels.service as hotels

    monkeypatch.setattr(hotels, "_http_get", _rate_limited)
    with pytest.raises(SerpApiError) as exc_info:
        search_serpapi_hotels(
            FAKE_KEY, "https://serpapi.test", destination="Somewhere",
            check_in_date="2026-10-08", check_out_date="2026-10-11")
    message = str(exc_info.value)
    assert FAKE_KEY not in message
    assert "api_key=[REDACTED]" in message
    assert "429" in message  # quota classification still detectable


def test_restaurant_429_error_redacts_key(monkeypatch):
    import backend.restaurants.service as restaurants

    monkeypatch.setattr(restaurants, "_http_get", _rate_limited)
    with pytest.raises(SerpApiRestaurantError) as exc_info:
        search_serpapi_restaurants(FAKE_KEY, "https://serpapi.test", destination="Somewhere")
    message = str(exc_info.value)
    assert FAKE_KEY not in message
    assert "api_key=[REDACTED]" in message
    assert "429" in message


def test_image_429_error_redacts_key(monkeypatch):
    import backend.images.service as images

    monkeypatch.setattr(images, "_http_get", _rate_limited)
    with pytest.raises(SerpApiImageError) as exc_info:
        images.search_serpapi_images(
            FAKE_KEY, "https://serpapi.test", location="Somewhere",
            destination=None, timeout_s=5.0)
    message = str(exc_info.value)
    assert FAKE_KEY not in message
    assert "api_key=[REDACTED]" in message
    assert "429" in message


def test_weather_error_redacts_key(monkeypatch):
    import backend.weather.service as weather

    def _boom(url, params, timeout_s):
        request = httpx.Request("GET", url, params={**(params or {}), "apikey": FAKE_KEY})
        return httpx.Response(500, request=request)

    monkeypatch.setattr(weather, "_httpx_get", _boom)
    with pytest.raises(WeatherProviderError) as exc_info:
        fetch_weather(9.9, 76.2, "https://weather.test", 5.0, 5,
                      date_str="2026-10-08", api_key=FAKE_KEY)
    message = str(exc_info.value)
    assert FAKE_KEY not in message
    assert "apikey=[REDACTED]" in message
