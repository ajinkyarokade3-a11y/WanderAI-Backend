"""Pydantic schemas for the Social Signals service."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    TRAVELER_REPORT = "TRAVELER_REPORT"
    WEATHER_REPORT = "WEATHER_REPORT"
    TRANSPORT_REPORT = "TRANSPORT_REPORT"
    ROAD_CONDITION = "ROAD_CONDITION"
    FLIGHT_DISRUPTION = "FLIGHT_DISRUPTION"
    TREND = "TREND"
    EMERGING_CONDITION = "EMERGING_CONDITION"


class SourceType(str, Enum):
    SOCIAL = "SOCIAL"
    NEWS = "NEWS"
    OFFICIAL = "OFFICIAL"


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class WeatherRelation(str, Enum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SocialSignal(BaseModel):
    """One normalized social signal."""

    title: Optional[str] = None
    summary: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    observed_at: Optional[str] = None
    relevance: Optional[str] = None
    source_type: Optional[SourceType] = None
    source_url: Optional[str] = None
    published_at: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    relevance_score: Optional[float] = None
    signal_type: Optional[SignalType] = None
    sentiment: Optional[Sentiment] = None
    weather_relation: Optional[WeatherRelation] = None
    confidence: Optional[Confidence] = None


class SocialData(BaseModel):
    """Normalized social signals response."""

    available: bool = True
    signals: List[SocialSignal] = Field(default_factory=list)
    overall_risk: Optional[str] = None
    observed_at: Optional[str] = None
    sources: Optional[List[str]] = None
    generated_at: Optional[str] = None
    status: Optional[str] = None
    total: Optional[int] = None


class SocialUnavailable(BaseModel):
    """Social signals unavailable response."""

    available: bool = False
    reason: Optional[str] = None
    error: Optional[Dict[str, str]] = None


SocialResponse = SocialData | SocialUnavailable
