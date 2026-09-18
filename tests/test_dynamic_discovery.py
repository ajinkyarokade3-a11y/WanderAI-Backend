"""Unit tests for bounded additional-activity research in dynamic discovery.

The discovery service must top up verified, destination-specific activities when
a research attempt returns fewer than required -- without fabricating candidates
or relaxing verification -- and stay fail-closed when enough verified candidates
do not exist.
"""
import secrets
import uuid
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config

from backend.database.connection import SessionLocal
from backend.dynamic_destination.service import (
    MAX_ACTIVITY_RESEARCH_ATTEMPTS,
    DynamicDestinationDiscoveryError,
    DynamicDestinationDiscoveryService,
)
from backend.models.models import Activity, Destination, Hotel, TransportOption


@pytest.fixture(scope="session", autouse=True)
def _migrated():
    command.upgrade(Config("alembic.ini"), "head")


def _evidence(label):
    return [{
        "url": f"https://example.com/{label}",
        "label": label,
        "supports": ["existence", "coordinates"],
    }]


def _activity(name, idx=0):
    return {
        "name": name,
        "category": "culture",
        "area": name,
        "description": f"Verified visit to {name}.",
        "duration_hours": 3.0,
        "price_per_person": 500.0,
        "currency": "INR",
        "difficulty_level": "easy",
        "latitude": 20.0 + idx * 0.1,
        "longitude": 70.0 + idx * 0.1,
        "evidence": _evidence(name.lower().replace(" ", "-")),
    }


def _payload(destination, activity_names):
    return {
        "destination": {
            "name": destination,
            "country": "India",
            "state_region": destination,
            "description": f"Verified travel catalog for {destination}.",
            "best_time_to_visit": "October to March",
            "latitude": 22.0,
            "longitude": 71.0,
            "regions": ["western India"],
            "evidence": _evidence(f"{destination.lower().replace(' ', '-')}-destination"),
        },
        "activities": [_activity(name, idx) for idx, name in enumerate(activity_names)],
        "hotels": [{
            "name": f"{destination} Heritage Stay",
            "category": "boutique",
            "address": f"Main Road, {destination}",
            "description": "Verified heritage hotel.",
            "price_per_night": 6500.0,
            "currency": "INR",
            "rating": 4.5,
            "latitude": 22.0,
            "longitude": 71.0,
            "evidence": _evidence("heritage-stay"),
        }],
        "transport_options": [{
            "name": f"{destination} regional private cab",
            "type": "private_cab",
            "route_from": "Traveler origin",
            "route_to": destination,
            "duration_hours": 5.0,
            "price": 12000.0,
            "currency": "INR",
            "capacity": 4,
            "latitude": 22.0,
            "longitude": 71.0,
            "features": ["driver"],
            "evidence": _evidence("regional-cab"),
        }],
    }


class _ScriptedResearch:
    """Fake research client replaying canned payloads while recording contexts."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.contexts = []

    def discover_destination_inventory(self, context):
        import copy
        self.contexts.append(dict(context))
        if len(self.contexts) <= len(self._payloads):
            return copy.deepcopy(self._payloads[len(self.contexts) - 1])
        return copy.deepcopy(self._payloads[-1])


def _trip(duration_days=6):
    return SimpleNamespace(
        duration_days=duration_days,
        total_budget=90000.0,
        currency="INR",
        traveler_count=2,
        pace="balanced",
        preferences=None,
    )


def _cleanup(session_id):
    db = SessionLocal()
    try:
        db.query(Activity).filter(Activity.discovery_session_id == session_id).delete()
        db.query(Hotel).filter(Hotel.discovery_session_id == session_id).delete()
        db.query(TransportOption).filter(TransportOption.discovery_session_id == session_id).delete()
        db.query(Destination).filter(Destination.discovery_session_id == session_id).delete()
        db.commit()
    finally:
        db.close()


def test_no_additional_research_when_five_plus_verified():
    destination = f"Testhaven {uuid.uuid4().hex[:8]}"
    session_id = secrets.token_hex(16)
    names = [f"Old Fort {i}" for i in range(6)]
    research = _ScriptedResearch([_payload(destination, names)])
    db = SessionLocal()
    try:
        result = DynamicDestinationDiscoveryService(db, research).discover_and_persist(
            _trip(duration_days=6), destination, session_id
        )
        assert result.inventory_source == "discovered"
        assert result.verification_status == "verified_candidate"
        stored = sorted(
            a.title for a in db.query(Activity).filter(
                Activity.discovery_session_id == session_id
            ).all()
        )
        assert stored == sorted(names)
    finally:
        db.close()
        _cleanup(session_id)
    assert len(research.contexts) == 1


def test_additional_research_tops_up_when_four_verified():
    destination = f"Testhaven {uuid.uuid4().hex[:8]}"
    session_id = secrets.token_hex(16)
    first = ["Fort A", "Fort B", "Fort C", "Fort D"]
    second = ["Fort C", "Fort D", "Fort E", "Fort F"]
    research = _ScriptedResearch([
        _payload(destination, first),
        _payload(destination, second),
    ])
    db = SessionLocal()
    try:
        result = DynamicDestinationDiscoveryService(db, research).discover_and_persist(
            _trip(duration_days=6), destination, session_id
        )
        assert result.inventory_source == "discovered"
        stored = sorted(
            a.title for a in db.query(Activity).filter(
                Activity.discovery_session_id == session_id
            ).all()
        )
        # Merged without duplicates: 4 + 2 new = 6 >= 5 required.
        assert stored == sorted(["Fort A", "Fort B", "Fort C", "Fort D", "Fort E", "Fort F"])
    finally:
        db.close()
        _cleanup(session_id)
    assert len(research.contexts) == 2
    follow_up = research.contexts[1]
    assert sorted(follow_up["already_verified_activity_names"]) == sorted(first)
    # Coverage target is one activity per day (6), not just the minimum (5).
    assert follow_up["additional_activities_needed"] == 2
    assert follow_up["destination"] == destination


def test_fail_closed_when_candidates_unverified():
    destination = f"Testhaven {uuid.uuid4().hex[:8]}"
    session_id = secrets.token_hex(16)
    bad = _payload(destination, ["Fort A", "Fort B"])
    bad["activities"][0]["evidence"] = [{"url": "https://example.com/fort-a"}]
    research = _ScriptedResearch([bad])
    db = SessionLocal()
    try:
        with pytest.raises(DynamicDestinationDiscoveryError, match="source URL alone is insufficient"):
            DynamicDestinationDiscoveryService(db, research).discover_and_persist(
                _trip(duration_days=6), destination, session_id
            )
    finally:
        db.close()
        _cleanup(session_id)
    # Invalid candidates fail fast: no retry can verify them.
    assert len(research.contexts) == 1
    db = SessionLocal()
    try:
        assert db.query(Destination).filter(
            Destination.discovery_session_id == session_id
        ).count() == 0
    finally:
        db.close()


def test_fail_closed_when_shortfall_persists():
    destination = f"Testhaven {uuid.uuid4().hex[:8]}"
    session_id = secrets.token_hex(16)
    research = _ScriptedResearch([_payload(destination, ["Fort A", "Fort B"])])
    db = SessionLocal()
    try:
        with pytest.raises(DynamicDestinationDiscoveryError, match="2 verified activities; 5 are required"):
            DynamicDestinationDiscoveryService(db, research).discover_and_persist(
                _trip(duration_days=6), destination, session_id
            )
    finally:
        db.close()
        _cleanup(session_id)
    # Bounded: initial attempt plus follow-ups, then fail-closed with no persist.
    assert len(research.contexts) == MAX_ACTIVITY_RESEARCH_ATTEMPTS
    db = SessionLocal()
    try:
        assert db.query(Destination).filter(
            Destination.discovery_session_id == session_id
        ).count() == 0
    finally:
        db.close()
