"""Bidirectional traveler<->operator chat: gating, shared thread, isolation.

Covers the full business flow through REAL APIs (no service bypass):
confirm (traveler) -> approve -> accept (operator) -> chat enabled ->
traveler+operator exchange on the SAME rows -> history survives a fresh
session -> cross-traveler access denied -> pre-confirm/pre-accept rejected
-> internal TripMessage rows stay traveler-invisible -> init idempotent ->
chronological ordering.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.auth.service import issue_traveler_token
from backend.database.connection import SessionLocal
from backend.main import app
from backend.models.models import TripMessage, User

client = TestClient(app)

OPERATOR_EMAIL = "ops.chat.test@tourflow.ai"
OPERATOR_PASSWORD = "ops-chat-password-123"


def _ensure_user(email, full_name, role):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            user = User(id=str(uuid.uuid4()), email=email, full_name=full_name,
                        role=role, is_active=True)
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()


@pytest.fixture()
def operator_headers(monkeypatch):
    monkeypatch.setenv("OPERATOR_LOGIN_PASSWORD", OPERATOR_PASSWORD)
    _ensure_user(OPERATOR_EMAIL, "Chat Operator", "operator")
    res = client.post("/api/auth/operator-login",
                      json={"email": OPERATOR_EMAIL, "password": OPERATOR_PASSWORD})
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['token']}"}


def _signup_traveler(name="Chat Traveler"):
    email = f"chat_{uuid.uuid4().hex[:8]}@example.com"
    res = client.post("/api/auth/traveler/signup",
                      json={"full_name": name, "email": email,
                            "password": "StrongPass123"})
    assert res.status_code == 201, res.text
    body = res.json()
    return body["user"], body["token"]


def _traveler_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _make_confirmable_trip(user_id):
    goa_id = client.get("/api/destinations/goa").json()["id"]
    res = client.post("/api/trips", json={
        "title": "Chat Flow Trip", "destination_id": goa_id,
        "duration_days": 2, "traveler_count": 2,
        "total_budget": 60000.0, "currency": "INR",
        "user_id": user_id,
        "start_date": "2026-11-10T00:00:00",
        "end_date": "2026-11-11T00:00:00",
    })
    assert res.status_code == 200, res.text
    return res.json()["id"]


def _eligible_trip(operator_headers, traveler_user, traveler_token):
    """Confirm -> approve -> accept; returns trip_id with chat enabled."""
    trip_id = _make_confirmable_trip(traveler_user["id"])
    assert client.post(f"/api/trips/{trip_id}/confirm",
                       json={"user_id": traveler_user["id"]}).status_code == 200
    assert client.post(f"/api/ops/trips/{trip_id}/approve",
                       json={}, headers=operator_headers).status_code == 200
    accept = client.post(f"/api/ops/trips/{trip_id}/accept",
                         json={}, headers=operator_headers)
    assert accept.status_code == 200, accept.text
    check = client.get(f"/api/trips/{trip_id}/chat",
                       headers=_traveler_headers(traveler_token))
    assert check.status_code == 200, check.text
    assert check.json()["enabled"] is True
    return trip_id


def test_chat_rejected_before_traveler_confirmation(operator_headers):
    user, token = _signup_traveler()
    trip_id = None
    try:
        trip_id = _make_confirmable_trip(user["id"])
        res = client.get(f"/api/trips/{trip_id}/chat",
                         headers=_traveler_headers(token))
        assert res.status_code == 409
        assert res.json()["detail"].endswith("waiting_for_traveler_confirmation")
        send = client.post(f"/api/trips/{trip_id}/chat/messages",
                           json={"body": "Hello?"},
                           headers=_traveler_headers(token))
        assert send.status_code == 409
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_chat_rejected_before_operator_acceptance(operator_headers):
    user, token = _signup_traveler()
    trip_id = None
    try:
        trip_id = _make_confirmable_trip(user["id"])
        assert client.post(f"/api/trips/{trip_id}/confirm",
                           json={"user_id": user["id"]}).status_code == 200
        assert client.post(f"/api/ops/trips/{trip_id}/approve",
                           json={}, headers=operator_headers).status_code == 200
        res = client.get(f"/api/trips/{trip_id}/chat",
                         headers=_traveler_headers(token))
        assert res.status_code == 409
        assert res.json()["detail"].endswith("waiting_for_operator_acceptance")
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")

def test_full_bidirectional_flow_same_conversation(operator_headers):
    from unittest.mock import patch

    import backend.images.service as images_service

    user, token = _signup_traveler()
    theaders = _traveler_headers(token)
    trip_id = None
    try:
        with patch.object(images_service, "get_real_images_for_location",
                          return_value=[]):
            trip_id = _eligible_trip(operator_headers, user, token)
        opened = client.get(f"/api/trips/{trip_id}/chat", headers=theaders).json()
        assert opened["trip_id"] == trip_id
        assert opened["enabled"] is True
        assert opened["messages"] == []
        assert opened["traveler"]["id"] == user["id"]
        sent = client.post(f"/api/trips/{trip_id}/chat/messages",
                           json={"body": "Can I change my hotel?"},
                           headers=theaders)
        assert sent.status_code == 200, sent.text
        assert sent.json()["sender_type"] == "traveler"
        assert sent.json()["sender_id"] == user["id"]
        seen = client.get(f"/api/ops/trips/{trip_id}/chat",
                          headers=operator_headers).json()
        assert [m["body"] for m in seen["messages"]] == ["Can I change my hotel?"]
        assert seen["unread_count"] == 1
        reply = client.post(f"/api/ops/trips/{trip_id}/chat/messages",
                            json={"body": "Yes, I can check alternatives."},
                            headers=operator_headers)
        assert reply.status_code == 200, reply.text
        assert reply.json()["sender_type"] == "operator"
        thread = client.get(f"/api/trips/{trip_id}/chat", headers=theaders).json()
        assert [m["sender_type"] for m in thread["messages"]] == [
            "traveler", "operator"]
        assert thread["messages"][0]["body"] == "Can I change my hotel?"
        assert thread["latest_message"]["body"] == "Yes, I can check alternatives."
        spoof = client.post(f"/api/trips/{trip_id}/chat/messages",
                            json={"body": "Trying spoof",
                                  "sender_type": "operator",
                                  "sender_id": "someone-else"},
                            headers=theaders)
        assert spoof.status_code == 422
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_messages_survive_fresh_session(operator_headers):
    from backend.models.models import TravelerOperatorChatMessage

    user, token = _signup_traveler()
    trip_id = None
    try:
        trip_id = _eligible_trip(operator_headers, user, token)
        client.post(f"/api/trips/{trip_id}/chat/messages",
                    json={"body": "Persist me"},
                    headers=_traveler_headers(token))
        db = SessionLocal()
        try:
            rows = db.query(TravelerOperatorChatMessage).filter(
                TravelerOperatorChatMessage.trip_id == trip_id).all()
            assert len(rows) == 1 and rows[0].body == "Persist me"
        finally:
            db.close()
        reread = client.get(
            f"/api/trips/{trip_id}/chat",
            headers=_traveler_headers(token)).json()
        assert [m["body"] for m in reread["messages"]] == ["Persist me"]
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_traveler_cannot_access_another_travelers_chat(operator_headers):
    user_a, token_a = _signup_traveler("Traveler A")
    user_b, token_b = _signup_traveler("Traveler B")
    trip_id = None
    try:
        trip_id = _eligible_trip(operator_headers, user_a, token_a)
        assert client.get(
            f"/api/trips/{trip_id}/chat",
            headers=_traveler_headers(token_b)).status_code == 404
        assert client.post(
            f"/api/trips/{trip_id}/chat/messages", json={"body": "Hi"},
            headers=_traveler_headers(token_b)).status_code == 404
        assert client.get(f"/api/trips/{trip_id}/chat").status_code == 401
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_operator_init_is_idempotent(operator_headers):
    user, token = _signup_traveler()
    trip_id = None
    try:
        trip_id = _eligible_trip(operator_headers, user, token)
        first = client.post(f"/api/ops/trips/{trip_id}/chat",
                            json={"body": "Welcome aboard!"},
                            headers=operator_headers)
        assert first.status_code == 200, first.text
        second = client.post(f"/api/ops/trips/{trip_id}/chat",
                             json={"body": "Different text"},
                             headers=operator_headers)
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
        thread = client.get(f"/api/ops/trips/{trip_id}/chat",
                            headers=operator_headers).json()
        assert len(thread["messages"]) == 1
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_internal_notes_stay_traveler_invisible(operator_headers):
    user, token = _signup_traveler()
    trip_id = None
    try:
        trip_id = _eligible_trip(operator_headers, user, token)
        client.post(f"/api/ops/trips/{trip_id}/messages", json={
            "trip_id": trip_id, "body": "INTERNAL VENDOR COST 9999",
            "category": "operational"})
        db = SessionLocal()
        try:
            db.add(TripMessage(trip_id=trip_id, operator_name="op",
                               category="operational",
                               body="INTERNAL private decision",
                               is_urgent=False))
            db.commit()
        finally:
            db.close()
        thread = client.get(
            f"/api/trips/{trip_id}/chat",
            headers=_traveler_headers(token)).json()
        blob = " ".join(m["body"] for m in thread["messages"])
        assert "INTERNAL" not in blob and "9999" not in blob
        assert "private decision" not in blob
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")


def test_chat_overview_lists_eligible_trips(operator_headers):
    user, token = _signup_traveler()
    trip_id = None
    try:
        trip_id = _eligible_trip(operator_headers, user, token)
        client.post(f"/api/trips/{trip_id}/chat/messages",
                    json={"body": "Overview check"},
                    headers=_traveler_headers(token))
        overview = client.get("/api/ops/chats/overview",
                              headers=operator_headers)
        assert overview.status_code == 200, overview.text
        entry = next((e for e in overview.json() if e["trip_id"] == trip_id),
                     None)
        assert entry is not None
        assert entry["unread_count"] == 1
        assert entry["latest_message"]["body"] == "Overview check"
    finally:
        if trip_id:
            client.delete(f"/api/trips/{trip_id}")
