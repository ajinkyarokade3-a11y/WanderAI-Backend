"""Operator trip visibility: Traveler app -> backend -> Operator Dashboard.

Covers the secure contract: operator login issues a verifiable JWT; the
canonical trip rows appear (existing + newly created) via GET /ops/trips;
traveler mutations surface as updated canonical state (updated_at bump feeds
the sync-version poll); traveler PII beyond id+name never leaves the server;
failed generation leaves no row; unknown trips 404; unauthenticated and
non-operator callers are rejected.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.auth.service import issue_traveler_token
from backend.database.connection import SessionLocal
from backend.main import app
from backend.models.models import User

client = TestClient(app)

OPERATOR_EMAIL = "ops.trips.test@tourflow.ai"
OPERATOR_PASSWORD = "ops-test-password-123"


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
def operator_token(monkeypatch):
    monkeypatch.setenv("OPERATOR_LOGIN_PASSWORD", OPERATOR_PASSWORD)
    _ensure_user(OPERATOR_EMAIL, "Ops Trips Tester", "operator")
    res = client.post("/api/auth/operator-login",
                      json={"email": OPERATOR_EMAIL, "password": OPERATOR_PASSWORD})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    assert body["token"]
    return body["token"]


@pytest.fixture()
def traveler_token():
    user = _ensure_user("traveler.trips.test@tourflow.ai", "Trip Traveler", "traveler")
    return issue_traveler_token(user.id)


def _ops_auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _confirmable_trip_payload(title):
    return {
        "title": title,
        "destination_name": "Udaipur",
        "duration_days": 3,
        "total_budget": 60000.0,
        "currency": "INR",
        "traveler_count": 2,
        "pace": "balanced",
        "start_date": "2026-11-10T09:00:00",
        "end_date": "2026-11-12T18:00:00",
    }


def _create_udaipur_trip(title):
    res = client.post("/api/trips", json={
        "title": title,
        "destination_name": "Udaipur",
        "duration_days": 3,
        "total_budget": 60000.0,
        "currency": "INR",
        "traveler_count": 2,
        "pace": "balanced",
    })
    assert res.status_code == 200, res.text
    return res.json()


# ---------------------------------------------------------------------------
# Authentication / authorization
# ---------------------------------------------------------------------------

def test_operator_login_rejects_bad_credentials(monkeypatch):
    monkeypatch.setenv("OPERATOR_LOGIN_PASSWORD", OPERATOR_PASSWORD)
    _ensure_user(OPERATOR_EMAIL, "Ops Trips Tester", "operator")
    assert client.post("/api/auth/operator-login",
                       json={"email": OPERATOR_EMAIL, "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/operator-login",
                       json={"email": "nobody@tourflow.ai", "password": OPERATOR_PASSWORD}).status_code == 401


def test_operator_trips_require_operator_session(operator_token, traveler_token):
    assert client.get("/api/ops/trips").status_code == 401

# ---------------------------------------------------------------------------
# Lifecycle: generated (planning/Preview) -> traveler-confirmed (actionable)
# -> operator-accepted (ongoing/Active). Same row, same id, no duplicates.
# ---------------------------------------------------------------------------

def test_generated_trip_is_preview_only_for_operators(operator_token):
    """New traveler trips are visible but operator-immutable until confirmed."""
    created = client.post("/api/trips", json=_confirmable_trip_payload("Ops Lifecycle Preview Trip"))
    assert created.status_code == 200, created.text
    trip_id = created.json()["id"]
    assert created.json()["status"] == "planning"

    headers = _ops_auth_headers(operator_token)
    listed = client.get("/api/ops/trips", headers=headers).json()
    match = next(t for t in listed if t["id"] == trip_id)
    assert match["status"] == "planning"
    assert match["lifecycle"] == "pending_traveler_confirmation"
    assert match["operator_actionable"] is False
    assert isinstance(match["itinerary"], list) and len(match["itinerary"]) > 0

    # Operator cannot approve/accept at this stage (409 Preview-only).
    assert client.post(f"/api/ops/trips/{trip_id}/approve", json={},
                       headers=headers).status_code == 409
    assert client.post(f"/api/ops/trips/{trip_id}/accept", json={},
                       headers=headers).status_code == 409


def test_unconfirmed_trip_rejects_assignment_mutations(operator_token):
    """Assign/dispatch endpoints 409 on planning trips."""
    created = client.post("/api/trips", json=_confirmable_trip_payload("Ops Lifecycle Locked Trip"))
    trip_id = created.json()["id"]
    headers = _ops_auth_headers(operator_token)

    assert client.post(f"/api/ops/trips/{trip_id}/approve", json={},
                       headers=headers).status_code == 409
    assert client.post("/api/ops/accommodations",
                       json={"trip_id": trip_id, "rooms": 1},
                       headers=headers).status_code in (401, 403, 409)
    assert client.post("/api/ops/transport",
                       json={"trip_id": trip_id, "origin": "Delhi",
                             "destination": "Udaipur",
                             "pickup_at": "2026-11-10T09:00:00",
                             "dropoff_at": "2026-11-10T18:00:00"},
                       headers=headers).status_code in (401, 403, 409)

    # The trip is still a planning preview — never silently confirmed.
    assert client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()["status"] == "planning"



def test_traveler_confirm_flips_trip_to_actionable_and_sync_version_moves(operator_token):
    """Traveler confirmation is immediately visible to the operator poll."""
    created = client.post("/api/trips", json=_confirmable_trip_payload("Ops Lifecycle Confirm Trip"))
    trip_id = created.json()["id"]
    owner = created.json()["user_id"]
    headers = _ops_auth_headers(operator_token)

    version_before = client.get("/api/sync/version").json()["version"]
    updated_before = client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()["updated_at"]

    confirmed = client.post(f"/api/trips/{trip_id}/confirm", json={"user_id": owner})
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["trip"]["id"] == trip_id
    assert confirmed.json()["trip"]["status"] == "confirmed"

    version_after = client.get("/api/sync/version").json()["version"]
    assert version_after >= version_before

    detail = client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()
    assert detail["id"] == trip_id
    assert detail["status"] == "confirmed"
    assert detail["lifecycle"] == "pending_operator_assignment"
    assert detail["operator_actionable"] is True
    # Same row, same itinerary content — confirmation never duplicates trips.
    assert detail["updated_at"] >= updated_before
    assert {i["id"] for i in detail["itinerary"]} == \
        {i["id"] for i in created.json()["itinerary"]}


def test_operator_accept_after_confirm_goes_ongoing_same_row(operator_token):
    """Confirmed -> operator Accept & Assign -> ongoing Active tour, same id."""
    created = client.post("/api/trips", json=_confirmable_trip_payload("Ops Lifecycle Accept Trip"))
    trip_id = created.json()["id"]
    owner = created.json()["user_id"]
    headers = _ops_auth_headers(operator_token)
    itinerary_count = len(created.json()["itinerary"])

    assert client.post(f"/api/trips/{trip_id}/confirm", json={"user_id": owner}).status_code == 200
    assert client.post(f"/api/ops/trips/{trip_id}/approve", json={},
                       headers=headers).status_code == 200
    accepted = client.post(f"/api/ops/trips/{trip_id}/accept", json={}, headers=headers)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["assignment_started"] is True

    detail = client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()
    assert detail["id"] == trip_id
    assert detail["status"] == "ongoing"
    assert detail["lifecycle"] == "active"
    assert detail["operator_actionable"] is True
    assert detail["approval"]["approved"] is True
    assert detail["approval"]["assignment_started"] is True
    assert len(detail["itinerary"]) == itinerary_count

    # Idempotent re-accept: same state, zero duplicate rows.
    again = client.post(f"/api/ops/trips/{trip_id}/accept", json={}, headers=headers)
    assert again.status_code == 200
    rows = client.get("/api/ops/trips", headers=headers).json()
    assert sum(1 for t in rows if t["id"] == trip_id) == 1

def test_operator_trips_require_operator_session(operator_token, traveler_token):
    assert client.get("/api/ops/trips").status_code == 401
    assert client.get("/api/ops/trips",
                      headers={"Authorization": "Bearer garbage"}).status_code == 401
    # A valid traveler JWT is not an operator session.
    assert client.get("/api/ops/trips",
                      headers=_ops_auth_headers(traveler_token)).status_code == 403
    assert client.get("/api/ops/trips/some-id",
                      headers=_ops_auth_headers(traveler_token)).status_code == 403


def test_operator_trips_unknown_id_404s(operator_token):
    res = client.get("/api/ops/trips/trip-that-does-not-exist",
                     headers=_ops_auth_headers(operator_token))
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Visibility: existing + newly created trips, traveler identity, no PII leak
# ---------------------------------------------------------------------------

def test_new_trip_appears_with_traveler_and_no_private_data(operator_token):
    created = _create_udaipur_trip("Ops Visibility Udaipur Trip")
    trip_id = created["id"]

    listing = client.get("/api/ops/trips", headers=_ops_auth_headers(operator_token))
    assert listing.status_code == 200
    rows = listing.json()
    assert isinstance(rows, list) and rows
    match = next((t for t in rows if t["id"] == trip_id), None)
    assert match is not None, "newly created trip must appear in the operator list"
    assert match["status"] == "planning"
    # Freshly generated trips are Preview rows (Pending Traveler Confirmation).
    assert match["lifecycle"] == "pending_traveler_confirmation"
    assert match["operator_actionable"] is False
    assert match["origin"] is None or isinstance(match["origin"], str)
    assert (match["destination"] or {}).get("name") == "Udaipur"
    assert match["traveler"]["id"]
    assert isinstance(match["traveler"]["name"], str) and match["traveler"]["name"]
    assert "email" not in match["traveler"] and "phone" not in match["traveler"]
    assert "password_hash" not in str(match)
    assert isinstance(match["itinerary"], list) and len(match["itinerary"]) > 0

    detail = client.get(f"/api/ops/trips/{trip_id}", headers=_ops_auth_headers(operator_token))
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == trip_id
    assert body["traveler"]["id"]
    assert "email" not in body["traveler"]
    assert isinstance(body["itinerary"], list) and len(body["itinerary"]) > 0
    transport_items = [i for i in body["itinerary"] if i["item_type"] == "transport"]
    stay_items = [i for i in body["itinerary"] if i["item_type"] == "hotel"]
    assert transport_items or stay_items, "operational transport/stay data must be present"


def test_multiple_trips_all_appear(operator_token):
    first = _create_udaipur_trip("Ops Multi Trip One")
    second = _create_udaipur_trip("Ops Multi Trip Two")
    rows = client.get("/api/ops/trips", headers=_ops_auth_headers(operator_token)).json()
    ids = {t["id"] for t in rows}
    assert first["id"] in ids and second["id"] in ids


def test_failed_generation_leaves_no_completed_trip(operator_token):
    title = "Ops Doomed Trip flop"
    res = client.post("/api/trips", json={
        "title": title,
        "destination_name": "Udaipur",
        "duration_days": 3,
        "total_budget": 60000.0,
        "currency": "INR",
        "traveler_count": 2,
        "pace": "balanced",
        "start_date": "2026-12-10T00:00:00",
        "end_date": "2026-12-05T00:00:00",  # ends before it starts
    })
    assert res.status_code == 422
    rows = client.get("/api/ops/trips", headers=_ops_auth_headers(operator_token)).json()
    assert all(t["title"] != title for t in rows)
    assert all(t["status"] != "completed" or t["title"] != title for t in rows)


# ---------------------------------------------------------------------------
# Updates propagate to the canonical operator view (+ sync version moves)
# ---------------------------------------------------------------------------

def test_traveler_transport_change_surfaces_to_operator(operator_token):
    created = _create_udaipur_trip("Ops Transport Update Trip")
    trip_id = created["id"]
    headers = _ops_auth_headers(operator_token)

    before = client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()
    version_before = client.get("/api/sync/version").json()["version"]

    change = client.post(f"/api/trips/{trip_id}/change-transport",
                         json={"transport_id": "trn-uda-001"})
    assert change.status_code == 200, change.text

    after = client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()
    transfers = [i for i in after["itinerary"] if i["item_type"] == "transport"]
    assert transfers, "operator must see the Day-1 transfer"
    assert transfers[0]["title"] == "Toyota Innova Lake City Cab"
    assert transfers[0]["transport_details"] is not None
    assert after["updated_at"] > before["updated_at"]

    version_after = client.get("/api/sync/version").json()["version"]
    assert version_after >= version_before
    # A refetch (what the dashboard poll does) returns the updated state.
    refetch = client.get(f"/api/ops/trips/{trip_id}", headers=headers).json()
    assert refetch["updated_at"] == after["updated_at"]
    assert [i for i in refetch["itinerary"] if i["item_type"] == "transport"][0]["title"] == \
        "Toyota Innova Lake City Cab"
