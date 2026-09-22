"""TourFlow AI Guide tests — grounded in real application data (no hardcoding).

Covers:
A. User name loaded dynamically (greeting uses real first name, never invented).
B. Active trip loaded correctly.
C. Conversation persists after refresh (history reload).
D. Previous messages included in AI context.
E. Different trips have isolated conversations.
F. User cannot access another user's conversation.
G. AI answers from today's itinerary.
H. AI answers from tomorrow's itinerary.
I. AI answers booking questions with real bookings.
J. AI does not invent unavailable information.
K. Gemini action cannot modify another user's trip.
L. Chat works when conversation history is empty.
"""

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import (
    Booking,
    Destination,
    GuideConversationSummary,
    GuideMessage,
    ItineraryItem,
    Trip,
    TripPreference,
    User,
)
from backend.guide.chat_service import GuideChatService
from backend.guide.context_service import GuideContextService

client = TestClient(app)


class _GeminiOff:
    def is_available(self):
        return False


@pytest.fixture()
def _no_gemini(monkeypatch):
    import backend.api.routes as routes
    monkeypatch.setattr(routes, "gemini_service", _GeminiOff())


# Per-test record of rows created by the factory helpers below. Teardown
# deletes ONLY these primary keys (never title-based matching), FK-safe
# order, best-effort per table, and always runs - even when the test fails.
# Nothing created before the fixture started can be touched.
_created_stack = []


@pytest.fixture(autouse=True)
def _cleanup_guide_test_rows():
    record = {"users": [], "trips": []}
    _created_stack.append(record)
    try:
        yield
    finally:
        _created_stack.remove(record)
        _delete_tracked_rows(record)


def _track(kind, row_id):
    if _created_stack and row_id is not None:
        _created_stack[-1][kind].append(row_id)


def _delete_tracked_rows(record):
    """Delete helper-created rows (and their children) by recorded PKs only."""
    trip_ids = [i for i in record["trips"] if i is not None]
    user_ids = [i for i in record["users"] if i is not None]
    if not trip_ids and not user_ids:
        return
    db = SessionLocal()
    try:
        steps = [
            (GuideMessage, GuideMessage.trip_id, trip_ids),
            (GuideConversationSummary, GuideConversationSummary.trip_id, trip_ids),
            (Booking, Booking.trip_id, trip_ids),
            (ItineraryItem, ItineraryItem.trip_id, trip_ids),
            (TripPreference, TripPreference.trip_id, trip_ids),
            (Trip, Trip.id, trip_ids),
            (User, User.id, user_ids),
        ]
        for model, column, ids in steps:
            if not ids:
                continue
            try:
                db.query(model).filter(column.in_(ids)).delete(
                    synchronize_session=False)
                db.commit()
            except Exception:
                db.rollback()
    finally:
        db.close()


def _signup(email=None, name="Test Traveler"):
    email = email or f"guide-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/traveler/signup", json={
        "full_name": name, "email": email, "password": "password123",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    _track("users", (body.get("user") or {}).get("id"))
    return body["user"], body["token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _make_trip(db, user_id, title="Guide Test Trip", dest_slug="manali",
               start_today=True, duration=3, budget=60000.0):
    dest = db.query(Destination).filter(Destination.slug == dest_slug).first()
    assert dest is not None
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) if start_today else None
    end = (start + timedelta(days=duration - 1)) if start else None
    trip = Trip(
        user_id=user_id, destination_id=dest.id, title=title,
        status="confirmed", start_date=start, end_date=end,
        duration_days=duration, total_budget=budget, currency="INR",
        traveler_count=2, pace="balanced",
    )
    db.add(trip)
    db.flush()
    db.add(TripPreference(
        trip_id=trip.id, budget_tier="moderate",
        interests=["nature", "adventure"], travel_companions="couple",
        accommodation_types=["boutique"], transport_preferences=["private_suv"],
        dietary_requirements=[],
    ))
    db.commit()
    db.refresh(trip)
    _track("trips", trip.id)
    return trip


def _add_item(db, trip_id, day, title, item_type="activity", **kw):
    row = ItineraryItem(
        trip_id=trip_id, day_number=day, order_index=kw.get("order_index", 1),
        item_type=item_type, title=title,
        description=kw.get("description"), start_time=kw.get("start_time"),
        end_time=kw.get("end_time"), cost=kw.get("cost", 0),
        status=kw.get("status", "confirmed"), location=kw.get("location"),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_a_greeting_uses_real_user_name(_no_gemini):
    user, token = _signup(name="Meera Krishnan")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"], title="Meera Himalayan Escape")
        trip_id = trip.id
    finally:
        db.close()
    r = client.get("/api/guide/greeting", params={"tripId": trip_id}, headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Meera" in body["greeting"]
    assert "Aarav" not in body["greeting"]
    assert "Priya" not in body["greeting"]
    assert "Meera Himalayan Escape" in body["greeting"]


def test_b_active_trip_loaded_without_trip_id(_no_gemini):
    user, token = _signup(name="Arjun Nair")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"], title="Arjun Goa Break", dest_slug="goa")
        trip_id = trip.id
    finally:
        db.close()
    r = client.get("/api/guide/greeting", headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trip_id"] == trip_id
    assert "Goa" in body["greeting"]


def test_c_conversation_persists_after_refresh(_no_gemini):
    user, token = _signup(name="Persist User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"])
        trip_id = trip.id
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "What should I do tomorrow?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    # Simulate page refresh: fresh GET must return the stored turn pair.
    h = client.get("/api/guide/history", params={"tripId": trip_id}, headers=_auth(token))
    assert h.status_code == 200, h.text
    msgs = h.json()["messages"]
    assert any(m["role"] == "user" and "tomorrow" in m["message"] for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)


def test_d_previous_messages_in_context(_no_gemini):
    user, token = _signup(name="Memory User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"])
        trip_id = trip.id
    finally:
        db.close()
    client.post("/api/guide/chat", json={"message": "I am worried about my hotel checkout", "tripId": trip_id},
                headers=_auth(token))
    db = SessionLocal()
    try:
        ctx = GuideContextService(db).buildContext(user["id"], trip_id)
    finally:
        db.close()
    assert any("hotel checkout" in m["message"] for m in ctx["recentMessages"])
    # Follow-up resolves via conversation memory ("what did I say earlier").
    r = client.post("/api/guide/chat", json={"message": "What did I say earlier about the hotel?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200
    assert "checkout" in r.json()["response"].lower() or "hotel" in r.json()["response"].lower()


def test_e_trips_have_isolated_conversations(_no_gemini):
    user, token = _signup(name="Isolation User")
    db = SessionLocal()
    try:
        a = _make_trip(db, user["id"], title="Trip Alpha")
        b = _make_trip(db, user["id"], title="Trip Beta")
        a_id, b_id = a.id, b.id
    finally:
        db.close()
    client.post("/api/guide/chat", json={"message": "Alpha-only secret phrase", "tripId": a_id},
                headers=_auth(token))
    hb = client.get("/api/guide/history", params={"tripId": b_id}, headers=_auth(token))
    assert hb.status_code == 200
    assert all("Alpha-only secret phrase" not in m["message"] for m in hb.json()["messages"])
    ha = client.get("/api/guide/history", params={"tripId": a_id}, headers=_auth(token))
    assert any("Alpha-only secret phrase" in m["message"] for m in ha.json()["messages"])


def test_unknown_trip_id_self_heals_with_active_trip(_no_gemini):
    """Explicit-but-wrong IDs 404 AND carry the user's own active trip
    (own data only) so the UI can auto-switch instead of dead-ending."""
    user, token = _signup(name="Self Heal User")
    trip_id = _manali_trip(user["id"])
    r = client.get("/api/guide/history", params={"tripId": "does-not-exist"},
                   headers=_auth(token))
    assert r.status_code == 404
    body = r.json()
    assert body["detail"] == "Trip not found"
    assert body["has_active_trip"] is True
    assert body["active_trip"]["trip_id"] == trip_id
    c = client.post("/api/guide/chat",
                    json={"message": "hi", "tripId": "does-not-exist"},
                    headers=_auth(token))
    assert c.status_code == 404
    assert c.json()["active_trip"]["trip_id"] == trip_id


def test_f_user_cannot_access_another_users_conversation(_no_gemini):
    user_a, token_a = _signup(name="Owner User")
    user_b, token_b = _signup(name="Intruder User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user_a["id"], title="Owner Private Trip")
        trip_id = trip.id
    finally:
        db.close()
    # Intruder chat on someone else's trip -> 404 (no existence leak, no access).
    r = client.post("/api/guide/chat", json={"message": "hello", "tripId": trip_id}, headers=_auth(token_b))
    assert r.status_code == 404
    h = client.get("/api/guide/history", params={"tripId": trip_id}, headers=_auth(token_b))
    assert h.status_code == 404


def test_g_answers_from_todays_itinerary(_no_gemini):
    user, token = _signup(name="Today User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"], title="Today Trip")
        trip_id = trip.id
        _add_item(db, trip_id, 1, "Solang Valley Paragliding", start_time="9:00 AM",
                  end_time="1:00 PM", location="Solang Valley", cost=2500)
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "What should I do today?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "Solang Valley" in r.json()["response"]


def test_h_answers_from_tomorrows_itinerary(_no_gemini):
    user, token = _signup(name="Tomorrow User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"], title="Tomorrow Trip")
        trip_id = trip.id
        _add_item(db, trip_id, 1, "Mall Road Evening Walk", start_time="5:00 PM", location="Mall Road", cost=0)
        _add_item(db, trip_id, 2, "Atal Tunnel Excursion", start_time="9:00 AM",
                  end_time="2:00 PM", location="Atal Tunnel", cost=1800)
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "What should I do tomorrow?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "Atal Tunnel" in r.json()["response"]
    # Reference resolution: "lunch" follow-up stays grounded, does not ask to repeat everything.
    r2 = client.post("/api/guide/chat", json={"message": "What about lunch?", "tripId": trip_id},
                     headers=_auth(token))
    assert r2.status_code == 200
    assert "Aarav" not in r2.json()["response"]


def test_i_answers_booking_questions(_no_gemini):
    user, token = _signup(name="Booking User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"], title="Booking Trip")
        trip_id = trip.id
        bk = Booking(trip_id=trip_id, item_type="hotel", amount=12000,
                     currency="INR", status="confirmed", payment_status="paid")
        db.add(bk)
        db.commit()
        db.refresh(bk)
        ref = bk.booking_reference
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "Show my bookings", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert ref in r.json()["response"]


def test_j_does_not_invent_unavailable_information(_no_gemini):
    user, token = _signup(name="Honest User")
    db = SessionLocal()
    try:
        # No itinerary items at all.
        trip = _make_trip(db, user["id"], title="Empty Trip")
        trip_id = trip.id
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "What should I do tomorrow?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    resp = r.json()["response"]
    assert "unavailable" in resp.lower()
    assert "Solang Valley" not in resp
    assert "Aarav" not in resp


def test_k_action_cannot_modify_another_users_trip():
    owner, _ = _signup(name="Action Owner")
    intruder, _ = _signup(name="Action Intruder")
    db = SessionLocal()
    try:
        trip = _make_trip(db, owner["id"], title="Victim Trip")
        victim_trip_id = trip.id
        item = _add_item(db, victim_trip_id, 1, "Victim Activity", start_time="9:00 AM")
        victim_item_id = item.id
        other = _make_trip(db, intruder["id"], title="Intruder Trip")
        intruder_trip = other
        svc = GuideChatService(db, _GeminiOff())
        # Forged item_id from another user's trip must be rejected.
        outcome = svc.execute_action(
            intruder["id"], intruder_trip,
            {"intent": "UPDATE_ITINERARY", "action": "MOVE_ITEM",
             "item_id": victim_item_id, "new_time": "11:00"},
        )
        assert outcome["applied"] is False
        db.refresh(item)
        assert item.start_time == "9:00 AM"
    finally:
        db.close()


def test_l_chat_works_with_empty_history(_no_gemini):
    user, token = _signup(name="Fresh User")
    db = SessionLocal()
    try:
        trip = _make_trip(db, user["id"], title="Fresh Trip")
        trip_id = trip.id
    finally:
        db.close()
    h = client.get("/api/guide/history", params={"tripId": trip_id}, headers=_auth(token))
    assert h.status_code == 200
    assert h.json()["messages"] == []
    r = client.post("/api/guide/chat", json={"message": "Hello, help me plan", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert r.json()["response"]
    assert r.json()["trip_id"] == trip_id


def test_no_active_trip_graceful(_no_gemini):
    user, token = _signup(name="No Trip User")
    r = client.get("/api/guide/greeting", headers=_auth(token))
    assert r.status_code == 200, r.text
    assert r.json()["has_active_trip"] is False
    assert "active trip" in r.json()["greeting"].lower()


def test_chat_requires_auth():
    r = client.post("/api/guide/chat", json={"message": "hi"})
    assert r.status_code == 401


def _create_trip_payload(dest_slug="manali"):
    dest_id = client.get(f"/api/destinations/{dest_slug}").json()["id"]
    return {"title": "Ownership Check", "destination_id": dest_id,
            "duration_days": 2, "traveler_count": 1,
            "total_budget": 30000.0, "currency": "INR"}


def test_trip_creation_uses_session_owner(_no_gemini):
    """Logged-in creation belongs to the session user, so the Guide sees it."""
    user, token = _signup(name="Owner Session")
    r = client.post("/api/trips", json=_create_trip_payload(), headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user_id"] == user["id"]
    g = client.get("/api/guide/greeting", headers=_auth(token))
    assert g.status_code == 200
    assert g.json()["trip_id"] == body["id"]
    assert "Ownership Check" in g.json()["greeting"]
    assert client.delete(f"/api/trips/{body['id']}").status_code == 200


def test_session_owner_beats_payload_user_id(_no_gemini):
    user, token = _signup(name="Session Wins")
    payload = _create_trip_payload()
    payload["user_id"] = "usr-alex-morgan-001"  # untrusted client value
    r = client.post("/api/trips", json=payload, headers=_auth(token))
    assert r.status_code == 200, r.text
    assert r.json()["user_id"] == user["id"]
    assert client.delete(f"/api/trips/{r.json()['id']}").status_code == 200


def test_anonymous_creation_legacy_preserved():
    r = client.post("/api/trips", json=_create_trip_payload())
    assert r.status_code == 200, r.text
    assert client.delete(f"/api/trips/{r.json()['id']}").status_code == 200


def _manali_trip(user_id):
    db = SessionLocal()
    try:
        trip = _make_trip(db, user_id, title="Manali Brain Check", dest_slug="manali")
        return trip.id
    finally:
        db.close()


def test_greeting_does_not_dump_itinerary(_no_gemini):
    user, token = _signup(name="Meera Krishnan")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "hii", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    resp = r.json()["response"]
    assert "Hi Meera" in resp
    assert "coming up" not in resp


def test_help_lists_capabilities(_no_gemini):
    user, token = _signup(name="Help User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "what can you do?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "booking" in r.json()["response"].lower()


def test_best_time_uses_real_destination_fact(_no_gemini):
    user, token = _signup(name="Season User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "best time to visit?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "October to June" in r.json()["response"]


def test_adventure_question_uses_real_catalog(_no_gemini):
    user, token = _signup(name="Thrill User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "show me adventure activities", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "Solang" in r.json()["response"]


def test_budget_breakdown_per_person(_no_gemini):
    user, token = _signup(name="Split User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "break down my budget per person", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    resp = r.json()["response"]
    assert "per person" in resp and "60,000" in resp


def test_trip_facts_duration(_no_gemini):
    user, token = _signup(name="Facts User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "how long is my trip?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "3 days" in r.json()["response"]


def test_weather_answered_honestly(_no_gemini):
    user, token = _signup(name="Rain User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "will it rain tomorrow?", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    resp = r.json()["response"].lower()
    assert "don't have live weather" in resp


def test_remove_activity_via_chat(_no_gemini):
    from backend.models.models import ItineraryItem
    user, token = _signup(name="Remove User")
    trip_id = _manali_trip(user["id"])
    db = SessionLocal()
    try:
        _add_item(db, trip_id, 1, "Test Remove Me", start_time="10:00 AM")
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "remove Test Remove Me", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] and body["action"]["applied"] is True
    assert "Removed" in body["response"]
    db = SessionLocal()
    try:
        assert db.query(ItineraryItem).filter(
            ItineraryItem.trip_id == trip_id, ItineraryItem.title == "Test Remove Me").count() == 0
    finally:
        db.close()


def test_vague_add_creates_nothing(_no_gemini):
    from backend.models.models import ItineraryItem
    user, token = _signup(name="Vague User")
    trip_id = _manali_trip(user["id"])
    db = SessionLocal()
    try:
        before = db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip_id).count()
    finally:
        db.close()
    r = client.post("/api/guide/chat", json={"message": "add something tomorrow", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "What should I change?" in r.json()["response"]
    db = SessionLocal()
    try:
        after = db.query(ItineraryItem).filter(ItineraryItem.trip_id == trip_id).count()
        assert after == before
    finally:
        db.close()


def test_change_hotel_via_chat(_no_gemini):
    user, token = _signup(name="Hotel Swap User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat",
                    json={"message": "change my hotel to Larisa", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] and body["action"]["applied"] is True
    assert "Larisa" in body["response"]
    db = SessionLocal()
    try:
        from backend.models.models import ItineraryItem
        hotel_items = db.query(ItineraryItem).filter(
            ItineraryItem.trip_id == trip_id, ItineraryItem.item_type == "hotel").all()
        assert any("Larisa" in (i.title or "") for i in hotel_items)
    finally:
        db.close()


def test_trip_card_in_chat_and_history(_no_gemini):
    user, token = _signup(name="Card User")
    trip_id = _manali_trip(user["id"])
    r = client.post("/api/guide/chat", json={"message": "hello", "tripId": trip_id},
                    headers=_auth(token))
    assert r.status_code == 200, r.text
    card = r.json()["trip_card"]
    assert card and card["total_budget"] == 60000.0 and card["destination"] == "Manali"
    h = client.get("/api/guide/history", params={"tripId": trip_id}, headers=_auth(token))
    assert h.status_code == 200
    assert h.json()["trip_card"]["trip_id"] == trip_id
