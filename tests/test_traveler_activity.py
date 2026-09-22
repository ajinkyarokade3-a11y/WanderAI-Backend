import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import Trip, Notification, Alert, ChangeHistory, TripMessage

client = TestClient(app)

def _signup(email=None):
    email = email or f"traveler_{uuid.uuid4().hex[:8]}@example.com"
    res = client.post("/api/auth/traveler/signup", json={"full_name":"Test Traveler","email":email,"password":"StrongPass123"})
    assert res.status_code == 201, res.text
    data=res.json()
    return data["user"], data["token"]

def _auth(token): return {"Authorization": f"Bearer {token}"}

def _create_trip(user_id):
    db=SessionLocal()
    trip=Trip(id=str(uuid.uuid4()), user_id=user_id, title="Test Trip", status="planning")
    db.add(trip); db.commit(); db.refresh(trip); db.close()
    return trip.id

def test_ownership_required():
    user_a, tok_a=_signup()
    user_b, tok_b=_signup()
    tid=_create_trip(user_a["id"])
    assert client.get(f"/api/traveler/trips/{tid}/activity", headers=_auth(tok_b)).status_code==404
    assert client.get(f"/api/traveler/trips/{tid}/activity/unread-count", headers=_auth(tok_b)).status_code==404
    assert client.get(f"/api/traveler/trips/{tid}/activity", headers={}).status_code==401

def test_event_filtering_and_unread():
    user, tok=_signup()
    tid=_create_trip(user["id"])
    db=SessionLocal()
    n1=Notification(trip_id=tid, user_id=user["id"], title="Welcome", message="hello", type="info", is_read=False)
    n2=Notification(trip_id=tid, user_id=user["id"], title="Done", message="done", type="success", is_read=True)
    a1=Alert(trip_id=tid, alert_type="weather", severity="critical", title="Storm", description="Storm alert", is_resolved=False)
    a2=Alert(trip_id=tid, alert_type="weather", severity="warning", title="Wind", description="Wind resolved", is_resolved=True)
    ch1=ChangeHistory(trip_id=tid, changed_by="user", action="transport_changed", field_changed="transport", new_value="Cab", reason="Traveler selected a catalog transport option.")
    ch2=ChangeHistory(trip_id=tid, changed_by="operator", action="replan_applied", field_changed="itinerary", new_value="X", reason="Operator approved an available destination activity.")
    # internal operator message should not leak
    msg=TripMessage(trip_id=tid, operator_name="op", category="operational", body="INTERNAL VENDOR COST 9999 and private decision", is_urgent=False)
    db.add_all([n1,n2,a1,a2,ch1,ch2,msg]); db.commit(); db.close()

    res=client.get(f"/api/traveler/trips/{tid}/activity", headers=_auth(tok))
    assert res.status_code==200, res.text
    body=res.json()
    assert "events" in body and "total" in body
    # ensure internal message not leaked
    all_text="".join([e["title"]+e["message"] for e in body["events"]])
    assert "INTERNAL VENDOR COST" not in all_text
    assert "private decision" not in all_text
    # old_value/new_value internal not leaked
    assert "9999" not in all_text
    # types present
    types={e["type"] for e in body["events"]}
    assert "notification" in types
    assert "disruption_alert" in types
    assert "transport_updated" in types
    assert "trip_replanned" in types
    for e in body["events"]:
        for field in ["id","type","title","message","trip_id","created_at","severity","is_read"]:
            assert field in e
        assert e["trip_id"]==tid
    # unread count
    unread=res.json()["unread_count"]
    # n1 unread, a1 unread, ch1 ch2 unread (2), n2 read, a2 read -> 4 unread
    assert unread==4
    res2=client.get(f"/api/traveler/trips/{tid}/activity/unread-count", headers=_auth(tok))
    assert res2.json()["count"]==4

def test_pagination():
    user, tok=_signup()
    tid=_create_trip(user["id"])
    db=SessionLocal()
    for i in range(5):
        db.add(Notification(trip_id=tid, user_id=user["id"], title=f"N{i}", message="m", type="info"))
    db.commit(); db.close()
    r1=client.get(f"/api/traveler/trips/{tid}/activity?limit=2&offset=0", headers=_auth(tok)).json()
    r2=client.get(f"/api/traveler/trips/{tid}/activity?limit=2&offset=2", headers=_auth(tok)).json()
    assert len(r1["events"])==2
    assert len(r2["events"])==2
    assert r1["total"]==5
    ids1={e["id"] for e in r1["events"]}
    ids2={e["id"] for e in r2["events"]}
    assert not ids1.intersection(ids2)

def test_type_filter():
    user, tok=_signup()
    tid=_create_trip(user["id"])
    db=SessionLocal()
    db.add(Notification(trip_id=tid, user_id=user["id"], title="N", message="m", type="info"))
    db.add(Alert(trip_id=tid, alert_type="weather", severity="warning", title="A", description="d"))
    db.commit(); db.close()
    res=client.get(f"/api/traveler/trips/{tid}/activity?type=notification", headers=_auth(tok)).json()
    assert all(e["type"]=="notification" for e in res["events"])

def test_stream_requires_auth_and_ownership():
    user, tok=_signup()
    tid=_create_trip(user["id"])
    assert client.get(f"/api/traveler/trips/{tid}/activity/stream").status_code==401
    user2, tok2=_signup()
    assert client.get(f"/api/traveler/trips/{tid}/activity/stream", headers=_auth(tok2)).status_code==404
    res=client.get(f"/api/traveler/trips/{tid}/activity/stream", headers=_auth(tok))
    assert res.status_code==200
    assert "text/event-stream" in res.headers.get("content-type","")

def test_trip_confirmed_event():
    user, tok=_signup()
    tid=_create_trip(user["id"])
    db=SessionLocal()
    trip=db.query(Trip).filter(Trip.id==tid).first()
    trip.confirmed_at=datetime.utcnow()
    trip.status="confirmed"
    db.commit(); db.close()
    res=client.get(f"/api/traveler/trips/{tid}/activity", headers=_auth(tok)).json()
    assert any(e["type"]=="trip_confirmed" for e in res["events"])
