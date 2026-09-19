import uuid
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.connection import SessionLocal
from backend.models.models import TravelerProfile, User

client = TestClient(app)

def _signup(email=None):
    email = email or f"traveler_{uuid.uuid4().hex[:8]}@example.com"
    res = client.post("/api/auth/traveler/signup", json={
        "full_name": "Test Traveler",
        "email": email,
        "password": "StrongPass123"
    })
    assert res.status_code == 201, res.text
    data = res.json()
    return data["user"], data["token"]

def _auth_header(token):
    return {"Authorization": f"Bearer {token}"}

def test_unauthenticated_access():
    assert client.get("/api/traveler/profile").status_code == 401
    assert client.put("/api/traveler/profile", json={}).status_code == 401
    assert client.patch("/api/traveler/profile", json={}).status_code == 401
    assert client.get("/api/traveler/preferences").status_code == 401
    assert client.put("/api/traveler/preferences", json={}).status_code == 401

def test_authenticated_profile_retrieval():
    _, token = _signup()
    res = client.get("/api/traveler/profile", headers=_auth_header(token))
    assert res.status_code == 200, res.text
    body = res.json()
    for field in ["user_id","full_name","email","phone","travel_style","dietary_preferences","fitness_level","preferred_currency","language","bio","created_at","updated_at"]:
        assert field in body
    assert body["travel_style"] == "balanced"

def test_automatic_profile_creation():
    user, token = _signup()
    # ensure profile deleted if exists
    db = SessionLocal()
    db.query(TravelerProfile).filter(TravelerProfile.user_id == user["id"]).delete()
    db.commit()
    db.close()
    # GET should recreate
    res = client.get("/api/traveler/profile", headers=_auth_header(token))
    assert res.status_code == 200
    db = SessionLocal()
    count = db.query(TravelerProfile).filter(TravelerProfile.user_id == user["id"]).count()
    db.close()
    assert count == 1
    # second GET should not duplicate
    res2 = client.get("/api/traveler/profile", headers=_auth_header(token))
    assert res2.status_code == 200
    db = SessionLocal()
    count2 = db.query(TravelerProfile).filter(TravelerProfile.user_id == user["id"]).count()
    db.close()
    assert count2 == 1

def test_profile_update():
    _, token = _signup()
    payload = {
        "full_name": "Updated Name",
        "phone": "+919999999999",
        "travel_style": "adventure",
        "dietary_preferences": ["vegetarian"],
        "fitness_level": "high",
        "preferred_currency": "USD",
        "language": "Hindi",
        "bio": "I love travel"
    }
    res = client.put("/api/traveler/profile", headers=_auth_header(token), json=payload)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["full_name"] == "Updated Name"
    assert body["travel_style"] == "adventure"
    assert body["preferred_currency"] == "USD"
    assert body["bio"] == "I love travel"

def test_partial_update_patch():
    _, token = _signup()
    # set initial
    client.put("/api/traveler/profile", headers=_auth_header(token), json={"travel_style":"luxury","bio":"initial"})
    res = client.patch("/api/traveler/profile", headers=_auth_header(token), json={"bio":"patched bio"})
    assert res.status_code == 200
    assert res.json()["bio"] == "patched bio"
    assert res.json()["travel_style"] == "luxury"

def test_preference_update():
    _, token = _signup()
    # get defaults
    res = client.get("/api/traveler/preferences", headers=_auth_header(token))
    assert res.status_code == 200
    assert "travel_style" in res.json()
    # update
    payload = {
        "travel_style": "cultural",
        "dietary_preferences": ["vegan","gluten_free"],
        "fitness_level": "low",
        "preferred_currency": "EUR",
        "language": "French"
    }
    res2 = client.put("/api/traveler/preferences", headers=_auth_header(token), json=payload)
    assert res2.status_code == 200, res2.text
    assert res2.json()["travel_style"] == "cultural"
    assert res2.json()["preferred_currency"] == "EUR"
    # verify profile reflects preferences
    prof = client.get("/api/traveler/profile", headers=_auth_header(token)).json()
    assert prof["travel_style"] == "cultural"
    assert prof["fitness_level"] == "low"

def test_invalid_preference_values():
    _, token = _signup()
    # invalid travel_style
    res = client.put("/api/traveler/preferences", headers=_auth_header(token), json={"travel_style":"invalid_style"})
    assert res.status_code == 422
    # invalid currency - lower case / too long
    res = client.put("/api/traveler/preferences", headers=_auth_header(token), json={"preferred_currency":"usd"})
    assert res.status_code == 422
    res = client.put("/api/traveler/preferences", headers=_auth_header(token), json={"preferred_currency":"TOOLONG"})
    assert res.status_code == 422
    # invalid language - single char
    res = client.put("/api/traveler/preferences", headers=_auth_header(token), json={"language":"x"})
    assert res.status_code == 422
    # invalid dietary_preferences - not array
    res = client.put("/api/traveler/profile", headers=_auth_header(token), json={"dietary_preferences":"vegetarian"})
    assert res.status_code == 422
    # invalid fitness_level
    res = client.put("/api/traveler/profile", headers=_auth_header(token), json={"fitness_level":"extreme"})
    assert res.status_code == 422

def test_traveler_isolation():
    user_a, token_a = _signup()
    user_b, token_b = _signup()
    # update A
    client.put("/api/traveler/profile", headers=_auth_header(token_a), json={"travel_style":"luxury","bio":"A bio"})
    client.put("/api/traveler/profile", headers=_auth_header(token_b), json={"travel_style":"budget","bio":"B bio"})
    # get each profile - must reflect own data
    prof_a = client.get("/api/traveler/profile", headers=_auth_header(token_a)).json()
    prof_b = client.get("/api/traveler/profile", headers=_auth_header(token_b)).json()
    assert prof_a["user_id"] == user_a["id"]
    assert prof_b["user_id"] == user_b["id"]
    assert prof_a["user_id"] != prof_b["user_id"]
    assert prof_a["travel_style"] == "luxury"
    assert prof_b["travel_style"] == "budget"
    assert prof_a["bio"] == "A bio"
    assert prof_b["bio"] == "B bio"
    # ensure no user_id acceptance from client - patch with no user_id should still be own
    # token A cannot get B data - already verified above
