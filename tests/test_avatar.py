"""Traveler avatar tests (DB-backed profile photos)."""
import uuid

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# 1x1 transparent PNG
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
       b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
       b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


def _signup(name="Avatar User"):
    email = f"avatar-{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/traveler/signup", json={
        "full_name": name, "email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    body = r.json()
    return body["user"], body["token"], email


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _cleanup(email):
    from backend.database.connection import SessionLocal
    from backend.models.models import TravelerProfile, User
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email).first()
        if u:
            db.query(TravelerProfile).filter(TravelerProfile.user_id == u.id).delete()
            db.delete(u)
            db.commit()
    finally:
        db.close()


def test_avatar_upload_get_delete():
    user, token, email = _signup()
    try:
        assert client.get("/api/traveler/profile", headers=_auth(token)).json()["has_avatar"] is False
        up = client.post("/api/traveler/avatar", headers=_auth(token),
                         files={"file": ("avatar.png", PNG, "image/png")})
        assert up.status_code == 200, up.text
        assert up.json()["avatar"] is True
        assert client.get("/api/traveler/profile", headers=_auth(token)).json()["has_avatar"] is True
        img = client.get(f"/api/traveler/avatar/{user['id']}")
        assert img.status_code == 200
        assert img.content == PNG
        assert img.headers["content-type"] == "image/png"
        assert client.delete("/api/traveler/avatar", headers=_auth(token)).status_code == 200
        assert client.get(f"/api/traveler/avatar/{user['id']}").status_code == 404
        assert client.get("/api/traveler/profile", headers=_auth(token)).json()["has_avatar"] is False
    finally:
        _cleanup(email)


def test_avatar_rejects_bad_uploads():
    user, token, email = _signup()
    try:
        bad = client.post("/api/traveler/avatar", headers=_auth(token),
                          files={"file": ("note.txt", b"hello", "text/plain")})
        assert bad.status_code == 422
        big = client.post("/api/traveler/avatar", headers=_auth(token),
                          files={"file": ("big.png", b"x" * (6 * 1024 * 1024), "image/png")})
        assert big.status_code == 422
        assert client.post("/api/traveler/avatar",
                           files={"file": ("a.png", PNG, "image/png")}).status_code == 401
        assert client.get("/api/traveler/avatar/no-such-user").status_code == 404
    finally:
        _cleanup(email)
