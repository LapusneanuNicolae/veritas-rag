"""Phase 5 tests: auth, JWT, multi-tenant isolation, caching."""
from fastapi.testclient import TestClient

from app.auth.security import create_access_token, decode_token, hash_password, verify_password
from app.main import app

client = TestClient(app)


def test_password_hash_roundtrip():
    h = hash_password("s3cret")
    assert verify_password("s3cret", h)
    assert not verify_password("wrong", h)


def test_jwt_roundtrip():
    token = create_access_token(sub="a@b.com", tenant_id="t1")
    payload = decode_token(token)
    assert payload["sub"] == "a@b.com" and payload["tenant_id"] == "t1"


def test_register_and_login():
    r = client.post("/auth/register", json={"email": "u1@example.com", "password": "pw"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "u1@example.com"


def test_tenant_isolation():
    # two tenants upload different docs; each only retrieves its own
    t1 = client.post("/auth/register", json={"email": "t1@x.com", "password": "pw"}).json()
    t2 = client.post("/auth/register", json={"email": "t2@x.com", "password": "pw"}).json()
    h1 = {"Authorization": f"Bearer {t1['access_token']}"}
    h2 = {"Authorization": f"Bearer {t2['access_token']}"}

    client.post("/api/documents/upload", headers=h1,
                files={"file": ("a.txt", b"Alpha tenant secret: the sky is blue.", "text/plain")})
    client.post("/api/documents/upload", headers=h2,
                files={"file": ("b.txt", b"Beta tenant secret: the grass is green.", "text/plain")})

    r1 = client.post("/api/ask", headers=h1, json={"question": "what colour is the sky"})
    docs1 = [c["document"] for c in r1.json()["citations"]]
    assert docs1 == ["a.txt"]  # tenant 1 never sees tenant 2's document


def test_answer_cache_hit():
    h = {"Authorization": f"Bearer "
         f"{client.post('/auth/register', json={'email': 'c@x.com', 'password': 'pw'}).json()['access_token']}"}
    client.post("/api/documents/upload", headers=h,
                files={"file": ("d.txt", b"Caching makes repeat answers fast.", "text/plain")})
    q = {"question": "what makes repeat answers fast?"}
    first = client.post("/api/ask", headers=h, json=q).json()
    second = client.post("/api/ask", headers=h, json=q).json()
    assert first["cached"] is False
    assert second["cached"] is True
