import os

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from main import app, ip_login_limiter, username_login_limiter

# Sin "with": no se ejecuta el lifespan, así que no se cargan CLIP, la BD ni MinIO.
# Basta para probar todo lo que ocurre antes de tocar esos servicios.
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_login_limits():
    ip_login_limiter.clear()
    username_login_limiter.clear()
    yield
    ip_login_limiter.clear()
    username_login_limiter.clear()


@pytest.fixture
def admin_headers():
    response = client.post("/api/admin/login", data={
        "username": os.environ["ADMIN_USERNAME"],
        "password": os.environ["ADMIN_PASSWORD"],
    })
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.mark.parametrize("method, path", [
    ("post", "/api/upload"),
    ("get", "/api/admin/images"),
    ("delete", "/api/admin/images/some-id"),
    ("get", "/api/admin/storage"),
])
def test_protected_endpoints_require_token(method, path):
    response = getattr(client, method)(path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_token_is_rejected():
    response = client.get("/api/admin/images", headers={"Authorization": "Bearer not-a-jwt"})

    assert response.status_code == 401


def test_login_with_wrong_password_fails():
    response = client.post("/api/admin/login", data={"username": "admin", "password": "wrong"})

    assert response.status_code == 401


def test_login_returns_bearer_token(admin_headers):
    assert admin_headers["Authorization"].startswith("Bearer ")


def test_upload_rejects_non_image_with_valid_token(admin_headers):
    response = client.post(
        "/api/upload",
        headers=admin_headers,
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_image_search_rejects_non_image():
    response = client.post(
        "/api/search/image",
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_old_debug_endpoint_is_gone():
    assert client.get("/api/debug/minio-connection").status_code == 404


def test_upload_with_banned_filename_returns_422_with_details(admin_headers):
    import io
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")

    response = client.post(
        "/api/upload",
        headers=admin_headers,
        files={"image": ("gun.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["status"] == "rejected"
    assert body["rejection_reason"] == "banned_words"
    assert body["banned_words"] == ["gun"]


def test_login_is_blocked_after_too_many_failures():
    for _ in range(settings.login_max_attempts_per_ip):
        response = client.post("/api/admin/login", data={"username": "admin", "password": "wrong"})
        assert response.status_code == 401

    # Bloqueado incluso con la contraseña correcta, hasta que expire la ventana
    response = client.post("/api/admin/login", data={
        "username": os.environ["ADMIN_USERNAME"],
        "password": os.environ["ADMIN_PASSWORD"],
    })

    assert response.status_code == 429
    assert 0 < int(response.headers["retry-after"]) <= settings.login_window_seconds


def test_successful_login_resets_failure_count():
    for _ in range(settings.login_max_attempts_per_ip - 1):
        client.post("/api/admin/login", data={"username": "admin", "password": "wrong"})
    ok = client.post("/api/admin/login", data={
        "username": os.environ["ADMIN_USERNAME"],
        "password": os.environ["ADMIN_PASSWORD"],
    })
    assert ok.status_code == 200

    # El contador volvió a cero: un nuevo fallo da 401, no 429
    response = client.post("/api/admin/login", data={"username": "admin", "password": "wrong"})

    assert response.status_code == 401
