from datetime import UTC, datetime, timedelta

import jwt
from fastapi.testclient import TestClient

from xpeech.agent.server.api import app
from xpeech.config.settings import JWTConfig, settings
from xpeech.utils.jwt_auth import create_access_token, decode_access_token

TEST_CONFIG = JWTConfig(secret_key="test-secret-key-that-is-at-least-32-bytes")


def test_access_token_is_valid_for_exactly_one_minute():
    now = datetime.now(UTC)
    token = create_access_token(config=TEST_CONFIG, now=now)
    claims = jwt.decode(
        token,
        TEST_CONFIG.secret_key,
        algorithms=[TEST_CONFIG.algorithm],
    )

    assert claims == {"exp": int((now + timedelta(seconds=60)).timestamp())}


def test_expired_access_token_is_rejected():
    token = create_access_token(
        config=TEST_CONFIG,
        now=datetime.now(UTC) - timedelta(seconds=61),
    )

    try:
        decode_access_token(token, config=TEST_CONFIG)
    except jwt.ExpiredSignatureError:
        pass
    else:
        raise AssertionError("expired token was accepted")


def test_api_requires_jwt():
    client = TestClient(app)

    api_response = client.post("/answer_question", data={"answer": "ok"}, headers={"x-session-id": "test"})

    assert api_response.status_code == 401
    assert api_response.headers["www-authenticate"] == "Bearer"


def test_api_rejects_token_signed_with_another_key():
    client = TestClient(app)
    token = create_access_token(config=TEST_CONFIG)

    response = client.post(
        "/answer_question",
        data={"answer": "ok"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-session-id": "test",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired bearer token"}


def test_token_endpoint_rejects_wrong_jwt_secret():
    client = TestClient(app)

    response = client.post(
        "/token",
        data={"username": "docs", "password": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect JWT secret"}


def test_token_endpoint_issues_one_minute_jwt_for_docs():
    client = TestClient(app)

    token_response = client.post(
        "/token",
        data={"username": "docs", "password": settings.jwt.secret_key},
    )

    assert token_response.status_code == 200
    assert token_response.headers["cache-control"] == "no-store"
    assert token_response.json()["token_type"] == "bearer"
    token = token_response.json()["access_token"]
    claims = decode_access_token(token)
    assert set(claims) == {"exp"}

    api_response = client.post(
        "/answer_question",
        data={"answer": "ok"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-session-id": "test",
        },
    )
    assert api_response.status_code == 200


def test_docs_are_public_and_use_oauth2_jwt_flow():
    client = TestClient(app)
    assert client.get("/docs").status_code == 200
    openapi_response = client.get("/openapi.json")

    assert openapi_response.status_code == 200
    schema = openapi_response.json()
    oauth2_security = schema["components"]["securitySchemes"]["OAuth2PasswordBearer"]
    assert oauth2_security["type"] == "oauth2"
    assert oauth2_security["flows"]["password"] == {
        "scopes": {},
        "tokenUrl": "token",
    }
    assert schema["paths"]["/chat"]["post"]["security"] == [{"OAuth2PasswordBearer": []}]
    assert "/token" not in schema["paths"]
