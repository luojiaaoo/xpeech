import asyncio
import shlex
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.server.api import app
from xpeech.agent.server.context import get_lark_user_access_token
from xpeech.agent import execute as execute_module
from xpeech.agent.tools import sandbox
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


def test_access_token_can_carry_lark_user_access_token():
    token = create_access_token(config=TEST_CONFIG, lark_user_access_token="u-request-token")

    assert decode_access_token(token, config=TEST_CONFIG)["lark_user_access_token"] == "u-request-token"


@pytest.mark.parametrize("lark_user_access_token", [123, False, {}, []])
def test_api_rejects_invalid_lark_user_access_token_claim(lark_user_access_token):
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60), "lark_user_access_token": lark_user_access_token},
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )
    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={"Authorization": f"Bearer {token}", "x-user-id": "test"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or expired bearer token"}


@pytest.mark.asyncio
async def test_chat_jwt_injects_isolated_shell_tokens_without_saving_them(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.settings.path, "cache_path", tmp_path / "cache")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", tmp_path / "sandbox-home")
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))
    monkeypatch.setenv("LARKSUITE_CLI_USER_ACCESS_TOKEN", "u-process-token")
    both_started = asyncio.Event()
    started = []
    observed = []

    async def run(message):
        started.append(message.user_id)
        if len(started) == 2:
            both_started.set()
        if len(started) <= 2:
            await asyncio.wait_for(both_started.wait(), timeout=5)
        await asyncio.sleep(0)
        args = shlex.split(sandbox.wrap_command("lark-cli whoami", tmp_path))
        environment = {args[index + 1]: args[index + 2] for index, value in enumerate(args) if value == "--setenv"}
        observed.append((message.user_id, environment.get("LARKSUITE_CLI_USER_ACCESS_TOKEN")))
        if get_lark_user_access_token() is None:
            index = args.index("--unsetenv")
            assert args[index + 1] == "LARKSUITE_CLI_USER_ACCESS_TOKEN"
        assert "lark_user_access_token" not in message.model_dump_json()
        yield {"event": "assistant", "context": "ok"}

    async def create(_user_id):
        return SimpleNamespace(workspace=tmp_path, run=run)

    monkeypatch.setattr(execute_module.AgentRunner, "create", create)

    async def send(client, user_id, lark_user_access_token):
        token = create_access_token(lark_user_access_token=lark_user_access_token)
        return await client.post(
            "/chat",
            headers={
                "Authorization": f"Bearer {token}",
                "x-user-id": user_id,
                "x-sender-name": "test-user",
            },
            data={"content": '[{"text":"hello"}]'},
        )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        responses = await asyncio.gather(
            send(client, "session-one", "u-first"),
            send(client, "session-two", "u-second"),
        )
        assert all(response.status_code == 200 for response in responses)
        assert set(observed) == {("session-one", "u-first"), ("session-two", "u-second")}
        response = await send(client, "session-one", None)

    assert response.status_code == 200
    assert observed[-1] == ("session-one", None)
    assert get_lark_user_access_token() is None


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

    api_response = client.post("/answer_question", data={"answer": "ok"}, headers={"x-user-id": "test"})

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
            "x-user-id": "test",
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
            "x-user-id": "test",
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
