import asyncio
import shlex
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent import execute as execute_module
from xpeech.agent.server.api import app
from xpeech.agent.server.context import get_lark_user_access_token
from xpeech.agent.tools import sandbox
from xpeech.config.settings import JWTConfig, settings
from xpeech.utils.channel_jwt import (
    ChannelJWTAuth,
    create_channel_access_token,
    decode_channel_access_token,
)
from xpeech.utils.internal_jwt import (
    create_internal_access_token,
    decode_internal_access_token,
)

TEST_SECRET_KEY = "test-secret-key-that-is-at-least-32-bytes"
TEST_CHANNEL_AUTH = ChannelJWTAuth(
    "test",
    "test-channel-key-that-is-at-least-32-bytes",
)


def test_example_jwt_config_has_three_flat_signing_keys():
    with Path("conf.toml.example").open("rb") as config_file:
        config = tomllib.load(config_file)

    jwt_config = JWTConfig.model_validate(config["jwt"])

    assert set(config["jwt"]) == {
        "secret_key",
        "web_client_secret_key",
        "feishu_secret_key",
        "algorithm",
        "access_token_expire_seconds",
    }
    assert jwt_config.model_dump() == {**config["jwt"], "channels": {}}
    assert "jwt" not in config["web_client"]
    assert "jwt" not in config["feishu"]


@pytest.mark.parametrize(
    "first, second",
    [
        ("secret_key", "web_client_secret_key"),
        ("secret_key", "feishu_secret_key"),
        ("web_client_secret_key", "feishu_secret_key"),
    ],
)
def test_jwt_config_rejects_reused_signing_keys(first, second):
    config = {
        "secret_key": TEST_SECRET_KEY,
        "web_client_secret_key": "test-web-channel-key-that-is-at-least-32-bytes",
        "feishu_secret_key": "test-feishu-channel-key-that-is-at-least-32-bytes",
    }
    config[second] = config[first]

    with pytest.raises(ValueError, match="must use distinct values"):
        JWTConfig.model_validate(config)


def test_access_token_is_valid_for_exactly_one_minute():
    now = datetime.now(UTC)
    token = create_internal_access_token(secret_key=TEST_SECRET_KEY, now=now)
    claims = jwt.decode(
        token,
        TEST_SECRET_KEY,
        algorithms=[settings.jwt.algorithm],
    )

    assert claims == {"exp": int((now + timedelta(seconds=60)).timestamp())}
    assert "kid" not in jwt.get_unverified_header(token)


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_internal_and_channel_tokens_use_shared_jwt_settings(monkeypatch, algorithm):
    monkeypatch.setattr(settings.jwt, "algorithm", algorithm)
    monkeypatch.setattr(settings.jwt, "access_token_expire_seconds", 30)
    now = datetime.now(UTC)
    internal_secret = "i" * 64
    channel_auth = ChannelJWTAuth("test", "c" * 64)

    internal_token = create_internal_access_token(secret_key=internal_secret, now=now)
    channel_token = create_channel_access_token(channel_auth, now=now)
    internal_claims = decode_internal_access_token(internal_token, secret_key=internal_secret)
    channel_claims = decode_channel_access_token(channel_token, channel_auth)

    assert jwt.get_unverified_header(internal_token)["alg"] == algorithm
    assert jwt.get_unverified_header(channel_token)["alg"] == algorithm
    expected_expiry = int((now + timedelta(seconds=30)).timestamp())
    assert internal_claims["exp"] == channel_claims["exp"] == expected_expiry


def test_access_token_can_carry_lark_user_access_token():
    token = create_channel_access_token(
        TEST_CHANNEL_AUTH,
        lark_user_access_token="u-request-token",
    )

    assert (
        decode_channel_access_token(token, TEST_CHANNEL_AUTH)[
            "lark_user_access_token"
        ]
        == "u-request-token"
    )


@pytest.mark.parametrize("lark_user_access_token", [123, False, {}, []])
def test_api_rejects_invalid_lark_user_access_token_claim(lark_user_access_token):
    token = jwt.encode(
        {
            "exp": datetime.now(UTC) + timedelta(seconds=60),
            "lark_user_access_token": lark_user_access_token,
        },
        settings.jwt.feishu_secret_key,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "feishu"},
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

    async def create(_user_id, **_kwargs):
        return SimpleNamespace(workspace=tmp_path, run=run)

    monkeypatch.setattr(execute_module.AgentRunner, "create", create)

    async def send(client, user_id, lark_user_access_token):
        token = create_channel_access_token(
            ChannelJWTAuth("feishu", settings.jwt.feishu_secret_key),
            lark_user_access_token=lark_user_access_token,
        )
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
    token = create_internal_access_token(
        secret_key=TEST_SECRET_KEY,
        now=datetime.now(UTC) - timedelta(seconds=61),
    )

    try:
        decode_internal_access_token(token, secret_key=TEST_SECRET_KEY)
    except jwt.ExpiredSignatureError:
        pass
    else:
        raise AssertionError("expired token was accepted")


def test_channel_access_token_only_requires_expiration():
    now = datetime.now(UTC)
    token = create_channel_access_token(TEST_CHANNEL_AUTH, now=now)

    claims = decode_channel_access_token(
        token,
        TEST_CHANNEL_AUTH,
    )

    assert claims["exp"] == int((now + timedelta(seconds=60)).timestamp())
    assert jwt.get_unverified_header(token)["alg"] == "HS256"
    assert jwt.get_unverified_header(token)["kid"] == TEST_CHANNEL_AUTH.channel
    assert set(claims) == {"exp"}


def test_channel_token_cannot_be_decoded_as_internal_token():
    token = create_channel_access_token(
        ChannelJWTAuth("feishu", settings.jwt.feishu_secret_key)
    )

    with pytest.raises(jwt.InvalidTokenError):
        decode_internal_access_token(token)


def test_api_rejects_lark_user_token_in_internal_jwt():
    token = jwt.encode(
        {
            "exp": datetime.now(UTC) + timedelta(seconds=60),
            "lark_user_access_token": "must-be-channel-scoped",
        },
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
    )

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "test",
        },
    )

    assert response.status_code == 401


def test_one_channel_key_cannot_impersonate_another_channel():
    token = create_channel_access_token(
        ChannelJWTAuth("feishu", settings.jwt.feishu_secret_key)
    )

    with pytest.raises(jwt.InvalidTokenError):
        decode_channel_access_token(
            token,
            ChannelJWTAuth("web_client", settings.jwt.web_client_secret_key),
        )


@pytest.mark.parametrize(
    "source, kid",
    [
        ("internal", "feishu"),
        ("internal", "web_client"),
        ("feishu", None),
        ("feishu", "web_client"),
        ("web_client", None),
        ("web_client", "feishu"),
    ],
)
def test_kid_cannot_select_a_different_signing_key(source, kid):
    keys = {
        "internal": settings.jwt.secret_key,
        "feishu": settings.jwt.feishu_secret_key,
        "web_client": settings.jwt.web_client_secret_key,
    }
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60)},
        keys[source],
        algorithm=settings.jwt.algorithm,
        headers={"kid": kid} if kid is not None else None,
    )

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={"Authorization": f"Bearer {token}", "x-user-id": "test"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize("headers", [{"kid": "unknown"}, {"kid": ""}])
def test_api_rejects_unknown_kid(headers):
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60)},
        settings.jwt.secret_key,
        algorithm=settings.jwt.algorithm,
        headers=headers,
    )

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={"Authorization": f"Bearer {token}", "x-user-id": "test"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "secret_key",
    [settings.jwt.feishu_secret_key, settings.jwt.web_client_secret_key],
)
def test_api_rejects_channel_token_without_kid(secret_key):
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60)},
        secret_key,
        algorithm=settings.jwt.algorithm,
    )

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={"Authorization": f"Bearer {token}", "x-user-id": "test"},
    )

    assert response.status_code == 401


def test_web_channel_key_cannot_supply_lark_user_token():
    token = create_channel_access_token(
        ChannelJWTAuth("web_client", settings.jwt.web_client_secret_key),
        lark_user_access_token="must-be-feishu-scoped",
    )

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "test",
        },
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "channel, secret_key",
    [
        ("feishu", settings.jwt.feishu_secret_key),
        ("web_client", settings.jwt.web_client_secret_key),
    ],
)
def test_api_accepts_each_configured_channel_jwt(channel, secret_key):
    token = create_channel_access_token(ChannelJWTAuth(channel, secret_key))

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "test",
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "kid, secret_key",
    [
        ("feishu", settings.jwt.feishu_secret_key),
        ("web_client", settings.jwt.web_client_secret_key),
    ],
)
@pytest.mark.parametrize("expired", [False, True])
def test_api_rejects_channel_tokens_with_missing_or_expired_exp(kid, secret_key, expired):
    claims = {}
    if expired:
        claims["exp"] = datetime.now(UTC) - timedelta(seconds=1)
    token = jwt.encode(
        claims,
        secret_key,
        algorithm=settings.jwt.algorithm,
        headers={"kid": kid},
    )

    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={
            "Authorization": f"Bearer {token}",
            "x-user-id": "test",
        },
    )

    assert response.status_code == 401


def test_api_requires_jwt():
    client = TestClient(app)

    api_response = client.post("/answer_question", data={"answer": "ok"}, headers={"x-user-id": "test"})

    assert api_response.status_code == 401
    assert api_response.headers["www-authenticate"] == "Bearer"


def test_api_rejects_token_signed_with_another_key():
    client = TestClient(app)
    token = create_channel_access_token(ChannelJWTAuth("feishu", TEST_SECRET_KEY))

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
        data={"username": "web_client", "password": "wrong-secret"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect JWT secret"}


@pytest.mark.parametrize("channel", ["feishu", "web_client"])
def test_token_endpoint_issues_channel_jwt_for_docs(channel):
    client = TestClient(app)
    keys = {
        "feishu": settings.jwt.feishu_secret_key,
        "web_client": settings.jwt.web_client_secret_key,
    }

    token_response = client.post(
        "/token",
        data={"username": channel, "password": keys[channel]},
    )

    assert token_response.status_code == 200
    assert token_response.headers["cache-control"] == "no-store"
    assert token_response.json()["token_type"] == "bearer"
    token = token_response.json()["access_token"]
    claims = decode_channel_access_token(token, ChannelJWTAuth(channel, keys[channel]))
    assert jwt.get_unverified_header(token)["kid"] == channel
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


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/chat"),
        ("POST", "/chat_sync"),
        ("POST", "/answer_question"),
        ("GET", "/background_message?channel=feishu"),
        ("GET", "/sessions/test/files?path=result.txt"),
        ("GET", "/skills"),
        ("POST", "/skills/import?user_id=test"),
        ("GET", "/statistics"),
        ("GET", "/statistics/timeseries"),
        ("GET", "/statistics/users"),
        ("GET", "/statistics/sessions"),
        ("GET", "/statistics/records/latest"),
        ("GET", "/statistics/updates"),
        ("GET", "/statistics/records"),
    ],
)
def test_business_endpoints_reject_internal_jwt(method, path):
    token = create_internal_access_token()

    response = TestClient(app).request(
        method,
        path,
        headers={"Authorization": f"Bearer {token}", "x-user-id": "test"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        "/statistics",
        "/statistics/timeseries",
        "/statistics/users",
        "/statistics/sessions",
        "/statistics/records/latest",
        "/statistics/updates",
        "/statistics/records",
    ],
)
def test_statistics_endpoints_reject_feishu_channel(path):
    token = create_channel_access_token(ChannelJWTAuth("feishu", settings.jwt.feishu_secret_key))

    response = TestClient(app).get(path, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 403


@pytest.mark.parametrize("channel", ["feishu", "web_client", "docs"])
def test_token_endpoint_does_not_accept_internal_key(channel):
    response = TestClient(app).post(
        "/token",
        data={"username": channel, "password": settings.jwt.secret_key},
    )

    assert response.status_code == 401


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
