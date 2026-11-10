import base64
import json
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote, unquote

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlalchemy.orm import registry
from sqlmodel import Field, Session, SQLModel, create_engine, func, select

from xpeech.config.settings import JWTChannelConfig, settings
from xpeech.subapp.channel.helper import BackendJWTParams, ChatStream
from xpeech.subapp.channel.web_client.dao import AuthenticationSession, User, WebClientDAO

web = import_module("xpeech.subapp.channel.web_client.app")
proxy = import_module("xpeech.subapp.channel.web_client.routes.proxy")
ORIGIN = "https://portal.example.test"
SESSION_ID = "018f0000-0000-7000-8000-000000000001"
NEXT_SESSION_ID = "018f0000-0000-7000-8000-000000000002"
EMBED_PATHS = [
    "/api/auth/embed", "/api/chat", "/api/chat_sync", "/api/answer_question", "/api/files",
    "/api/history/sessions", f"/api/history/sessions/{SESSION_ID}",
]


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(web, "PBKDF2_ITERATIONS", 1)
    monkeypatch.setattr(settings.jwt, "channels", {
        "crm": JWTChannelConfig(secret_key="crm-test-channel-key-at-least-64-characters-for-sha512-signing-tests"),
    })
    app = web.create_app(web.WebConfig(
        backend_url="http://backend.test", database_path=tmp_path / "users.db",
        static_dir=tmp_path / "missing", system_name="Assistant",
    ))
    with TestClient(app) as client:
        yield client


def signed_identity(user_id="alice", channel_name="crm", *, secret=None, claims=None, algorithm=None):
    if secret is None:
        channel = settings.jwt.channels.get(channel_name)
        secret = channel.secret_key if channel is not None else "unconfigured-channel-key-at-least-32-characters"
    payload = claims if claims is not None else {
        "sub": user_id, "exp": int((datetime.now(UTC) + timedelta(seconds=60)).timestamp()),
    }
    headers = {"kid": channel_name} if channel_name is not None else {}
    return jwt.encode(payload, secret, algorithm=algorithm or settings.jwt.algorithm, headers=headers)


def bearer(token=None, **extra):
    return {
        "Authorization": f"Bearer {token if token is not None else signed_identity()}",
        "x-session-id": SESSION_ID, "x-sender-name": quote("张三", safe=""), **extra,
    }


def send(client, path, headers):
    if path in ("/api/chat", "/api/chat_sync"):
        return client.post(path, headers=headers, data={"content": "[]"})
    if path == "/api/answer_question":
        return client.post(path, headers=headers, data={"answer": "yes"})
    return client.get(path, headers=headers, params={"path": "report.txt"} if path == "/api/files" else {})


def stream_mock(monkeypatch):
    chat = AsyncMock(side_effect=lambda *_args, **_kwargs: ChatStream(
        httpx.Response(200, content=b'data: {"event":"assistant_end","context":""}\n\n'),
        SimpleNamespace(aclose=AsyncMock()),
    ))
    sync = AsyncMock(return_value=httpx.Response(200, json=[{"event": "assistant", "context": "hello"}]))
    monkeypatch.setattr(proxy, "open_chat", chat)
    monkeypatch.setattr(proxy, "open_chat_sync", sync)
    return SimpleNamespace(chat=chat, sync=sync)


def test_embed_validation_is_stateless_and_does_not_create_users_or_sessions(client, tmp_path, monkeypatch):
    for method in ("get_user_by_user_id", "create_user", "get_user_for_session", "create_session", "delete_session"):
        monkeypatch.setattr(WebClientDAO, method, AsyncMock(side_effect=AssertionError("Must not access local auth data")))
    stream = stream_mock(monkeypatch)
    answer = AsyncMock(return_value=httpx.Response(200, content=b"ok"))
    file = AsyncMock(return_value=httpx.Response(200, content=b"file"))
    monkeypatch.setattr(proxy, "submit_question", answer)
    monkeypatch.setattr(proxy, "fetch_file", file)
    token = signed_identity("external_visitor")
    headers = bearer(token)
    identity = client.get("/api/auth/embed", headers=headers)
    assert identity.status_code == 200
    assert identity.json()["user_id"] == "external_visitor"
    assert 0 < identity.json()["expires_in"] <= 60
    assert set(identity.json()) == {"user_id", "expires_in"}
    assert identity.headers["cache-control"] == "no-store"
    assert "set-cookie" not in identity.headers
    for sender in ("张三", "李四"):
        assert send(client, "/api/chat", {**headers, "x-sender-name": quote(sender, safe="")}).status_code == 200
        assert stream.chat.await_args.args[:2] == (BackendJWTParams(token), sender)
    assert send(client, "/api/answer_question", headers).status_code == 200
    assert send(client, "/api/files", headers).status_code == 200
    for mocked in (answer, file):
        assert mocked.await_args.args[0] == BackendJWTParams(token)
    assert client.post("/api/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/auth/embed", headers=headers).status_code == 200
    engine = create_engine(f"sqlite:///{(tmp_path / 'users.db').as_posix()}")
    try:
        with Session(engine) as database:
            assert database.exec(select(User.user_id)).all() == ["admin"]
            assert database.exec(select(func.count()).select_from(AuthenticationSession)).one() == 0
        assert "embedded_sessions" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()
    assert "/api/auth/embed/session" not in client.app.openapi()["paths"]
    for method in (client.post, client.delete):
        assert method("/api/auth/embed/session").status_code == 404


def test_explicit_jwt_does_not_reuse_local_account_or_admin_cookie(client):
    assert client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"}).status_code == 200
    local = client.post("/api/admin/users", json={
        "user_id": "alice", "username": "LocalAlice", "password": "alice-password", "is_admin": True,
    }).json()
    headers = bearer()
    assert client.get("/api/auth/embed", headers=headers).json()["user_id"] == "alice"
    for path in ("/api/auth/me", "/api/admin/users", "/api/statistics"):
        assert client.get(path, headers=headers).status_code == 403
    assert client.patch("/api/auth/password", headers=headers, json={"new_password": "new-password"}).status_code == 403
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer invalid"}).status_code == 401
    assert client.patch(f"/api/admin/users/{local['id']}", json={"is_active": False}).status_code == 200
    assert client.get("/api/auth/embed", headers=headers).status_code == 200
    assert client.delete(f"/api/admin/users/{local['id']}").status_code == 204
    assert client.get("/api/auth/embed", headers=headers).status_code == 200
    assert client.post("/api/auth/logout", headers=headers).status_code == 204
    assert client.get("/api/auth/me").json()["user_id"] == "admin"
    assert client.get("/api/auth/embed", headers=bearer(signed_identity("admin"))).status_code == 200
    assert client.get("/api/admin/users", headers=bearer(signed_identity("admin"))).status_code == 403


@pytest.mark.parametrize("origin", [ORIGIN, "https://other.test", "https://unconfigured.test", "http://localhost:3000"])
def test_cors_and_embedding_work_without_origin_configuration(client, origin):
    for path, method in (("/api/auth/embed", "GET"), ("/api/chat", "POST"), ("/api/admin/users/1", "PATCH")):
        response = client.options(path, headers={
            "Origin": origin, "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type,authorization,x-session-id,x-sender-name",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"
        assert "x-sender-name" in response.headers["access-control-allow-headers"]
        assert "access-control-allow-credentials" not in response.headers
    headers = {**bearer(), "Origin": origin}
    assert client.get("/api/auth/embed", headers=headers).status_code == 200
    invalid = client.get("/api/auth/embed", headers={"Authorization": "Bearer invalid", "Origin": origin})
    assert invalid.status_code == 401
    assert invalid.headers["access-control-allow-origin"] == "*"
    assert client.get("/?embed=1").headers["content-security-policy"] == "frame-ancestors *"
    sdk = client.get("/embed.js", headers={"Origin": origin})
    assert sdk.status_code == 200
    assert sdk.headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize("user_id", ["", "  ", "x" * 129, "a\nb", "../admin", "user/name", "alice@example.test", "用户"])
def test_invalid_subjects_are_rejected(client, user_id):
    assert client.get("/api/auth/embed", headers=bearer(signed_identity(user_id))).status_code == 401


@pytest.mark.parametrize("user_id", ["alice", "admin", "Alice Smith", "0"])
def test_business_jwt_is_forwarded_unchanged_for_chat_answers_and_files(client, monkeypatch, user_id):
    name = "crm.portal-v2"
    secret = "custom-crm-portal-key-at-least-64-characters-for-sha512-signing-tests"
    monkeypatch.setattr(settings.jwt, "channels", {name: JWTChannelConfig(secret_key=secret)})
    token = signed_identity(user_id, name, claims={
        "sub": user_id, "exp": int((datetime.now(UTC) + timedelta(seconds=10)).timestamp()), "source": "portal",
    })
    headers = bearer(token, **{"x-user-id": "spoofed-user", "x-channel-name": "web_client"})
    calls = []

    def backend(request):
        calls.append(request)
        backend_token = request.headers["authorization"].removeprefix("Bearer ")
        assert backend_token == token
        assert "x-channel-name" not in request.headers
        claims = jwt.decode(backend_token, secret, algorithms=[settings.jwt.algorithm])
        assert claims["sub"] == user_id
        assert jwt.get_unverified_header(backend_token)["kid"] == name
        if request.url.path == "/chat_sync":
            assert request.headers["accept"] == "application/json"
            assert unquote(request.headers["x-sender-name"]) == "张三"
            # 同步执行由后端生成独立会话，不透传 x-session-id。
            assert "x-session-id" not in request.headers
            return httpx.Response(200, json=[{"event": "assistant", "context": "hello"}])
        return httpx.Response(200, content=b'data: {"event":"assistant_end","context":""}\n\n')

    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(**kwargs, transport=httpx.MockTransport(backend)))
    for path in EMBED_PATHS[1:]:
        assert send(client, path, headers).status_code == 200
    assert [request.url.path for request in calls] == ["/chat", "/chat_sync", "/answer_question", "/files"]
    assert unquote(calls[0].headers["x-sender-name"]) == "张三"
    assert calls[0].headers["x-session-id"] == SESSION_ID


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat_sync", "/api/answer_question"])
@pytest.mark.parametrize("session_id", [None, "", "default", "invalid", "018f0000-0000-4000-8000-000000000001", SESSION_ID.upper()])
def test_embedded_requests_require_canonical_uuidv7(client, monkeypatch, path, session_id):
    monkeypatch.setattr(proxy.httpx, "AsyncClient", Mock(side_effect=AssertionError("Must not contact backend")))
    headers = bearer()
    if session_id is None:
        headers.pop("x-session-id")
    else:
        headers["x-session-id"] = session_id
    assert send(client, path, headers).status_code == 422


def test_two_embedded_conversations_keep_distinct_ids(client, monkeypatch):
    stream = stream_mock(monkeypatch)
    headers = bearer()
    for session_id in (SESSION_ID, SESSION_ID, NEXT_SESSION_ID):
        assert send(client, "/api/chat", {**headers, "x-session-id": session_id}).status_code == 200
    assert [call.kwargs["session_id"] for call in stream.chat.await_args_list] == [SESSION_ID, SESSION_ID, NEXT_SESSION_ID]


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat_sync"])
@pytest.mark.parametrize("sender", [None, "", "%20", "%FF"])
def test_embedded_sender_name_is_required_and_validated(client, monkeypatch, sender, path):
    stream = stream_mock(monkeypatch)
    headers = bearer()
    if sender is None:
        headers.pop("x-sender-name")
    else:
        headers["x-sender-name"] = sender
    assert send(client, path, headers).status_code == 422
    stream.chat.assert_not_awaited()
    stream.sync.assert_not_awaited()


@pytest.mark.parametrize("channel", [None, "unknown", "internal", "feishu", "web_client", "", "bad name"])
def test_jwt_must_use_a_configured_custom_channel(client, channel):
    assert client.get("/api/auth/embed", headers=bearer(signed_identity(channel_name=channel))).status_code == 401


@pytest.mark.parametrize("path", EMBED_PATHS)
@pytest.mark.parametrize("claims", [
    {"sub": "alice"}, {"exp": 9999999999}, {"sub": 0, "exp": 9999999999},
    {"sub": "alice", "exp": 1}, {"sub": "alice", "exp": True},
    {"sub": "alice", "exp": "9999999999"}, {"sub": "alice", "exp": float("inf")},
    {"sub": "alice", "exp": float("nan")}, {"sub": "alice", "exp": 1e100},
    {"sub": "alice", "exp": 9999999999, "nbf": 9999999998},
])
def test_invalid_claims_are_rejected_on_every_request(client, monkeypatch, path, claims):
    monkeypatch.setattr(proxy.httpx, "AsyncClient", Mock(side_effect=AssertionError("Must not contact backend")))
    assert send(client, path, bearer(signed_identity(claims=claims))).status_code == 401


def test_signature_tampering_wrong_keys_and_algorithm_substitution_are_rejected(client):
    valid = signed_identity()
    parts = valid.split(".")
    claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
    claims["sub"] = "admin"
    parts[1] = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    for token in (
        ".".join(parts), signed_identity("admin", secret="wrong-signing-key-at-least-32-characters"),
        signed_identity("admin", algorithm="HS512"),
        jwt.encode({"sub": "admin", "exp": 9999999999}, "", algorithm="none", headers={"kid": "crm"}),
        "not.a.jwt", "embed_legacy-token", "x" * 16385,
    ):
        assert client.get("/api/auth/embed", headers=bearer(token)).status_code == 401


def test_jwt_expiration_is_checked_again_for_each_request(client, monkeypatch):
    stream = stream_mock(monkeypatch)
    expires_at = int((datetime.now(UTC) + timedelta(seconds=60)).timestamp())
    headers = bearer(signed_identity(claims={"sub": "alice", "exp": expires_at}))
    assert client.get("/api/auth/embed", headers=headers).status_code == 200
    assert send(client, "/api/chat", headers).status_code == 200
    after_expiration = datetime.fromtimestamp(expires_at + 1, UTC)
    monkeypatch.setattr(jwt.api_jwt, "datetime", SimpleNamespace(now=lambda tz: after_expiration))
    for path in EMBED_PATHS:
        assert send(client, path, headers).status_code == 401
    assert stream.chat.await_count == 1
    assert stream.sync.await_count == 0


def test_global_algorithm_is_used_but_no_new_lifetime_or_token_is_issued(client, monkeypatch):
    monkeypatch.setattr(settings.jwt, "algorithm", "HS512")
    monkeypatch.setattr(settings.jwt, "access_token_expire_seconds", 45)
    headers = bearer(signed_identity(claims={"sub": "alice", "exp": 9999999999, "aud": "business", "is_admin": True}))
    response = client.get("/api/auth/embed", headers=headers)
    assert response.status_code == 200
    assert response.json()["user_id"] == "alice"
    assert response.json()["expires_in"] > 45
    assert client.get("/api/admin/users", headers=headers).status_code == 403


def test_embedded_jwt_cannot_authenticate_through_cookies(client):
    headers = bearer()
    client.cookies.set("xpeech_session", headers["Authorization"].removeprefix("Bearer "))
    for path in EMBED_PATHS:
        assert send(client, path, {}).status_code == 401
    assert client.get("/api/auth/embed", headers=headers).status_code == 200


@pytest.mark.parametrize("session_id", [None, "00000000-0000-7000-8000-000000000000"])
def test_local_cookie_login_still_signs_backend_jwt_for_main_session(client, monkeypatch, session_id):
    secret = "web-client-test-key-at-least-64-characters-for-sha512-signing-tests"
    monkeypatch.setitem(settings.jwt.channels, "web_client", JWTChannelConfig(secret_key=secret))
    assert client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"}).status_code == 200
    calls = []

    def backend(request):
        calls.append(request)
        token = request.headers["authorization"].removeprefix("Bearer ")
        assert jwt.get_unverified_header(token)["kid"] == "web_client"
        assert jwt.decode(token, secret, algorithms=[settings.jwt.algorithm])["sub"] == "admin"
        if request.url.path == "/chat_sync":
            assert "x-session-id" not in request.headers  # 同步执行由后端生成独立会话
        else:
            assert request.headers["x-session-id"] == "00000000-0000-7000-8000-000000000000"
        return httpx.Response(200, content=b'data: {"event":"assistant_end","context":""}\n\n')

    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(**kwargs, transport=httpx.MockTransport(backend)))
    headers = {"x-channel-name": "crm"}
    if session_id is not None:
        headers["x-session-id"] = session_id
    for path in ("/api/chat", "/api/chat_sync", "/api/answer_question"):
        assert send(client, path, headers).status_code == 200
    assert len(calls) == 3
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat_sync", "/api/answer_question"])
@pytest.mark.parametrize("session_id", [SESSION_ID, "not-a-uuid"])
def test_local_cookie_requests_accept_uuidv7_sessions(client, monkeypatch, path, session_id):
    monkeypatch.setitem(
        settings.jwt.channels,
        "web_client",
        JWTChannelConfig(secret_key="web-client-test-key-at-least-64-characters-for-sha512-signing-tests"),
    )
    stream_mock(monkeypatch)
    monkeypatch.setattr(proxy, "submit_question", AsyncMock(return_value=httpx.Response(200, content=b"ok")))
    assert client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"}).status_code == 200
    expected_status = 200 if session_id == SESSION_ID else 422
    assert send(client, path, {"x-session-id": session_id}).status_code == expected_status


@pytest.mark.asyncio
async def test_old_local_session_schema_is_preserved_without_creating_embedded_table(tmp_path):
    path = tmp_path / "old-users.db"

    class LegacyModel(SQLModel, registry=registry()):
        pass

    class LegacySession(LegacyModel, table=True):
        __tablename__ = "sessions"

        token_hash: str = Field(primary_key=True)
        user_id: int
        expires_at: datetime
        created_at: datetime

    engine = create_engine(f"sqlite:///{path.as_posix()}")
    try:
        LegacySession.metadata.create_all(engine)
        with Session(engine) as database:
            database.add(LegacySession(
                token_hash="old-session", user_id=1,
                expires_at=datetime(2999, 1, 1), created_at=datetime(2026, 1, 1),
            ))
            database.commit()
    finally:
        engine.dispose()
    dao = WebClientDAO(path)
    try:
        await dao.initialize(lambda: "admin-hash")
        assert (await dao.get_user_for_session("old-session")).user_id == "admin"
        await dao.initialize(lambda: "unused")
        assert (await dao.get_user_for_session("old-session")).user_id == "admin"
        async with dao.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
            assert "embedded_sessions" not in tables
    finally:
        await dao.close()
