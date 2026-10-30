from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.config.settings import EmbedConfig, JWTChannelConfig, settings
from xpeech.subapp.channel.helper import ChatStream

web = import_module("xpeech.subapp.channel.web_client.app")
proxy = import_module("xpeech.subapp.channel.web_client.routes.proxy")

ORIGIN = "https://portal.example.test"
OTHER_ORIGIN = "https://other.example.test"
SESSION_ID = "018f0000-0000-7000-8000-000000000001"


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(web, "PBKDF2_ITERATIONS", 1)
    monkeypatch.setattr(settings.jwt, "channels", {
        "crm": JWTChannelConfig(secret_key="crm-test-channel-key-at-least-32-characters"),
    })
    app = web.create_app(web.WebConfig(
        backend_url="http://backend.test", database_path=tmp_path / "users.db",
        static_dir=tmp_path / "missing", system_name="Assistant",
        embed=EmbedConfig(allowed_origins=[ORIGIN, OTHER_ORIGIN]),
    ))
    with TestClient(app) as client:
        yield client


def login_embedded(client, user_id="alice", origin=ORIGIN, token=None):
    headers = {"Origin": origin}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post("/api/auth/embed/session", json={"user_id": user_id, "channel_name": "crm"}, headers=headers)


def test_existing_account_is_used_without_admin_cookie_privileges(client):
    assert client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"}).status_code == 200
    created = client.post("/api/admin/users", json={
        "user_id": "alice", "username": "Alice", "password": "alice-password",
    }).json()
    result = login_embedded(client, "alice")
    assert result.status_code == 200
    assert "set-cookie" not in result.headers
    assert result.headers["access-control-allow-origin"] == ORIGIN
    assert result.headers["cache-control"] == "no-store"
    body = result.json()
    assert body["user"]["user_id"] == "alice"
    assert body["user"]["id"] == created["id"]
    assert body["user"]["username"] == "Alice"
    assert body["user"]["is_admin"] is False
    bearer = {"Authorization": f"Bearer {body['token']}"}
    assert client.get("/api/auth/me", headers=bearer).json()["id"] == body["user"]["id"]
    assert client.get("/api/admin/users", headers=bearer).status_code == 403
    assert client.get("/api/statistics", headers=bearer).status_code == 403
    assert client.patch("/api/auth/password", json={"new_password": "new-password"}, headers=bearer).status_code == 403
    # Invalid explicit credentials must not fall back to the valid administrator cookie.
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer invalid"}).status_code == 401
    assert client.post("/api/auth/logout", headers=bearer).status_code == 204
    assert client.get("/api/auth/me", headers=bearer).status_code == 401
    assert client.get("/api/auth/me").json()["user_id"] == "admin"


def test_actual_user_id_is_shared_across_allowed_origins_and_refresh_revokes_old_token(client):
    alice = login_embedded(client).json()
    again = login_embedded(client, token=alice["token"]).json()
    bob = login_embedded(client, "bob").json()
    other_site = login_embedded(client, origin=OTHER_ORIGIN).json()
    assert again["user"]["id"] == alice["user"]["id"]
    assert alice["user"]["user_id"] == "alice"
    assert bob["user"]["user_id"] == "bob"
    assert other_site["user"]["id"] == alice["user"]["id"]
    assert bob["user"]["id"] != alice["user"]["id"]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {alice['token']}"}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {again['token']}"}).status_code == 200
    assert client.delete("/api/auth/embed/session", headers={
        "Origin": ORIGIN, "Authorization": f"Bearer {again['token']}",
    }).status_code == 204
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {again['token']}"}).status_code == 401
    for headers in ({}, {"Origin": "https://attacker.test"}):
        response = client.post("/api/auth/embed/session", json={"user_id": "alice", "channel_name": "crm"}, headers=headers)
        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers


def test_cors_is_limited_to_embed_session_routes(client):
    assert "/api/auth/embed/session" in client.app.openapi()["paths"]
    preflight = client.options("/api/auth/embed/session", headers={
        "Origin": ORIGIN, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization",
    })
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in preflight.headers
    denied = client.options("/api/auth/embed/session", headers={
        "Origin": "https://attacker.test", "Access-Control-Request-Method": "POST",
    })
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers
    assert "access-control-allow-origin" not in client.get("/api/admin/users", headers={"Origin": ORIGIN}).headers
    invalid = client.post("/api/auth/embed/session", json={"user_id": ""}, headers={"Origin": ORIGIN})
    assert invalid.status_code == 422
    assert invalid.headers["access-control-allow-origin"] == ORIGIN
    policy = client.get("/?embed=1").headers["content-security-policy"]
    assert policy == f"frame-ancestors {ORIGIN} {OTHER_ORIGIN}"
    sdk = client.get("/embed.js", headers={"Origin": ORIGIN})
    assert sdk.status_code == 200
    assert sdk.headers["content-type"].startswith("application/javascript")
    assert sdk.headers["access-control-allow-origin"] == "*"


@pytest.mark.parametrize("user_id", ["", "  ", "x" * 129, "a\nb", "../admin", "user/name"])
def test_rejects_invalid_identity(client, user_id):
    assert login_embedded(client, user_id).status_code == 422


def test_default_embed_requires_an_allowed_origin(tmp_path):
    app = web.create_app(web.WebConfig(
        backend_url="http://backend.test", database_path=tmp_path / "default.db",
        static_dir=tmp_path / "missing", system_name="Assistant",
    ))
    with TestClient(app) as client:
        assert "/api/auth/embed/session" in client.app.openapi()["paths"]
        assert login_embedded(client).status_code == 403
        assert client.get("/?embed=1").headers["content-security-policy"] == "frame-ancestors 'none'"


def test_admin_account_can_sign_in_through_embedding(client):
    account = login_embedded(client, "admin")
    assert account.status_code == 200
    assert account.json()["user"]["is_admin"] is True
    headers = {"Authorization": f"Bearer {account.json()['token']}"}
    assert client.get("/api/auth/me", headers=headers).json()["user_id"] == "admin"
    assert client.get("/api/admin/users", headers=headers).status_code == 200


def test_promotion_preserves_embed_access_and_disabled_accounts_are_blocked(client):
    account = login_embedded(client).json()
    client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"})
    client.patch(f"/api/admin/users/{account['user']['id']}", json={"is_admin": True})
    assert login_embedded(client).status_code == 200
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {account['token']}"}).json()["is_admin"] is True
    client.patch(f"/api/admin/users/{account['user']['id']}", json={"is_admin": False, "is_active": False})
    assert login_embedded(client).status_code == 403
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {account['token']}"}).status_code == 401


def test_existing_local_accounts_with_embed_prefix_still_use_password_login(client):
    client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"})
    assert client.post("/api/admin/users", json={
        "user_id": "embed_employee", "username": "Employee", "password": "employee-password",
    }).status_code == 201
    assert client.post("/api/auth/login", json={
        "user_id": "embed_employee", "password": "employee-password",
    }).status_code == 200
    assert client.patch("/api/auth/password", json={"new_password": "updated-password"}).status_code == 204


@pytest.mark.parametrize("user_id", ["alice", "admin"])
def test_embedded_chat_answer_and_file_use_the_actual_user_id(client, monkeypatch, user_id):
    account = login_embedded(client, user_id).json()
    bearer = {"Authorization": f"Bearer {account['token']}", "x-session-id": SESSION_ID, "x-channel-name": "crm"}
    stream = AsyncMock(return_value=ChatStream(
        httpx.Response(200, content=b'data: {"event":"assistant","context":"ok"}\n\n'),
        SimpleNamespace(aclose=AsyncMock()),
    ))
    answer = AsyncMock(return_value=httpx.Response(200, content=b"ok"))
    file = AsyncMock(return_value=httpx.Response(200, content=b"file"))
    monkeypatch.setattr(proxy, "open_chat_stream", stream)
    monkeypatch.setattr(proxy, "submit_question", answer)
    monkeypatch.setattr(proxy, "fetch_file", file)
    assert client.post("/api/chat", headers=bearer, data={"content": '[{"text":"hello"}]'}).status_code == 200
    assert client.post("/api/answer_question", headers=bearer, data={"answer": "yes"}).status_code == 200
    assert client.get("/api/files", headers=bearer, params={"path": "report.pdf"}).content == b"file"
    assert client.get("/api/files", params={"path": "report.pdf"}).status_code == 401
    for mocked in (stream, answer, file):
        assert mocked.await_args.args[0] == user_id
    assert stream.await_args.kwargs["session_id"] == SESSION_ID
    assert answer.await_args.kwargs["session_id"] == SESSION_ID


@pytest.mark.parametrize("path", ["/api/chat", "/api/answer_question"])
@pytest.mark.parametrize("session_id", [
    None, "", "default", "not-a-uuid", "018f0000-0000-4000-8000-000000000001", SESSION_ID.upper(),
])
def test_embedded_requests_require_canonical_uuidv7(client, monkeypatch, path, session_id):
    account = login_embedded(client).json()
    stream = AsyncMock()
    answer = AsyncMock()
    monkeypatch.setattr(proxy, "open_chat_stream", stream)
    monkeypatch.setattr(proxy, "submit_question", answer)
    headers = {"Authorization": f"Bearer {account['token']}", "x-channel-name": "crm"}
    if session_id is not None:
        headers["x-session-id"] = session_id
    response = client.post(path, headers=headers, data={"content": "[]", "answer": "yes"})
    assert response.status_code == 422
    stream.assert_not_awaited()
    answer.assert_not_awaited()


def test_two_embedded_conversations_forward_distinct_ids(client, monkeypatch):
    account = login_embedded(client).json()
    stream = AsyncMock(side_effect=lambda *_args, **_kwargs: ChatStream(
        httpx.Response(200, content=b'data: {"event":"assistant_end","context":""}\n\n'),
        SimpleNamespace(aclose=AsyncMock()),
    ))
    monkeypatch.setattr(proxy, "open_chat_stream", stream)
    second_id = "018f0000-0000-7000-8000-000000000002"
    for session_id in (SESSION_ID, SESSION_ID, second_id):
        assert client.post("/api/chat", data={"content": "[]"}, headers={
            "Authorization": f"Bearer {account['token']}", "x-session-id": session_id, "x-channel-name": "crm",
        }).status_code == 200
    assert [call.kwargs["session_id"] for call in stream.await_args_list] == [SESSION_ID, SESSION_ID, second_id]


def test_origin_configuration_requires_exact_origins():
    assert EmbedConfig().allowed_origins == []
    for origins in (["*"], ["https://portal.test/path"], ["https://user@portal.test"], ["null"]):
        with pytest.raises(ValueError):
            EmbedConfig(allowed_origins=origins)
    assert EmbedConfig(allowed_origins=["https://PORTAL.test:443/"]).allowed_origins == ["https://portal.test"]


@pytest.mark.parametrize("user_id", ["alice", "admin"])
def test_embedded_channel_signs_all_backend_requests_with_its_own_key(client, monkeypatch, user_id):
    name = "crm.portal-v2"
    secret = "custom-crm-portal-key-at-least-32-characters"
    monkeypatch.setattr(settings.jwt, "channels", {name: JWTChannelConfig(secret_key=secret)})
    response = client.post("/api/auth/embed/session", json={
        "user_id": user_id, "channel_name": name,
    }, headers={"Origin": ORIGIN})
    assert response.status_code == 200
    headers = {
        "Authorization": f"Bearer {response.json()['token']}",
        "x-channel-name": name, "x-session-id": SESSION_ID,
    }
    calls = []

    def backend(request):
        calls.append(request)
        token = request.headers["authorization"].removeprefix("Bearer ")
        assert jwt.get_unverified_header(token)["kid"] == name
        jwt.decode(token, secret, algorithms=[settings.jwt.algorithm], options={"require": ["exp"]})
        assert request.headers["x-user-id"] == user_id
        return httpx.Response(200, content=b'data: {"event":"assistant_end","context":""}\n\n')

    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: async_client(
        **kwargs, transport=httpx.MockTransport(backend),
    ))
    assert client.post("/api/chat", headers=headers, data={"content": "[]"}).status_code == 200
    assert client.post("/api/answer_question", headers=headers, data={"answer": "yes"}).status_code == 200
    assert client.get("/api/files", headers=headers, params={"path": "report.txt"}).status_code == 200
    expected_paths = ["/chat", "/answer_question", f"/sessions/{user_id}/files"]
    if user_id == "admin":
        assert client.get("/api/statistics", headers=headers).status_code == 200
        expected_paths.append("/statistics")
    assert [request.url.path for request in calls] == expected_paths
    assert calls[0].headers["x-session-id"] == calls[1].headers["x-session-id"] == SESSION_ID


@pytest.mark.parametrize("name", [None, "unknown", "internal", "feishu", "web_client", "", "bad name"])
def test_unconfigured_embed_channels_are_rejected_before_forwarding(client, monkeypatch, name):
    if name is None:
        assert client.post("/api/auth/embed/session", json={"user_id": "alice"}, headers={"Origin": ORIGIN}).status_code == 422
    assert client.post("/api/auth/embed/session", json={
        "user_id": "alice", "channel_name": name,
    }, headers={"Origin": ORIGIN}).status_code == 422
    account = login_embedded(client).json()
    stream = AsyncMock()
    monkeypatch.setattr(proxy, "open_chat_stream", stream)
    headers = {"Authorization": f"Bearer {account['token']}", "x-session-id": SESSION_ID}
    if name is not None:
        headers["x-channel-name"] = name
    assert client.post("/api/chat", headers=headers, data={"content": "[]"}).status_code == 422
    stream.assert_not_awaited()


@pytest.mark.parametrize("path", ["/api/answer_question", "/api/files", "/api/statistics"])
@pytest.mark.parametrize("name", [None, "web_client", "feishu"])
def test_other_embedded_proxies_never_fall_back_to_builtin_channels(client, monkeypatch, path, name):
    account = login_embedded(client, "admin").json()
    headers = {"Authorization": f"Bearer {account['token']}", "x-session-id": SESSION_ID}
    if name is not None:
        headers["x-channel-name"] = name
    answer = AsyncMock()
    file = AsyncMock()
    monkeypatch.setattr(proxy, "submit_question", answer)
    monkeypatch.setattr(proxy, "fetch_file", file)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", Mock(side_effect=AssertionError("Must not contact backend")))
    response = client.post(path, headers=headers, data={"answer": "yes"}) if path == "/api/answer_question" else client.get(
        path, headers=headers, params={"path": "report.txt"},
    )
    assert response.status_code == 422
    answer.assert_not_awaited()
    file.assert_not_awaited()


def test_cookie_login_cannot_override_backend_channel(client, monkeypatch):
    client.post("/api/auth/login", json={"user_id": "admin", "password": "admin123456"})
    stream = AsyncMock()
    monkeypatch.setattr(proxy, "open_chat_stream", stream)
    assert client.post("/api/chat", headers={"x-channel-name": "web_client"}, data={"content": "[]"}).status_code == 403
    stream.assert_not_awaited()
