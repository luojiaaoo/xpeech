import asyncio
import json
from importlib import import_module
from unittest.mock import AsyncMock

import pytest

from xpeech.config.settings import ExtraUserInfoConfig, OAuth2Config, settings
from xpeech.subapp.channel.web_client.models import OAuth2WebConfig, WebConfig
from xpeech.subapp.channel.helper import ChatStream

settings_module = import_module("xpeech.config.settings")


@pytest.mark.asyncio
async def test_sender_name_and_organization_share_command_result(monkeypatch):
    execute = AsyncMock(return_value='{"data":{"org":"company/team","name":"Global Alice"}}')
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    config = ExtraUserInfoConfig(
        command_extra_user_info_template="lookup ${user_id}",
        user_organization_claim=".data.org",
        session_metadata_claims={"department": ".data.org", "display_name": ".data.name"},
    )
    organization, sender_name, metadata = await asyncio.gather(
        config.resolve_user_organization("alice"), config.resolve_sender_name("alice", ".data.name"),
        config.resolve_session_metadata("alice"),
    )
    assert (organization, sender_name) == ("company/team", "Global Alice")
    assert metadata == {"department": "company/team", "display_name": "Global Alice"}
    assert execute.await_count == 1


@pytest.mark.asyncio
async def test_session_metadata_preserves_strings_and_isolates_users(monkeypatch):
    execute = AsyncMock(side_effect=['{"number":42,"note":"  "}', '{"number":7,"note":"Bob"}'])
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    config = ExtraUserInfoConfig(
        command_extra_user_info_template="lookup $user_id",
        session_metadata_claims={"employee_number": ".number | tostring", "note": ".note"},
    )
    assert await config.resolve_session_metadata("alice") == {"employee_number": "42", "note": "  "}
    assert await config.resolve_session_metadata("bob") == {"employee_number": "7", "note": "Bob"}
    assert execute.await_count == 2
    assert await ExtraUserInfoConfig().resolve_session_metadata("alice") == {}
    assert execute.await_count == 2


def test_session_metadata_claims_require_command_and_valid_jq():
    with pytest.raises(ValueError, match="command_extra_user_info_template"):
        ExtraUserInfoConfig(session_metadata_claims={"department": ".department"})
    with pytest.raises(ValueError, match="jq"):
        ExtraUserInfoConfig(command_extra_user_info_template="lookup $user_id", session_metadata_claims={"department": "["})


@pytest.mark.parametrize("payload", ['{}', '{"name":null}', '{"name":42}', '{"name":" "}', '{"name":["a","b"]}'])
@pytest.mark.asyncio
async def test_extra_sender_name_rejects_invalid_claim(monkeypatch, payload):
    monkeypatch.setattr(settings_module, "execute_command_template", AsyncMock(return_value=payload))
    config = ExtraUserInfoConfig(command_extra_user_info_template="lookup $user_id")
    with pytest.raises(ValueError):
        await config.resolve_sender_name("alice", ".name")


@pytest.mark.asyncio
async def test_invalid_json_is_not_cached(monkeypatch):
    execute = AsyncMock(side_effect=['not json', '{"name":"Alice"}'])
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    config = ExtraUserInfoConfig(command_extra_user_info_template="lookup $user_id")
    with pytest.raises(ValueError):
        await config.resolve_sender_name("alice", ".name")
    assert await config.resolve_sender_name("alice", ".name") == "Alice"
    assert execute.await_count == 2


def test_oauth_username_claim_is_required():
    with pytest.raises(ValueError, match="username_claim"):
        OAuth2Config()
    assert OAuth2Config(username_claim=".name").username_source == "oauth2"
    with pytest.raises(ValueError, match="username_claim"):
        OAuth2Config(username_claim=None)
    with pytest.raises(ValueError, match="username_source"):
        OAuth2Config(username_claim=".name", username_source="unknown")


@pytest.mark.parametrize("claim", ["name", "[", ""])
def test_global_claim_validation(claim):
    with pytest.raises(ValueError):
        ExtraUserInfoConfig(command_extra_user_info_template="lookup $user_id", user_organization_claim=claim)


@pytest.mark.parametrize("local_claim", [".name", ".missing"])
@pytest.mark.parametrize("username_source", ["oauth2", "extra_user_info"])
@pytest.mark.parametrize("existing_user", [False, True, "disabled"])
@pytest.mark.parametrize("display_type", ["link", "qrcode"])
def test_oauth_login_creates_local_account_by_default(monkeypatch, tmp_path, local_claim, existing_user, display_type, username_source):
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlsplit
    import httpx
    from fastapi.testclient import TestClient
    from xpeech.subapp.channel.web_client.dao import User

    routes = import_module("xpeech.subapp.channel.web_client.routes.auth")
    web = import_module("xpeech.subapp.channel.web_client.app")
    proxy = import_module("xpeech.subapp.channel.web_client.routes.proxy")
    oauth_username = "OAuth Alice" * 8
    execute = AsyncMock(return_value='{"name":"Backend Alice"}')
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    monkeypatch.setattr(settings, "extra_user_info", ExtraUserInfoConfig(
        command_extra_user_info_template="lookup $user_id", session_metadata_claims={"employee_name": ".name"},
    ))

    class OAuthClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return SimpleNamespace(status_code=200, json=lambda: {"access_token": "token"})

        async def get(self, *args, **kwargs):
            return SimpleNamespace(status_code=200, json=lambda: {"sub": "alice", "name": oauth_username})

    monkeypatch.setattr(routes.httpx, "AsyncClient", OAuthClient)
    local_user = User(
        id=1, user_id="alice", username="Database Alice", password_hash="hash",
        is_admin=True, is_active=existing_user != "disabled",
    )
    users = {"alice": local_user} if existing_user else {}
    sessions = {}

    async def create_user(**values):
        assert values.pop("overwrite_username") is True
        if "alice" in users:
            users["alice"].username = values["username"]
            return users["alice"]
        user = User(id=2, **values)
        users[user.user_id] = user
        return user

    from threading import Event
    poll_started = Event()
    release_poll = Event()
    block_poll = False

    async def get_user(user_id):
        if block_poll:
            poll_started.set()
            await asyncio.to_thread(release_poll.wait, 5)
        return users.get(user_id)

    async def update_user(user_pk, **values):
        users["alice"].username = values["username"]
        return users["alice"]

    async def create_session(**values):
        sessions[values["token_hash"]] = users["alice"]

    async def delete_session(token_hash):
        sessions.pop(token_hash, None)

    dao = SimpleNamespace(
        initialize=AsyncMock(), close=AsyncMock(),
        get_user_by_user_id=AsyncMock(side_effect=get_user),
        create_user=AsyncMock(side_effect=create_user),
        update_user=AsyncMock(side_effect=update_user),
        create_session=AsyncMock(side_effect=create_session),
        delete_session=AsyncMock(side_effect=delete_session),
        get_user_for_session=AsyncMock(side_effect=lambda token_hash: sessions.get(token_hash)),
        change_password=AsyncMock(return_value=True),
    )
    monkeypatch.setattr(web, "WebClientDAO", lambda engine: dao)
    monkeypatch.setattr(web, "_password_matches", lambda *args: True)
    monkeypatch.setattr(web, "_password_hash", lambda *args: "hash")
    chat = AsyncMock(return_value=httpx.Response(200, json=[]))
    monkeypatch.setattr(proxy, "open_chat_sync", chat)
    stream = AsyncMock(side_effect=lambda *args, **kwargs: ChatStream(
        httpx.Response(200, content=b""), SimpleNamespace(aclose=AsyncMock()),
    ))
    monkeypatch.setattr(proxy, "open_chat", stream)
    app = web.create_app(WebConfig(
        backend_url="http://backend.test", static_dir=tmp_path / "missing", system_name="Test",
        session_history_path=tmp_path / "history",
        oauth2=(OAuth2WebConfig(
            provider_name="Test", client_id="client", client_secret="secret", display_type=display_type,
            authorization_url="https://example.test/auth", token_url="https://example.test/token",
            userinfo_url="https://example.test/userinfo", redirect_uri=None, scopes=(),
            user_id_claim=".sub", username_claim=local_claim, username_source=username_source, use_pkce=False,
            token_auth_method="client_secret_post", extra_authorization_params={},
        ),),
    ))
    with TestClient(app) as client:
        attempt = client.post("/api/auth/oauth2/qr", json={"provider_name": "Test"}).json()
        state = parse_qs(urlsplit(attempt["authorization_url"]).query)["state"][0]
        callback = client.get(
            "/api/auth/oauth2/callback", params={"state": state, "code": "code"}, follow_redirects=False,
        )
        if local_claim == ".missing":
            assert callback.status_code == 400
            assert execute.await_count == (1 if username_source == "extra_user_info" else 0)
            dao.create_user.assert_not_awaited()
            return
        if existing_user == "disabled":
            login = client.post("/api/auth/login", json={"user_id": "alice", "password": "password"})
            assert login.status_code == 401
            dao.create_session.assert_not_awaited()
        if display_type == "link":
            if existing_user == "disabled":
                assert callback.status_code == 400
                assert client.cookies.get("xpeech_session") is None
                repeated = client.get("/api/auth/oauth2/callback", params={"state": state, "code": "code"})
                assert repeated.status_code == 400
                return
            assert callback.status_code == 303
        else:
            assert callback.status_code == 200
            poll_body = {"login_id": attempt["login_id"], "poll_token": attempt["poll_token"]}
            if not existing_user and local_claim == ".name":
                from concurrent.futures import ThreadPoolExecutor
                block_poll = True
                with ThreadPoolExecutor(max_workers=1) as pool:
                    first = pool.submit(client.post, "/api/auth/oauth2/poll", json=poll_body)
                    try:
                        assert poll_started.wait(5)
                        duplicate = client.post("/api/auth/oauth2/poll", json=poll_body)
                        assert duplicate.status_code == 404
                    finally:
                        release_poll.set()
                    result = first.result(timeout=5)
                block_poll = False
            else:
                result = client.post("/api/auth/oauth2/poll", json=poll_body)
            assert client.post("/api/auth/oauth2/poll", json=poll_body).status_code == 404
            if existing_user == "disabled":
                assert result.status_code == 401
                assert client.cookies.get("xpeech_session") is None
                return
            assert result.status_code == 200
            assert result.json()["user"]["id"] is not None
        expected_name = oauth_username[:64] if username_source == "oauth2" else "Backend Alice"
        expected = expected_name
        identity = client.get("/api/auth/me")
        assert identity.status_code == 200
        assert identity.json()["username"] == expected
        assert identity.json()["is_admin"] is bool(existing_user)
        assert identity.json()["id"] is not None
        assert client.patch("/api/auth/password", json={"new_password": "password123"}).status_code == 204
        response = client.post("/api/chat_sync", data={
            "content": "[]", "session_metadata": '{"employee_name":"client-name","trace":"sync"}',
        }, headers={
            "x-session-id": "018f0000-0000-7000-8000-000000000001",
        })
        assert response.status_code == 200
        assert chat.await_args.args[1] == expected
        assert json.loads(chat.await_args.args[3]) == {"employee_name": "Backend Alice", "trace": "sync"}
        response = client.post("/api/chat", data={"content": "[]", "session_metadata": '{"trace":"http"}'}, headers={
            "x-session-id": "018f0000-0000-7000-8000-000000000001",
        })
        assert response.status_code == 200
        assert json.loads(stream.await_args.args[3]) == {"employee_name": "Backend Alice", "trace": "http"}
        with client.websocket_connect("/api/chat/ws") as socket:
            socket.send_json({
                "type": "start", "session_id": "018f0000-0000-7000-8000-000000000002", "content": "[]",
                "session_metadata": {"employee_name": "client-name", "trace": "ws"},
            })
            assert socket.receive_json()["type"] == "done"
        assert json.loads(stream.await_args.args[3]) == {"employee_name": "Backend Alice", "trace": "ws"}
        assert client.post("/api/auth/logout").status_code == 204
        assert client.get("/api/auth/me").status_code == 401
        dao.create_user.assert_awaited_once()
        assert dao.create_user.await_args.kwargs["username"] == expected
        assert dao.create_user.await_args.kwargs["is_admin"] is False
        assert dao.create_user.await_args.kwargs["overwrite_username"] is True
        dao.create_session.assert_awaited_once()
        dao.delete_session.assert_awaited_once()
        assert dao.get_user_by_user_id.await_count == (1 if display_type == "qrcode" else 0)
        assert dao.get_user_for_session.await_count > 0
        if existing_user:
            login = client.post("/api/auth/login", json={"user_id": "alice", "password": "password"})
            assert login.json()["username"] == expected
            assert dao.create_session.await_count == 2
    execute.assert_awaited_once()
