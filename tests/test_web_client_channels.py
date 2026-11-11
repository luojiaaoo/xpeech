from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from xpeech.config.settings import JWTChannelConfig, JWTConfig, WebExtraChannelConfig, settings
from xpeech.subapp.channel.helper import ChatStream, _backend_headers
from xpeech.utils.channel_jwt import UserJWTAuth

web = import_module("xpeech.subapp.channel.web_client.app")
proxy = import_module("xpeech.subapp.channel.web_client.routes.proxy")
oauth = import_module("xpeech.subapp.channel.web_client.routes.auth")
SESSION_ID = "018f0000-0000-7000-8000-000000000001"


@pytest.fixture
def configured_channels(monkeypatch):
    channels = {
        "web_client": JWTChannelConfig(
            secret_key="web-channel-test-secret-key-with-at-least-32-characters",
            allow_extra_channels=[WebExtraChannelConfig(
                channel="crm", system_name="客户服务助手",
                recommended_questions=["如何查询客户资料？", "整理跟进计划"],
            )],
        ),
        "crm": JWTChannelConfig(secret_key="crm-channel-test-secret-key-with-at-least-32-characters"),
        "private": JWTChannelConfig(secret_key="private-channel-test-secret-key-with-at-least-32-characters"),
    }
    monkeypatch.setattr(settings.jwt, "channels", channels)
    return channels


def test_extra_channels_require_unique_configured_custom_channels(configured_channels):
    def validate():
        return JWTConfig(secret_key="internal-test-secret-key-at-least-32-characters", channels=configured_channels)

    validate()
    extras = configured_channels["web_client"].allow_extra_channels
    extras.append(extras[0])
    with pytest.raises(ValueError, match="duplicate"):
        validate()
    extras.pop()
    for channel in ["unknown", "internal", "web_client", "feishu"]:
        extras[0] = WebExtraChannelConfig(channel=channel, system_name="测试系统")
        with pytest.raises(ValueError, match="configured custom"):
            validate()
    extras.clear()
    configured_channels["crm"].allow_extra_channels = [WebExtraChannelConfig(channel="private", system_name="私有")]
    with pytest.raises(ValueError, match="only be configured"):
        validate()


@pytest.fixture
def oauth_display_type(request):
    return getattr(request, "param", "qrcode")


@pytest.fixture
def channel_client(postgres_database, configured_channels, monkeypatch, tmp_path, oauth_display_type):
    monkeypatch.setattr(web, "PBKDF2_ITERATIONS", 1)
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Assistant</title>")
    app = web.create_app(web.WebConfig(
        backend_url="http://backend.test", static_dir=static, system_name="默认助手",
        session_history_path=tmp_path / "history",
        oauth2=(web.OAuth2WebConfig(
            provider_name="test", display_type=oauth_display_type, client_id="client", client_secret="secret",
            authorization_url="https://oauth.test/authorize", token_url="https://oauth.test/token",
            userinfo_url="https://oauth.test/user", redirect_uri=None, scopes=(),
            user_id_claim=".sub", username_claim=".name", use_pkce=True,
            token_auth_method="client_secret_post", extra_authorization_params={},
        ),),
    ))
    with TestClient(app) as client:
        yield client


def login(client, channel="crm"):
    response = client.post("/api/auth/login", json={
        "user_id": "admin", "password": "admin123456", "channel": channel,
    })
    assert response.status_code == 200
    assert response.json()["channel"] == channel


def test_login_branding_persists_and_apis_cannot_override_the_channel(channel_client):
    client = channel_client
    config = client.get("/api/config?channel=crm")
    assert config.status_code == 200
    assert config.headers["cache-control"] == "no-store"
    assert config.json()["system_name"] == "客户服务助手"
    assert config.json()["recommended_questions"] == ["如何查询客户资料？", "整理跟进计划"]
    assert "secret" not in config.text
    assert client.get("/api/config").json()["channel"] == "web_client"
    login(client)
    assert client.get("/api/config").json() == config.json()
    assert client.get("/api/auth/me").json()["channel"] == "crm"
    assert client.get("/api/history/sessions?channel=private").status_code == 400
    assert client.get("/api/auth/me?channel=web_client").status_code == 400
    assert client.get("/api/auth/me").json()["channel"] == "crm"
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/config").json()["channel"] == "web_client"


def test_removing_an_allowed_channel_disables_its_existing_web_sessions(channel_client, configured_channels):
    login(channel_client)
    configured_channels["web_client"].allow_extra_channels.clear()
    assert channel_client.get("/api/auth/me").status_code == 422
    assert channel_client.get("/api/config").status_code == 422


@pytest.mark.parametrize("channel", ["", "unknown", "private", "internal", "feishu"])
def test_web_login_rejects_channels_outside_the_allowlist(channel_client, channel):
    assert channel_client.get("/api/config", params={"channel": channel}).status_code == 422
    assert channel_client.post("/api/auth/login", json={
        "user_id": "admin", "password": "admin123456", "channel": channel,
    }).status_code == 422
    assert channel_client.post("/api/auth/oauth2/qr", json={
        "provider_name": "test", "channel": channel,
    }).status_code == 422


@pytest.mark.parametrize("url", [
    "/?channel=crm&embed=1", "/?embed=0&channel=crm", "/?embed=0",
    "/?channel=crm&channel=web_client", "/chat?channel=crm",
])
def test_channel_and_embed_are_exclusive_and_channel_is_only_a_login_parameter(channel_client, url):
    assert channel_client.get(url).status_code == 400


def test_custom_channel_is_used_for_http_websocket_jwt_and_isolated_history(channel_client, monkeypatch):
    opener = AsyncMock(side_effect=lambda *_args, **_kwargs: ChatStream(
        httpx.Response(200, content=b'data: {"event":"assistant","context":"answer"}\n\n'),
        SimpleNamespace(aclose=AsyncMock()),
    ))
    monkeypatch.setattr(proxy, "open_chat", opener)
    file_fetch = AsyncMock(return_value=httpx.Response(200, content=b"report"))
    answer = AsyncMock(return_value=httpx.Response(200, json={"message": "Answer received"}))
    monkeypatch.setattr(proxy, "fetch_file", file_fetch)
    monkeypatch.setattr(proxy, "submit_question", answer)
    client = channel_client
    login(client)
    assert client.get("/api/files", params={"path": "report.txt"}).content == b"report"
    assert client.post("/api/answer_question", data={"answer": "yes"}).status_code == 200
    assert client.post("/api/chat", data={"content": "[]"}, headers={"x-session-id": SESSION_ID}).status_code == 200
    assert client.get("/api/history/sessions").json()[0]["channel"] == "crm"
    with client.websocket_connect("/api/chat/ws") as socket:
        socket.send_json({"type": "start", "session_id": SESSION_ID, "content": "[]"})
        assert socket.receive_json()["type"] == "event"
        assert socket.receive_json()["type"] == "done"
    for call in [*opener.await_args_list, file_fetch.await_args, answer.await_args]:
        headers = _backend_headers(call.args[0])
        auth = UserJWTAuth(headers["authorization"].removeprefix("Bearer "))
        assert (auth.channel, auth.user_id) == ("crm", "admin")
    login(client, "web_client")
    assert client.get("/api/history/sessions").json() == []
    login(client)
    assert len(client.get(f"/api/history/sessions/{SESSION_ID}").json()["events"]) == 4


@pytest.mark.parametrize("oauth_display_type", ["qrcode", "link"], indirect=True)
def test_oauth2_binds_channel_to_attempt_and_preserves_it_after_callback(channel_client, monkeypatch, oauth_display_type):
    class OAuthClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return httpx.Response(200, json={"access_token": "provider-token"})

        async def get(self, *args, **kwargs):
            return httpx.Response(200, json={"sub": "admin", "name": "Admin"})

    monkeypatch.setattr(oauth.httpx, "AsyncClient", OAuthClient)
    qr = channel_client.post("/api/auth/oauth2/qr", json={"provider_name": "test", "channel": "crm"}).json()
    state = parse_qs(urlsplit(qr["authorization_url"]).query)["state"][0]
    callback = channel_client.get("/api/auth/oauth2/callback", params={"state": state, "code": "ok"}, follow_redirects=False)
    assert callback.status_code == (303 if oauth_display_type == "link" else 200)
    if oauth_display_type == "qrcode":
        polled = channel_client.post("/api/auth/oauth2/poll", json={
            "login_id": qr["login_id"], "poll_token": qr["poll_token"], "channel": "web_client",
        })
        assert polled.json()["user"]["channel"] == "crm"
    assert channel_client.get("/api/config").json()["system_name"] == "客户服务助手"
    assert channel_client.get("/api/auth/me").json()["channel"] == "crm"
