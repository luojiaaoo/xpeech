from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from xpeech.config.settings import JWTChannelConfig, JWTConfig, WebExtraChannelConfig, settings
from xpeech.subapp.channel.helper import ChatStream, _backend_headers
from xpeech.subapp.channel.web_client.dao import User
from xpeech.subapp.channel.web_client.session import authorize_web_channel
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


@pytest.mark.parametrize("prefix", ["", " ", "\t\n"])
def test_organization_filter_rejects_empty_prefixes(prefix):
    with pytest.raises(ValueError):
        WebExtraChannelConfig(channel="crm", system_name="CRM", organization_filter=[prefix])


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["web_client", "crm"])
async def test_unrestricted_channels_skip_organization_lookup(configured_channels, monkeypatch, channel):
    lookup = AsyncMock()
    monkeypatch.setattr(type(settings.extra_user_info), "resolve_user_organization", lookup)
    assert await authorize_web_channel(channel, "admin") == channel
    lookup.assert_not_awaited()


@pytest.fixture
def organization_client(configured_channels, monkeypatch, tmp_path, oauth_display_type):
    user = User(id=1, user_id="admin", username="Admin", password_hash="unused", is_admin=True)
    dao = SimpleNamespace(
        initialize=AsyncMock(), close=AsyncMock(), create_session=AsyncMock(),
        get_user_by_user_id=AsyncMock(return_value=user),
        get_user_for_session=AsyncMock(return_value=user),
        create_user=AsyncMock(return_value=user),
    )
    monkeypatch.setattr(web, "WebClientDAO", lambda engine: dao)
    monkeypatch.setattr(web, "_password_matches", lambda *args: True)
    configured_channels["web_client"].allow_extra_channels[0].organization_filter = ["公司/销售/", "公司/客服/"]
    lookup = AsyncMock(return_value="公司/客服/支持")
    monkeypatch.setattr(type(settings.extra_user_info), "resolve_user_organization", lookup)
    app = web.create_app(web.WebConfig(
        backend_url="http://backend.test", static_dir=tmp_path / "static", system_name="默认助手",
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
        yield client, dao, lookup


@pytest.mark.parametrize("organization", ["公司/销售/客户", "公司/客服/支持", "公司/研发/", None])
def test_organization_filter_checks_login_and_existing_sessions(organization_client, organization):
    client, dao, lookup = organization_client
    lookup.return_value = organization
    allowed = organization is not None and organization.startswith(("公司/销售/", "公司/客服/"))
    # Branding remains available before the user's identity is known.
    assert client.get("/api/config?channel=crm").status_code == 200
    response = client.post("/api/auth/login", json={"user_id": "admin", "password": "password", "channel": "crm"})
    assert response.status_code == (200 if allowed else 403)
    if not allowed:
        assert "set-cookie" not in response.headers
        dao.create_session.assert_not_awaited()
        return
    lookup.assert_awaited_with("admin")
    assert client.get("/api/auth/me").status_code == 200
    lookup.return_value = "公司/研发/"
    assert client.get("/api/auth/me").status_code == 403
    with client.websocket_connect("/api/chat/ws") as socket:
        socket.send_json({"type": "start", "session_id": SESSION_ID, "content": "[]"})
        error = socket.receive_json()
        assert (error["type"], error["status"]) == ("error", 403)
    login(client, "web_client")
    assert client.get("/api/auth/me").status_code == 200


@pytest.mark.parametrize("oauth_display_type", ["qrcode", "link"], indirect=True)
@pytest.mark.parametrize("allowed", [True, False])
def test_organization_filter_checks_oauth_callback_and_poll(organization_client, monkeypatch, oauth_display_type, allowed):
    client, dao, lookup = organization_client
    lookup.return_value = "公司/客服/支持" if allowed else None
    provider = AsyncMock()
    provider.__aenter__.return_value = provider
    provider.post.return_value = httpx.Response(200, json={"access_token": "token"})
    provider.get.return_value = httpx.Response(200, json={"sub": "admin", "name": "Admin"})
    monkeypatch.setattr(oauth.httpx, "AsyncClient", lambda **kwargs: provider)
    qr = client.post("/api/auth/oauth2/qr", json={"provider_name": "test", "channel": "crm"}).json()
    state = parse_qs(urlsplit(qr["authorization_url"]).query)["state"][0]
    response = client.get("/api/auth/oauth2/callback", params={"state": state, "code": "ok"}, follow_redirects=False)
    if not allowed and oauth_display_type == "link":
        assert response.status_code == 400
        assert "所属组织无权访问" in response.text
        assert "set-cookie" not in response.headers
        dao.create_session.assert_not_awaited()
    elif oauth_display_type == "link":
        assert response.status_code == 303
        assert client.get("/api/auth/me").status_code == 200
    else:
        assert response.status_code == 200
        # Organization can change between authorization and session creation.
        lookup.return_value = "公司/研发/"
        response = client.post("/api/auth/oauth2/poll", json={"login_id": qr["login_id"], "poll_token": qr["poll_token"]})
        assert response.status_code == 403
        assert "set-cookie" not in response.headers
        dao.create_session.assert_not_awaited()


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
        inject_prompt=web.InjectPromptWebConfig(enabled=True, command_template="printf ${state}"),
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


def test_config_channel_comes_from_url_even_with_a_custom_channel_cookie(configured_channels, monkeypatch, tmp_path):
    dao = SimpleNamespace(
        initialize=AsyncMock(), close=AsyncMock(),
        get_user_for_session=AsyncMock(return_value=SimpleNamespace(channel="crm")),
    )
    monkeypatch.setattr(web, "WebClientDAO", lambda engine: dao)
    app = web.create_app(web.WebConfig(
        backend_url="http://backend.test", static_dir=tmp_path / "static", system_name="默认助手",
        session_history_path=tmp_path / "history",
    ))
    with TestClient(app) as client:
        client.cookies.set("xpeech_session", "crm-login-token")
        default = client.get("/api/config").json()
        assert (default["channel"], default["system_name"]) == ("web_client", "默认助手")
        custom = client.get("/api/config?channel=crm").json()
        assert (custom["channel"], custom["system_name"]) == ("crm", "客户服务助手")
        assert custom["recommended_questions"] == ["如何查询客户资料？", "整理跟进计划"]
    dao.get_user_for_session.assert_not_awaited()


def test_url_selects_branding_independently_of_login_and_apis_cannot_override_the_channel(channel_client):
    client = channel_client
    config = client.get("/api/config?channel=crm")
    assert config.status_code == 200
    assert config.headers["cache-control"] == "no-store"
    assert config.json()["system_name"] == "客户服务助手"
    assert config.json()["recommended_questions"] == ["如何查询客户资料？", "整理跟进计划"]
    assert "secret" not in config.text
    default_config = client.get("/api/config").json()
    assert default_config["channel"] == "web_client"
    assert default_config["system_name"] == "默认助手"
    login(client)
    assert client.get("/api/config").json() == default_config
    assert client.get("/api/config?channel=crm").json() == config.json()
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
    assert channel_client.get("/api/config?channel=crm").status_code == 422
    assert channel_client.get("/api/config").json()["channel"] == "web_client"


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
    assert client.post("/api/answer_question", data={"answer": "yes"}, headers={"x-session-id": SESSION_ID}).status_code == 200
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
@pytest.mark.parametrize("entry_state", [None, "state-token-1234567890"])
def test_oauth2_binds_channel_to_attempt_and_preserves_it_after_callback(channel_client, monkeypatch, oauth_display_type, entry_state):
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
    body = {"provider_name": "test", "channel": "crm"}
    if entry_state is not None:
        body["state"] = entry_state
    qr_response = channel_client.post("/api/auth/oauth2/qr", json=body)
    assert qr_response.status_code == 200
    qr = qr_response.json()
    state = parse_qs(urlsplit(qr["authorization_url"]).query)["state"][0]
    callback = channel_client.get("/api/auth/oauth2/callback", params={"state": state, "code": "ok"}, follow_redirects=False)
    assert callback.status_code == (303 if oauth_display_type == "link" else 200)
    if oauth_display_type == "link":
        location = urlsplit(callback.headers["location"])
        expected = {"channel": ["crm"]}
        if entry_state is not None:
            expected["state"] = [entry_state]
        assert location.path == "/"
        assert parse_qs(location.query) == expected
        assert channel_client.get(callback.headers["location"]).status_code == 200
    if oauth_display_type == "qrcode":
        polled = channel_client.post("/api/auth/oauth2/poll", json={
            "login_id": qr["login_id"], "poll_token": qr["poll_token"], "channel": "web_client",
        })
        assert polled.json()["user"]["channel"] == "crm"
    assert channel_client.get("/api/config?channel=crm").json()["system_name"] == "客户服务助手"
    assert channel_client.get("/api/config").json()["channel"] == "web_client"
    assert channel_client.get("/api/auth/me").json()["channel"] == "crm"
