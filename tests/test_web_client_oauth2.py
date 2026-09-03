import re
from importlib import import_module
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

web_client_app = import_module("xpeech.channel.web_client.app")
oauth_routes = import_module("xpeech.channel.web_client.routes.auth")
settings_module = import_module("xpeech.config.settings")
OAuth2WebConfig = web_client_app.OAuth2WebConfig
InjectPromptWebConfig = web_client_app.InjectPromptWebConfig
WebConfig = web_client_app.WebConfig
OAuth2SettingsConfig = settings_module.OAuth2Config
WebClientSettingsConfig = settings_module.WebClientConfig
create_app = web_client_app.create_app
oauth2_claim = oauth_routes.oauth2_claim
oauth2_filter = oauth_routes.oauth2_filter
resolve_injected_prompt = oauth_routes.resolve_injected_prompt


class FakeOAuthResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def test_oauth2_claim_resolves_nested_feishu_userinfo():
    userinfo = {
        "code": 0,
        "data": {
            "employee_no": "oauth-user-42",
            "name": "OAuth User",
        },
    }

    assert oauth2_claim(userinfo, ".data.employee_no") == "oauth-user-42"
    assert oauth2_claim(userinfo, ".data.name") == "OAuth User"
    assert oauth2_claim(
        {"data": {"items": [{"id": "oauth-user-42"}]}}, ".data.items[0].id"
    ) == "oauth-user-42"
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_claim(userinfo, "data.employee_no")  # bare dotted path is not valid jq
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_claim(userinfo, ".data.employee_no.value")  # cannot index a string
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_claim(userinfo, ".data.missing")
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_claim({"sub": 42}, ".sub")  # non-string claim value
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_claim({"tags": ["a", "b"]}, ".tags[]")  # must yield exactly one value


def test_oauth2_filter_allows_when_any_output_is_produced():
    userinfo = {"data": {"department": "R&D", "active": True, "roles": ["user"]}}

    # any output allows login, even false or null
    assert oauth2_filter(userinfo, '.data.department == "R&D"') is True
    assert oauth2_filter(userinfo, '.data.department != "R&D"') is True  # [false]
    assert oauth2_filter(userinfo, ".data.missing") is True  # [null]
    assert oauth2_filter(userinfo, ".data.active") is True
    # denial requires the expression to produce no output, e.g. via select()
    assert oauth2_filter(userinfo, 'select(.data.department == "R&D")') is True
    assert oauth2_filter(
        {"data": {"department": "HR", "roles": []}},
        'select(.data.department == "R&D")',
    ) is False
    # iterating an empty array produces no output and denies login
    assert oauth2_filter({"data": {"roles": []}}, '.data.roles[] | select(. == "admin")') is False
    # iterating null is a jq runtime error and must raise
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_filter(userinfo, '.data.missing[] | . == "x"')
    with pytest.raises(oauth_routes.OAuth2ClaimError):
        oauth2_filter(userinfo, "department == 1")  # bare identifier is not valid jq


def test_oauth2_jq_expression_settings_are_validated_at_startup(tmp_path: Path):
    WebClientSettingsConfig(database_path=tmp_path / "users.db")

    with pytest.raises(ValueError, match="jq 表达式"):
        OAuth2SettingsConfig(provider_name="XX", userinfo_filter="department == 1")
    with pytest.raises(ValueError, match="jq 表达式"):
        OAuth2SettingsConfig(provider_name="XX", session_id_claim="sub")
    with pytest.raises(ValueError, match="jq 表达式"):
        OAuth2SettingsConfig(provider_name="XX", username_claim="data.name")
    assert (
        OAuth2SettingsConfig(
            provider_name="XX",
            session_id_claim=".data.employee_no",
            username_claim=".data.name",
            userinfo_filter='.data.department == "R&D"',
        ).userinfo_filter
        == '.data.department == "R&D"'
    )
    assert (
        OAuth2SettingsConfig(provider_name="XX").userinfo_filter is None
    )


def test_web_client_oauth2_settings_are_a_list_with_unique_provider_names(
    tmp_path: Path,
):
    config = WebClientSettingsConfig(database_path=tmp_path / "users.db")
    assert config.oauth2 == []

    with pytest.raises(ValueError, match="provider_name must be unique"):
        WebClientSettingsConfig(
            database_path=tmp_path / "users.db",
            oauth2=[
                OAuth2SettingsConfig(provider_name="Provider"),
                OAuth2SettingsConfig(provider_name=" provider "),
            ],
        )


@pytest.mark.asyncio
async def test_injected_prompt_command_replaces_state_without_length_limit():
    state = "state-token-1234567890"
    prompt_prefix = "前缀" * 300

    prompt = await resolve_injected_prompt(
        InjectPromptWebConfig(
            enabled=True,
            command_template=f"printf {prompt_prefix}${{state}}",
        ),
        state,
    )

    assert prompt == f"{prompt_prefix}{state}"
    assert len(prompt) > 512


@pytest.mark.parametrize("display_type", ["link", "qrcode"])
def test_oauth2_login_maps_to_an_existing_web_user(
    tmp_path: Path,
    monkeypatch,
    display_type: str,
):
    oauth_requests: list[tuple[str, str, dict[str, object]]] = []

    class FakeOAuthClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url: str, **kwargs):
            oauth_requests.append(("POST", url, kwargs))
            return FakeOAuthResponse({"access_token": "provider-access-token"})

        async def get(self, url: str, **kwargs):
            oauth_requests.append(("GET", url, kwargs))
            return FakeOAuthResponse(
                {
                    "code": 0,
                    "data": {
                        "employee_no": "oauth-user-42",
                        "name": "OAuth User",
                    },
                }
            )

    monkeypatch.setattr(web_client_app, "PBKDF2_ITERATIONS", 1)
    monkeypatch.setattr(oauth_routes.httpx, "AsyncClient", FakeOAuthClient)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "oauth-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_oauth",
            inject_prompt=InjectPromptWebConfig(
                enabled=True,
                command_template="printf unused-$state",
            ),
            oauth2=(
                OAuth2WebConfig(
                    provider_name="XX",
                    display_type=display_type,
                    client_id="oauth-client",
                    client_secret="oauth-secret",
                    authorization_url="https://login.example.test/authorize",
                    token_url="https://login.example.test/token",
                    userinfo_url="https://login.example.test/userinfo",
                    redirect_uri="https://assistant.example.test/api/auth/oauth2/callback",
                    scopes=("openid", "profile"),
                    session_id_claim=".data.employee_no",
                    username_claim=".data.name",
                    use_pkce=True,
                    token_auth_method="client_secret_post",
                    extra_authorization_params={"prompt": "login"},
                ),
            ),
        )
    )

    with TestClient(app) as client:
        public_config = client.get("/api/config").json()
        assert public_config["oauth2"] == [
            {
                "enabled": True,
                "provider_name": "XX",
                "display_type": display_type,
            }
        ]
        assert public_config["inject_prompt"] == {"enabled": True}

        assert client.post(
            "/api/auth/login",
            json={"session_id": "admin", "password": "admin123456"},
        ).status_code == 200
        assert client.post(
            "/api/admin/users",
            json={
                "username": "OAuthUser",
                "session_id": "oauth-user-42",
                "password": "local-password",
                "is_admin": False,
            },
        ).status_code == 201
        client.post("/api/auth/logout")

        entry_state = "state.token_~1234-567890"
        qr_response = client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "XX", "state": entry_state},
        )
        assert qr_response.status_code == 200
        qr_login = qr_response.json()
        repeated_qr_login = client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "XX", "state": entry_state},
        ).json()
        assert repeated_qr_login["login_id"] != qr_login["login_id"]
        assert repeated_qr_login["authorization_url"] != qr_login["authorization_url"]
        assert repeated_qr_login["poll_token"] != qr_login["poll_token"]
        qr_login = repeated_qr_login
        authorization_query = parse_qs(
            urlsplit(qr_login["authorization_url"]).query
        )
        assert authorization_query["client_id"] == ["oauth-client"]
        expected_redirect_uri = "https://assistant.example.test/api/auth/oauth2/callback"
        assert authorization_query["redirect_uri"] == [expected_redirect_uri]
        oauth2_state = authorization_query["state"][0]
        state_prefix = f"{entry_state}_-_"
        assert oauth2_state.startswith(state_prefix)
        state_token = oauth2_state[len(state_prefix) :]
        assert re.fullmatch(r"[A-Za-z0-9_-]{43}", state_token)
        assert authorization_query["scope"] == ["openid profile"]
        assert authorization_query["prompt"] == ["login"]
        assert authorization_query["code_challenge_method"] == ["S256"]

        poll_body = {
            "login_id": qr_login["login_id"],
            "poll_token": qr_login["poll_token"],
        }
        assert client.post("/api/auth/oauth2/poll", json=poll_body).json() == {
            "status": "pending"
        }
        assert client.post(
            "/api/auth/oauth2/poll",
            json={**poll_body, "poll_token": "not-the-browser-secret"},
        ).status_code == 404

        callback_params = {
            "state": authorization_query["state"][0],
            "code": "oauth-code",
        }
        callback = client.get(
            "/api/auth/oauth2/callback",
            params=callback_params,
            follow_redirects=False,
        )
        if display_type == "link":
            assert callback.status_code == 303
            callback_location = urlsplit(callback.headers["location"])
            assert callback_location.path == "/"
            assert parse_qs(callback_location.query)["state"] == [entry_state]
            assert client.get("/api/auth/me").json()["session_id"] == "oauth-user-42"

            repeated_callback = client.get(
                "/api/auth/oauth2/callback",
                params={"state": authorization_query["state"][0], "code": "oauth-code"},
            )
            assert repeated_callback.status_code == 400
            assert client.post("/api/auth/oauth2/poll", json=poll_body).status_code == 404
        else:
            assert callback.status_code == 200
            assert "授权成功" in callback.text
            assert "3 秒后自动关闭" in callback.text
            assert "`${seconds} 秒后自动关闭`" in callback.text
            assert "window.close()" in callback.text
            assert "script-src 'nonce-" in callback.headers["content-security-policy"]

            repeated_callback = client.get(
                "/api/auth/oauth2/callback",
                params={"state": authorization_query["state"][0], "code": "oauth-code"},
            )
            assert repeated_callback.status_code == 200

            approved = client.post("/api/auth/oauth2/poll", json=poll_body)
            assert approved.status_code == 200
            assert approved.json()["status"] == "approved"
            assert approved.json()["user"]["session_id"] == "oauth-user-42"
            assert "user_prefix" not in approved.json()
            assert client.get("/api/auth/me").json()["session_id"] == "oauth-user-42"

        injected = client.get("/api/auth/inject-prompt", params={"state": entry_state})
        assert injected.status_code == 200
        assert injected.json()["user_prefix"] == f"unused-{entry_state}"
        assert injected.headers["cache-control"] == "no-store"

        assert len(oauth_requests) == 2

        assert client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "XX", "state": "x" * 129},
        ).status_code == 422
        assert client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "XX", "state": "state+token-1234567890"},
        ).status_code == 422

    token_request = oauth_requests[0]
    assert token_request[0:2] == ("POST", "https://login.example.test/token")
    assert token_request[2]["data"]["code"] == "oauth-code"
    assert token_request[2]["data"]["client_id"] == "oauth-client"
    assert token_request[2]["data"]["client_secret"] == "oauth-secret"
    assert token_request[2]["data"]["code_verifier"]
    assert token_request[2]["data"]["redirect_uri"] == expected_redirect_uri
    assert oauth_requests[1] == (
        "GET",
        "https://login.example.test/userinfo",
        {
            "headers": {
                "Accept": "application/json",
                "Authorization": "Bearer provider-access-token",
            }
        },
    )


def test_oauth2_login_uses_the_selected_provider(tmp_path: Path, monkeypatch):
    oauth_requests: list[tuple[str, str, dict[str, object]]] = []

    class FakeOAuthClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url: str, **kwargs):
            oauth_requests.append(("POST", url, kwargs))
            return FakeOAuthResponse({"access_token": "second-access-token"})

        async def get(self, url: str, **kwargs):
            oauth_requests.append(("GET", url, kwargs))
            return FakeOAuthResponse({"sub": "multi-provider-user", "name": "User"})

    def provider(name: str, client_id: str, base_url: str) -> OAuth2WebConfig:
        return OAuth2WebConfig(
            provider_name=name,
            display_type="qrcode",
            client_id=client_id,
            client_secret=f"{client_id}-secret",
            authorization_url=f"{base_url}/authorize",
            token_url=f"{base_url}/token",
            userinfo_url=f"{base_url}/userinfo",
            redirect_uri=None,
            scopes=("openid",),
            session_id_claim=".sub",
            username_claim=".name",
            use_pkce=True,
            token_auth_method="client_secret_post",
            extra_authorization_params={},
            auto_create_users=True,
        )

    monkeypatch.setattr(web_client_app, "PBKDF2_ITERATIONS", 1)
    monkeypatch.setattr(oauth_routes.httpx, "AsyncClient", FakeOAuthClient)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "multi-provider-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            oauth2=(
                provider("First", "first-client", "https://first.example.test"),
                provider("Second", "second-client", "https://second.example.test"),
            ),
        )
    )

    with TestClient(app) as client:
        assert client.get("/api/config").json()["oauth2"] == [
            {"enabled": True, "provider_name": "First", "display_type": "qrcode"},
            {"enabled": True, "provider_name": "Second", "display_type": "qrcode"},
        ]
        assert client.post("/api/auth/oauth2/qr", json={}).status_code == 422
        assert client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "missing"},
        ).status_code == 404

        create_response = client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": " second "},
        )
        assert create_response.status_code == 200
        oauth2_login = create_response.json()
        authorization_url = urlsplit(oauth2_login["authorization_url"])
        assert authorization_url.netloc == "second.example.test"
        assert parse_qs(authorization_url.query)["client_id"] == ["second-client"]

        callback = client.get(
            "/api/auth/oauth2/callback",
            params={
                "state": parse_qs(authorization_url.query)["state"][0],
                "code": "oauth-code",
            },
        )
        assert callback.status_code == 200

    assert oauth_requests[0][0:2] == (
        "POST",
        "https://second.example.test/token",
    )
    assert oauth_requests[0][2]["data"]["client_id"] == "second-client"
    assert oauth_requests[0][2]["data"]["client_secret"] == "second-client-secret"
    assert oauth_requests[1][0:2] == (
        "GET",
        "https://second.example.test/userinfo",
    )


def test_oauth2_endpoints_are_hidden_when_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(web_client_app, "PBKDF2_ITERATIONS", 1)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "oauth-disabled-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
        )
    )

    with TestClient(app) as client:
        assert client.get("/api/config").json()["oauth2"] == []
        assert client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "XX"},
        ).status_code == 404
        assert client.post(
            "/api/auth/login",
            json={"session_id": "admin", "password": "admin123456"},
        ).status_code == 200
        assert client.get(
            "/api/auth/inject-prompt",
            params={"state": "state-token-1234567890"},
        ).status_code == 404
