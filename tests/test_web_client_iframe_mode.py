import re
from importlib import import_module
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient

web_client_app = import_module("xpeech.subapp.channel.web_client.app")
oauth_routes = import_module("xpeech.subapp.channel.web_client.routes.auth")
OAuth2WebConfig = web_client_app.OAuth2WebConfig
InjectPromptWebConfig = web_client_app.InjectPromptWebConfig
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app


class FakeOAuthResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def fake_oauth_client(oauth_requests: list, userinfo_payload: dict[str, object]):
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
            return FakeOAuthResponse(userinfo_payload)

    return FakeOAuthClient


def iframe_oauth2_app(tmp_path: Path, monkeypatch, database_name: str):
    monkeypatch.setattr(web_client_app, "PBKDF2_ITERATIONS", 1)
    return create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / database_name,
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            inject_prompt=InjectPromptWebConfig(
                enabled=True,
                command_template="printf unused-$state",
            ),
            oauth2=(
                OAuth2WebConfig(
                    provider_name="XX",
                    display_type="qrcode",
                    client_id="oauth-client",
                    client_secret="oauth-secret",
                    authorization_url="https://login.example.test/authorize",
                    token_url="https://login.example.test/token",
                    userinfo_url="https://login.example.test/userinfo",
                    redirect_uri="https://assistant.example.test/api/auth/oauth2/callback",
                    scopes=("openid", "profile"),
                    user_id_claim=".sub",
                    username_claim=".name",
                    use_pkce=True,
                    token_auth_method="client_secret_post",
                    extra_authorization_params={"prompt": "login"},
                    auto_create_users=True,
                ),
            ),
        )
    )


def test_iframe_mode_login_returns_token_without_cookie(tmp_path: Path):
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "iframe-login-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_iframe",
        )
    )

    with TestClient(app) as client:
        login = client.post(
            "/api/auth/login",
            json={"user_id": "admin", "password": "admin123456", "iframe_mode": True},
        )
        assert login.status_code == 200
        token = login.json().pop("token")
        assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
        assert "set-cookie" not in {key.lower() for key in login.headers}

        bearer = {"Authorization": f"Bearer {token}"}
        me = client.get("/api/auth/me", headers=bearer)
        assert me.status_code == 200
        assert me.json()["user_id"] == "admin"
        assert client.get("/api/auth/me").status_code == 401
        assert client.get(
            "/api/auth/me", headers={"Authorization": "Bearer wrong-token"}
        ).status_code == 401

        changed = client.patch(
            "/api/auth/password",
            json={"new_password": "new-password"},
            headers=bearer,
        )
        assert changed.status_code == 204
        assert client.get("/api/auth/me", headers=bearer).status_code == 200

        assert client.post("/api/auth/logout", headers=bearer).status_code == 204
        assert client.get("/api/auth/me", headers=bearer).status_code == 401

        # 普通登录保持 cookie 会话，不向 JS 暴露 token
        normal = client.post(
            "/api/auth/login",
            json={"user_id": "admin", "password": "new-password"},
        )
        assert normal.status_code == 200
        assert "token" not in normal.json()
        assert "set-cookie" in {key.lower() for key in normal.headers}
        assert client.get("/api/auth/me").status_code == 200


def test_iframe_mode_oauth2_returns_token_in_fragment(tmp_path: Path, monkeypatch):
    oauth_requests: list = []
    monkeypatch.setattr(
        oauth_routes.httpx,
        "AsyncClient",
        fake_oauth_client(oauth_requests, {"sub": "iframe-user", "name": "Iframe User"}),
    )
    app = iframe_oauth2_app(tmp_path, monkeypatch, "iframe-oauth-users.db")

    with TestClient(app) as client:
        entry_state = "state.token_~1234-567890"
        created = client.post(
            "/api/auth/oauth2/qr",
            json={
                "provider_name": "XX",
                "state": entry_state,
                "iframe_mode": True,
            },
        )
        assert created.status_code == 200
        authorization_query = parse_qs(
            urlsplit(created.json()["authorization_url"]).query
        )
        # iframe_mode 静默授权：覆盖服务商配置的交互式 prompt
        assert authorization_query["prompt"] == ["none"]

        # iframe_mode 统一走跳转流程，即使服务商配置为 qrcode 也不依赖扫码轮询
        callback = client.get(
            "/api/auth/oauth2/callback",
            params={"state": authorization_query["state"][0], "code": "oauth-code"},
            follow_redirects=False,
        )
        assert callback.status_code == 303
        assert "set-cookie" not in {key.lower() for key in callback.headers}
        location = urlsplit(callback.headers["location"])
        assert location.path == "/"
        query = parse_qs(location.query)
        assert query["iframe_mode"] == ["1"]
        assert query["oauth2provider"] == ["XX"]
        assert query["state"] == [entry_state]
        fragment_token = location.fragment.removeprefix("token=")
        assert re.fullmatch(r"[A-Za-z0-9_-]{43}", fragment_token)

        me = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {fragment_token}"}
        )
        assert me.status_code == 200
        assert me.json()["user_id"] == "iframe-user"

        # 授权失败：错误带回 SPA 退回登录页，而不是结果页
        failed = client.post(
            "/api/auth/oauth2/qr",
            json={"provider_name": "XX", "iframe_mode": True},
        )
        failed_state = parse_qs(urlsplit(failed.json()["authorization_url"]).query)[
            "state"
        ][0]
        failed_callback = client.get(
            "/api/auth/oauth2/callback",
            params={"state": failed_state, "error": "login_required"},
            follow_redirects=False,
        )
        assert failed_callback.status_code == 303
        failed_location = urlsplit(failed_callback.headers["location"])
        failed_query = parse_qs(failed_location.query)
        assert failed_query["iframe_mode"] == ["1"]
        assert failed_query["oauth2provider"] == ["XX"]
        assert failed_query["auth_error"] == ["login_required"]
        assert failed_location.fragment == ""
