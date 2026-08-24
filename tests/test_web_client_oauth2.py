from importlib import import_module
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient

web_client_app = import_module("xpeech.channel.web_client.app")
OAuth2WebConfig = web_client_app.OAuth2WebConfig
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app
oauth2_claim = web_client_app._oauth2_claim


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

    assert oauth2_claim(userinfo, "data.employee_no") == "oauth-user-42"
    assert oauth2_claim(userinfo, "data.name") == "OAuth User"
    assert oauth2_claim(userinfo, "data.missing") is None
    assert oauth2_claim(userinfo, "data.employee_no.value") is None


def test_oauth2_qr_login_maps_to_an_existing_web_user(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(web_client_app.httpx, "AsyncClient", FakeOAuthClient)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "oauth-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_oauth",
            oauth2=OAuth2WebConfig(
                provider_name="XX",
                display_type="link",
                client_id="oauth-client",
                client_secret="oauth-secret",
                authorization_url="https://login.example.test/authorize",
                token_url="https://login.example.test/token",
                userinfo_url="https://login.example.test/userinfo",
                redirect_uri="https://assistant.example.test/api/auth/oauth2/callback",
                scopes=("openid", "profile"),
                session_id_claim="data.employee_no",
                username_claim="data.name",
                use_pkce=True,
                token_auth_method="client_secret_post",
                extra_authorization_params={"prompt": "login"},
            ),
        )
    )

    with TestClient(app) as client:
        public_config = client.get("/api/config").json()
        assert public_config["oauth2"] == {
            "enabled": True,
            "provider_name": "XX",
            "display_type": "link",
        }

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

        qr_response = client.post("/api/auth/oauth2/qr")
        assert qr_response.status_code == 200
        qr_login = qr_response.json()
        authorization_query = parse_qs(
            urlsplit(qr_login["authorization_url"]).query
        )
        assert authorization_query["client_id"] == ["oauth-client"]
        assert authorization_query["redirect_uri"] == [
            "https://assistant.example.test/api/auth/oauth2/callback"
        ]
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

        callback = client.get(
            "/api/auth/oauth2/callback",
            params={"state": authorization_query["state"][0], "code": "oauth-code"},
        )
        assert callback.status_code == 200
        assert "授权成功" in callback.text
        repeated_callback = client.get(
            "/api/auth/oauth2/callback",
            params={"state": authorization_query["state"][0], "code": "oauth-code"},
        )
        assert repeated_callback.status_code == 200
        assert len(oauth_requests) == 2

        approved = client.post("/api/auth/oauth2/poll", json=poll_body)
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"
        assert approved.json()["user"]["session_id"] == "oauth-user-42"
        assert client.get("/api/auth/me").json()["session_id"] == "oauth-user-42"

    token_request = oauth_requests[0]
    assert token_request[0:2] == ("POST", "https://login.example.test/token")
    assert token_request[2]["data"]["code"] == "oauth-code"
    assert token_request[2]["data"]["client_id"] == "oauth-client"
    assert token_request[2]["data"]["client_secret"] == "oauth-secret"
    assert token_request[2]["data"]["code_verifier"]
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
        public_oauth2 = client.get("/api/config").json()["oauth2"]
        assert public_oauth2 == {
            "enabled": False,
            "provider_name": "OAuth2",
            "display_type": "qrcode",
        }
        assert client.post("/api/auth/oauth2/qr").status_code == 404
