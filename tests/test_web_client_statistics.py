from importlib import import_module
from pathlib import Path
from typing import ClassVar

import httpx
from fastapi.testclient import TestClient

web_client_app = import_module("xpeech.channel.web_client.app")
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app


class FakeAsyncClient:
    requests: ClassVar[list[dict]] = []

    def __init__(self, *, timeout: int):
        assert timeout == 30

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return None

    async def get(self, url: str, *, headers: dict[str, str], params: list[tuple[str, str]]):
        self.requests.append({"url": url, "headers": headers, "params": params})
        return httpx.Response(
            200,
            json={"data": [{"id": 20, "user_question": "问题", "model_response": "回答"}]},
            headers={"Content-Type": "application/json", "Cache-Control": "no-store"},
            request=httpx.Request("GET", url),
        )


def test_web_client_proxies_authenticated_statistics_requests(tmp_path: Path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr(web_client_app.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_statistics",
        )
    )

    with TestClient(app) as client:
        unauthorized = client.get("/api/statistics")
        login = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123456"},
        )
        created_user = client.post(
            "/api/admin/users",
            json={"username": "viewer", "password": "viewer123456", "is_admin": False},
        )
        proxied = client.get(
            "/api/statistics/records/latest",
            params=[("limit", "20"), ("tag", "first"), ("tag", "second")],
        )
        passthrough = client.get("/api/statistics/future/report", params={"range": "week"})
        client.post("/api/auth/logout")
        viewer_login = client.post(
            "/api/auth/login",
            json={"username": "viewer", "password": "viewer123456"},
        )
        forbidden = client.get("/api/statistics")

    assert unauthorized.status_code == 401
    assert login.status_code == 200
    assert "xpeech_session_statistics=" in login.headers["set-cookie"]
    assert "xpeech_session=" not in login.headers["set-cookie"]
    assert created_user.status_code == 201
    assert viewer_login.status_code == 200
    assert forbidden.status_code == 403
    assert proxied.status_code == 200
    assert proxied.json()["data"][0]["user_question"] == "问题"
    assert proxied.headers["cache-control"] == "no-store"
    assert passthrough.status_code == 200
    assert len(FakeAsyncClient.requests) == 2
    first_request, second_request = FakeAsyncClient.requests
    assert first_request["url"] == "http://backend.test/statistics/records/latest"
    assert first_request["params"] == [("limit", "20"), ("tag", "first"), ("tag", "second")]
    assert first_request["headers"]["authorization"].startswith("Bearer ")
    assert first_request["headers"]["x-session-id"] == "web_admin"
    assert second_request["url"] == "http://backend.test/statistics/future/report"
    assert second_request["params"] == [("range", "week")]
