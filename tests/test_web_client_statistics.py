from importlib import import_module
from pathlib import Path
from typing import ClassVar

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.utils.internal_jwt import decode_internal_access_token

pytestmark = pytest.mark.usefixtures("postgres_database")

web_client_app = import_module("xpeech.subapp.channel.web_client.app")
proxy_routes = import_module("xpeech.subapp.channel.web_client.routes.proxy")
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
        if url.endswith("/yaml"):
            return httpx.Response(
                200, content=b"- role: user\n  content: hello\n",
                headers={
                    "Content-Type": "application/yaml",
                    "Content-Disposition": 'attachment; filename="00000000-0000-4000-8000-000000000001.yaml"',
                    "Cache-Control": "no-store",
                },
                request=httpx.Request("GET", url),
            )
        if url.endswith("/records/export"):
            return httpx.Response(
                200,
                content="\ufeffID,渠道\r\n1,feishu\r\n".encode(),
                headers={
                    "Content-Type": "text/csv; charset=utf-8",
                    "Content-Disposition": 'attachment; filename="conversation-records.csv"',
                    "Cache-Control": "no-store",
                },
                request=httpx.Request("GET", url),
            )
        return httpx.Response(
            200,
            json={"data": [{"id": 20, "user_question": "问题", "model_response": "回答"}]},
            headers={"Content-Type": "application/json", "Cache-Control": "no-store"},
            request=httpx.Request("GET", url),
        )


def test_web_client_proxies_authenticated_statistics_requests(tmp_path: Path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr(proxy_routes.httpx, "AsyncClient", FakeAsyncClient)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_statistics",
        )
    )

    with TestClient(app) as client:
        unauthorized = client.get("/api/statistics")
        login = client.post(
            "/api/auth/login",
            json={"user_id": "admin", "password": "admin123456"},
        )
        protected_admin = client.patch(
            f"/api/admin/users/{login.json()['id']}",
            json={"username": "renamed-admin", "user_id": "renamed-admin"},
        )
        created_user = client.post(
            "/api/admin/users",
            json={
                "username": "viewer",
                "user_id": "viewer-session",
                "password": "viewer123456",
                "is_admin": False,
            },
        )
        updated_user = client.patch(
            f"/api/admin/users/{created_user.json()['id']}",
            json={"username": "viewer-updated", "user_id": "viewer-session-updated"},
        )
        proxied = client.get(
            "/api/statistics/records/latest",
            params=[("limit", "20"), ("tag", "first"), ("tag", "second")],
        )
        passthrough = client.get("/api/statistics/future/report", params={"range": "week"})
        exported = client.get(
            "/api/statistics/records/export",
            params=[("channel", "feishu"), ("user_id", "alice"), ("user_id", "bob")],
        )
        yaml_file = client.get("/api/statistics/records/20/yaml")
        client.post("/api/auth/logout")
        viewer_login = client.post(
            "/api/auth/login",
            json={"user_id": "viewer-session-updated", "password": "viewer123456"},
        )
        forbidden = client.get("/api/statistics")
        forbidden_yaml = client.get("/api/statistics/records/20/yaml")

    assert unauthorized.status_code == 401
    assert login.status_code == 200
    assert "xpeech_session_statistics=" in login.headers["set-cookie"]
    assert "xpeech_session=" not in login.headers["set-cookie"]
    assert protected_admin.status_code == 400
    assert created_user.status_code == 201
    assert created_user.json()["user_id"] == "viewer-session"
    assert updated_user.status_code == 200
    assert updated_user.json()["username"] == "viewer-updated"
    assert updated_user.json()["user_id"] == "viewer-session-updated"
    assert viewer_login.status_code == 200
    assert forbidden.status_code == 403
    assert forbidden_yaml.status_code == 403
    assert proxied.status_code == 200
    assert proxied.json()["data"][0]["user_question"] == "问题"
    assert proxied.headers["cache-control"] == "no-store"
    assert passthrough.status_code == 200
    assert exported.status_code == 200
    assert exported.content.decode("utf-8-sig") == "ID,渠道\r\n1,feishu\r\n"
    assert exported.headers["content-type"] == "text/csv; charset=utf-8"
    assert exported.headers["content-disposition"] == 'attachment; filename="conversation-records.csv"'
    assert exported.headers["cache-control"] == "no-store"
    assert yaml_file.status_code == 200
    assert yaml_file.content == b"- role: user\n  content: hello\n"
    assert yaml_file.headers["content-type"] == "application/yaml"
    assert yaml_file.headers["content-disposition"].endswith('00000000-0000-4000-8000-000000000001.yaml"')
    assert yaml_file.headers["cache-control"] == "no-store"
    assert len(FakeAsyncClient.requests) == 4
    first_request, second_request, export_request, yaml_request = FakeAsyncClient.requests
    assert yaml_request["url"] == "http://backend.test/statistics/records/20/yaml"
    assert first_request["url"] == "http://backend.test/statistics/records/latest"
    assert first_request["params"] == [("limit", "20"), ("tag", "first"), ("tag", "second")]
    assert first_request["headers"]["authorization"].startswith("Bearer ")
    assert second_request["url"] == "http://backend.test/statistics/future/report"
    assert second_request["params"] == [("range", "week")]
    assert export_request["params"] == [("channel", "feishu"), ("user_id", "alice"), ("user_id", "bob")]
    for request in FakeAsyncClient.requests:
        assert "x-user-id" not in request["headers"]
        token = request["headers"]["authorization"].removeprefix("Bearer ")
        claims = decode_internal_access_token(token)
        assert set(claims) == {"exp"}
        assert "kid" not in jwt.get_unverified_header(token)
