"""Browser integrations can access the backend from any origin using JWTs."""

import pytest
from fastapi.testclient import TestClient

from xpeech.agent.server.api import app


@pytest.mark.parametrize("origin", ["https://portal.example.test", "http://localhost:3000"])
@pytest.mark.parametrize("method,path", [("POST", "/chat"), ("GET", "/files"), ("DELETE", "/skills/example")])
def test_browser_preflight_allows_any_origin_method_and_headers(origin, method, path):
    response = TestClient(app).options(path, headers={
        "Origin": origin,
        "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "authorization,content-type,x-session-id,x-custom-header",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert method in response.headers["access-control-allow-methods"]
    assert "x-custom-header" in response.headers["access-control-allow-headers"]
    assert "access-control-allow-credentials" not in response.headers


def test_cors_exposes_context_headers_and_preserves_authentication():
    client = TestClient(app)
    headers = {"Origin": "https://unconfigured.test", "x-session-id": "default"}
    health = client.get("/health", headers=headers)
    assert health.status_code == 200
    assert health.headers["access-control-allow-origin"] == "*"
    assert health.headers["access-control-expose-headers"] == "*"
    assert health.headers["x-request-id"]
    assert health.headers["x-session-id"] == "default"

    rejected = client.post("/chat", headers=headers, data={"content": "[]"})
    assert rejected.status_code == 401
    assert rejected.headers["access-control-allow-origin"] == "*"
    assert rejected.headers["access-control-expose-headers"] == "*"
