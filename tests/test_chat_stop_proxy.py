"""Stop forwarding preserves the existing cookie/JWT user and session contract."""

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xpeech.subapp.channel.web_client import backend
from xpeech.subapp.channel.web_client.dao import User
from xpeech.subapp.channel.web_client.history import JsonlEventHistoryRepository
from xpeech.subapp.channel.web_client.routes.proxy import create_proxy_router
from xpeech.utils.channel_jwt import UserJWTAuth, UserTokenParams, create_token

SESSION_ID = "018f0000-0000-7000-8000-000000000001"


@pytest.mark.parametrize("identity", ["cookie", "jwt"])
@pytest.mark.parametrize("upstream_status", [200, 401])
@pytest.mark.parametrize("action", ["stop", "status"])
def test_chat_control_proxy_forwards_verified_identity_session_and_response(
    tmp_path, monkeypatch, identity, upstream_status, action,
):
    token = create_token(UserTokenParams("web_client", user_id="alice"))
    user = (
        UserJWTAuth(token) if identity == "jwt"
        else User(user_id="alice", username="Alice", password_hash="unused")
    )

    def current_user():
        return user

    app = FastAPI()
    app.include_router(create_proxy_router(
        "http://backend.test", current_user, current_user,
        JsonlEventHistoryRepository(tmp_path), "test_cookie",
    ))
    requests = []
    method = "POST" if action == "stop" else "GET"
    payload = {"stopped" if action == "stop" else "active": True} if upstream_status == 200 else {"detail": "Expired token"}

    def respond(request):
        requests.append(request)
        assert request.url.path == f"/chat/{action}"
        assert request.method == method
        assert request.headers["x-session-id"] == SESSION_ID
        assert "x-user-id" not in request.headers
        assert "x-sender-name" not in request.headers
        forwarded_token = request.headers["authorization"].removeprefix("Bearer ")
        assert UserJWTAuth(forwarded_token).user_id == "alice"
        if identity == "jwt":
            assert forwarded_token == token
        return httpx.Response(upstream_status, json=payload)

    client_class = httpx.AsyncClient
    monkeypatch.setattr(backend.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    client = TestClient(app)
    response = client.request(method, f"/api/chat/{action}", headers={"x-session-id": SESSION_ID, "x-user-id": "bob"})
    assert response.status_code == upstream_status
    assert response.json() == payload
    if action == "status":
        assert response.headers["cache-control"] == "no-store"
        for invalid_id in (None, "", "not-a-uuid"):
            invalid_headers = {} if invalid_id is None else {"x-session-id": invalid_id}
            assert client.get("/api/chat/status", headers=invalid_headers).status_code == 422
    assert len(requests) == 1
    for path in ("/api/chat", "/api/chat_sync", "/api/chat/stop", "/api/answer_question"):
        parameters = app.openapi()["paths"][path]["post"]["parameters"]
        session_header = next(parameter for parameter in parameters if parameter["name"] == "x-session-id")
        assert session_header["required"] is True
        for invalid_id in (None, "", "default", "not-a-uuid", SESSION_ID.upper()):
            headers = {} if invalid_id is None else {"x-session-id": invalid_id}
            invalid = client.post(path, headers=headers, data={"content": "[]", "answer": "yes"})
            assert invalid.status_code == 422
        assert len(requests) == 1
