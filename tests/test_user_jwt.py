"""External user identity and trusted internal request parameters."""

import asyncio
from unittest.mock import Mock

import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.loop import AgentLoop, QuestionEvent
from xpeech.agent.server.api import app
from xpeech.config.settings import settings
from xpeech.exceptions import PathProtectionError
from xpeech.utils import channel_jwt
from xpeech.utils.channel_jwt import ChannelJWTAuth, ChannelTokenParams, UserJWTAuth, UserTokenParams, create_token
from xpeech.utils.internal_jwt import create_internal_access_token


def user_headers(user_id="alice"):
    token = create_token(UserTokenParams("web_client", user_id=user_id))
    return {"Authorization": f"Bearer {token}", "x-session-id": "default"}


def test_user_identity_is_validated_once_and_read_only(monkeypatch):
    token = create_token(UserTokenParams("web_client", user_id="alice"))
    decode = Mock(wraps=jwt.decode)
    monkeypatch.setattr(channel_jwt.jwt, "decode", decode)
    auth = UserJWTAuth(token)
    assert isinstance(auth, ChannelJWTAuth)
    assert auth.user_id == "alice"
    assert auth.claims["sub"] == "alice"
    with pytest.raises(AttributeError):
        auth.user_id = "bob"
    assert auth.user_id == "alice"
    assert decode.call_count == 1


@pytest.mark.parametrize("subject", [None, "", 123, False, [], {}, "../bob", "a/b", "x" * 129])
def test_user_auth_rejects_missing_or_invalid_subject(subject):
    claims = {"exp": 9999999999}
    if subject is not None:
        claims["sub"] = subject
    token = jwt.encode(
        claims, settings.jwt.channels["web_client"].secret_key,
        algorithm=settings.jwt.algorithm, headers={"kid": "web_client"},
    )
    with pytest.raises(jwt.InvalidTokenError):
        UserJWTAuth(token)
    response = TestClient(app).post(
        "/answer_question", data={"answer": "yes"},
        headers={"Authorization": f"Bearer {token}", "x-user-id": "alice", "x-session-id": "default"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize("user_id", ["", "../bob", "x" * 129])
def test_signer_rejects_invalid_user_id(user_id):
    with pytest.raises(PathProtectionError):
        create_token(UserTokenParams("web_client", user_id=user_id))


@pytest.mark.parametrize("method, path", [
    ("POST", "/chat"), ("POST", "/chat_sync"), ("POST", "/answer_question"),
    ("GET", "/files?path=report.txt"), ("GET", "/skills"), ("POST", "/skills/import"),
])
@pytest.mark.parametrize("user_id", [None, "alice"])
def test_user_endpoints_reject_channel_only_tokens(method, path, user_id):
    token = create_token(ChannelTokenParams("web_client"))
    headers = {"Authorization": f"Bearer {token}"}
    if user_id is not None:
        headers["x-user-id"] = user_id
    response = TestClient(app).request(method, path, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_answer_uses_and_returns_jwt_identity(monkeypatch):
    alice = QuestionEvent(event=asyncio.Event())
    bob = QuestionEvent(event=asyncio.Event())
    monkeypatch.setattr(AgentLoop, "SESSION_QUESTION_EVENT", {("alice", "default"): alice, ("bob", "default"): bob})
    response = TestClient(app).post(
        "/answer_question?user_id=bob", data={"answer": "yes", "user_id": "bob"},
        headers={**user_headers(), "x-user-id": "bob"},
    )
    assert response.status_code == 200
    assert alice.answer == "yes"
    assert alice.event.is_set()
    assert not bob.event.is_set()
    assert response.headers["x-user-id"] == "alice"
    assert response.headers["x-channel"] == "web_client"


def test_file_download_is_bound_to_subject(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path)
    for user_id in ("alice", "bob"):
        (tmp_path / user_id).mkdir()
        (tmp_path / user_id / "report.txt").write_text(user_id)
    client = TestClient(app)
    headers = {**user_headers(), "x-user-id": "bob"}
    response = client.get("/files?user_id=bob", params={"path": "report.txt"}, headers=headers)
    assert response.status_code == 200
    assert response.text == "alice"
    assert client.get("/files", params={"path": str(tmp_path / "bob/report.txt")}, headers=headers).status_code == 403
    assert client.get("/sessions/bob/files", params={"path": "report.txt"}, headers=headers).status_code == 404


def test_external_openapi_has_no_user_identity_parameters():
    schema = app.openapi()
    for path, method in (("/chat", "post"), ("/chat_sync", "post"), ("/answer_question", "post"),
                         ("/files", "get"), ("/skills", "get"), ("/skills/import", "post")):
        names = {parameter["name"] for parameter in schema["paths"][path][method].get("parameters", [])}
        assert names.isdisjoint({"user_id", "x-user-id"})


def test_background_message_ignores_user_header(monkeypatch):
    from unittest.mock import AsyncMock

    from xpeech.agent.server.routes import background_messages

    poll = AsyncMock(return_value=None)
    monkeypatch.setattr(background_messages, "scheduler_request", poll)
    token = create_token(ChannelTokenParams("feishu"))
    response = TestClient(app).get(
        "/background_message", headers={"Authorization": f"Bearer {token}", "x-user-id": "ignored"},
    )
    assert response.status_code == 204
    poll.assert_awaited_once_with("GET", "/background_message?channel=feishu")


@pytest.mark.parametrize("user_id", ["alice", "", "trusted/internal-user"])
def test_internal_user_parameters_are_used_directly(user_id, monkeypatch):
    from xpeech.subapp.scheduler import app as scheduler_service

    observed = []

    def list_tasks(user_id):
        observed.append(user_id)
        return []

    monkeypatch.setattr(scheduler_service, "list_scheduled_tasks", list_tasks)
    token = create_internal_access_token()
    response = TestClient(scheduler_service.app).get(
        "/schedules", headers={"Authorization": f"Bearer {token}", "x-user-id": user_id},
    )
    assert response.status_code == 200
    assert observed == [user_id]
