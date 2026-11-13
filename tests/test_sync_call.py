from pathlib import Path

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.api import app
from xpeech.agent.server.dependencies import CHAT_GUARD, settings
from xpeech.agent.server.schema import InputText
from xpeech.utils.channel_jwt import UserTokenParams, create_token

UUID7_SESSION_ID = "018f0000-0000-7000-8000-000000000001"
BUSY_UUID7_SESSION_ID = "018f0000-0000-7000-8000-000000000002"
GENERATED_SESSION_ID = "018f0000-0000-7000-8000-0000000000ff"


def chat_headers(**extra):
    token = create_token(UserTokenParams("web_client", user_id="user"))
    headers = {
        "Authorization": f"Bearer {token}",
        "x-session-id": "00000000-0000-7000-8000-000000000000",
        "x-sender-name": "Alice",
    }
    headers.update(extra)
    return headers


class FakeRunner:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.messages = []

    async def run_not_interactive(self, message):
        self.messages.append(message)
        return [
            {"event": "assistant", "context": "final answer"},
            {"event": "token_usage", "context": "usage"},
        ]


@pytest.fixture
def fake_runner(monkeypatch, tmp_path):
    runner = FakeRunner(tmp_path)
    created = {}

    async def create(cls, user_id, **kwargs):
        created["user_id"] = user_id
        created["kwargs"] = kwargs
        return runner

    monkeypatch.setattr(AgentRunner, "create", classmethod(create))
    runner.created = created
    return runner


def test_chat_sync_returns_collected_events(fake_runner, monkeypatch):
    monkeypatch.setattr("xpeech.agent.execute.generate_session_id", lambda: GENERATED_SESSION_ID)
    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(**{"x-session-id": UUID7_SESSION_ID}),
        data={
            "content": '[{"text":"hello"}]',
            "session_metadata": '{"channel":"api","trace":"t1"}',
        },
    )

    assert response.status_code == 200
    assert response.json() == [
        {"event": "assistant", "context": "final answer"},
        {"event": "token_usage", "context": "usage"},
    ]
    assert fake_runner.created == {
        "user_id": "user",
        "kwargs": {"not_interactive": True, "channel": "web_client"},
    }
    [message] = fake_runner.messages
    # 会话 ID 由服务端生成，忽略请求头里的 x-session-id。
    assert message.session_id == GENERATED_SESSION_ID
    assert message.sender_name == "Alice"
    assert message.channel == "web_client"
    assert message.session_metadata == {"trace": "t1"}
    assert message.content == [InputText(text="hello")]


def test_backend_requires_sender_name(fake_runner):
    headers = chat_headers()
    headers.pop("x-sender-name")
    response = TestClient(app).post("/chat_sync", headers=headers, data={"content": "[]"})
    assert response.status_code == 422
    assert fake_runner.messages == []
    assert fake_runner.created == {}


def test_channel_comes_only_from_jwt_kid_and_ignores_request_input(fake_runner):
    response = TestClient(app).post(
        "/chat_sync?channel=feishu",
        headers=chat_headers(),
        data={
            "content": '[{"text":"hello"}]',
            "session_metadata": '{"channel":"feishu"}',
        },
    )

    assert response.status_code == 200
    [message] = fake_runner.messages
    assert message.channel == "web_client"
    assert message.session_metadata == {}


def test_channel_jwt_signed_with_another_channels_key_is_rejected(fake_runner):
    forged = jwt.encode(
        {"exp": 9999999999}, settings.jwt.channels["web_client"].secret_key,
        algorithm=settings.jwt.algorithm, headers={"kid": "feishu"},
    )

    response = TestClient(app).post(
        "/chat_sync",
        headers={"Authorization": f"Bearer {forged}", "x-user-id": "user", "x-sender-name": "Alice"},
        data={"content": '[{"text":"hello"}]'},
    )

    assert response.status_code == 401
    assert fake_runner.messages == []


def test_chat_sync_initializes_non_interactive_tools_without_session_header(
    fake_runner,
):
    headers = chat_headers()
    del headers["x-session-id"]

    response = TestClient(app).post(
        "/chat_sync",
        headers=headers,
        data={"content": '[{"text":"hello"}]'},
    )

    assert response.status_code == 200
    assert fake_runner.created["kwargs"] == {"not_interactive": True, "channel": "web_client"}


def test_chat_sync_recognizes_json_output_keyword(monkeypatch, fake_runner):
    monkeypatch.setattr(settings.llm, "support_json_output", True)

    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(),
        data={"content": '[{"text":"hello::请严格以JSON格式回复::"}]'},
    )

    assert response.status_code == 200
    assert fake_runner.messages[0].json_output is True


def test_chat_sync_rejects_unsupported_json_output_before_running(monkeypatch, fake_runner):
    monkeypatch.setattr(settings.llm, "support_json_output", False)

    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(),
        data={"content": '[{"text":"hello::请严格以JSON格式回复::"}]'},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The configured model does not support JSON output"}
    assert fake_runner.messages == []


def test_chat_sync_saves_attachments_to_workspace(fake_runner, tmp_path):
    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(),
        data={"content": '[{"text":"check the report"}]'},
        files=[("files", ("report.txt", b"report content"))],
    )

    assert response.status_code == 200
    [message] = fake_runner.messages
    assert len(message.files) == 1
    saved = Path(message.files[0])
    assert saved.name == "report.txt"
    assert saved.read_bytes() == b"report content"
    assert tmp_path in saved.parents


@pytest.mark.asyncio
async def test_chat_sync_ignores_the_session_chat_guard(fake_runner):
    assert await CHAT_GUARD.try_acquire("user", BUSY_UUID7_SESSION_ID)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/chat_sync",
                headers=chat_headers(),
                data={"content": '[{"text":"hello"}]'},
            )
    finally:
        await CHAT_GUARD.release("user", BUSY_UUID7_SESSION_ID)

    # chat_sync 不占用会话并发锁：同一会话正在 /chat 时也可并发执行。
    assert response.status_code == 200
    assert len(fake_runner.messages) == 1


def test_chat_sync_uses_subject_even_when_request_names_another_user(fake_runner, monkeypatch):
    from xpeech.agent.server.context import get_user_id

    original_run = fake_runner.run_not_interactive

    async def run(message):
        assert get_user_id() == "user"
        return await original_run(message)

    monkeypatch.setattr(fake_runner, "run_not_interactive", run)
    response = TestClient(app).post(
        "/chat_sync?user_id=other-user",
        headers=chat_headers(**{"x-user-id": "other-user"}),
        data={"content": '[{"text":"hello"}]', "user_id": "other-user"},
    )
    assert response.status_code == 200
    assert fake_runner.created["user_id"] == "user"
    assert fake_runner.messages[0].user_id == "user"
    assert response.headers["x-user-id"] == "user"
    assert response.headers["x-channel"] == "web_client"
    assert get_user_id() is None
