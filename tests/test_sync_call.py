from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.api import app
from xpeech.agent.server.dependencies import CHAT_GUARD, settings
from xpeech.agent.server.schema import InputText
from xpeech.utils.channel_jwt import ChannelJWTAuth, create_channel_access_token


def chat_headers(**extra):
    token = create_channel_access_token(
        ChannelJWTAuth("web_client", settings.jwt.web_client_secret_key)
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "x-user-id": "user",
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


def test_chat_sync_returns_collected_events(fake_runner):
    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(**{"x-session-id": "chat-1"}),
        data={
            "content": '[{"text":"hello"}]',
            "session_metadata": '{"channel":"api"}',
        },
    )

    assert response.status_code == 200
    assert response.json() == [
        {"event": "assistant", "context": "final answer"},
        {"event": "token_usage", "context": "usage"},
    ]
    assert fake_runner.created == {
        "user_id": "user",
        "kwargs": {"not_interactive": True, "channel_tags": []},
    }
    [message] = fake_runner.messages
    assert message.session_id == "chat-1"
    assert message.sender_name == "Alice"
    assert message.session_metadata == {"channel": "api"}
    assert message.content == [InputText(text="hello")]


@pytest.mark.parametrize(
    "session_id",
    [None, "default", "chat-1"],
)
def test_chat_sync_initializes_non_interactive_tools_for_every_session(
    fake_runner,
    session_id,
):
    headers = chat_headers()
    if session_id is not None:
        headers["x-session-id"] = session_id

    response = TestClient(app).post(
        "/chat_sync",
        headers=headers,
        data={"content": '[{"text":"hello"}]'},
    )

    assert response.status_code == 200
    assert fake_runner.created["kwargs"] == {"not_interactive": True, "channel_tags": []}


def test_chat_sync_passes_json_output_flag(monkeypatch, fake_runner):
    monkeypatch.setattr(settings.llm, "support_json_output", True)

    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(),
        data={"content": '[{"text":"hello"}]', "json_output": "true"},
    )

    assert response.status_code == 200
    assert fake_runner.messages[0].json_output is True


def test_chat_sync_rejects_unsupported_json_output_before_running(monkeypatch, fake_runner):
    monkeypatch.setattr(settings.llm, "support_json_output", False)

    response = TestClient(app).post(
        "/chat_sync",
        headers=chat_headers(),
        data={"content": '[{"text":"hello"}]', "json_output": "true"},
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
async def test_chat_sync_conflicts_with_active_session(fake_runner):
    assert await CHAT_GUARD.try_acquire("user", "busy-session")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post(
                "/chat_sync",
                headers=chat_headers(**{"x-session-id": "busy-session"}),
                data={"content": '[{"text":"hello"}]'},
            )
    finally:
        await CHAT_GUARD.release("user", "busy-session")

    assert response.status_code == 409
    assert fake_runner.messages == []
