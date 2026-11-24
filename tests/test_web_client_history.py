import base64
import json
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from uuid6 import uuid7

from xpeech.config.settings import JWTChannelConfig, WebClientConfig, settings
from xpeech.subapp.channel.web_client.app import WebConfig, create_app
from xpeech.subapp.channel.web_client.history import JsonlEventHistoryRepository
from xpeech.subapp.channel.web_client.routes import proxy

SESSION_ID = "018f0000-0000-7000-8000-000000000001"
NEXT_SESSION_ID = "018f0000-0000-7000-8000-000000000002"
SECRET = "test-channel-key-at-least-64-characters-for-sha512-signing-tests"


@pytest.mark.asyncio
async def test_event_repository_appends_filters_channels_and_cleans_expired_files(tmp_path: Path):
    repository = JsonlEventHistoryRepository(tmp_path, max_history_days=1)
    events = [
        {"event": "user", "context": "hello"},
        {"event": "assistant", "context": "world"},
        {"event": "user", "context": "continue"},
    ]
    append_event = repository.event_appender("alice", SESSION_ID, channel="web_client")
    for event in events:
        await append_event(event)
    append_embedded = repository.event_appender("alice", SESSION_ID, channel="crm")
    await append_embedded({"event": "user", "context": "embedded"})
    append_next_session = repository.event_appender("alice", NEXT_SESSION_ID, channel="web_client")
    await append_next_session({"event": "user", "context": "new"})

    assert await repository.load("alice", SESSION_ID, channel="web_client") == events
    sessions = await repository.list_channel_sessions("alice", channel="web_client")
    assert [item["session_id"] for item in sessions] == [NEXT_SESSION_ID, SESSION_ID]
    assert sessions[1]["preview"] == "hello"
    assert (tmp_path / "alice" / f"channel=web_client&session_id={SESSION_ID}.jsonl").is_file()
    assert (tmp_path / "alice" / f"channel=crm&session_id={SESSION_ID}.jsonl").is_file()

    expired = time.time() - 2 * 86400
    path = tmp_path / "alice" / f"channel=web_client&session_id={SESSION_ID}.jsonl"
    os.utime(path, (expired, expired))
    remaining = await repository.list_channel_sessions("alice", channel="web_client")
    assert [item["session_id"] for item in remaining] == [NEXT_SESSION_ID]
    assert not path.exists()


def test_web_history_retention_must_be_positive():
    with pytest.raises(ValidationError, match="max_history_days"):
        WebClientConfig(max_history_days=0)


@pytest.fixture
def history_client(tmp_path: Path, monkeypatch, postgres_database):
    monkeypatch.setattr("xpeech.subapp.channel.web_client.app.PBKDF2_ITERATIONS", 1)
    monkeypatch.setattr(settings.jwt, "channels", {
        "web_client": JWTChannelConfig(secret_key=SECRET),
        "crm": JWTChannelConfig(secret_key=SECRET),
    })
    history_path = tmp_path / "event-history"

    def open_stream(_auth, _sender, content, _metadata, _url, *, files, session_id):
        turn_id = str(uuid7())
        display = "\n".join([
            *(block["text"] for block in json.loads(content)),
            *(f"📎 {file[0]}" for _, file in files),
        ])
        events = MagicMock()
        events.__aiter__.return_value = [
            {"event": "user", "context": display, "turn_id": turn_id},
            {"event": "assistant", "context": "回答", "turn_id": turn_id},
            {"event": "assistant_end", "context": "", "turn_id": turn_id},
        ]
        return nullcontext((httpx.Response(200), events))

    opener = MagicMock(side_effect=open_stream)
    monkeypatch.setattr(proxy, "open_chat_stream", opener)
    app = create_app(WebConfig(
        backend_url="http://backend.test",

        session_history_path=history_path,
        static_dir=tmp_path / "missing",
        system_name="Assistant",
    ))
    with TestClient(app) as client:
        yield client, history_path, opener


def login(client):
    assert client.post(
        "/api/auth/login",
        json={"user_id": "admin", "password": "admin123456"},
    ).status_code == 200


def send_local_chat(client, text="你好", **extra):
    payload = websocket_message(SESSION_ID, text, files=[{
        "name": "报告.txt", "type": "text/plain", "data": base64.b64encode(b"attachment").decode(),
    }])
    payload.update(extra)
    with client.websocket_connect("/api/chat/ws") as socket:
        socket.send_json(payload)
        return receive_websocket_turn(socket)


def test_logged_in_user_can_load_and_continue_the_same_session(history_client):
    client, history_path, opener = history_client
    login(client)

    assert send_local_chat(client)[-1]["type"] == "done"
    assert send_local_chat(client, "继续")[-1]["type"] == "done"
    assert [call.kwargs["session_id"] for call in opener.call_args_list] == [SESSION_ID, SESSION_ID]

    sessions = client.get("/api/history/sessions")
    assert sessions.status_code == 200
    assert sessions.headers["cache-control"] == "no-store"
    assert [item["session_id"] for item in sessions.json()] == [SESSION_ID]
    assert sessions.json()[0]["channel"] == "web_client"
    assert sessions.json()[0]["preview"] == "你好\n📎 报告.txt"

    history = client.get(f"/api/history/sessions/{SESSION_ID}")
    assert history.status_code == 200
    assert [event["event"] for event in history.json()["events"]] == [
        "user", "assistant", "assistant_end", "user", "assistant", "assistant_end",
    ]
    assert history.json()["events"][0]["context"] == "你好\n📎 报告.txt"
    saved = [
        json.loads(line)
        for line in (history_path / "admin" / f"channel=web_client&session_id={SESSION_ID}.jsonl")
        .read_text()
        .splitlines()
    ]
    assert saved == history.json()["events"]


def test_embedded_and_login_history_use_their_verified_channels(history_client):
    client, history_path, _ = history_client
    login(client)
    assert send_local_chat(client)[-1]["type"] == "done"
    token = jwt.encode(
        {"sub": "admin", "exp": datetime.now(UTC) + timedelta(minutes=1)},
        SECRET,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "crm"},
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "x-session-id": SESSION_ID,
        "x-sender-name": "embedded-user",
    }

    embedded_sessions = client.get("/api/history/sessions", headers=headers)
    assert embedded_sessions.status_code == 200
    assert embedded_sessions.json() == []
    assert client.get(f"/api/history/sessions/{SESSION_ID}", headers=headers).json()["events"] == []

    assert send_local_chat(client, "嵌入消息", token=token, sender_name="embedded-user", files=[])[-1]["type"] == "done"
    assert client.get("/api/history/sessions", headers=headers).json()[0]["preview"] == "嵌入消息"
    user_event = client.get(f"/api/history/sessions/{SESSION_ID}", headers=headers).json()["events"][0]
    assert user_event["event"] == "user"
    assert user_event["context"] == "嵌入消息"
    assert (history_path / "admin" / f"channel=crm&session_id={SESSION_ID}.jsonl").is_file()
    assert len(client.get("/api/history/sessions").json()) == 1


def test_history_requires_authentication_and_a_canonical_session_id(history_client):
    client, _, _ = history_client
    assert client.get("/api/history/sessions").status_code == 401
    login(client)
    assert client.get("/api/history/sessions/not-a-uuid").status_code == 422
    assert client.get(f"/api/history/sessions/{NEXT_SESSION_ID}").json() == {
        "session_id": NEXT_SESSION_ID,
        "channel": "web_client",
        "events": [],
    }


def websocket_message(session_id: str, text: str, **extra):
    return {
        "type": "start",
        "session_id": session_id,
        "content": json.dumps([{"text": text}], ensure_ascii=False),
        "token": None,
        "sender_name": None,
        "session_metadata": {"source": "websocket-test"},
        "files": [],
        **extra,
    }


def receive_websocket_turn(websocket):
    messages = []
    while True:
        message = websocket.receive_json()
        messages.append(message)
        if message["type"] in {"done", "error"}:
            return messages


def test_each_reply_uses_a_session_websocket_and_saves_before_done(history_client):
    client, history_path, opener = history_client
    login(client)

    with client.websocket_connect("/api/chat/ws") as websocket:
        websocket.send_json(websocket_message(
            SESSION_ID, "会话 A",
            files=[{
                "name": "报告.txt",
                "type": "text/plain",
                "data": base64.b64encode(b"attachment").decode(),
            }],
        ))
        first = receive_websocket_turn(websocket)
        assert [message["type"] for message in first] == ["event", "event", "event", "done"]
        assert all(message["session_id"] == SESSION_ID for message in first)
        events = [message["event"] for message in first[:-1]]
        turn_id = events[0]["turn_id"]
        assert events[0] == {
            "event": "user", "context": "会话 A\n📎 报告.txt", "turn_id": turn_id,
        }
        assert events[1] == {"event": "assistant", "context": "回答", "turn_id": turn_id}
        assert client.get(f"/api/history/sessions/{SESSION_ID}").json()["events"] == events

    with client.websocket_connect("/api/chat/ws") as websocket:
        websocket.send_json(websocket_message(NEXT_SESSION_ID, "会话 B"))
        second = receive_websocket_turn(websocket)
        assert [message["type"] for message in second] == ["event", "event", "event", "done"]
        assert client.get(f"/api/history/sessions/{NEXT_SESSION_ID}").json()["events"][0]["context"] == "会话 B"

    with client.websocket_connect("/api/chat/ws") as websocket:
        websocket.send_json(websocket_message(SESSION_ID, "继续 A"))
        continued = receive_websocket_turn(websocket)
        assert continued[-1]["type"] == "done"

    assert [call.kwargs["session_id"] for call in opener.call_args_list] == [
        SESSION_ID, NEXT_SESSION_ID, SESSION_ID,
    ]
    assert json.loads(opener.call_args_list[0].args[3]) == {"source": "websocket-test"}
    assert opener.call_args_list[0].kwargs["files"][0][1][0] == "报告.txt"
    assert (history_path / "admin" / f"channel=web_client&session_id={SESSION_ID}.jsonl").is_file()


def test_websocket_embedded_auth_uses_json_token_and_rejects_invalid_credentials(history_client):
    client, history_path, _ = history_client
    token = jwt.encode(
        {"sub": "alice", "exp": datetime.now(UTC) + timedelta(minutes=1)},
        SECRET,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "crm"},
    )

    with client.websocket_connect("/api/chat/ws") as websocket:
        websocket.send_json(websocket_message(
            SESSION_ID, "嵌入消息", token=token, sender_name="embedded-user",
        ))
        messages = receive_websocket_turn(websocket)
        assert messages[-1]["type"] == "done"

    with client.websocket_connect("/api/chat/ws") as websocket:
        websocket.send_json(websocket_message(NEXT_SESSION_ID, "不能发送", token="invalid"))
        error = receive_websocket_turn(websocket)[-1]
        assert error["type"] == "error"
        assert error["status"] == 401

    assert client.get(f"/api/history/sessions/{SESSION_ID}").status_code == 401
    assert (history_path / "alice" / f"channel=crm&session_id={SESSION_ID}.jsonl").is_file()
    assert not (history_path / "alice" / f"channel=crm&session_id={NEXT_SESSION_ID}.jsonl").exists()
