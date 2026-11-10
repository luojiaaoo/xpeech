import base64
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from xpeech.config.settings import JWTChannelConfig, WebClientConfig, settings
from xpeech.subapp.channel.helper import ChatStream
from xpeech.subapp.channel.web_client.app import WebConfig, create_app
from xpeech.subapp.channel.web_client.history import YamlEventHistoryRepository
from xpeech.subapp.channel.web_client.routes import proxy

SESSION_ID = "018f0000-0000-7000-8000-000000000001"
NEXT_SESSION_ID = "018f0000-0000-7000-8000-000000000002"
SECRET = "test-channel-key-at-least-64-characters-for-sha512-signing-tests"


def test_sse_collector_handles_split_multibyte_data_and_multiline_events():
    collector = proxy._SSEEventCollector()
    payload = (
        'event: message\ndata: {"event":"assistant",\ndata: "context":"你好"}\n\n'
        'data: {"event":"assistant_end","context":""}\n\n'
    ).encode()
    for byte in payload:
        collector.feed(bytes([byte]))
    collector.finish()

    assert collector.events == [
        {"event": "assistant", "context": "你好"},
        {"event": "assistant_end", "context": ""},
    ]


@pytest.mark.asyncio
async def test_event_repository_appends_filters_channels_and_cleans_expired_files(tmp_path: Path):
    repository = YamlEventHistoryRepository(tmp_path, max_history_days=1)
    first = [{"event": "user", "context": "hello"}, {"event": "assistant", "context": "world"}]
    second = [{"event": "user", "context": "continue"}]
    await repository.append("alice", SESSION_ID, first, channel="web_client")
    await repository.append("alice", SESSION_ID, second, channel="web_client")
    await repository.append("alice", SESSION_ID, [{"event": "user", "context": "embedded"}], channel="crm")
    await repository.append("alice", NEXT_SESSION_ID, [{"event": "user", "context": "new"}], channel="web_client")

    assert await repository.load("alice", SESSION_ID, channel="web_client") == [*first, *second]
    sessions = await repository.list_channel_sessions("alice", channel="web_client")
    assert [item["session_id"] for item in sessions] == [NEXT_SESSION_ID, SESSION_ID]
    assert sessions[1]["preview"] == "hello"
    assert (tmp_path / "alice" / f"channel=web_client&session_id={SESSION_ID}.yaml").is_file()
    assert (tmp_path / "alice" / f"channel=crm&session_id={SESSION_ID}.yaml").is_file()

    expired = time.time() - 2 * 86400
    path = tmp_path / "alice" / f"channel=web_client&session_id={SESSION_ID}.yaml"
    os.utime(path, (expired, expired))
    remaining = await repository.list_channel_sessions("alice", channel="web_client")
    assert [item["session_id"] for item in remaining] == [NEXT_SESSION_ID]
    assert not path.exists()


def test_web_history_retention_must_be_positive():
    with pytest.raises(ValidationError, match="max_history_days"):
        WebClientConfig(database_path="users.db", max_history_days=0)


@pytest.fixture
def history_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("xpeech.subapp.channel.web_client.app.PBKDF2_ITERATIONS", 1)
    monkeypatch.setattr(settings.jwt, "channels", {
        "web_client": JWTChannelConfig(secret_key=SECRET),
        "crm": JWTChannelConfig(secret_key=SECRET),
    })
    history_path = tmp_path / "event-history"
    opener = AsyncMock(side_effect=lambda *_args, **_kwargs: ChatStream(
        httpx.Response(
            200,
            content=(
                'data: {"event":"assistant","context":"回答"}\n\n'
                'data: {"event":"assistant_end","context":""}\n\n'
            ).encode(),
            headers={"Content-Type": "text/event-stream"},
        ),
        SimpleNamespace(aclose=AsyncMock()),
    ))
    monkeypatch.setattr(proxy, "open_chat", opener)
    app = create_app(WebConfig(
        backend_url="http://backend.test",
        database_path=tmp_path / "users.db",
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


def send_local_chat(client, text="你好"):
    return client.post(
        "/api/chat",
        headers={"x-session-id": SESSION_ID},
        data={"content": json.dumps([{"text": text}])},
        files={"files": ("报告.txt", b"attachment", "text/plain")},
    )


def test_logged_in_user_can_load_and_continue_the_same_session(history_client):
    client, history_path, opener = history_client
    login(client)

    assert send_local_chat(client).status_code == 200
    assert send_local_chat(client, "继续").status_code == 200
    assert [call.kwargs["session_id"] for call in opener.await_args_list] == [SESSION_ID, SESSION_ID]

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
    saved = yaml.safe_load(
        (history_path / "admin" / f"channel=web_client&session_id={SESSION_ID}.yaml").read_text()
    )
    assert saved == history.json()["events"]


def test_embedded_and_login_history_use_their_verified_channels(history_client):
    client, history_path, _ = history_client
    login(client)
    assert send_local_chat(client).status_code == 200
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

    assert client.post(
        "/api/chat",
        headers=headers,
        data={"content": json.dumps([{"text": "嵌入消息"}])},
    ).status_code == 200
    assert client.get("/api/history/sessions", headers=headers).json()[0]["preview"] == "嵌入消息"
    assert client.get(f"/api/history/sessions/{SESSION_ID}", headers=headers).json()["events"][0] == {
        "event": "user", "context": "嵌入消息",
    }
    assert (history_path / "admin" / f"channel=crm&session_id={SESSION_ID}.yaml").is_file()
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


def websocket_message(request_id: str, session_id: str, text: str, **extra):
    return {
        "type": "start",
        "request_id": request_id,
        "session_id": session_id,
        "content": json.dumps([{"text": text}], ensure_ascii=False),
        "token": None,
        "sender_name": None,
        "session_metadata": {"source": "websocket-test"},
        "files": [],
        **extra,
    }


def receive_websocket_turn(websocket, request_id: str):
    messages = []
    while True:
        message = websocket.receive_json()
        messages.append(message)
        if message["request_id"] == request_id and message["type"] in {"done", "error"}:
            return messages


def test_one_websocket_multiplexes_sessions_and_saves_before_done(history_client):
    client, history_path, opener = history_client
    login(client)

    with client.websocket_connect("/api/chat/ws") as websocket:
        websocket.send_json(websocket_message(
            "request-a", SESSION_ID, "会话 A",
            files=[{
                "name": "报告.txt",
                "type": "text/plain",
                "data": base64.b64encode(b"attachment").decode(),
            }],
        ))
        websocket.send_json(websocket_message("request-b", NEXT_SESSION_ID, "会话 B"))
        received = []
        completed = set()
        while completed != {"request-a", "request-b"}:
            message = websocket.receive_json()
            received.append(message)
            if message["type"] == "done":
                completed.add(message["request_id"])
        first = [message for message in received if message["request_id"] == "request-a"]
        second = [message for message in received if message["request_id"] == "request-b"]

        assert [message["type"] for message in first] == ["event", "event", "done"]
        assert all(message["session_id"] == SESSION_ID for message in first)
        assert first[0]["event"] == {"event": "assistant", "context": "回答"}
        assert [message["type"] for message in second] == ["event", "event", "done"]
        assert client.get(f"/api/history/sessions/{SESSION_ID}").json()["events"][0] == {
            "event": "user", "context": "会话 A\n📎 报告.txt",
        }
        assert client.get(f"/api/history/sessions/{NEXT_SESSION_ID}").json()["events"][0]["context"] == "会话 B"

        websocket.send_json(websocket_message("request-a-2", SESSION_ID, "继续 A"))
        continued = receive_websocket_turn(websocket, "request-a-2")
        assert continued[-1]["type"] == "done"

    assert [call.kwargs["session_id"] for call in opener.await_args_list] == [
        SESSION_ID, NEXT_SESSION_ID, SESSION_ID,
    ]
    assert json.loads(opener.await_args_list[0].args[3]) == {"source": "websocket-test"}
    assert opener.await_args_list[0].kwargs["files"][0][1][0] == "报告.txt"
    assert (history_path / "admin" / f"channel=web_client&session_id={SESSION_ID}.yaml").is_file()


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
            "embedded-request", SESSION_ID, "嵌入消息",
            token=token, sender_name="embedded-user",
        ))
        messages = receive_websocket_turn(websocket, "embedded-request")
        assert messages[-1]["type"] == "done"
        websocket.send_json(websocket_message(
            "invalid-request", NEXT_SESSION_ID, "不能发送", token="invalid",
        ))
        error = receive_websocket_turn(websocket, "invalid-request")[-1]
        assert error["type"] == "error"
        assert error["status"] == 401

    assert client.get(f"/api/history/sessions/{SESSION_ID}").status_code == 401
    assert (history_path / "alice" / f"channel=crm&session_id={SESSION_ID}.yaml").is_file()
    assert not (history_path / "alice" / f"channel=crm&session_id={NEXT_SESSION_ID}.yaml").exists()
