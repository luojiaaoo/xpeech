from importlib import import_module
from types import SimpleNamespace
from typing import Annotated
from unittest.mock import AsyncMock
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from xpeech.config.settings import ExtraUserInfoConfig, JWTChannelConfig, settings
from xpeech.subapp.channel.helper import BackendJWTParams, ChatStream
from xpeech.subapp.channel.web_client.history import JsonlEventHistoryRepository
from xpeech.subapp.channel.web_client.routes import proxy
from xpeech.subapp.channel.web_client.session import require_embedded_jwt
from xpeech.utils.channel_jwt import UserTokenParams, create_token
from xpeech.utils.command_template import CommandTemplateError

settings_module = import_module("xpeech.config.settings")
SESSION_ID = "018f0000-0000-7000-8000-000000000001"
PAYLOAD = '{"data":{"name":"  目录姓名  "}}'


@pytest.mark.parametrize(("provided", "lookup", "expected", "status_code"), [
    (None, PAYLOAD, "目录姓名", 200),
    ("主站姓名", PAYLOAD, "主站姓名", 200),
    (None, None, None, 422),
    (None, CommandTemplateError("failed", operation="lookup", returncode=1), None, 422),
    (None, '{"data":{"name":null}}', None, 502),
    (" ", PAYLOAD, None, 422),
])
def test_embedded_sender_name_in_http_and_websocket(tmp_path, monkeypatch, provided, lookup, expected, status_code):
    extra = ExtraUserInfoConfig() if lookup is None else ExtraUserInfoConfig(
        command_extra_user_info_template="lookup $user_id", username_claim=".data.name",
    )
    monkeypatch.setattr(settings, "extra_user_info", extra)
    monkeypatch.setattr(settings.jwt, "channels", {"crm": JWTChannelConfig(secret_key="test-channel-key-" * 4)})
    execute = AsyncMock(return_value=lookup)
    if isinstance(lookup, CommandTemplateError):
        execute.side_effect = lookup
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    sync = AsyncMock(return_value=httpx.Response(200, json=[]))
    stream = AsyncMock(return_value=ChatStream(httpx.Response(200, content=b""), SimpleNamespace(aclose=AsyncMock())))
    monkeypatch.setattr(proxy, "open_chat_sync", sync)
    monkeypatch.setattr(proxy, "open_chat", stream)

    async def current_user(authorization: Annotated[str | None, Header()] = None, token: str | None = None):
        return require_embedded_jwt(authorization)

    app = FastAPI()
    app.include_router(proxy.create_proxy_router(
        "http://backend.test", current_user, current_user, JsonlEventHistoryRepository(tmp_path), "unused",
    ))
    token = create_token(UserTokenParams(channel="crm", user_id="alice"))
    headers = {"Authorization": f"Bearer {token}", "x-session-id": SESSION_ID}
    start = {"type": "start", "session_id": SESSION_ID, "token": token, "content": "[]"}
    if provided is not None:
        headers["x-sender-name"] = start["sender_name"] = quote(provided)

    with TestClient(app) as client:
        response = client.post("/api/chat_sync", headers=headers, data={"content": "[]"})
        assert response.status_code == status_code
        with client.websocket_connect("/api/chat/ws") as socket:
            socket.send_json(start)
            frame = socket.receive_json()
            if status_code == 200:
                assert frame["type"] == "done"
            else:
                assert (frame["type"], frame["status"]) == ("error", status_code)

    if expected is not None:
        assert sync.await_args.args[:2] == stream.await_args.args[:2] == (BackendJWTParams(token), expected)
    else:
        sync.assert_not_awaited()
        stream.assert_not_awaited()
    if provided is None and lookup is not None:
        assert execute.await_count == (2 if isinstance(lookup, CommandTemplateError) else 1)
        assert all(call.args[1] == {"user_id": "alice"} for call in execute.await_args_list)
    else:
        execute.assert_not_awaited()
