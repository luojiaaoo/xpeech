import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from xpeech.subapp.channel.helper import ChatStream

pytestmark = pytest.mark.usefixtures("postgres_database")

web_client_app = import_module("xpeech.subapp.channel.web_client.app")
proxy_routes = import_module("xpeech.subapp.channel.web_client.routes.proxy")
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app


@pytest.mark.parametrize("text", ["hello", "hello::请严格以JSON格式回复::"])
@pytest.mark.parametrize("endpoint", ["chat", "chat_sync"])
@pytest.mark.parametrize("status_code", [200, 400, 409])
def test_web_client_forwards_chat_metadata_without_channel_injection(
    tmp_path: Path, monkeypatch, text, endpoint, status_code,
):
    body = (
        b'[{"event":"assistant","context":"hello"}]' if endpoint == "chat_sync"
        else b'data: {"event":"command","context":"/new"}\n\n'
    )
    media_type = "application/json" if endpoint == "chat_sync" else "text/event-stream"
    response = httpx.Response(
        status_code,
        content=body,
        headers={"Content-Type": media_type},
    )
    backend_client = SimpleNamespace(aclose=AsyncMock())
    if endpoint == "chat_sync":
        opener = AsyncMock(return_value=response)
        monkeypatch.setattr(proxy_routes, "open_chat_sync", opener)
    else:
        opener = AsyncMock(return_value=ChatStream(response, backend_client))
        monkeypatch.setattr(proxy_routes, "open_chat", opener)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
        )
    )
    schema = app.openapi()
    form = schema["paths"][f"/api/{endpoint}"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    form_fields = schema["components"]["schemas"][form["$ref"].rsplit("/", 1)[1]]["properties"]
    assert "json_output" not in form_fields

    with TestClient(app) as client:
        assert client.post(
            "/api/auth/login",
            json={"user_id": "admin", "password": "admin123456"},
        ).status_code == 200
        proxied = client.post(
            f"/api/{endpoint}",
            data={
                "content": json.dumps([{"text": text}], ensure_ascii=False),
                "session_metadata": json.dumps(
                    {"channel": "spoofed", "source": "browser"}
                ),
            },
        )

    assert proxied.status_code == status_code
    assert proxied.content == body
    assert proxied.headers["content-type"].startswith(media_type)
    assert opener.await_args.args[4] == f"http://backend.test/{endpoint}"
    assert "json_output" not in opener.await_args.kwargs
    assert json.loads(opener.await_args.args[2]) == [{"text": text}]
    assert json.loads(opener.await_args.args[3]) == {
        "channel": "spoofed",
        "source": "browser",
    }
    if endpoint == "chat_sync":
        # 同步执行由后端生成独立会话，不透传会话 ID。
        assert "session_id" not in opener.await_args.kwargs
    else:
        backend_client.aclose.assert_awaited_once()
