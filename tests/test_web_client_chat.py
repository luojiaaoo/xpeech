import json
from importlib import import_module
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.usefixtures("postgres_database")

web_client_app = import_module("xpeech.subapp.channel.web_client.app")
proxy_routes = import_module("xpeech.subapp.channel.web_client.routes.proxy")
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app


@pytest.mark.parametrize("text", ["hello", "hello::请严格以JSON格式回复::"])
@pytest.mark.parametrize("status_code", [200, 400, 409])
def test_web_client_forwards_chat_metadata_without_channel_injection(
    tmp_path: Path, monkeypatch, text, status_code,
):
    body = b'[{"event":"assistant","context":"hello"}]'
    media_type = "application/json"
    response = httpx.Response(
        status_code,
        content=body,
        headers={"Content-Type": media_type},
    )
    opener = AsyncMock(return_value=response)
    monkeypatch.setattr(proxy_routes, "open_chat_sync", opener)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
        )
    )
    schema = app.openapi()
    form = schema["paths"]["/api/chat_sync"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    form_fields = schema["components"]["schemas"][form["$ref"].rsplit("/", 1)[1]]["properties"]
    assert "json_output" not in form_fields

    with TestClient(app) as client:
        assert client.post(
            "/api/auth/login",
            json={"user_id": "admin", "password": "admin123456"},
        ).status_code == 200
        proxied = client.post(
            "/api/chat_sync",
            headers={"x-session-id": "018f0000-0000-7000-8000-000000000001"},
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
    assert opener.await_args.args[4] == "http://backend.test/chat_sync"
    assert "json_output" not in opener.await_args.kwargs
    assert json.loads(opener.await_args.args[2]) == [{"text": text}]
    assert json.loads(opener.await_args.args[3]) == {
        "channel": "spoofed",
        "source": "browser",
    }
    assert "session_id" not in opener.await_args.kwargs
