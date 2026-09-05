import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from xpeech.channel.helper import ChatStream


web_client_app = import_module("xpeech.channel.web_client.app")
proxy_routes = import_module("xpeech.channel.web_client.routes.proxy")
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app


def test_web_client_adds_channel_to_chat_metadata(tmp_path: Path, monkeypatch):
    response = httpx.Response(
        200,
        content=b'data: {"event":"command","context":"/new"}\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    backend_client = SimpleNamespace(aclose=AsyncMock())
    open_chat_stream = AsyncMock(return_value=ChatStream(response, backend_client))
    monkeypatch.setattr(proxy_routes, "open_chat_stream", open_chat_stream)
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "chat-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
        )
    )

    with TestClient(app) as client:
        assert client.post(
            "/api/auth/login",
            json={"session_id": "admin", "password": "admin123456"},
        ).status_code == 200
        proxied = client.post(
            "/api/chat",
            data={
                "content": '[{"text":"hello"}]',
                "session_metadata": json.dumps(
                    {"channel": "spoofed", "source": "browser"}
                ),
            },
        )

    assert proxied.status_code == 200
    assert json.loads(open_chat_stream.await_args.args[3]) == {
        "channel": "web",
        "source": "browser",
    }
