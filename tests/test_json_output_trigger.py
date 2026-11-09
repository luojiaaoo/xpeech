"""验证专用标记只对本次用户消息开启强制 JSON 输出。"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.api import app
from xpeech.agent.server.dependencies import settings
from xpeech.utils.channel_jwt import UserTokenParams, create_token


@pytest.fixture
def chat_requests(monkeypatch, tmp_path):
    messages = []

    async def run(*, message):
        messages.append(message)
        yield {"event": "assistant", "context": '{"ok":true}'}

    async def run_not_interactive(message):
        return [event async for event in run(message=message)]

    create = AsyncMock(
        return_value=SimpleNamespace(
            workspace=tmp_path,
            run=run,
            run_not_interactive=run_not_interactive,
        )
    )
    monkeypatch.setattr(AgentRunner, "create", create)
    monkeypatch.setattr(settings.llm, "support_json_output", True)
    token = create_token(UserTokenParams("web_client", user_id="user"))
    headers = {
        "Authorization": f"Bearer {token}",
        "x-session-id": "00000000-0000-7000-8000-000000000000",
        "x-sender-name": "Alice",
    }
    return SimpleNamespace(client=TestClient(app), messages=messages, create=create, headers=headers)


@pytest.mark.parametrize("endpoint", ["/chat", "/chat_sync"])
@pytest.mark.parametrize(
    ("texts", "enabled"),
    [
        (["::请严格以JSON格式回复::，列出新闻源"], True),
        (["列出新闻源。::请严格以JSON格式回复::"], True),
        (["列出新闻源", "::请严格以JSON格式回复::"], True),
        (["请输出 JSON 格式，解释 json 用法"], False),
        (["请严格以JSON格式回复"], False),
        (["::请严格以JSON格式回复"], False),
        (["请严格以JSON格式回复::"], False),
        (["请以JSON格式回复"], False),
        (["::请严格以json格式回复::"], False),
        (["::请严格以JSON", "格式回复::"], False),
    ],
)
def test_keyword_enables_only_current_request(chat_requests, endpoint, texts, enabled):
    content = [{"text": text} for text in texts]
    response = chat_requests.client.post(
        endpoint,
        headers=chat_requests.headers,
        data={"content": json.dumps(content, ensure_ascii=False)},
    )

    assert response.status_code == 200
    assert chat_requests.messages[0].json_output is enabled
    assert [block.text for block in chat_requests.messages[0].content] == texts

    # 同一会话的下一条普通消息不能继承前一条消息的标记。
    response = chat_requests.client.post(
        endpoint,
        headers=chat_requests.headers,
        data={"content": '[{"text":"继续"}]'},
    )
    assert response.status_code == 200
    assert chat_requests.messages[1].json_output is False


@pytest.mark.parametrize("endpoint", ["/chat", "/chat_sync"])
def test_unsupported_model_rejects_keyword_before_creating_runner(chat_requests, monkeypatch, endpoint):
    monkeypatch.setattr(settings.llm, "support_json_output", False)

    response = chat_requests.client.post(
        endpoint,
        headers=chat_requests.headers,
        data={"content": '[{"text":"列出新闻源。::请严格以JSON格式回复::"}]'},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The configured model does not support JSON output"}
    chat_requests.create.assert_not_awaited()


@pytest.mark.parametrize("endpoint", ["/chat", "/chat_sync"])
@pytest.mark.parametrize("legacy_flag", ["true", "false", "invalid"])
def test_removed_api_flag_cannot_control_json_output(chat_requests, monkeypatch, endpoint, legacy_flag):
    monkeypatch.setattr(settings.llm, "support_json_output", False)
    response = chat_requests.client.post(
        endpoint,
        headers=chat_requests.headers,
        data={"content": '[{"text":"普通消息"}]', "json_output": legacy_flag},
    )

    assert response.status_code == 200
    assert chat_requests.messages[0].json_output is False


@pytest.mark.parametrize("endpoint", ["/chat", "/chat_sync"])
def test_chat_openapi_has_no_json_output_parameter(endpoint):
    schema = app.openapi()
    form = schema["paths"][endpoint]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    fields = schema["components"]["schemas"][form["$ref"].rsplit("/", 1)[1]]["properties"]

    assert "json_output" not in fields
