import asyncio
import json
from html import unescape
from pathlib import Path
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from lark_channel import (
    CardActionEvent,
    CardActionPayload,
    Conversation,
    EventOperator,
    FeishuChannel,
    FileContent,
    Identity,
    ImageContent,
    InboundMessage,
    MediaContent,
    PostContent,
    SendResult,
    TextContent,
)
from lark_channel.channel.safety.pipeline import SafetyPipeline

import xpeech.subapp.channel.feishu.bridge as feishu_bridge
import xpeech.subapp.channel.feishu.delivery as feishu_delivery
import xpeech.subapp.channel.feishu.inbound as feishu_inbound
from xpeech.subapp.channel.feishu.bridge import FeishuBridge
from xpeech.subapp.channel.feishu.cards import (
    build_feishu_background_task_card,
    build_feishu_question_card,
)
from xpeech.subapp.channel.feishu.channel import XpeechFeishuChannel
from xpeech.subapp.channel.feishu.config import DEFAULT_SESSION_ID
from xpeech.subapp.channel.schema import ChatEvent, ChatEventType, FileData, Message, TextData


def _inbound_message(
    content,
    *,
    message_id: str = "om_message",
    chat_type: str = "p2p",
) -> InboundMessage:
    return InboundMessage(
        id=message_id,
        create_time=123_000,
        conversation=Conversation(chat_id="oc_chat", chat_type=chat_type),
        sender=Identity(open_id="ou_sender", display_name="Alice"),
        content=content,
    )


def _bridge_with_channel(
    channel,
    *,
    employee_no: str = "E1001",
    email: str | None = "alice@example.com",
) -> FeishuBridge:
    bridge = object.__new__(FeishuBridge)
    bridge.chat_url = "http://backend.test"
    bridge.channel = channel
    channel.block_batch_scope = MagicMock()
    channel.unblock_batch_scope = MagicMock()
    bridge.session_update_message_id = {}
    bridge.question_cards = {}
    bridge.get_user_identity = AsyncMock(return_value=(employee_no, email))
    return bridge


def _remember_question(bridge, message_id, question):
    card = feishu_bridge._QuestionCard(message_id, json.dumps(question))
    bridge.question_cards[message_id] = card
    return card


def test_channel_uses_expected_compatibility_configuration():
    bridge = FeishuBridge(
        chat_url="http://backend.test",
        app_id="cli_test",
        app_secret="secret",
    )

    assert bridge.channel.config.policy.dm_policy == "open"
    assert bridge.channel.config.policy.group_policy == "open"
    assert bridge.channel.config.policy.require_mention is True
    assert bridge.channel.config.safety.dedup.ttl_seconds == 43_200
    assert bridge.channel.config.outbound.retry.max_attempts == 5
    assert bridge.channel.config.security.mode == "compat"


@pytest.mark.asyncio
async def test_sdk_batches_busy_messages_and_card_answer_unblocks_agent(tmp_path, monkeypatch):
    bridge = FeishuBridge("http://backend.test", "cli_test", "secret")
    channel = bridge.channel
    channel._bg_loop = asyncio.get_running_loop()
    bridge.get_user_identity = AsyncMock(return_value=("E1001", "alice@example.com"))
    channel.download_resource_to_file = AsyncMock(return_value=tmp_path / "report.txt")
    channel.update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_question"))
    question_card = _remember_question(bridge, "om_question", {
        "type": "form", "fields": [{"type": "input", "name": "answer", "label": "回答"}],
    })
    first_started = asyncio.Event()
    answered = asyncio.Event()
    batches = []

    async def consume(messages):
        batches.append(messages)
        if len(batches) == 1:
            first_started.set()
            await answered.wait()

    async def answer(*_args, **_kwargs):
        answered.set()

    bridge.consume = consume
    notify_question = AsyncMock(side_effect=answer)
    monkeypatch.setattr(feishu_bridge, "notify_question", notify_question)
    safety = channel.config.safety
    pipeline = SafetyPipeline(
        loop=asyncio.get_running_loop(),
        on_message=bridge._on_message,
        policy=channel.config.policy,
        dedup_config=safety.dedup,
        batch_config=safety.text_batch,
        media_batch_config=safety.media_batch,
        queue_config=safety.chat_queue,
    )
    channel._safety = pipeline
    contents = [
        TextContent(text="first"),
        TextContent(text="second" * 1000),
        FileContent(file_key="file_report", file_name="report.txt"),
        TextContent(text="third"),
    ]
    messages = [_inbound_message(content, message_id=f"om_{index}") for index, content in enumerate(contents)]
    for message in messages:
        message.create_time = int(time() * 1000)
    action = SimpleNamespace(
        event=SimpleNamespace(
            context=SimpleNamespace(open_message_id="om_question", open_chat_id="oc_chat"),
            operator=SimpleNamespace(open_id="ou_sender"),
            action=SimpleNamespace(tag="button", value={}, form_value={"answer": "yes"}),
        )
    )
    try:
        await pipeline.push_message(messages[0])
        await asyncio.wait_for(first_started.wait(), timeout=2)
        for message in messages[1:]:
            await pipeline.push_message(message)
        assert len(batches) == 1

        # These go through the real SDK card dispatcher, including its dedup gate.
        await channel._handle_interaction_event(action)
        await channel._handle_interaction_event(action)
        await asyncio.wait_for(answered.wait(), timeout=1)
        await asyncio.wait_for(pipeline.dispose(), timeout=2)
    finally:
        answered.set()
        await pipeline.dispose()

    assert [[message.message_id for message in batch] for batch in batches] == [
        ["om_0"],
        ["om_1", "om_2", "om_3"],
    ]
    assert batches[1][0].content == [TextData(text="second" * 1000)]
    assert batches[1][1].content == [FileData(file=tmp_path / "report.txt")]
    assert batches[1][2].content == [TextData(text="third")]
    notify_question.assert_awaited_once_with(
        "E1001",
        {"answer": "yes"},
        bridge.chat_url,
        "feishu",
        session_id=DEFAULT_SESSION_ID,
    )
    channel.update_card.assert_awaited_once_with(
        "om_question", build_feishu_question_card(question_card.context, disabled=True, answers={"answer": "yes"}),
    )


@pytest.mark.parametrize(
    ("event_id", "scope", "expected"),
    [
        ("card:answer", "oc_chat", "card-action:oc_chat"),
        ("card:answer", "", "card-action:card:answer"),
        ("comment:doc:comment", "doc", "doc"),
    ],
)
@pytest.mark.asyncio
async def test_card_queue_override_preserves_parent_safety(monkeypatch, event_id, scope, expected):
    parent_safety = AsyncMock()
    monkeypatch.setattr(FeishuChannel, "_through_action_safety", parent_safety)
    channel = XpeechFeishuChannel(app_id="cli_test", app_secret="secret")
    handler = AsyncMock()

    await channel._through_action_safety(event_id=event_id, queue_scope=scope, handler=handler)

    parent_safety.assert_awaited_once_with(
        event_id=event_id,
        queue_scope=expected,
        handler=handler,
    )


def test_background_task_card_contains_result_content():
    assert build_feishu_background_task_card(
        "**scheduled result**",
        schedule_type="cron",
        schedule_value="0 9 * * *",
    ) == {
        "schema": "2.0",
        "header": {
            "title": {"tag": "plain_text", "content": "定时任务执行结果"},
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": "**调度类型**：cron\n**调度表达式**：`0 9 * * *`",
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                },
                {
                    "tag": "markdown",
                    "content": "**scheduled result**",
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                },
            ]
        },
    }


def test_background_task_card_escapes_dynamic_markup():
    raw = '**结果**\n[链接](https://example.test)\n<rss>甲 &amp; 乙</rss>'
    card = build_feishu_background_task_card(raw, schedule_type="<cron>", schedule_value="<expression>")
    metadata, result = card["body"]["elements"]
    for element in (metadata, result):
        assert "<" not in element["content"] and ">" not in element["content"]
    assert unescape(result["content"]) == raw
    assert unescape(metadata["content"]) == "**调度类型**：<cron>\n**调度表达式**：`<expression>`"
    assert "**结果**" in result["content"] and "[链接](https://example.test)" in result["content"]


@pytest.mark.asyncio
async def test_background_poll_immediately_sends_markdown_card():
    request = httpx.Request("GET", "http://backend.test/background_message")
    response = httpx.Response(
        200,
        request=request,
        json={
            "channel": "feishu",
            "open_id": "ou_receiver",
            "content": "**scheduled result**",
            "schedule_type": "cron",
            "schedule_value": "0 9 * * *",
        },
    )
    client = SimpleNamespace(get=AsyncMock(return_value=response))
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_background"))
    bridge = _bridge_with_channel(SimpleNamespace(send=send))

    assert await bridge.poll_background_message_once(client) is True

    client.get.assert_awaited_once()
    assert client.get.await_args.args == ("http://backend.test/background_message",)
    assert client.get.await_args.kwargs["headers"]["authorization"].startswith("Bearer ")
    send.assert_awaited_once_with(
        to="ou_receiver",
        message={
            "card": build_feishu_background_task_card(
                "**scheduled result**",
                schedule_type="cron",
                schedule_value="0 9 * * *",
            ),
        },
    )


@pytest.mark.asyncio
async def test_background_poll_treats_no_content_as_empty():
    request = httpx.Request("GET", "http://backend.test/background_message")
    client = SimpleNamespace(
        get=AsyncMock(return_value=httpx.Response(204, request=request)),
    )
    send = AsyncMock()
    bridge = _bridge_with_channel(SimpleNamespace(send=send))

    assert await bridge.poll_background_message_once(client) is False
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_poll_retries_failures_and_remains_cancellable(monkeypatch):
    class ClientContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        feishu_bridge.httpx,
        "AsyncClient",
        lambda **_kwargs: ClientContext(),
    )
    sleep = AsyncMock()
    monkeypatch.setattr(feishu_bridge.asyncio, "sleep", sleep)
    bridge = _bridge_with_channel(SimpleNamespace())
    bridge.poll_background_message_once = AsyncMock(
        side_effect=[RuntimeError("backend unavailable"), asyncio.CancelledError()],
    )

    with pytest.raises(asyncio.CancelledError):
        await bridge.poll_background_messages()

    assert bridge.poll_background_message_once.await_count == 2
    sleep.assert_awaited_once_with(feishu_bridge.BACKGROUND_MESSAGE_RETRY_SECONDS)


@pytest.mark.asyncio
async def test_bridge_lifecycle_runs_background_poll_task():
    started = set()
    cancelled = set()

    async def wait_until_cancelled(name):
        started.add(name)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(name)

    async def connect():
        await asyncio.sleep(0)

    bridge = _bridge_with_channel(
        SimpleNamespace(
            connect=connect,
            disconnect=AsyncMock(),
        )
    )
    bridge.poll_background_messages = lambda: wait_until_cancelled("background")

    await bridge.start()

    assert started == {"background"}
    assert cancelled == {"background"}
    bridge.channel.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_user_identity_queries_contact_api_and_caches_result():
    response = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(user=SimpleNamespace(employee_no="E1001", email="alice@example.com")),
    )
    get_user = AsyncMock(return_value=response)
    client = SimpleNamespace(contact=SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(aget=get_user))))
    bridge = object.__new__(FeishuBridge)
    bridge.channel = SimpleNamespace(client=client)

    first = await bridge.get_user_identity("ou_sender")
    second = await bridge.get_user_identity("ou_sender")

    assert first == second == ("E1001", "alice@example.com")
    get_user.assert_awaited_once()
    request = get_user.await_args.args[0]
    assert request.user_id == "ou_sender"
    assert request.user_id_type == "open_id"


@pytest.mark.parametrize(
    ("response", "error_message"),
    [
        (
            SimpleNamespace(success=lambda: False, code=41050, msg="no user authority", data=None),
            "Failed to get Feishu user identity",
        ),
        (
            SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="success",
                data=SimpleNamespace(user=SimpleNamespace(employee_no=None, email="alice@example.com")),
            ),
            "Feishu user has no employee_no",
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_user_identity_rejects_api_failure_or_missing_fields(response, error_message: str):
    get_user = AsyncMock(return_value=response)
    client = SimpleNamespace(contact=SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(aget=get_user))))
    bridge = object.__new__(FeishuBridge)
    bridge.channel = SimpleNamespace(client=client)

    with pytest.raises(RuntimeError, match=error_message):
        await FeishuBridge.get_user_identity.__wrapped__(bridge, "ou_missing")


@pytest.mark.parametrize(
    ("email", "enterprise_email", "expected_email"),
    [
        ("alice@example.com", "alice@company.example", "alice@example.com"),
        (None, "alice@company.example", "alice@company.example"),
        (None, None, None),
    ],
)
@pytest.mark.asyncio
async def test_get_user_identity_supports_enterprise_or_missing_email(email, enterprise_email, expected_email):
    response = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(user=SimpleNamespace(employee_no="E1001", email=email, enterprise_email=enterprise_email)),
    )
    get_user = AsyncMock(return_value=response)
    client = SimpleNamespace(contact=SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(aget=get_user))))
    bridge = object.__new__(FeishuBridge)
    bridge.channel = SimpleNamespace(client=client)

    identity = await FeishuBridge.get_user_identity.__wrapped__(bridge, "ou_sender")

    assert identity == ("E1001", expected_email)


@pytest.mark.asyncio
async def test_parse_text_message_uses_employee_number_and_feishu_user_metadata():
    bridge = _bridge_with_channel(SimpleNamespace())

    message = await bridge._parse_msg(_inbound_message(TextContent(text="hello")))

    assert message.message_id == "om_message"
    assert message.chat_id == "oc_chat"
    assert message.user_id == "E1001"
    assert message.sender_name == "Alice"
    assert message.timestamp == 123
    assert message.content == [TextData(text="hello")]
    assert message.session_metadata == {
        "open_id": "ou_sender",
        "email": "alice@example.com",
    }


@pytest.mark.asyncio
async def test_parse_text_message_allows_missing_email():
    bridge = _bridge_with_channel(SimpleNamespace(), email=None)

    message = await bridge._parse_msg(_inbound_message(TextContent(text="hello")))

    assert message.user_id == "E1001"
    assert message.session_metadata == {
        "open_id": "ou_sender",
    }


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.asyncio
async def test_on_message_waits_for_consumption_and_releases_busy_scope(cancelled):
    channel = SimpleNamespace(block_batch_scope=MagicMock(), unblock_batch_scope=MagicMock())
    bridge = _bridge_with_channel(channel)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=123,
        session_metadata={},
    )
    bridge._parse_msg = AsyncMock(return_value=message)
    started = asyncio.Event()
    release = asyncio.Event()

    async def consume(messages):
        assert messages == [message]
        started.set()
        await release.wait()

    bridge.consume = consume
    async with asyncio.TaskGroup() as tasks:
        task = tasks.create_task(bridge._on_message(_inbound_message(TextContent(text="hello"))))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()
        channel.block_batch_scope.assert_called_once_with("oc_chat")
        channel.unblock_batch_scope.assert_not_called()
        bridge.session_update_message_id["E1001"] = "om_progress"
        if cancelled:
            task.cancel()
        else:
            release.set()

    assert task.cancelled() is cancelled
    channel.unblock_batch_scope.assert_called_once_with("oc_chat")
    assert bridge.session_update_message_id == {}


@pytest.mark.asyncio
async def test_on_message_ignores_unsupported_message(monkeypatch: pytest.MonkeyPatch):
    bridge = _bridge_with_channel(SimpleNamespace())
    warning = MagicMock()
    monkeypatch.setattr(feishu_bridge, "logger", SimpleNamespace(warning=warning))
    inbound_message = _inbound_message(TextContent(text="hello"), chat_type="group")

    await bridge._on_message(inbound_message)

    bridge.channel.unblock_batch_scope.assert_called_once_with("oc_chat")
    warning.assert_called_once_with(
        "Unsupported Feishu message: chat_type=group, content_type=TextContent, message_id=om_message",
    )


@pytest.mark.asyncio
async def test_channel_send_logs_message_type(monkeypatch: pytest.MonkeyPatch):
    send_result = SendResult(success=True, message_id="om_sent")
    bridge = _bridge_with_channel(SimpleNamespace())
    bridge.send = AsyncMock(return_value=send_result)
    info = MagicMock()
    monkeypatch.setattr(feishu_delivery, "logger", SimpleNamespace(info=info))

    await bridge.channel_send(
        to="oc_chat",
        message={"text": "hello"},
        opts={"reply_to": "om_parent"},
        user_id="E1001",
        sender_name="Alice",
        message_type=ChatEventType.ASSISTANT,
        iter_content=None,
    )

    bridge.send.assert_awaited_once_with(
        to="oc_chat",
        message={"text": "hello"},
        iter_content=None,
        opts={"reply_to": "om_parent"},
    )
    info.assert_called_once_with(
        "Feishu message sending user_id={} sender_name={} message_type={}",
        "E1001",
        "Alice",
        ChatEventType.ASSISTANT,
    )


@pytest.mark.parametrize(
    ("content", "resource_type", "file_name", "downloaded_name"),
    [
        (ImageContent(image_key="img_key"), "image", None, "img_key.png"),
        (FileContent(file_key="file_key", file_name="report.pdf"), "file", "report.pdf", "report.pdf"),
        (MediaContent(file_key="video_key", file_name="demo.mp4"), "video", "demo.mp4", "demo.mp4"),
    ],
)
@pytest.mark.asyncio
async def test_parse_resource_message_uses_channel_download_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content,
    resource_type: str,
    file_name: str | None,
    downloaded_name: str,
):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "E1001" / downloaded_name
    download_resource = AsyncMock(return_value=downloaded_path)
    bridge = _bridge_with_channel(SimpleNamespace(download_resource_to_file=download_resource))

    message = await bridge._parse_msg(_inbound_message(content))

    expected_kwargs = {
        "resource_type": resource_type,
        "message_id": "om_message",
        "dest_dir": tmp_path / "E1001",
    }
    if file_name:
        expected_kwargs["file_name"] = file_name
    file_key = content.image_key if resource_type == "image" else content.file_key
    download_resource.assert_awaited_once_with(file_key, **expected_kwargs)
    assert message.content == [FileData(file=downloaded_path)]


@pytest.mark.asyncio
async def test_parse_post_reads_content_v2_document_and_preserves_attachment_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "E1001" / "img_post.png"
    download_resource = AsyncMock(return_value=downloaded_path)
    bridge = _bridge_with_channel(SimpleNamespace(download_resource_to_file=download_resource))
    post = PostContent(
        post={
            "content_v2": [
                [
                    {"tag": "text", "text": "before"},
                    {"tag": "a", "text": "link", "href": "https://example.com"},
                ],
                [{"tag": "img", "image_key": "img_post"}],
                [{"tag": "text", "text": "after"}],
            ],
        }
    )

    message = await bridge._parse_msg(_inbound_message(post))

    assert message.content == [
        TextData(text="before\n[link](https://example.com)"),
        TextData(text="[Attachment: img_post.png]"),
        FileData(file=downloaded_path),
        TextData(text="after"),
    ]
    download_resource.assert_awaited_once_with(
        "img_post",
        resource_type="image",
        message_id="om_message",
        dest_dir=tmp_path / "E1001",
    )


@pytest.mark.asyncio
async def test_card_action_uses_typed_form_value(monkeypatch: pytest.MonkeyPatch):
    notify_question = AsyncMock()
    monkeypatch.setattr(feishu_bridge, "notify_question", notify_question)
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {
        "type": "form", "fields": [
            {"type": "input", "name": "answer", "label": "回答", "default_value": "initial"},
            {"type": "select", "name": "choice", "label": "单选", "options": [{"label": "A", "value": "a"}]},
            {"type": "multi_select", "name": "tags", "label": "多选", "options": [{"label": "B", "value": "b"}]},
            {"type": "date", "name": "day", "label": "日期"},
            {"type": "datetime", "name": "when", "label": "时间"},
        ],
    })
    answers = {
        "answer": "typed", "choice": "a", "tags": ["b"],
        "day": "2026-11-12 +0800", "when": "2026-11-12 09:30 +0800",
    }
    event = CardActionEvent(
        message_id="om_message",
        chat_id="oc_chat",
        operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(form_value=answers),
        raw={"event": {"action": {"form_value": {"answer": "raw"}}}},
    )

    await bridge._on_card_action(event)

    notify_question.assert_awaited_once_with(
        "E1001",
        answers,
        "http://backend.test",
        "feishu",
        session_id=DEFAULT_SESSION_ID,
    )
    bridge.get_user_identity.assert_awaited_once_with("ou_sender")
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(question_card.context, disabled=True, answers=answers),
    )
    elements = update_card.await_args.args[1]["body"]["elements"][0]["elements"]
    fields = {element["name"]: element for element in elements if "name" in element}
    assert all(field["disabled"] for field in fields.values())
    assert fields["answer"]["default_value"] == "typed"
    assert fields["choice"]["initial_option"] == "a"
    assert fields["tags"]["selected_values"] == ["b"]
    assert fields["day"]["initial_date"] == "2026-11-12"
    assert fields["when"]["initial_datetime"] == "2026-11-12 09:30"
    submit = elements[-1]["columns"][0]["elements"][0]
    assert submit["disabled"] is True
    assert submit["text"]["content"] == "提交"


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["confirm", "cancel"])
async def test_confirmation_button_callback_submits_decision(monkeypatch, decision):
    notify_question = AsyncMock()
    monkeypatch.setattr(feishu_bridge, "notify_question", notify_question)
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {"type": "confirm", "description": "继续执行？"})
    event = CardActionEvent(
        message_id="om_message",
        chat_id="oc_chat",
        operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"type": "confirm", "decision": decision, "tool_call_id": None}),
    )
    await bridge._on_card_action(event)
    notify_question.assert_awaited_once_with(
        "E1001", {"decision": decision}, "http://backend.test", "feishu", session_id=DEFAULT_SESSION_ID,
    )
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(question_card.context, disabled=True),
    )
    card = update_card.await_args.args[1]
    assert card["body"]["elements"][0]["text"]["content"] == "继续执行？"
    assert all(column["elements"][0]["disabled"] for column in card["body"]["elements"][-1]["columns"])
    await bridge._on_card_action(event)
    assert notify_question.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [SendResult(success=False), RuntimeError("Feishu unavailable")])
async def test_card_update_failure_does_not_block_answer_submission(monkeypatch, failure):
    notify = AsyncMock()
    monkeypatch.setattr(feishu_bridge, "notify_question", notify)

    async def update(*_args, **_kwargs):
        notify.assert_awaited_once()
        if isinstance(failure, Exception):
            raise failure
        return failure

    update_card = AsyncMock(side_effect=update)
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    card = _remember_question(bridge, "om_message", {"type": "confirm"})
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"decision": "confirm"}),
    )

    await bridge._on_card_action(event)
    await bridge._on_card_action(event)

    notify.assert_awaited_once_with(
        "E1001", {"decision": "confirm"}, bridge.chat_url, "feishu", session_id=DEFAULT_SESSION_ID,
    )
    assert card.disabled is True
    assert update_card.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_answer_does_not_disable_card_and_allows_retry(monkeypatch, confirmation, error_type):
    update_card = AsyncMock(return_value=SendResult(success=True))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question = {"type": "confirm"} if confirmation else {
        "type": "form", "fields": [{"type": "input", "name": "answer", "label": "回答"}],
    }
    card = _remember_question(bridge, "om_message", question)
    answers = {"decision": "confirm"} if confirmation else {"answer": "保留我的回答"}
    error = error_type("submission interrupted")
    notify = AsyncMock(side_effect=[error, None])
    monkeypatch.setattr(feishu_bridge, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(form_value=answers),
    )

    with pytest.raises(error_type) as exc:
        await bridge._on_card_action(event)
    assert exc.value is error
    assert card.disabled is False
    assert card.submitting is False
    update_card.assert_not_awaited()

    await bridge._on_card_action(event)
    await bridge._on_card_action(event)
    assert card.disabled is True
    assert notify.await_count == 2
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(card.context, disabled=True, answers=answers),
    )


@pytest.mark.asyncio
async def test_pending_submission_ignores_duplicate_clicks_without_disabling_card(monkeypatch):
    update_card = AsyncMock(return_value=SendResult(success=True))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    card = _remember_question(bridge, "om_message", {"type": "confirm"})
    started = asyncio.Event()
    release = asyncio.Event()

    async def submit(*_args, **_kwargs):
        update_card.assert_not_awaited()
        started.set()
        await release.wait()

    notify = AsyncMock(side_effect=submit)
    monkeypatch.setattr(feishu_bridge, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"decision": "confirm"}),
    )
    callback = asyncio.create_task(bridge._on_card_action(event))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert card.submitting is True
        assert card.disabled is False
        await asyncio.wait_for(bridge._on_card_action(event), timeout=1)
        notify.assert_awaited_once()
        update_card.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(callback, timeout=1)
    assert card.submitting is False
    assert card.disabled is True
    update_card.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_success", [False, True])
async def test_failed_submission_does_not_change_a_finished_confirmation(monkeypatch, finish_success):
    update_card = AsyncMock(return_value=SendResult(success=finish_success))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    card = _remember_question(bridge, "om_message", {"type": "confirm", "tool_call_id": "call-1"})
    tool_result = {"success": True, "result": "已执行"}

    async def notify(*_args, **_kwargs):
        await bridge._finish_confirmation_card(card, tool_result)
        raise RuntimeError("submission response lost")

    monkeypatch.setattr(feishu_bridge, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"decision": "confirm"}),
    )

    with pytest.raises(RuntimeError):
        await bridge._on_card_action(event)
    assert card.disabled is True
    assert card.submitting is False
    assert update_card.await_count == 1
    assert update_card.await_args.args == (
        "om_message", build_feishu_question_card(card.context, tool_result=tool_result),
    )


@pytest.mark.parametrize(
    "result",
    [
        SendResult(success=False),
        SendResult(success=True, message_id=None),
    ],
)
@pytest.mark.asyncio
async def test_failed_progress_send_does_not_cache_invalid_message_id(result: SendResult):
    send = AsyncMock(return_value=result)
    bridge = _bridge_with_channel(SimpleNamespace(send=send))

    with pytest.raises(RuntimeError, match="Feishu send progress card"):
        await bridge.channel_send(
            to="oc_chat",
            message={"card": {}},
            opts=None,
            user_id="feishu_oc_chat",
            sender_name="Alice",
            message_type=ChatEventType.TOOL_CALL,
            iter_content=None,
        )

    assert "feishu_oc_chat" not in bridge.session_update_message_id


@pytest.mark.asyncio
async def test_successful_progress_send_caches_message_id():
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_progress"))
    bridge = _bridge_with_channel(SimpleNamespace(send=send))

    await bridge.channel_send(
        to="oc_chat",
        message={"card": {}},
        opts=None,
        user_id="feishu_oc_chat",
        sender_name="Alice",
        message_type=ChatEventType.TOOL_CALL,
        iter_content=None,
    )

    assert bridge.session_update_message_id == {"feishu_oc_chat": "om_progress"}


def test_thinking_card_wraps_content_in_collapsible_panel():
    async def chunks():
        yield "thinking"

    content = chunks()
    bridge = object.__new__(FeishuBridge)

    message, iter_content = bridge._format_chat_event(ChatEvent(event=ChatEventType.THINKING, context=content))

    assert iter_content is content
    panel = message["card"]["body"]["elements"][0]
    assert panel == {
        "tag": "collapsible_panel",
        "element_id": "panel",
        "expanded": False,
        "header": {
            "title": {"tag": "plain_text", "content": "查看思考过程"},
            "expanded_title": {"tag": "plain_text", "content": "收起思考过程"},
        },
        "elements": [
            {
                "tag": "markdown",
                "margin": "0px 0px 0px 0px",
                "content": "思考中...",
                "text_size": "notation",
                "text_align": "left",
                "icon": {"tag": "standard_icon", "token": "tab-more_outlined", "color": "green"},
                "element_id": "main",
            }
        ],
    }


@pytest.mark.asyncio
async def test_streaming_thinking_uses_custom_card(monkeypatch: pytest.MonkeyPatch):
    async def chunks():
        yield "thinking"

    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    card = {"schema": "2.0", "body": {"elements": [{"tag": "markdown", "element_id": "main"}]}}
    create_card_instance = AsyncMock(return_value="card_thinking")
    send_card_by_reference = AsyncMock(return_value=SendResult(success=True, message_id="om_stream"))
    update_card_element_content = AsyncMock()
    finish_streaming_card = AsyncMock()
    update_card = AsyncMock()
    bridge = _bridge_with_channel(
        SimpleNamespace(
            create_card_instance=create_card_instance,
            send_card_by_reference=send_card_by_reference,
            update_card_element_content=update_card_element_content,
            finish_streaming_card=finish_streaming_card,
            update_card=update_card,
        )
    )
    await bridge.channel_send(
        to="oc_chat",
        message={"card": card},
        opts=None,
        user_id="feishu_oc_chat",
        sender_name="Alice",
        message_type=ChatEventType.THINKING,
        iter_content=chunks(),
    )

    create_card_instance.assert_awaited_once_with(card)
    send_card_by_reference.assert_awaited_once_with("oc_chat", "card_thinking")
    update_card_element_content.assert_awaited_once_with(
        "card_thinking",
        "main",
        "thinking",
        sequence=1,
    )
    finish_streaming_card.assert_awaited_once_with("card_thinking", sequence=2)
    update_card.assert_not_awaited()
    assert "feishu_oc_chat" not in bridge.session_update_message_id


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [ChatEventType.THINKING, ChatEventType.ASSISTANT])
@pytest.mark.parametrize("interval", [0, 60])
async def test_streaming_cards_escape_split_markup_in_updates_and_final_flush(monkeypatch, event_type, interval):
    tokens = ['**结果**\n[链接](https://example.test)\n<', 'rss>甲 &', 'amp; 乙</rss>']

    async def chunks():
        for token in tokens:
            yield token

    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", interval)
    monkeypatch.setattr(feishu_delivery.asyncio, "sleep", AsyncMock())
    channel = SimpleNamespace(
        create_card_instance=AsyncMock(return_value="stream-card"),
        send_card_by_reference=AsyncMock(return_value=SendResult(success=True, message_id="stream-message")),
        update_card_element_content=AsyncMock(),
        finish_streaming_card=AsyncMock(),
    )
    bridge = _bridge_with_channel(channel)
    message, iter_content = bridge._format_chat_event(ChatEvent(event=event_type, context=chunks()))
    await bridge.send("oc_chat", message, iter_content)

    updates = [call.args[2] for call in channel.update_card_element_content.await_args_list]
    assert len(updates) == (len(tokens) if interval == 0 else 1)
    assert all("<" not in content and ">" not in content for content in updates)
    assert unescape(updates[-1]) == "".join(tokens)
    assert "**结果**" in updates[-1] and "[链接](https://example.test)" in updates[-1]
    if interval == 0:
        assert [unescape(content) for content in updates] == ["".join(tokens[:i]) for i in range(1, 4)]
    channel.finish_streaming_card.assert_awaited_once_with("stream-card", sequence=len(updates) + 1)


@pytest.mark.asyncio
async def test_consume_sends_one_sdk_batch_as_one_prompt(monkeypatch):
    messages = [
        Message(
            message_id=f"om_message_{index}",
            chat_id="oc_chat",
            user_id="E1001",
            sender_name="Alice",
            content=[TextData(text=text)],
            timestamp=0,
            session_metadata={},
        )
        for index, text in enumerate(("first", "second"), start=1)
    ]
    requests = []

    async def chat_events(request_messages, chat_url, channel_name, *, session_id):
        requests.append((request_messages, chat_url, channel_name, session_id))
        if False:
            yield

    monkeypatch.setattr(feishu_bridge, "iter_chat_events", chat_events)
    channel = SimpleNamespace(add_reaction=AsyncMock())
    bridge = _bridge_with_channel(channel)

    await bridge.consume(messages)

    assert requests == [
        (messages, "http://backend.test/chat", "feishu", DEFAULT_SESSION_ID)
    ]
    assert [call.args[0] for call in channel.add_reaction.await_args_list] == [
        "om_message_1",
        "om_message_2",
    ]


@pytest.mark.asyncio
async def test_consume_streams_assistant_chunks_as_reply(monkeypatch: pytest.MonkeyPatch):
    async def chunks():
        for chunk in ["hello", " ", "world"]:
            yield chunk

    async def streaming_chat_events(*_args, **_kwargs):
        yield ChatEvent(event=ChatEventType.ASSISTANT, context=chunks())

    monkeypatch.setattr(feishu_bridge, "iter_chat_events", streaming_chat_events)
    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    create_card_instance = AsyncMock(return_value="card_assistant")
    send_card_by_reference = AsyncMock(return_value=SendResult(success=True, message_id="om_stream"))
    update_card_element_content = AsyncMock()
    finish_streaming_card = AsyncMock()
    channel = SimpleNamespace(
        add_reaction=AsyncMock(),
        create_card_instance=create_card_instance,
        send_card_by_reference=send_card_by_reference,
        update_card_element_content=update_card_element_content,
        finish_streaming_card=finish_streaming_card,
    )
    bridge = _bridge_with_channel(channel)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )
    bridge.session_update_message_id["E1001"] = "om_progress"

    await bridge.consume([message])

    created_card = create_card_instance.await_args.args[0]
    assert created_card["body"]["elements"][0]["icon"] == {
        "tag": "standard_icon",
        "token": "robot_filled",
        "color": "red",
    }
    send_card_by_reference.assert_awaited_once_with(
        "oc_chat",
        "card_assistant",
        reply_to="om_message",
    )
    assert [await_call.args[2] for await_call in update_card_element_content.await_args_list] == [
        "hello",
        "hello ",
        "hello world",
    ]
    finish_streaming_card.assert_awaited_once_with("card_assistant", sequence=4)
    assert "E1001" not in bridge.session_update_message_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (409, "Session 'E1001' already has an active chat request"),
        (500, "Backend unavailable"),
    ],
)
async def test_consume_returns_backend_http_error_to_user(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    detail: str,
):
    request = httpx.Request("POST", "http://backend.test/chat")
    response = httpx.Response(status_code, request=request, json={"detail": detail})

    async def busy_chat_events(*_args, **_kwargs):
        raise httpx.HTTPStatusError("Conflict", request=request, response=response)
        yield

    monkeypatch.setattr(feishu_bridge, "iter_chat_events", busy_chat_events)
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_busy"))
    channel = SimpleNamespace(add_reaction=AsyncMock(), send=send)
    bridge = _bridge_with_channel(channel)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    await bridge.consume([message])

    send.assert_awaited_once()
    assert send.await_args.kwargs["to"] == "oc_chat"
    assert send.await_args.kwargs["opts"] == {"reply_to": "om_message"}
    sent_message = send.await_args.kwargs["message"]
    assert isinstance(sent_message, dict)
    error_element = sent_message["card"]["body"]["elements"][0]
    assert error_element["content"] == f"**[错误]** {status_code}: {detail}"
    assert error_element["icon"] == {
        "tag": "standard_icon",
        "token": "warning_outlined",
        "color": "red",
    }
    assert "streaming_mode" not in sent_message["card"].get("config", {})


@pytest.mark.asyncio
async def test_execution_result_cannot_be_overwritten_by_submitted_confirmation_state(monkeypatch):
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {"type": "confirm", "tool_call_id": "call-1"})
    tool_result = {"success": False, "result": "执行出错"}
    final_card = build_feishu_question_card(question_card.context, tool_result=tool_result)

    async def notify(*_args, **_kwargs):
        update_card.assert_not_awaited()
        await bridge._finish_confirmation_card(question_card, tool_result)

    monkeypatch.setattr(feishu_bridge, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={
            "type": "confirm", "decision": "confirm", "tool_call_id": "call-1",
        }),
    )
    await bridge._on_card_action(event)
    update_card.assert_awaited_once_with("om_message", final_card)


@pytest.mark.asyncio
async def test_confirmation_timeout_before_identity_lookup_finishes_keeps_final_card(monkeypatch):
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {"type": "confirm", "tool_call_id": "call-1"})
    identity_started = asyncio.Event()
    release_identity = asyncio.Event()

    async def identity(_open_id):
        identity_started.set()
        await release_identity.wait()
        return "E1001", None

    bridge.get_user_identity = identity
    notify = AsyncMock()
    monkeypatch.setattr(feishu_bridge, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"type": "confirm", "decision": "confirm"}),
    )
    callback = asyncio.create_task(bridge._on_card_action(event))
    try:
        await asyncio.wait_for(identity_started.wait(), timeout=1)
        tool_result = {"success": False, "result": "用户确认超时，工具未执行。"}
        await bridge._finish_confirmation_card(question_card, tool_result)
    finally:
        release_identity.set()
        await asyncio.wait_for(callback, timeout=1)
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(question_card.context, tool_result=tool_result),
    )
    notify.assert_not_awaited()
