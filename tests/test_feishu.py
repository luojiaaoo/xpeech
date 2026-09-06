import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from lark_channel import (
    CardActionEvent,
    CardActionPayload,
    Conversation,
    EventOperator,
    FileContent,
    Identity,
    ImageContent,
    InboundMessage,
    MediaContent,
    PostContent,
    SendResult,
    TextContent,
)

import xpeech.channel.feishu.bridge as feishu_bridge
import xpeech.channel.feishu.delivery as feishu_delivery
import xpeech.channel.feishu.inbound as feishu_inbound
from xpeech.channel.feishu.bridge import FeishuBridge
from xpeech.channel.feishu.cards import FINISH_CARD_CONTENT, build_feishu_markdown_card
from xpeech.channel.schema import ChatEvent, ChatEventType, FileData, Message, TextData


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
    bridge.session_update_message_id = {}
    bridge.get_user_identity = AsyncMock(return_value=(employee_no, email))
    return bridge


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


def test_background_markdown_card_contains_result_content():
    assert build_feishu_markdown_card("**scheduled result**") == {
        "schema": "2.0",
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": "**scheduled result**",
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                }
            ]
        },
    }


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
        },
    )
    client = SimpleNamespace(get=AsyncMock(return_value=response))
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_background"))
    bridge = _bridge_with_channel(SimpleNamespace(send=send))

    assert await bridge.poll_background_message_once(client) is True

    client.get.assert_awaited_once()
    assert client.get.await_args.args == ("http://backend.test/background_message",)
    assert client.get.await_args.kwargs["params"] == {"channel": "feishu"}
    assert client.get.await_args.kwargs["headers"]["authorization"].startswith("Bearer ")
    send.assert_awaited_once_with(
        to="ou_receiver",
        message={"card": build_feishu_markdown_card("**scheduled result**")},
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
    bridge.session_tasks = {}
    bridge.poll_sessions = lambda: wait_until_cancelled("sessions")
    bridge.poll_background_messages = lambda: wait_until_cancelled("background")

    await bridge.start()

    assert started == {"sessions", "background"}
    assert cancelled == {"sessions", "background"}
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
    assert message.session_id == "E1001"
    assert message.sender_name == "Alice"
    assert message.timestamp == 123
    assert message.content == [TextData(text="hello")]
    assert message.session_metadata == {
        "channel": "feishu",
        "open_id": "ou_sender",
        "email": "alice@example.com",
    }


@pytest.mark.asyncio
async def test_parse_text_message_allows_missing_email():
    bridge = _bridge_with_channel(SimpleNamespace(), email=None)

    message = await bridge._parse_msg(_inbound_message(TextContent(text="hello")))

    assert message.session_id == "E1001"
    assert message.session_metadata == {
        "channel": "feishu",
        "open_id": "ou_sender",
    }


@pytest.mark.asyncio
async def test_on_message_logs_when_message_is_added(monkeypatch: pytest.MonkeyPatch):
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        session_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=123,
        session_metadata={},
    )
    bridge = object.__new__(FeishuBridge)
    bridge.receive_queues = {}
    bridge._parse_msg = AsyncMock(return_value=message)
    info = MagicMock()
    monkeypatch.setattr(feishu_bridge, "logger", SimpleNamespace(info=info))

    await bridge._on_message(_inbound_message(TextContent(text="hello")))

    assert bridge.receive_queues["E1001"].get_nowait() == message
    info.assert_called_once_with(
        "Feishu message added session_id={} chat_id={} message_id={} sender_name={} queue_size={}",
        "E1001",
        "oc_chat",
        "om_message",
        "Alice",
        1,
    )


@pytest.mark.asyncio
async def test_on_message_ignores_unsupported_message(monkeypatch: pytest.MonkeyPatch):
    bridge = _bridge_with_channel(SimpleNamespace())
    bridge.receive_queues = {}
    warning = MagicMock()
    monkeypatch.setattr(feishu_bridge, "logger", SimpleNamespace(warning=warning))
    inbound_message = _inbound_message(TextContent(text="hello"), chat_type="group")

    await bridge._on_message(inbound_message)

    assert bridge.receive_queues == {}
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
        session_id="E1001",
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
        "Feishu message sending session_id={} sender_name={} message_type={}",
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
async def test_parse_post_reads_locale_document_and_preserves_attachment_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "E1001" / "img_post.png"
    download_resource = AsyncMock(return_value=downloaded_path)
    bridge = _bridge_with_channel(SimpleNamespace(download_resource_to_file=download_resource))
    post = PostContent(
        post={
            "zh_cn": {
                "title": "",
                "content": [
                    [
                        {"tag": "text", "text": "before"},
                        {"tag": "a", "text": "link", "href": "https://example.com"},
                    ],
                    [{"tag": "img", "image_key": "img_post"}],
                    [{"tag": "text", "text": "after"}],
                ],
            }
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
    event = CardActionEvent(
        message_id="om_message",
        chat_id="oc_chat",
        operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(form_value={"answer": "typed"}),
        raw={"event": {"action": {"form_value": {"answer": "raw"}}}},
    )

    await bridge._on_card_action(event)

    notify_question.assert_awaited_once_with(
        "E1001",
        {"answer": "typed"},
        "http://backend.test",
    )
    bridge.get_user_identity.assert_awaited_once_with("ou_sender")
    update_card.assert_awaited_once_with("om_message", FINISH_CARD_CONTENT)


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
            session_id="feishu_oc_chat",
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
        session_id="feishu_oc_chat",
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
        session_id="feishu_oc_chat",
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
async def test_consume_streams_assistant_chunks_as_reply(monkeypatch: pytest.MonkeyPatch):
    async def chunks():
        for chunk in ["hello", " ", "world"]:
            yield chunk

    async def streaming_chat_events(*_args):
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
    bridge.receive_queues = {"E1001": asyncio.Queue()}
    bridge.receive_queues["E1001"].put_nowait(
        Message(
            message_id="om_message",
            chat_id="oc_chat",
            session_id="E1001",
            sender_name="Alice",
            content=[TextData(text="hello")],
            timestamp=0,
            session_metadata={},
        )
    )
    bridge.session_update_message_id["E1001"] = "om_progress"

    await bridge.consume("E1001", idle_timeout=0)

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

    async def busy_chat_events(*_args):
        raise httpx.HTTPStatusError("Conflict", request=request, response=response)
        yield

    monkeypatch.setattr(feishu_bridge, "iter_chat_events", busy_chat_events)
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_busy"))
    channel = SimpleNamespace(add_reaction=AsyncMock(), send=send)
    bridge = _bridge_with_channel(channel)
    bridge.receive_queues = {"E1001": asyncio.Queue()}
    bridge.receive_queues["E1001"].put_nowait(
        Message(
            message_id="om_message",
            chat_id="oc_chat",
            session_id="E1001",
            sender_name="Alice",
            content=[TextData(text="hello")],
            timestamp=0,
            session_metadata={},
        )
    )

    await bridge.consume("E1001", idle_timeout=0)

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
