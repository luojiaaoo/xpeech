from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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

from xpeech.channel import feishu
from xpeech.channel.feishu import FINISH_CARD_CONTENT, FeishuBridge
from xpeech.channel.schema import ChatEventType, FileData, TextData


def _inbound_message(content, *, message_id: str = "om_message") -> InboundMessage:
    return InboundMessage(
        id=message_id,
        create_time=123_000,
        conversation=Conversation(chat_id="oc_chat", chat_type="p2p"),
        sender=Identity(open_id="ou_sender", display_name="Alice"),
        content=content,
    )


def _bridge_with_channel(channel) -> FeishuBridge:
    bridge = object.__new__(FeishuBridge)
    bridge.chat_url = "http://backend.test"
    bridge.channel = channel
    bridge.session_update_message_id = {}
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


@pytest.mark.asyncio
async def test_parse_text_message_preserves_sender_and_session_metadata():
    bridge = _bridge_with_channel(SimpleNamespace())

    message = await bridge._parse_msg(_inbound_message(TextContent(text="hello")))

    assert message.message_id == "om_message"
    assert message.chat_id == "oc_chat"
    assert message.session_id == "feishu_oc_chat"
    assert message.sender_name == "Alice"
    assert message.timestamp == 123
    assert message.content == [TextData(text="hello")]
    assert message.session_metadata == {"sender_id": "ou_sender"}


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
    monkeypatch.setattr(feishu, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "feishu_oc_chat" / downloaded_name
    download_resource = AsyncMock(return_value=downloaded_path)
    bridge = _bridge_with_channel(SimpleNamespace(download_resource_to_file=download_resource))

    message = await bridge._parse_msg(_inbound_message(content))

    expected_kwargs = {
        "resource_type": resource_type,
        "message_id": "om_message",
        "dest_dir": tmp_path / "feishu_oc_chat",
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
    monkeypatch.setattr(feishu, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "feishu_oc_chat" / "img_post.png"
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
        dest_dir=tmp_path / "feishu_oc_chat",
    )


@pytest.mark.asyncio
async def test_card_action_uses_typed_form_value(monkeypatch: pytest.MonkeyPatch):
    notify_question = AsyncMock()
    monkeypatch.setattr(feishu, "notify_question", notify_question)
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
        "feishu_oc_chat",
        {"answer": "typed"},
        "http://backend.test",
    )
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
            message_type=ChatEventType.THINKING,
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
        message_type=ChatEventType.THINKING,
    )

    assert bridge.session_update_message_id == {"feishu_oc_chat": "om_progress"}
