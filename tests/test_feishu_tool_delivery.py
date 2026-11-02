import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lark_channel import SendResult

from xpeech.subapp.channel.feishu import bridge as bridge_module
from xpeech.subapp.channel.feishu import delivery
from xpeech.subapp.channel.feishu.bridge import FeishuBridge
from xpeech.subapp.channel.feishu.channel import XpeechFeishuChannel
from xpeech.subapp.channel.feishu.delivery import ToolProgress
from xpeech.subapp.channel.feishu.formatting import OUTPUT_EVENT_TYPES
from xpeech.subapp.channel.schema import ChatEvent, ChatEventType, Message, TextData


def tool_event(event_type, entries):
    return ChatEvent(event=event_type, context=json.dumps(entries))


def bridge_with_channel():
    bridge = object.__new__(FeishuBridge)
    bridge.chat_url = "http://backend.test"
    bridge.session_update_message_id = {}
    bridge.add_reaction = AsyncMock()
    bridge.channel = SimpleNamespace(
        create_card_instance=AsyncMock(side_effect=["tools-card", "assistant-card"]),
        send_card_by_reference=AsyncMock(return_value=SendResult(success=True, message_id="message")),
        update_card_element_content=AsyncMock(),
        patch_card_element=AsyncMock(),
        finish_streaming_card=AsyncMock(),
    )
    return bridge


def message():
    return Message(
        message_id="input",
        chat_id="chat",
        user_id="alice",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )


@pytest.mark.asyncio
async def test_all_batches_share_one_collapsed_streaming_card(monkeypatch):
    bridge = bridge_with_channel()
    channel = bridge.channel
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)

    async def assistant_text():
        yield "continuing"

    async def events(*_args):
        yield tool_event(
            ChatEventType.TOOL_CALL,
            [
                ["slow", "shell", {"command": "slow"}],
                ["fast", "shell", {"command": "fast"}],
            ],
        )
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, [["fast", "shell", "fast result", 0]])
        first = channel.update_card_element_content.await_args
        assert first.args[:2] == ("tools-card", "main")
        assert "fast result" in first.args[2]
        assert "slow result" not in first.args[2]
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, [["slow", "shell", "slow result", 1]])
        yield ChatEvent(event=ChatEventType.ASSISTANT, context=assistant_text())
        yield tool_event(ChatEventType.TOOL_CALL, [["fast", "echo", {"text": "second round"}]])
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, [["fast", "echo", "second result", 0]])

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    await bridge.consume([message()])

    assert channel.create_card_instance.await_count == 2
    card = channel.create_card_instance.await_args_list[0].args[0]
    assert card["config"]["streaming_mode"] is True
    [panel] = card["body"]["elements"]
    assert panel["tag"] == "collapsible_panel"
    assert panel["expanded"] is False
    assert "slow" in panel["header"]["title"]["content"]
    assert "fast" in panel["header"]["title"]["content"]
    channel.patch_card_element.assert_awaited_once()
    patch = channel.patch_card_element.await_args
    assert patch.args[:2] == ("tools-card", "panel")
    assert list(patch.args[2]) == ["header"]
    title = patch.args[2]["header"]["title"]["content"]
    assert all(text in title for text in ("slow", "fast", "echo", "second round"))
    updates = [call for call in channel.update_card_element_content.await_args_list if call.args[0] == "tools-card"]
    assert len(updates) == 4
    assert [call.kwargs["sequence"] for call in updates] == [1, 2, 4, 5]
    assert patch.kwargs["sequence"] == 3
    assert all(text in updates[-1].args[2] for text in ("fast result", "slow result", "second result"))
    assert "执行中…" not in updates[-1].args[2]
    assert sum(call.args == ("chat", "tools-card") for call in channel.send_card_by_reference.await_args_list) == 1
    assert any(
        call.args == ("tools-card",) and call.kwargs == {"sequence": 6}
        for call in channel.finish_streaming_card.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("collapsed", [False, True])
async def test_tool_delivery_obeys_shared_style_and_truncation_configuration(monkeypatch, collapsed):
    output = {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL],
        "title": "自定义工具标题",
        "text_size": "normal",
        "text_align": "center",
        "icon": None,
        "max_argument_chars": 8,
        "max_result_chars": 12,
    }
    if not collapsed:
        output.pop("title")
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, output)
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    bridge = bridge_with_channel()
    progress = ToolProgress()
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL, [["id", "echo", "a" * 30]]), progress)
    await bridge.send_tool_progress(
        "chat", tool_event(ChatEventType.TOOL_CALL_RESULT, [["id", "echo", "x" * 30, 0]]), progress
    )

    card = bridge.channel.create_card_instance.await_args.args[0]
    [element] = card["body"]["elements"]
    if collapsed:
        assert "自定义工具标题" in element["header"]["title"]["content"]
        assert "echo aaaaaaa…" in element["header"]["title"]["content"]
    else:
        assert element["tag"] == "markdown"
    body = element["elements"][0] if collapsed else element
    assert body["text_size"] == "normal"
    assert body["text_align"] == "center"
    assert "icon" not in body
    bridge.channel.update_card_element_content.assert_awaited_once_with(
        "tools-card",
        "main",
        "**echo**\n" + "x" * 11 + "…",
        sequence=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_tool_card_stream_is_finished_when_consumption_fails_or_is_cancelled(monkeypatch, cancel):
    bridge = bridge_with_channel()
    waiting = asyncio.Event()

    async def events(*_args):
        yield tool_event(ChatEventType.TOOL_CALL, [["call", "shell", {}]])
        waiting.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError("backend disconnected")

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    task = asyncio.create_task(bridge.consume([message()]))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=1)
    finally:
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task

    bridge.channel.finish_streaming_card.assert_awaited_once_with("tools-card", sequence=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [False, True])
async def test_header_patch_uses_sdk_partial_update_and_checks_response(monkeypatch, success):
    channel = XpeechFeishuChannel(app_id="cli_test", app_secret="test-secret")
    patch = AsyncMock(return_value=SimpleNamespace(success=lambda: success, code=0 if success else 1, msg="test"))
    monkeypatch.setattr(channel.client.cardkit.v1.card_element, "apatch", patch)
    header = {"header": {"title": {"tag": "plain_text", "content": "工具调用"}}}
    if success:
        await channel.patch_card_element("card", "tools", header, sequence=3)
    else:
        with pytest.raises(RuntimeError, match="update card element failed"):
            await channel.patch_card_element("card", "tools", header, sequence=3)
    request = patch.await_args.args[0]
    assert request.card_id == "card"
    assert request.element_id == "tools"
    assert request.request_body.sequence == 3
    assert json.loads(request.request_body.partial_element) == header
