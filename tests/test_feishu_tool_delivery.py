import asyncio
import json
from copy import deepcopy
from itertools import count
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lark_channel import SendResult

from xpeech.agent.loop import AgentLoop
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent.tool_executor import ToolExecutor
from xpeech.agent.tools import shell as shell_module
from xpeech.agent.tools.execution_request import ExecutionDecision, build_execution_request_tool
from xpeech.provider.schema import ToolCallRequest
from xpeech.subapp.channel.feishu import bridge as bridge_module
from xpeech.subapp.channel.feishu import delivery
from xpeech.subapp.channel.feishu.bridge import FeishuBridge
from xpeech.subapp.channel.feishu.channel import XpeechFeishuChannel
from xpeech.subapp.channel.feishu.delivery import ToolProgress
from xpeech.subapp.channel.feishu.formatting import OUTPUT_EVENT_TYPES
from xpeech.subapp.channel.schema import ChatEvent, ChatEventType, Message, TextData
from xpeech.utils.security.sandbox.schema import CommandResult


def tool_event(event_type, entries):
    return ChatEvent(event=event_type, context=json.dumps(entries))


def bridge_with_channel():
    bridge = object.__new__(FeishuBridge)
    bridge.chat_url = "http://backend.test"
    bridge.session_update_message_id = {}
    bridge.question_cards = {}
    bridge.add_reaction = AsyncMock()
    card_ids = count()
    bridge.channel = SimpleNamespace(
        create_card_instance=AsyncMock(side_effect=lambda _card: f"card-{next(card_ids)}"),
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
@pytest.mark.parametrize("with_assistant", [False, True])
async def test_each_tool_has_its_own_card_and_result_updates_by_id(monkeypatch, with_assistant):
    bridge = bridge_with_channel()
    channel = bridge.channel
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)

    async def assistant_text():
        yield "continuing"

    async def events(*_args, **_kwargs):
        yield tool_event(ChatEventType.TOOL_CALL, [
            {"tool_call_id": "slow", "name": "shell", "arguments": {"command": "slow", "execution_title": "slow"}},
            {"tool_call_id": "fast", "name": "shell", "arguments": {"command": "fast", "execution_title": "fast"}},
        ])
        assert channel.create_card_instance.await_count == 2
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": "fast", "name": "shell", "result": "fast result", "success": True,
        })
        first = channel.update_card_element_content.await_args
        assert first.args == ("card-1", "main", "fast result")
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": "slow", "name": "shell", "result": "slow result", "success": True,
        })
        if with_assistant:
            yield ChatEvent(event=ChatEventType.ASSISTANT, context=assistant_text())
        yield tool_event(ChatEventType.TOOL_CALL, [
            {"tool_call_id": "fast", "name": "echo", "arguments": {"text": "second round"}},
        ])
        channel.finish_streaming_card.assert_any_await("card-0", sequence=2)
        channel.finish_streaming_card.assert_any_await("card-1", sequence=2)
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": "fast", "name": "echo", "result": "second result", "success": True,
        })

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    await bridge.consume([message()])
    assert channel.create_card_instance.await_count == (4 if with_assistant else 3)
    for index, command in enumerate(["slow", "fast"]):
        card = channel.create_card_instance.await_args_list[index].args[0]
        assert card["config"]["streaming_mode"] is True
        [panel] = card["body"]["elements"]
        assert panel["tag"] == "collapsible_panel" and panel["expanded"] is False
        title = panel["header"]["title"]["content"]
        assert command in title and ("fast" if command == "slow" else "slow") not in title
        assert title == f'> 工具调用 · shell {{"command":"{command}","execution_title":"{command}"}}'
        assert panel["elements"][0]["content"] == "执行中…"
    second_card_id = "card-3" if with_assistant else "card-2"
    channel.update_card_element_content.assert_any_await(
        second_card_id, "main", "second result", sequence=1,
    )
    channel.finish_streaming_card.assert_any_await(second_card_id, sequence=2)
    assert channel.finish_streaming_card.await_count == (4 if with_assistant else 3)
    channel.patch_card_element.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_delivery_obeys_shared_style_and_truncation_configuration(monkeypatch):
    output = {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL],
        "title": "自定义工具标题",
        "text_size": "normal",
        "text_align": "center",
        "icon": None,
        "max_argument_chars": 8,
        "max_result_chars": 12,
    }
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, output)
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    bridge = bridge_with_channel()
    progress = ToolProgress()
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL, [{"tool_call_id": "id", "name": "echo", "arguments": "a" * 30}]), progress)
    await bridge.send_tool_progress(
        "chat",
        tool_event(
            ChatEventType.TOOL_CALL_RESULT,
            {"tool_call_id": "id", "name": "echo", "result": "x" * 30, "duration_seconds": 0},
        ),
        progress,
    )

    card = bridge.channel.create_card_instance.await_args.args[0]
    [element] = card["body"]["elements"]
    assert element["header"]["title"]["content"] == "> 自定义工具标题 · echo aaaaaaa…"
    assert element["tag"] == "collapsible_panel"
    assert "a" * 30 not in element["header"]["title"]["content"]
    body = element["elements"][0]
    assert body["text_size"] == "normal"
    assert body["text_align"] == "center"
    assert "icon" not in body
    bridge.channel.update_card_element_content.assert_awaited_once_with(
        "card-0",
        "main",
        "x" * 11 + "…",
        sequence=1,
    )


@pytest.mark.asyncio
async def test_rss_batch_results_are_escaped_before_card_updates(monkeypatch):
    bridge = bridge_with_channel()
    channel = bridge.channel
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    rss = '<?xml version="1.0"?><rss><channel><title>A &amp; B</title></channel></rss>'

    async def reject_raw_markup(_card_id, _element_id, content, *, sequence):
        assert "<" not in content and ">" not in content

    channel.update_card_element_content.side_effect = reject_raw_markup

    async def assistant_text():
        yield "All RSS results received"

    async def events(*_args, **_kwargs):
        yield tool_event(
            ChatEventType.TOOL_CALL,
            [{"tool_call_id": str(i), "name": "web_fetch", "arguments": {"url": "https://example.test/rss"}} for i in range(10)],
        )
        for i in range(10):
            yield tool_event(
                ChatEventType.TOOL_CALL_RESULT,
                {"tool_call_id": str(i), "name": "web_fetch", "result": rss, "duration_seconds": 0},
            )
        yield ChatEvent(event=ChatEventType.ASSISTANT, context=assistant_text())

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    await bridge.consume([message()])

    updates = [call for call in channel.update_card_element_content.await_args_list if call.args[0] in {f"card-{i}" for i in range(10)}]
    assert len(updates) == 10
    assert {call.args[0] for call in updates} == {f"card-{i}" for i in range(10)}
    assert channel.create_card_instance.await_count == 11
    assert all("&lt;rss&gt;" in call.args[2] for call in updates)
    channel.update_card_element_content.assert_any_await(
        "card-10", "main", "All RSS results received", sequence=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("failed_card", [None, "card-0", "card-1"])
async def test_tool_card_stream_is_finished_when_consumption_fails_or_is_cancelled(
    monkeypatch, cancel, failed_card,
):
    bridge = bridge_with_channel()
    waiting = asyncio.Event()

    async def finish(card_id, *, sequence):
        if card_id == failed_card:
            raise RuntimeError("card close failed")

    bridge.channel.finish_streaming_card.side_effect = finish

    async def events(*_args, **_kwargs):
        yield tool_event(ChatEventType.TOOL_CALL, [
            {"tool_call_id": call_id, "name": "shell", "arguments": {}} for call_id in ["a", "b"]
        ])
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

    bridge.channel.finish_streaming_card.assert_any_await("card-0", sequence=1)
    bridge.channel.finish_streaming_card.assert_any_await("card-1", sequence=1)
    assert bridge.channel.finish_streaming_card.await_count == 2


@pytest.mark.asyncio
async def test_tool_card_finish_failure_does_not_interrupt_the_next_batch(monkeypatch):
    bridge = bridge_with_channel()
    bridge.channel.finish_streaming_card.side_effect = [RuntimeError("card close failed"), None, None]

    async def events(*_args, **_kwargs):
        yield tool_event(ChatEventType.TOOL_CALL, [
            {"tool_call_id": call_id, "name": "shell", "arguments": {}} for call_id in ["a", "b"]
        ])
        yield tool_event(ChatEventType.TOOL_CALL, [
            {"tool_call_id": "c", "name": "echo", "arguments": {}},
        ])

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    await bridge.consume([message()])

    assert bridge.channel.create_card_instance.await_count == 3
    assert [call.args[0] for call in bridge.channel.finish_streaming_card.await_args_list] == [
        "card-0", "card-1", "card-2",
    ]


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


@pytest.mark.asyncio
async def test_confirmation_cards_are_updated_by_tool_call_id_with_out_of_order_results(monkeypatch):
    bridge = bridge_with_channel()
    channel = bridge.channel
    channel.send = AsyncMock(side_effect=[
        SendResult(success=True, message_id="question-b"),
        SendResult(success=True, message_id="question-a"),
    ])
    channel.update_card = AsyncMock(return_value=SendResult(success=True))
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)

    async def events(*_args, **_kwargs):
        yield tool_event(ChatEventType.TOOL_CALL, [
            {"tool_call_id": "a", "name": "shell", "arguments": {"command": "A"}},
            {"tool_call_id": "b", "name": "shell", "arguments": {"command": "B"}},
        ])
        for call_id in ["b", "a"]:
            yield tool_event(ChatEventType.QUESTION, {
                "type": "confirm", "title": f"确认 {call_id}", "description": "执行命令",
                "tool_call_id": call_id, "tool_name": "shell",
            })
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": "b", "name": "shell", "result": "RuntimeError: B failed", "success": False, "duration_seconds": 0,
        })
        yield tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": "a", "name": "shell", "result": "A done", "success": True, "duration_seconds": 1,
        })

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    await bridge.consume([message()])
    updates = channel.update_card.await_args_list
    assert len(updates) == 2
    assert [call.args[0] for call in updates] == ["question-b", "question-a"]
    assert updates[0].args[1]["header"]["template"] == "red"
    assert updates[1].args[1]["header"]["template"] == "blue"
    assert "RuntimeError: B failed" in json.dumps(updates[0].args[1])
    assert "A done" in json.dumps(updates[1].args[1])
    assert "RuntimeError" not in json.dumps(updates[1].args[1])
    for update in updates:
        buttons = update.args[1]["body"]["elements"][-1]["columns"]
        assert all(column["elements"][0]["disabled"] for column in buttons)
    assert bridge.question_cards == {}
    for call, call_id in zip(channel.send.await_args_list, ["b", "a"], strict=True):
        card = call.args[1]["card"]
        button = card["body"]["elements"][-1]["columns"][0]["elements"][0]
        assert button["behaviors"][0]["value"]["type"] == "confirm"
        assert button["behaviors"][0]["value"]["tool_call_id"] == call_id


@pytest.mark.asyncio
async def test_repeated_tool_result_replaces_matching_id_instead_of_appending(monkeypatch):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    progress = ToolProgress()
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL, [
        {"tool_call_id": "a", "name": "shell", "arguments": {}},
        {"tool_call_id": "b", "name": "shell", "arguments": {}},
    ]), progress)
    for call_id, result in [("b", "old B"), ("a", "A"), ("b", "new B")]:
        await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": call_id, "name": "shell", "result": result, "duration_seconds": 0,
        }), progress)
    content = bridge.channel.update_card_element_content.await_args.args[2]
    assert "new B" in content and "old B" not in content
    assert "A" not in content
    assert bridge.channel.update_card_element_content.await_args.args[0] == "card-1"
    bridge.channel.update_card_element_content.assert_any_await("card-0", "main", "A", sequence=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("expanded", [False, True])
async def test_failure_only_colors_matching_title_and_preserves_folded_values(monkeypatch, expanded):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL], "expanded": expanded, "expanded_title": "收起工具结果",
    })
    progress = ToolProgress()
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL, [
        {"tool_call_id": call_id, "name": "shell", "arguments": {
            "command": "<same>&", "execution_title": "相同标题", "execution_description": "相同说明",
        }} for call_id in ["a", "b"]
    ]), progress)
    original_cards = deepcopy([call.args[0] for call in bridge.channel.create_card_instance.await_args_list])
    for call_id, success in [("b", False), ("a", True)]:
        await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": call_id, "name": "shell", "result": "<output>&", "success": success,
        }), progress)
    updates = bridge.channel.update_card_element_content.await_args_list
    assert [call.args[0] for call in updates] == ["card-1", "card-0"]
    assert updates[0].args[2] == updates[1].args[2]
    assert "&lt;output&gt;&amp;" in updates[0].args[2]
    assert '<font' not in updates[0].args[2]
    patch = bridge.channel.patch_card_element.await_args
    assert patch.args[:2] == ("card-1", "panel")
    assert patch.args[2] == {"header": {
        "title": {"tag": "markdown", "content": '<font color="red">&gt; 工具调用 · shell {"command":"&lt;same&gt;&amp;","execution_title":"相同标题","execution_description":"相同说明"}</font>'},
        "expanded_title": {"tag": "markdown", "content": '<font color="red">∨ 工具调用 · shell {"command":"&lt;same&gt;&amp;","execution_title":"相同标题","execution_description":"相同说明"}</font>'},
    }}
    assert patch.kwargs["sequence"] == 2
    assert [call.args[0] for call in bridge.channel.create_card_instance.await_args_list] == original_cards
    # 成功重试恢复原始标题，不重建卡片或设置展开状态。
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
        "tool_call_id": "b", "name": "shell", "result": "recovered", "success": True,
    }), progress)
    assert bridge.channel.patch_card_element.await_args.args == (
        "card-1", "panel", {"header": original_cards[1]["body"]["elements"][0]["header"]},
    )
    assert bridge.channel.update_card_element_content.await_args.args[2].endswith("recovered")
    assert bridge.channel.create_card_instance.await_count == 2


@pytest.mark.asyncio
async def test_failed_result_truncation_does_not_add_color_tags(monkeypatch):
    bridge = bridge_with_channel()
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL], "max_chars": 15,
    })
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
        "tool_call_id": "a", "name": "shell", "result": "<&" * 20, "success": False,
    }), ToolProgress())
    card = bridge.channel.create_card_instance.await_args.args[0]
    panel = card["body"]["elements"][0]
    assert panel["elements"][0]["content"] == "&lt;&amp;&lt;…"
    assert '<font color="red">' in panel["header"]["title"]["content"]


@pytest.mark.asyncio
async def test_failed_tool_xml_code_keeps_fences_and_only_colors_its_title(monkeypatch):
    bridge = bridge_with_channel()
    progress = ToolProgress()
    code = '```xml\n<root>&amp; <at id="all" /></root>\n```\n'
    for call_id, success in [("a", False), ("b", True)]:
        await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": call_id, "name": "shell", "result": code + "正文 <outside/>", "success": success,
        }), progress)
    cards = [call.args[0] for call in bridge.channel.create_card_instance.await_args_list]
    for card in cards:
        assert card["body"]["elements"][0]["elements"][0]["content"] == (
            code + "正文 &lt;outside/&gt;"
        )
    assert '<font color="red">' in cards[0]["body"]["elements"][0]["header"]["title"]["content"]
    assert '<font' not in cards[1]["body"]["elements"][0]["header"]["title"]["content"]


@pytest.mark.asyncio
async def test_streamed_xml_code_is_preserved_across_split_fences_and_tags(monkeypatch):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)

    async def chunks():
        for token in ["正文 <outside/>\n`", "``xml\n<root>&", "amp;</root>\n`", "``\n<outside/>"]:
            yield token

    message, stream = bridge._format_chat_event(ChatEvent(event=ChatEventType.ASSISTANT, context=chunks()))
    await bridge.send("chat", message, stream)
    updates = [call.args[2] for call in bridge.channel.update_card_element_content.await_args_list]
    prefix = "正文 &lt;outside/&gt;\n"
    assert updates == [
        prefix + "`",
        prefix + "```xml\n<root>&",
        prefix + "```xml\n<root>&amp;</root>\n`",
        prefix + "```xml\n<root>&amp;</root>\n```\n&lt;outside/&gt;",
    ]


@pytest.mark.asyncio
async def test_truncated_tool_code_block_does_not_swallow_the_next_result(monkeypatch):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL], "max_result_chars": 20,
    })
    progress = ToolProgress()
    code = "```xml\n<root>" + "x" * 30 + "</root>\n```"
    for call_id, result, success in [("a", code, True), ("b", "<error/>", False)]:
        await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
            "tool_call_id": call_id, "name": "shell", "result": result, "success": success,
        }), progress)
    cards = [call.args[0] for call in bridge.channel.create_card_instance.await_args_list]
    assert cards[0]["body"]["elements"][0]["elements"][0]["content"] == code[:19] + "…\n```"
    assert cards[1]["body"]["elements"][0]["elements"][0]["content"] == "&lt;error/&gt;"


@pytest.mark.asyncio
async def test_shell_command_failure_only_colors_its_card_title(tmp_path, monkeypatch):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)

    async def run(command, **_kwargs):
        return (
            CommandResult(b"", b"bash: line 1: aaaa: command not found\n", 127)
            if command == "aaaa" else CommandResult(b"aaaa\n", b"", 0)
        )

    sandbox = SimpleNamespace(idle_timeout=60, run=run)
    monkeypatch.setattr(shell_module.sandbox, "SandboxExecutor", lambda *_args, **_kwargs: sandbox)
    shell = build_execution_request_tool(
        shell_module.build_shell_tools(tmp_path, skills_loader=SkillsLoader(tmp_path)),
        lambda _: ExecutionDecision.ALLOW,
    )
    calls = [
        ToolCallRequest(id=call_id, name="shell", arguments={
            "command": command, "execution_description": "测试命令退出状态",
        })
        for call_id, command in [("failed", "aaaa"), ("succeeded", "echo aaaa")]
    ]
    agent = AgentLoop.__new__(AgentLoop)
    agent.tool_executor = ToolExecutor(workspace=tmp_path, max_result_chars=10_000, mapping_tool_call_funcs={"shell": shell})
    results = {}

    async def events(*_args, **_kwargs):
        async for event in agent.tool_call(SimpleNamespace(tool_calls=calls, content=""), [], 0, "alice", "session", "Alice"):
            if event["event"] == "tool_call_result":
                result = json.loads(event["context"])
                results[result["tool_call_id"]] = result["success"]
            yield ChatEvent(**event)

    monkeypatch.setattr(bridge_module, "iter_chat_events", events)
    await bridge.consume([message()])

    assert results == {"failed": False, "succeeded": True}
    updates = {call.args[0]: call.args[2] for call in bridge.channel.update_card_element_content.await_args_list}
    assert "RuntimeError: STDERR:" in updates["card-0"]
    assert "Exit code: 127" in updates["card-0"]
    assert "Exit code: 0" in updates["card-1"]
    assert all('<font' not in content for content in updates.values())
    patch = bridge.channel.patch_card_element.await_args
    assert patch.args[:2] == ("card-0", "panel")
    assert '<font color="red">' in patch.args[2]["header"]["title"]["content"]


@pytest.mark.asyncio
async def test_empty_tool_output_is_not_left_running(monkeypatch):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    progress = ToolProgress()
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL, [
        {"tool_call_id": "empty", "name": "shell", "arguments": {}},
    ]), progress)
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
        "tool_call_id": "empty", "name": "shell", "result": "", "success": True,
    }), progress)
    assert bridge.channel.update_card_element_content.await_args.args[2] == "（无输出）"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["shell", "read_file", "custom_tool"])
async def test_all_tools_show_unmodified_arguments_in_header_and_only_output_when_expanded(monkeypatch, tool_name):
    bridge = bridge_with_channel()
    monkeypatch.setattr(delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    progress = ToolProgress()
    arguments = {
        "command": "mkdir -p test", "timeout_ms": 10000,
        "execution_title": "创建 test 文件夹", "execution_description": "创建名为 test 的文件夹。",
    }
    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL, [
        {"tool_call_id": "create-test", "name": tool_name, "arguments": arguments},
    ]), progress)
    card = bridge.channel.create_card_instance.await_args.args[0]
    [summary] = card["body"]["elements"]
    title = f'工具调用 · {tool_name} ' + json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    assert summary["header"] == {
        "title": {"tag": "plain_text", "content": f"> {title}"},
        "expanded_title": {"tag": "plain_text", "content": f"∨ {title}"},
    }
    assert summary["elements"][0]["content"] == "执行中…"
    assert "execution_title" in title and "execution_description" in title

    await bridge.send_tool_progress("chat", tool_event(ChatEventType.TOOL_CALL_RESULT, {
        "tool_call_id": "create-test", "name": tool_name, "result": "Exit code: 0", "success": True,
    }), progress)
    content = bridge.channel.update_card_element_content.await_args.args[2]
    assert content == "Exit code: 0"
