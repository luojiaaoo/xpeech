import json

import pytest

from xpeech.subapp.channel.feishu.formatting import OUTPUT_EVENT_TYPES, FeishuEventFormatter
from xpeech.subapp.channel.schema import ChatEvent, ChatEventType


def tool_event(event_type, entries):
    return ChatEvent(event=event_type, context=json.dumps(entries, ensure_ascii=False))


@pytest.mark.parametrize("event_type", [ChatEventType.THINKING, ChatEventType.TOOL_CALL])
def test_collapsed_streaming_card_uses_shared_output_configuration(monkeypatch, event_type):
    output = {
        **OUTPUT_EVENT_TYPES[event_type],
        "title": "自定义标题",
        "expanded_title": "自定义展开标题",
        "expanded": True,
        "text_size": "normal",
        "text_align": "center",
        "icon": {"tag": "standard_icon", "token": "robot_filled", "color": "red"},
    }
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, event_type, output)
    message, _ = FeishuEventFormatter()._format_chat_event(ChatEvent(event=event_type, context="结果"))
    card = message["card"]
    assert card["config"]["streaming_mode"] is True
    [panel] = card["body"]["elements"]
    assert panel["tag"] == "collapsible_panel"
    assert panel["expanded"] is True
    assert panel["header"] == {
        "title": {"tag": "plain_text", "content": "自定义标题"},
        "expanded_title": {"tag": "plain_text", "content": "自定义展开标题"},
    }
    [body] = panel["elements"]
    assert body["text_size"] == "normal"
    assert body["text_align"] == "center"
    assert body["icon"] == output["icon"]


def test_tool_results_are_stream_data_without_a_separate_card_format():
    formatter = FeishuEventFormatter()
    event = tool_event(ChatEventType.TOOL_CALL_RESULT, [["id", "shell", "result", 0]])
    assert ChatEventType.TOOL_CALL_RESULT not in OUTPUT_EVENT_TYPES
    assert formatter._format_chat_event(event) is None
    assert formatter._format_tool_event(event) == "**shell**\nresult"


def test_truncation_limits_are_configured_in_output_event_types(monkeypatch):
    output = {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL],
        "max_argument_chars": 10,
        "max_result_chars": 20,
        "max_title_chars": 35,
        "max_chars": 40,
    }
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, output)
    formatter = FeishuEventFormatter()
    summary = formatter._format_tool_event(tool_event(ChatEventType.TOOL_CALL, [["id", "echo", "参" * 30]]))
    text = formatter._format_tool_event(tool_event(ChatEventType.TOOL_CALL_RESULT, [["id", "echo", "果" * 30, 0]]))
    assert summary == "echo " + "参" * 9 + "…"
    assert text == "**echo**\n" + "果" * 19 + "…"
    message, _ = formatter._format_chat_event(
        ChatEvent(event=ChatEventType.TOOL_CALL, context=text * 10),
        title=summary * 10,
    )
    [panel] = message["card"]["body"]["elements"]
    title = panel["header"]["title"]["content"]
    content = panel["elements"][0]["content"]
    assert len(title) == 35 and title.endswith("…")
    assert len(content) == 40 and content.endswith("…")


@pytest.mark.parametrize("value", ["", None, ["item"], {"found": [1, 2]}])
def test_empty_and_structured_results_are_preserved(value):
    text = FeishuEventFormatter()._format_tool_event(
        tool_event(ChatEventType.TOOL_CALL_RESULT, [["id", "search", value, 0]]),
    )
    expected = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    assert text == "**search**\n" + expected


def test_default_limits_bound_a_large_unicode_card():
    message, _ = FeishuEventFormatter()._format_chat_event(
        ChatEvent(event=ChatEventType.TOOL_CALL, context="😃" * 30_000),
        title="😃" * 5_000,
    )
    assert len(json.dumps(message, ensure_ascii=False).encode("utf-8")) < 30_000


@pytest.mark.parametrize("raw", ["not JSON", '{"unexpected":true}', '[["id"]]'])
@pytest.mark.parametrize("event_type", [ChatEventType.TOOL_CALL, ChatEventType.TOOL_CALL_RESULT])
def test_malformed_tool_events_remain_visible(raw, event_type):
    assert FeishuEventFormatter()._format_tool_event(ChatEvent(event=event_type, context=raw)) == raw
