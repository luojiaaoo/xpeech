import json
from html import unescape

import pytest

from xpeech.subapp.channel.feishu.formatting import OUTPUT_EVENT_TYPES, FeishuEventFormatter, ensure_markdown
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
        "title": {"tag": "plain_text", "content": "> 自定义标题"},
        "expanded_title": {"tag": "plain_text", "content": "∨ 自定义标题"},
    }
    [body] = panel["elements"]
    assert body["text_size"] == "normal"
    assert body["text_align"] == "center"
    assert body["icon"] == output["icon"]


def test_tool_results_are_stream_data_without_a_separate_card_format():
    formatter = FeishuEventFormatter()
    event = tool_event(
        ChatEventType.TOOL_CALL_RESULT,
        {"tool_call_id": "id", "name": "shell", "result": "result", "duration_seconds": 0},
    )
    assert ChatEventType.TOOL_CALL_RESULT not in OUTPUT_EVENT_TYPES
    assert formatter._format_chat_event(event) is None
    assert formatter._format_tool_result("result") == "result"


@pytest.mark.parametrize("raw", [
    '<?xml version="1.0"?><rss><channel><title>A &amp; B</title></channel></rss>',
    '<feed xmlns="http://www.w3.org/2005/Atom"><title>新闻</title></feed>',
    '<html><body><![CDATA[<at id="all">test</at>]]></body></html>',
])
def test_tool_card_displays_markup_as_text_without_changing_tool_results(raw):
    formatter = FeishuEventFormatter()
    event = tool_event(
        ChatEventType.TOOL_CALL_RESULT,
        {"tool_call_id": "id", "name": "web_fetch", "result": raw, "duration_seconds": 0},
    )
    result_text = formatter._format_tool_result(raw)
    assert result_text == raw
    message, _ = formatter._format_chat_event(ChatEvent(event=ChatEventType.TOOL_CALL, context=result_text))
    content = message["card"]["body"]["elements"][0]["elements"][0]["content"]
    assert "<" not in content and ">" not in content
    assert unescape(content) == result_text
    assert json.loads(event.context)["result"] == raw


def test_tool_card_truncation_keeps_html_entities_complete(monkeypatch):
    monkeypatch.setitem(OUTPUT_EVENT_TYPES, ChatEventType.TOOL_CALL, {
        **OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL], "max_chars": 8,
    })
    message, _ = FeishuEventFormatter()._format_chat_event(
        ChatEvent(event=ChatEventType.TOOL_CALL, context="<&" * 100),
    )
    content = message["card"]["body"]["elements"][0]["elements"][0]["content"]
    assert len(content) <= 8
    assert content == "&lt;…"


@pytest.mark.parametrize("fence", ["```", "````", "~~~"])
@pytest.mark.parametrize("closed", [False, True])
def test_xml_code_blocks_preserve_source_while_surrounding_text_is_escaped(fence, closed):
    code = f'{fence}xml\n<root value="a & b">&amp; <at id="all" /></root>\n'
    suffix = ""
    if closed:
        code += fence + "\n"
        suffix = "正文 <outside>&"
    raw = "**标题** <outside>&\n" + code + suffix
    assert ensure_markdown(raw) == (
        "**标题** &lt;outside&gt;&amp;\n" + code
        + ("正文 &lt;outside&gt;&amp;" if closed else "")
    )


def test_shorter_or_mismatched_fences_do_not_end_a_code_block():
    code = "````xml\n<one/>\n```\n<two/>\n~~~~\n<three/>\n````\n"
    assert ensure_markdown(code + "<outside/>") == code + "&lt;outside/&gt;"


def test_multiple_code_blocks_and_crlf_line_endings_are_preserved():
    raw = "```xml\r\n<first/>&\r\n```\r\n<outside/>\r\n```html\r\n<div/>\r\n```"
    assert ensure_markdown(raw) == raw.replace("<outside/>", "&lt;outside/&gt;")


def test_truncation_preserves_literal_ampersands_inside_code_and_complete_entities_outside():
    code = "```xml\n<root>&literal</root>\n```\n"
    limit = len("```xml\n<root>&lit") + 1
    assert ensure_markdown(code, limit) == "```xml\n<root>&lit…"
    assert ensure_markdown(code + "<&" * 10, len(code) + 8) == code + "&lt;…"


@pytest.mark.parametrize("event_type", [ChatEventType.ERROR, ChatEventType.COMMAND, ChatEventType.TOKEN_USAGE])
def test_non_streaming_card_escapes_dynamic_markup_and_preserves_markdown(event_type):
    formatter = FeishuEventFormatter()
    raw = '**标题**\n[链接](https://example.test)\n<rss>甲 &amp; 乙</rss>'
    context = json.dumps({"tokens": raw}) if event_type == ChatEventType.TOKEN_USAGE else raw
    event = ChatEvent(event=event_type, context=context)
    message, _ = formatter._format_chat_event(event)
    content = message["card"]["body"]["elements"][0]["content"]
    assert "<" not in content and ">" not in content
    assert unescape(content) == formatter._format_chat_event_content(event)
    assert "**标题**" in content and "[链接](https://example.test)" in content


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
    summary = formatter._format_tool_title({"name": "echo", "arguments": "参" * 30})
    text = formatter._format_tool_result("果" * 30)
    assert summary == "echo " + "参" * 9 + "…"
    assert text == "果" * 19 + "…"
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
    text = FeishuEventFormatter()._format_tool_result(value)
    expected = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    assert text == expected


def test_default_limits_bound_a_large_unicode_card():
    message, _ = FeishuEventFormatter()._format_chat_event(
        ChatEvent(event=ChatEventType.TOOL_CALL, context="😃" * 30_000),
        title="😃" * 5_000,
    )
    assert len(json.dumps(message, ensure_ascii=False).encode("utf-8")) < 30_000
