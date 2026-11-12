from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Iterator
from html import escape
from typing import Any, NotRequired, TypedDict

from ..schema import ChatEvent, ChatEventType


def plain_text(content: str) -> dict[str, str]:
    return {"tag": "plain_text", "content": content}


_CODE_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)")


def _markdown_parts(content: str, *, close_unfinished: bool = False) -> Iterator[tuple[str, bool]]:
    """分出围栏代码块；流式输出中尚未闭合的代码块延伸到当前末尾。"""
    fence = ""
    lines: list[str] = []
    for line in content.splitlines(keepends=True):
        match = _CODE_FENCE.match(line)
        if fence:
            lines.append(line)
            if (match and match[1][0] == fence[0] and len(match[1]) >= len(fence)
                    and not match[2].strip()):
                yield "".join(lines), True
                lines = []
                fence = ""
        elif match and (match[1][0] != "`" or "`" not in match[2]):
            if lines:
                yield "".join(lines), False
            lines = [line]
            fence = match[1]
        else:
            lines.append(line)
    if lines:
        text = "".join(lines)
        if fence and close_unfinished:
            # 独立工具结果可能被截断；补齐围栏，避免吞掉下一条结果。
            text += ("" if text.endswith(("\r", "\n")) else "\n") + fence
        yield text, bool(fence)


def ensure_markdown(content: str, max_chars: int | None = None) -> str:
    """保留代码块原文，仅转义正文中的 XML/HTML，避免被解析为飞书标签。"""
    parts = [(text if code else escape(text, quote=False), code) for text, code in _markdown_parts(content)]
    content = "".join(text for text, _ in parts)
    if max_chars is not None and len(content) > max_chars:
        if max_chars <= 0:
            return ""
        remaining = max_chars - 1
        truncated = []
        for text, code in parts:
            prefix = text[:remaining]
            if not code and len(prefix) < len(text):
                # 只在转义过的正文中移除残缺实体，代码中的 & 是原始字符。
                last_entity = prefix.rfind("&")
                if last_entity > prefix.rfind(";"):
                    prefix = prefix[:last_entity]
            truncated.append(prefix)
            remaining -= len(text)
            if remaining <= 0:
                break
        content = "".join(truncated) + "…"
    return content


class OutputEventType(TypedDict):
    content: str | None | type(Ellipsis)
    text_size: NotRequired[str | dict[str, Any]]
    text_align: NotRequired[str]
    icon: NotRequired[dict[str, Any] | None]
    title: NotRequired[str]
    expanded_title: NotRequired[str]
    expanded: NotRequired[bool]
    max_title_chars: NotRequired[int]
    max_chars: NotRequired[int]
    max_argument_chars: NotRequired[int]
    max_result_chars: NotRequired[int]


# 图标：https://open.feishu.cn/document/feishu-cards/enumerations-for-icons
# None 按事件内容格式化；... 表示流式；str 表示固定文本。
OUTPUT_EVENT_TYPES: dict[ChatEventType, OutputEventType] = {
    ChatEventType.THINKING: {
        "content": ...,
        "title": "查看思考过程",
        "expanded_title": "收起思考过程",
        "expanded": False,
        "text_size": "notation",
        "text_align": "left",
        "icon": {
            "tag": "standard_icon",
            "token": "tab-more_outlined",
            "color": "green",
        },
    },
    ChatEventType.ASSISTANT: {
        "content": ...,
        "text_size": "normal",
        "text_align": "left",
        "icon": {
            "tag": "standard_icon",
            "token": "robot_filled",
            "color": "red",
        },
    },
    ChatEventType.ERROR: {
        "content": None,
        "text_size": "normal",
        "text_align": "left",
        "icon": {
            "tag": "standard_icon",
            "token": "warning_outlined",
            "color": "red",
        },
    },
    ChatEventType.COMMAND: {
        "content": None,
        "text_size": "notation",
        "text_align": "center",
        "icon": {
            "tag": "standard_icon",
            "token": "command_outlined",
            "color": "turquoise",
        },
    },
    ChatEventType.TOOL_CALL: {
        "content": ...,
        "title": "工具调用",
        "expanded": False,
        "text_size": "notation",
        "text_align": "left",
        "icon": {
            "tag": "standard_icon",
            "token": "select-up_outlined",
            "color": "wathet",
        },
        "max_title_chars": 800,
        "max_chars": 6_000,
        "max_argument_chars": 160,
        "max_result_chars": 2_000,
    },
    ChatEventType.TOKEN_USAGE: {
        "content": None,
        "text_size": "notation",
        "text_align": "left",
        "icon": None,
    },
}


class FeishuEventFormatter:
    """根据标准化聊天事件构建飞书卡片。"""

    def _format_chat_event_content(self, event: ChatEvent) -> str:
        content = event.context
        if event.event == ChatEventType.ASSISTANT:
            return "输出中..."
        if event.event == ChatEventType.COMMAND:
            return f"**[命令]** {content}"
        if event.event == ChatEventType.THINKING:
            return "思考中..."
        if event.event == ChatEventType.ERROR:
            return f"**[错误]** {content}"
        if event.event == ChatEventType.TOOL_CALL:
            return content or "执行中…"
        if event.event == ChatEventType.TOKEN_USAGE:
            return f"**[词元]** {self._format_token_usage(content)}"
        return f"[{event.event}]\n{content}"

    def _format_chat_event(
        self,
        event: ChatEvent,
        *,
        title: str | None = None,
    ) -> tuple[dict[str, Any], AsyncIterator[str] | None] | None:
        if event.event not in OUTPUT_EVENT_TYPES:
            return None

        output = OUTPUT_EVENT_TYPES[event.event]
        content = output["content"]
        text = content if isinstance(content, str) else self._format_chat_event_content(event)
        text = ensure_markdown(text, output.get("max_chars"))
        element = {
            "tag": "markdown",
            "margin": "0px 0px 0px 0px",
            "content": text,
            "text_size": output.get("text_size", "normal"),
            "text_align": output.get("text_align", "left"),
        }
        if output.get("icon") is not None:
            element["icon"] = output["icon"]

        card_element = element
        if "title" in output:
            header = {
                "title": plain_text(
                    self._truncate(
                        output["title"] + (f" · {title}" if title else ""),
                        output.get("max_title_chars"),
                    )
                ),
            }
            if "expanded_title" in output:
                header["expanded_title"] = plain_text(output["expanded_title"])
            card_element = {
                "tag": "collapsible_panel",
                "element_id": "panel",
                "expanded": output.get("expanded", False),
                "header": header,
                "elements": [element],
            }

        if content is ...:
            element["element_id"] = "main"
            return (
                {
                    "card": {
                        "schema": "2.0",
                        "config": {
                            "streaming_mode": True,
                            "summary": {"content": ""},
                        },
                        "body": {"elements": [card_element]},
                    }
                },
                event.context if isinstance(event.context, AsyncIterator) else None,
            )

        return (
            {
                "card": {
                    "schema": "2.0",
                    "body": {"elements": [card_element]},
                }
            },
            None,
        )

    def _format_token_usage(self, context: str) -> str:
        data: dict[str, str] = json.loads(context)
        return " | ".join(f"{key}：{value}" for key, value in data.items())

    @staticmethod
    def _truncate(text: str, max_chars: int | None) -> str:
        if max_chars is None or len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 1)] + "…"

    def _format_tool_title(self, call: dict[str, Any]) -> str:
        """生成单个工具调用的标题。"""
        arguments = call["arguments"]
        copy = arguments if isinstance(arguments, dict) else {}
        title = copy.get("execution_title") or copy.get("execution_description") or call["name"]
        return self._truncate(title, OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL].get("max_argument_chars"))

    def _format_tool_result(self, call: dict[str, Any], result: Any) -> str:
        """生成单个工具调用的结果文本。"""
        output = OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL]
        name = self._format_tool_title(call)
        arguments = call["arguments"]
        description = arguments.get("execution_description") if isinstance(arguments, dict) else None
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        text = "".join(part for part, _ in _markdown_parts(
            self._truncate(text, output.get("max_result_chars")), close_unfinished=True,
        ))
        if description and description != name:
            text = f"{description}\n\n{text}"
        return f"**{name}**\n{text}"
