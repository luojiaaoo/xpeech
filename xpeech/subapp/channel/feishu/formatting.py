from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, NotRequired, TypedDict

from ..schema import ChatEvent, ChatEventType
from .cards import plain_text


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
        text = self._truncate(text, output.get("max_chars"))
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

    def _format_tool_event(self, event: ChatEvent) -> str:
        """将工具事件转换为摘要或结果文本，样式统一使用 TOOL_CALL 配置。"""
        output = OUTPUT_EVENT_TYPES[ChatEventType.TOOL_CALL]
        is_result = event.event == ChatEventType.TOOL_CALL_RESULT
        limit = output.get("max_result_chars" if is_result else "max_argument_chars")
        try:
            entries = json.loads(event.context)
        except (json.JSONDecodeError, TypeError):
            entries = None
        if not isinstance(entries, list) or not all(
            isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[1], str) for entry in entries
        ):
            return self._truncate(str(event.context), limit)
        lines = []
        for _, name, value, *_ in entries:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            name = self._truncate(name, output.get("max_argument_chars"))
            text = self._truncate(text, limit)
            lines.append(f"**{name}**\n{text}" if is_result else f"{name} {text}")
        return ("\n\n" if is_result else " | ").join(lines)
