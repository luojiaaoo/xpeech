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


# 图标：https://open.feishu.cn/document/feishu-cards/enumerations-for-icons
# None 不发送；... 表示流式；str 表示直接输出。
OUTPUT_EVENT_TYPES: dict[ChatEventType, OutputEventType] = {
    ChatEventType.THINKING: {
        "content": ...,
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
        "content": "我需要调用工具处理一下。",
        "text_size": "notation",
        "text_align": "center",
        "icon": {
            "tag": "standard_icon",
            "token": "select-up_outlined",
            "color": "wathet",
        },
    },
    ChatEventType.TOOL_CALL_RESULT: {
        "content": "工具处理完成，我继续整理结果。",
        "text_size": "notation",
        "text_align": "center",
        "icon": {
            "tag": "standard_icon",
            "token": "bitableform_outlined",
            "color": "yellow",
        },
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
            return f"**[调用工具]** {self._format_tool_call_event(content)}"
        if event.event == ChatEventType.TOOL_CALL_RESULT:
            return f"**[工具调用结果]** {self._format_json_context(content)}"
        if event.event == ChatEventType.TOKEN_USAGE:
            return f"**[词元]** {self._format_token_usage(content)}"
        return f"[{event.event}]\n{content}"

    def _format_chat_event(
        self,
        event: ChatEvent,
    ) -> tuple[dict[str, Any], AsyncIterator[str] | None] | None:
        if event.event not in OUTPUT_EVENT_TYPES:
            return None

        output = OUTPUT_EVENT_TYPES[event.event]
        content = output["content"]
        text = content if isinstance(content, str) else self._format_chat_event_content(event)
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
        if event.event == ChatEventType.THINKING:
            card_element = {
                "tag": "collapsible_panel",
                "expanded": False,
                "header": {
                    "title": plain_text("查看思考过程"),
                    "expanded_title": plain_text("收起思考过程"),
                },
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
                event.context,
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

    def _format_tool_call_event(self, content: str) -> str:
        try:
            tool_calls = json.loads(content)
        except json.JSONDecodeError:
            return content

        if not isinstance(tool_calls, list):
            return self._format_json_value(tool_calls)

        lines: list[str] = []
        for index, tool_call in enumerate(tool_calls, start=1):
            if isinstance(tool_call, list) and len(tool_call) >= 3:
                lines.append(f"{index}. {tool_call[1]}\n```json\n{self._format_json_value(tool_call[2])}\n```")
            else:
                lines.append(f"{index}. {self._format_json_value(tool_call)}")
        return "\n\n".join(lines)

    def _format_json_context(self, context: str) -> str:
        try:
            value = json.loads(context)
        except json.JSONDecodeError:
            return context
        return f"```json\n{self._format_json_value(value)}\n```"

    def _format_token_usage(self, context: str) -> str:
        data: dict[str, str] = json.loads(context)
        return " | ".join(f"{key}：{value}" for key, value in data.items())

    def _format_json_value(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=4)
