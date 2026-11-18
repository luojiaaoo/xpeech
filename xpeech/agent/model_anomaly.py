"""模型异常行为的非阻塞兜底。"""

import json
from typing import Any

from ..provider.schema import ToolCallRequest


class ModelAnomalyGuard:
    """在一次 run 内跟踪模型异常行为，并生成模型可见的提醒。"""

    REPEATED_TOOL_CALL_THRESHOLD = 3
    REPEATED_TOOL_CALL_PROMPT = (
        "You have called {tool_name} with the same input {observed_count} times in a row.\n"
        "Do not repeat the exact same tool call again unless the user explicitly asked you to retry it unchanged.\n"
        "Use the existing result to take a different next step, explain the blocker, or ask the user for guidance."
    )

    def __init__(self) -> None:
        self._previous_tool_call: tuple[str, str] | None = None
        self._repeated_tool_call_count = 0
        self._pending_reminders: list[dict[str, Any]] = []

    def prepare_messages(self, messages_yaml: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将待发送提醒临时追加到本次请求末尾，消费后不写回历史。"""
        messages = [*messages_yaml, *self._pending_reminders]
        self._pending_reminders = []
        return messages

    def observe_tool_calls(self, tool_calls: list[ToolCallRequest]) -> None:
        """跟踪调用并暂存下一次请求的提醒；无工具调用时清空状态。"""
        if not tool_calls:
            self._previous_tool_call = None
            self._repeated_tool_call_count = 0
            self._pending_reminders = []
            return

        for tool_call in tool_calls:
            signature = (
                tool_call.name,
                json.dumps(tool_call.arguments, sort_keys=True, ensure_ascii=False),
            )
            if signature == self._previous_tool_call:
                self._repeated_tool_call_count += 1
            else:
                self._previous_tool_call = signature
                self._repeated_tool_call_count = 1

            if self._repeated_tool_call_count == self.REPEATED_TOOL_CALL_THRESHOLD:
                reminder = {
                    "role": "user",
                    "content": self.REPEATED_TOOL_CALL_PROMPT.format(
                        tool_name=tool_call.name,
                        observed_count=self._repeated_tool_call_count,
                    ),
                }
                self._pending_reminders.append(reminder)
