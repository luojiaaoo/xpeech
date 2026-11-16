"""按分项估算展示用的上下文占用，并生成 token_usage 事件。"""

import json
from dataclasses import dataclass
from typing import Any

from ..provider.litellm_provider import LiteLLMProvider
from ..utils.helper import token_counter
from .helper import strip_internal_message_metadata
from .prompt.helper import remove_system_messages


def format_token_count(value: int) -> str:
    """统一换算为 k，并向下取整。"""
    return f"{value // 1_000}k"


@dataclass
class TokenUsage:
    """保留组装时的分项统计；展示估算独立于压缩预算的整包计数。"""

    provider: LiteLLMProvider
    tools: list[str]
    compression_threshold: int
    context_tokens: dict[str, int]

    async def build_event(
        self,
        messages: list[dict[str, Any]],
        *,
        input_tokens: int,
        output_tokens: int,
        model_call_count: int,
        duration_s: float,
    ) -> dict[str, str]:
        """补充最终对话和工具的独立计数，汇总中文统计字段。"""
        tool_tokens = await self.provider.chat_count_tool_tokens(tools=self.tools)
        components = {
            **self.context_tokens,
            "对话消息": await token_counter(strip_internal_message_metadata(remove_system_messages(messages))),
            "系统工具": tool_tokens["system_tools"],
            "MCP 工具": tool_tokens["mcp_tools"],
        }
        total = sum(components.values())
        limit = self.compression_threshold
        ratio = min(1, total / limit)

        def usage_part(tokens: int) -> dict[str, int | float]:
            return {"词元数": tokens, "占比": min(1, tokens / limit)}

        details = {
            **{name: usage_part(tokens) for name, tokens in components.items()},
            "剩余空间": usage_part(max(0, limit - total)),
        }
        context_usage = f"{ratio:.2%} ({format_token_count(total)} / {format_token_count(limit)})"
        if total > self.compression_threshold * 0.9:
            context_usage += " ♻️ 即将达到最大上下文"
        return {
            "event": "token_usage",
            "context": json.dumps({
                "上下文使用率": context_usage,
                "上下文明细": details,
                "输入 Token": format_token_count(input_tokens),
                "输出 Token": format_token_count(output_tokens),
                "总 Token": format_token_count(input_tokens + output_tokens),
                "会话时长": f"{duration_s:.0f}秒",
                "大模型请求次数": str(model_call_count),
            }, ensure_ascii=False),
        }
