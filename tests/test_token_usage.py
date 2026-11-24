import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xpeech.agent.token_usage import TokenUsage, format_token_count


@pytest.mark.parametrize(("tokens", "expected"), [
    (0, "0k"), (999, "0k"), (1000, "1k"), (1234, "1k"),
    (999994, "999k"), (999995, "999k"), (1234567, "1234k"),
])
def test_format_token_count_preserves_unit_boundaries(tokens, expected):
    assert format_token_count(tokens) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("system_tokens", [10, 146380, 210000])
async def test_usage_event_uses_display_counts_without_recounting_full_context(monkeypatch, system_tokens):
    from xpeech.agent import token_usage as usage_module

    counter = AsyncMock(return_value=100)
    monkeypatch.setattr(usage_module, "token_counter", counter)
    provider = SimpleNamespace(
        default_model="test", default_context_token=200_000,
        chat_count_tool_tokens=AsyncMock(return_value={"system_tools": 200, "mcp_tools": 300}),
    )
    context_tokens = {"系统提示词": system_tokens, "用户上下文": 20}
    usage = TokenUsage(provider=provider, compression_threshold=147_000, context_tokens=context_tokens)
    event = await usage.build_event([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "<system-reminder>identity</system-reminder>"},
        {"role": "user", "content": "hello", "timestamp": 1.0},
        {"role": "assistant", "content": "answer"},
    ], input_tokens=1000, output_tokens=234, model_call_count=2, duration_s=3)

    counter.assert_awaited_once_with([
        {"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"},
    ])
    provider.chat_count_tool_tokens.assert_awaited_once_with()
    metrics = json.loads(event["context"])
    assert event["event"] == "token_usage"
    details = metrics["上下文明细"]
    assert set(details) == {"系统提示词", "用户上下文", "对话消息", "系统工具", "MCP 工具", "剩余空间"}
    assert details["系统提示词"]["词元数"] == system_tokens
    assert details["用户上下文"] == {"词元数": 20, "占比": 20 / 147000}
    assert all(0 <= part["占比"] <= 1 for part in details.values())
    total = system_tokens + 620
    assert metrics["上下文使用率"].startswith(f"{min(1, total / 147000):.2%} ({total // 1000}k / 147k)")
    assert details["剩余空间"]["词元数"] == max(0, 147000 - total)
    if system_tokens > 200000:
        assert metrics["上下文使用率"].startswith("100.00%")
        assert details["系统提示词"]["占比"] == 1
        assert details["剩余空间"] == {"词元数": 0, "占比": 0}
    assert (metrics["输入 Token"], metrics["输出 Token"], metrics["总 Token"]) == (1000, 234, 1234)
    assert "原始词元数" not in metrics
    assert (metrics["会话时长"], metrics["大模型请求次数"]) == ("3秒", 2)
    assert context_tokens == {"系统提示词": system_tokens, "用户上下文": 20}
