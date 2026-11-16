from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

import xpeech.agent.compression as compression_module

from xpeech.agent.compression import ConversationCompressor
from xpeech.agent.helper import is_timestamped_user_message, strip_internal_message_metadata
from xpeech.agent.prompt.helper import remove_system_messages
from xpeech.provider.schema import LLMResponse


def make_response(content: str) -> LLMResponse:
    async def chunks():
        yield "content", content

    return LLMResponse(iter_mix_chunks=chunks())


@pytest.fixture(autouse=True)
def summary_budget(monkeypatch):
    monkeypatch.setattr(compression_module, "SUMMARY_TOKENS", 1)


class TestMessageUtils:
    def test_keep_messages_for_days_returns_complete_turns(self):
        messages = [
            {"role": "user", "timestamp": 1.0, "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "timestamp": 10 * 86400, "content": "recent"},
            {"role": "assistant", "content": "recent reply"},
        ]
        historical, recent = ConversationCompressor._keep_messages_for_days(7, messages)
        assert historical == messages[:2]
        assert recent == messages[2:]

    def test_remove_system_messages_keeps_user_requests_and_quoted_reminders(self):
        actual_messages = [
            {"role": "user", "timestamp": 123.0, "content": "<system-reminder>example</system-reminder>"},
            {"role": "user", "content": "Explain <system-reminder>example</system-reminder>"},
            {"role": "assistant", "content": "<system-reminder>example</system-reminder>"},
            {"role": "tool", "content": "<system-reminder>example</system-reminder>"},
        ]
        messages = [
            {"role": "system", "content": "fixed rules"},
            {"role": "user", "content": "<system-reminder>current identity</system-reminder>"},
            {"role": "user", "content": "<system-reminder>current memory</system-reminder>"},
            *actual_messages,
        ]

        assert remove_system_messages(messages) == actual_messages

    def test_strip_internal_metadata_does_not_mutate_history(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "timestamp": 123.0, "content": "hello"},
            {"role": "user", "content": "generated"},
            {"role": "user", "content": "<system-reminder>skill catalog</system-reminder>"},
        ]

        cleaned = strip_internal_message_metadata(messages)

        assert "timestamp" not in cleaned[1]
        assert messages[1]["timestamp"] == 123.0
        assert is_timestamped_user_message(messages[1])
        assert not is_timestamped_user_message(messages[2])
        assert cleaned[3] == {"role": "user", "content": "<system-reminder>skill catalog</system-reminder>"}
        assert messages[3] == cleaned[3]


class TestConversationCompressor:
    @pytest.mark.asyncio
    async def test_days_strategy_takes_priority_over_turn_limit(self):
        async def count_tokens(*, messages):
            return sum(len(message["content"]) for message in messages)

        chat = AsyncMock(return_value=make_response("x"))
        compressor = ConversationCompressor(
            chat=chat, max_accept_tokens=30, target_tokens=12,
            token_counter=count_tokens, recent_turns_to_keep=1,
        )
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "timestamp": 1.0, "content": "o" * 40},
            {"role": "user", "timestamp": 10 * 86400, "content": "new1"},
            {"role": "user", "timestamp": 11 * 86400, "content": "new2"},
        ]
        compressed = await compressor.compress(messages)
        assert compressed == [messages[0], {"role": "assistant", "content": "x"}, *messages[-2:]]
        chat.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("content,finish_reason", [("", "stop"), ("partial", "length")])
    async def test_invalid_summary_preserves_original_messages(self, content, finish_reason):
        response = make_response(content)
        response.set_finish_reason(finish_reason)
        chat = AsyncMock(return_value=response)

        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        compressor = ConversationCompressor(
            chat=chat, max_accept_tokens=30, target_tokens=20, token_counter=count_tokens,
        )
        messages = [{"role": "user", "timestamp": 1.0, "content": "a" * 40}]
        original = deepcopy(messages)

        with pytest.raises(ValueError):
            await compressor.compress(messages)

        assert messages == original
        chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_oversized_summary_warns_and_returns_result(self, monkeypatch):
        async def count_tokens(*, messages):
            return sum(len(message["content"]) for message in messages)

        warning = Mock()
        monkeypatch.setattr(compression_module.logger, "warning", warning)
        compressor = ConversationCompressor(
            chat=AsyncMock(return_value=make_response("x" * 30)),
            max_accept_tokens=40, target_tokens=20, token_counter=count_tokens,
        )
        compressed = await compressor.compress([
            {"role": "user", "timestamp": 1.0, "content": "a" * 50},
        ])
        assert compressed == [{"role": "assistant", "content": "x" * 30}]
        warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_oversized_fixed_context_stops_before_summarizing(self):
        async def count_tokens(*, messages):
            return 10 + sum(len(message["content"]) for message in messages)  # 包括工具定义。

        chat = AsyncMock()
        compressor = ConversationCompressor(
            chat=chat, max_accept_tokens=30, target_tokens=20, token_counter=count_tokens,
        )
        messages = [
            {"role": "system", "content": "fixed rules"},
            {"role": "user", "timestamp": 1.0, "content": "a" * 40},
        ]
        with pytest.raises(ValueError, match="固定上下文"):
            await compressor.compress(messages)
        chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_should_compress_at_maximum_threshold(self):
        async def count_tokens(*, messages):
            return len(messages)

        async def summarize(**_kwargs):
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            max_accept_tokens=3,
            target_tokens=1,
            token_counter=count_tokens,
        )

        assert not await compressor.should_compress([{}, {}])
        assert await compressor.should_compress([{}, {}, {}])

    @pytest.mark.asyncio
    async def test_summarizes_old_messages_and_keeps_recent_turn(self):
        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        summarized_messages = None
        summary_max_tokens = None

        async def summarize(**kwargs):
            nonlocal summarized_messages, summary_max_tokens
            summarized_messages = kwargs["messages"]
            summary_max_tokens = kwargs["parameters"].max_tokens
            return make_response("x")

        compressor = ConversationCompressor(
            chat=summarize,
            max_accept_tokens=5_000,
            target_tokens=12,
            token_counter=count_tokens,
        )
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "timestamp": 1.0, "content": "old1"},
            {"role": "assistant", "content": "old2"},
            {"role": "user", "timestamp": 2.0, "content": "new1"},
            {"role": "assistant", "content": "new2"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed[0] == {"role": "system", "content": "s"}
        assert compressed[1] == {"role": "assistant", "content": "x"}
        assert compressed[-1] == messages[-1]
        assert await compressor._is_within_target(compressed)
        assert summarized_messages is not None
        assert summarized_messages[-1]["content"] == "Please summarize the history messages."
        assert summary_max_tokens == 1

    @pytest.mark.asyncio
    async def test_level_one_summarizes_history_and_keeps_recent_days(self):
        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        summarize_calls = 0
        summarized_messages = None

        async def summarize(**kwargs):
            nonlocal summarize_calls, summarized_messages
            summarize_calls += 1
            summarized_messages = kwargs["messages"]
            return make_response("x")

        compressor = ConversationCompressor(
            chat=summarize,
            max_accept_tokens=10,
            target_tokens=7,
            token_counter=count_tokens,
        )
        now = 1_000_000.0
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "timestamp": now - 8 * 24 * 60 * 60, "content": "old"},
            {"role": "assistant", "content": "old"},
            {"role": "user", "timestamp": now, "content": "new"},
            {"role": "assistant", "content": "a"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed == [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "timestamp": now, "content": "new"},
            {"role": "assistant", "content": "a"},
        ]
        assert summarize_calls == 1
        assert summarized_messages is not None
        assert any(message["content"] == "old" for message in summarized_messages)
        assert all(message["content"] != "new" for message in summarized_messages)

    @pytest.mark.asyncio
    async def test_summary_budget_is_reserved_before_retaining_recent_turns(self, monkeypatch):
        monkeypatch.setattr(compression_module, "SUMMARY_TOKENS", 7)
        summarize_calls = 0

        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages if message.get("role") != "system")

        async def summarize(**_kwargs):
            nonlocal summarize_calls
            summarize_calls += 1
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            max_accept_tokens=10,
            target_tokens=10,
            token_counter=count_tokens,
        )
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "timestamp": 1.0, "content": "old1"},
            {"role": "assistant", "content": "old2"},
            {"role": "user", "timestamp": 2.0, "content": "new1"},
            {"role": "assistant", "content": "new2"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed == [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "summary"},
        ]
        assert summarize_calls == 1
        assert await compressor._is_within_target(compressed)

    @pytest.mark.asyncio
    async def test_compression_never_keeps_an_orphan_tool_message(self):
        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        async def summarize(**_kwargs):
            return make_response("x")

        compressor = ConversationCompressor(
            chat=summarize,
            max_accept_tokens=10,
            target_tokens=8,
            token_counter=count_tokens,
        )
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "timestamp": 1.0, "content": "old"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "tool"},
            {"role": "user", "timestamp": 2.0, "content": "new"},
            {"role": "assistant", "content": "ok"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed == [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "timestamp": 2.0, "content": "new"},
            {"role": "assistant", "content": "ok"},
        ]
        assert not any(message["role"] == "tool" for message in compressed)

    @pytest.mark.asyncio
    async def test_fallback_summarizes_an_oversized_latest_turn(self, monkeypatch):
        monkeypatch.setattr(compression_module, "SUMMARY_TOKENS", 7)
        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        async def summarize(**_kwargs):
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            max_accept_tokens=10,
            target_tokens=10,
            token_counter=count_tokens,
        )

        compressed = await compressor.compress(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "timestamp": 1.0, "content": "too-large"},
            ]
        )

        assert compressed[0] == {"role": "system", "content": "s"}
        assert compressed[1] == {"role": "assistant", "content": "summary"}
        assert await compressor._is_within_target(compressed)
