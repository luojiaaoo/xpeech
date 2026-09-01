import pytest

from xpeech.agent.compression import ConversationCompressor
from xpeech.agent.helper import is_timestamped_user_message, strip_internal_message_metadata
from xpeech.provider.schema import LLMResponse


def make_response(content: str) -> LLMResponse:
    async def chunks():
        yield "content", content

    return LLMResponse(iter_mix_chunks=chunks())


class TestMessageUtils:
    def test_strip_internal_metadata_does_not_mutate_history(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "timestamp": 123.0, "content": "hello"},
            {"role": "user", "content": "generated"},
        ]

        cleaned = strip_internal_message_metadata(messages)

        assert "timestamp" not in cleaned[1]
        assert messages[1]["timestamp"] == 123.0
        assert is_timestamped_user_message(messages[1])
        assert not is_timestamped_user_message(messages[2])

    def test_keep_messages_for_days_returns_historical_and_recent_segments(self):
        messages = [
            {"role": "user", "timestamp": 1.0, "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "timestamp": 10 * 24 * 60 * 60, "content": "recent"},
            {"role": "assistant", "content": "recent reply"},
        ]

        historical_messages, recent_messages = ConversationCompressor._keep_messages_for_days(7, messages)

        assert historical_messages == messages[:2]
        assert recent_messages == messages[2:]


class TestConversationCompressor:
    @pytest.mark.asyncio
    async def test_should_compress_at_maximum_threshold(self):
        async def count_tokens(*, messages):
            return len(messages)

        async def summarize(**_kwargs):
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            summary_tokens=100,
            max_accept_tokens=3,
            target_tokens=1,
            token_counter=count_tokens,
        )

        assert not await compressor.should_compress([{}, {}])
        assert await compressor.should_compress([{}, {}, {}])

    @pytest.mark.asyncio
    async def test_level_two_summarizes_old_messages_and_keeps_recent_turn(self):
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
            summary_tokens=100,
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
        assert summary_max_tokens == 100

    @pytest.mark.asyncio
    async def test_level_one_summarizes_history_and_keeps_the_latest_days(self):
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
            summary_tokens=100,
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
    async def test_summary_that_exceeds_target_falls_back_to_complete_recent_suffix(self):
        summarize_calls = 0

        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages if message.get("role") != "system")

        async def summarize(**_kwargs):
            nonlocal summarize_calls
            summarize_calls += 1
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            summary_tokens=100,
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
            {"role": "user", "timestamp": 2.0, "content": "new1"},
            {"role": "assistant", "content": "new2"},
        ]
        assert summarize_calls == 1
        assert await compressor._is_within_target(compressed)

    @pytest.mark.asyncio
    async def test_compression_never_keeps_an_orphan_tool_message(self):
        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        async def summarize(**_kwargs):
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            summary_tokens=100,
            max_accept_tokens=10,
            target_tokens=7,
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
            {"role": "user", "timestamp": 2.0, "content": "new"},
            {"role": "assistant", "content": "ok"},
        ]
        assert await compressor._is_within_target(compressed)

    @pytest.mark.asyncio
    async def test_fallback_summarizes_an_oversized_latest_turn(self):
        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages)

        async def summarize(**_kwargs):
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            summary_tokens=100,
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
