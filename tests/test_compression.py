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
    async def test_level_one_truncates_only_old_tool_results(self):
        summarize_calls = 0

        async def count_tokens(*, messages):
            return sum(len(message.get("content", "")) for message in messages if message.get("role") == "tool")

        async def summarize(**_kwargs):
            nonlocal summarize_calls
            summarize_calls += 1
            return make_response("summary")

        compressor = ConversationCompressor(
            chat=summarize,
            summary_tokens=100,
            max_accept_tokens=10_000,
            target_tokens=3_000,
            token_counter=count_tokens,
            tool_result_max_chars=1_000,
        )
        old_result = "a" * 1_500
        recent_result = "b" * 1_500
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "timestamp": 1.0, "content": "one"},
            {"role": "tool", "content": old_result},
            {"role": "user", "timestamp": 2.0, "content": "two"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "timestamp": 3.0, "content": "three"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "timestamp": 4.0, "content": "four"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "timestamp": 5.0, "content": "five"},
            {"role": "tool", "content": recent_result},
        ]

        compressed = await compressor.compress(messages)

        assert compressed[2]["content"] == old_result[:1_000]
        assert compressed[-1]["content"] == recent_result
        assert messages[2]["content"] == old_result
        assert summarize_calls == 0

    @pytest.mark.asyncio
    async def test_level_three_summarizes_old_messages_and_keeps_recent_turn(self):
        async def count_tokens(*, messages):
            if len(messages) == 1 and messages[0].get("role") == "user":
                return 0
            return 10_000

        summarized_messages = None
        summary_max_tokens = None

        async def summarize(**kwargs):
            nonlocal summarized_messages, summary_max_tokens
            summarized_messages = kwargs["messages"]
            summary_max_tokens = kwargs["parameters"].max_tokens
            return make_response("history summary")

        compressor = ConversationCompressor(
            chat=summarize,
            summary_tokens=100,
            max_accept_tokens=5_000,
            target_tokens=1_000,
            token_counter=count_tokens,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "timestamp": 1.0, "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "timestamp": 10 * 24 * 60 * 60, "content": "recent"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed[0] == {"role": "system", "content": "system"}
        assert compressed[1] == {"role": "assistant", "content": "history summary"}
        assert compressed[-1] == messages[-1]
        assert summarized_messages is not None
        assert summarized_messages[-1]["content"] == "Please summarize the history messages."
        assert summary_max_tokens == 100
