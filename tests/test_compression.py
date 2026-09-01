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
    async def test_level_two_summarizes_old_messages_and_keeps_recent_turn(self):
        async def count_tokens(*, messages):
            user_messages = [message for message in messages if is_timestamped_user_message(message)]
            if len(user_messages) == 4 and messages[0].get("role") != "system":
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
            recent_turns_to_keep=4,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "timestamp": 1.0, "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "timestamp": 2.0, "content": "two"},
            {"role": "user", "timestamp": 3.0, "content": "three"},
            {"role": "user", "timestamp": 4.0, "content": "four"},
            {"role": "user", "timestamp": 5.0, "content": "recent"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed[0] == {"role": "system", "content": "system"}
        assert compressed[1] == {"role": "assistant", "content": "history summary"}
        assert compressed[-1] == messages[-1]
        assert summarized_messages is not None
        assert summarized_messages[-1]["content"] == "Please summarize the history messages."
        assert summary_max_tokens == 100

    @pytest.mark.asyncio
    async def test_level_three_drops_oldest_messages_without_considering_roles(self):
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
            target_tokens=5,
            token_counter=count_tokens,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "timestamp": 1.0, "content": "12345678"},
            {"role": "assistant", "content": "abcdefgh"},
            {"role": "tool", "content": "ijkl"},
        ]

        compressed = await compressor.compress(messages)

        assert compressed == [
            {"role": "system", "content": "system"},
            {"role": "tool", "content": "ijkl"},
        ]
        assert summarize_calls == 0
