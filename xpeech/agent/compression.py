from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from loguru import logger

from ..provider.schema import LLMParameters, LLMResponse
from ..utils.helper import token_counter
from .helper import is_timestamped_user_message
from .prompt.compress import SUMMARY_PROMPT
from .prompt.helper import prepend_system_messages, split_system_messages

Message = dict[str, Any]


class ConversationCompressor:
    """负责执行对话压缩策略，不处理长期记忆。"""

    def __init__(
        self,
        *,
        chat: Callable[..., Awaitable[LLMResponse]],
        summary_tokens: int,
        max_accept_tokens: int,
        target_tokens: int,
        token_counter: Callable[..., Awaitable[int]] = token_counter,
        recent_turns_to_keep: int = 20,
    ) -> None:
        """初始化压缩器及各级压缩阈值。"""
        self._chat = chat
        self._summary_tokens = summary_tokens
        self._max_accept_tokens = max_accept_tokens
        self._target_tokens = target_tokens
        self._token_counter = token_counter
        self._recent_turns_to_keep = recent_turns_to_keep

    async def should_compress(self, messages: list[Message]) -> bool:
        """判断当前消息的令牌数是否达到压缩阈值。"""
        total_tokens = await self._token_counter(messages=messages)
        return total_tokens >= self._max_accept_tokens

    async def _is_within_target(self, messages: list[Message]) -> bool:
        """判断当前消息的令牌数是否已低于目标阈值。"""
        total_tokens = await self._token_counter(messages=messages)
        return total_tokens < self._target_tokens

    @staticmethod
    def _split_recent_user_messages(messages: list[Message], keep_count: int) -> int:
        """返回需要保留的最近若干条用户消息的起始索引。"""
        count = 0
        for index in range(len(messages) - 1, -1, -1):
            if not is_timestamped_user_message(messages[index]):
                continue
            count += 1
            if count == keep_count:
                return index
        return 0

    @staticmethod
    def _keep_messages_for_days(days: int, messages: list[Message]) -> list[Message]:
        """保留最近指定天数内的历史消息及全部系统消息。"""
        system_messages, history_messages = split_system_messages(messages)
        last_timestamp = None
        split_index = 0
        for index in range(len(history_messages) - 1, -1, -1):
            if not is_timestamped_user_message(history_messages[index]):
                continue
            if last_timestamp is None:
                last_timestamp = history_messages[index]["timestamp"]
            timestamp = history_messages[index]["timestamp"]
            if last_timestamp - timestamp > timedelta(days=days).total_seconds():
                break
            split_index = index
        return prepend_system_messages(history_messages[split_index:], system_messages)

    async def _summarize_messages(self, messages: list[Message]) -> list[Message]:
        """调用模型总结历史消息，并恢复原有系统消息。"""
        system_messages, history_messages = split_system_messages(messages)
        messages_for_summary = [
            {"role": "system", "content": SUMMARY_PROMPT},
            *history_messages,
            {"role": "user", "content": "Please summarize the history messages."},
        ]
        try:
            response = await self._chat(
                messages=messages_for_summary,
                parameters=LLMParameters(max_tokens=self._summary_tokens),
                remove_all_tools=True,
            )
            await response.flush()
            summary = response.content
        except Exception:
            logger.exception("Failed to summarize history")
            raise
        return prepend_system_messages([{"role": "assistant", "content": summary}], system_messages)

    async def compress(self, messages: list[Message]) -> list[Message]:
        """逐级压缩消息，直至满足目标上下文大小。"""
        logger.info("Compressing messages messages={}", len(messages))

        for days in range(7, 1, -1):
            messages = self._keep_messages_for_days(days, messages)
            if await self._is_within_target(messages):
                logger.info("Compression finished level=1 messages={}", len(messages))
                return messages

        for keep_count in range(self._recent_turns_to_keep, 3, -1):
            split_index = self._split_recent_user_messages(messages, keep_count)
            recent_messages = messages[split_index:]
            if await self._is_within_target(recent_messages):
                logger.info("Compression level=2 summarizing history")
                compressed_messages = await self._summarize_messages(messages[:split_index]) + recent_messages
                logger.info("Compression finished level=2 messages={}", len(compressed_messages))
                return compressed_messages

        for keep_count in range(len(messages)):
            recent_messages = messages[keep_count:]
            if await self._is_within_target(recent_messages):
                logger.info("Compression level=3 dropping oldest messages")
                compressed_messages = await self._summarize_messages(messages[:keep_count]) + recent_messages
                logger.info("Compression finished level=3 messages={}", len(compressed_messages))
                return compressed_messages

        raise RuntimeError("Conversation compression did not produce a result")
