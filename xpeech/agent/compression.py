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
        return prepend_system_messages([{"role": "assistant", "content": summary or "(nothing)"}], system_messages)

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

    @staticmethod
    def _turn_start_indexes(messages: list[Message]) -> list[int]:
        """返回可安全作为保留后缀起点的索引。

        一个原始用户消息及其后直到下一个原始用户消息前的所有内容视为一个
        完整回合。只从这种边界截断，避免留下没有对应 tool call 的 tool 消息。
        """
        return [index for index, message in enumerate(messages) if is_timestamped_user_message(message)]

    async def compress(self, messages: list[Message]) -> list[Message]:
        """压缩到目标大小，同时保留最长的、结构完整的最近对话后缀。"""
        logger.info("Compressing messages messages={}", len(messages))

        # 一级：按时间窗口丢弃过旧上下文。每次都基于原始消息计算，避免
        # 在一级失败后提前丢掉内容，导致后续摘要无法覆盖它们。
        for days in range(7, 1, -1):
            recent_days_messages = self._keep_messages_for_days(days, messages)
            if await self._is_within_target(recent_days_messages):
                logger.info("Compression finished by days={} messages={}", days, len(recent_days_messages))
                return recent_days_messages

        system_messages, history_messages = split_system_messages(messages)
        turn_start_indexes = self._turn_start_indexes(history_messages)

        # 二级：从最多保留的最近回合数开始，按完整用户回合逐步缩小。
        # 第一个能放入的后缀就是此策略下保留消息最多的候选。
        for split_index in turn_start_indexes[-self._recent_turns_to_keep :]:
            recent_messages = history_messages[split_index:]
            retained_messages = prepend_system_messages(recent_messages, system_messages)
            if not await self._is_within_target(retained_messages):
                continue

            # 只有最终的“系统消息 + 摘要 + 最近消息”仍能放入目标时才保留摘要。
            if split_index:
                summarized_messages = await self._summarize_messages(
                    prepend_system_messages(history_messages[:split_index], system_messages)
                )
                compressed_messages = summarized_messages + recent_messages
                if await self._is_within_target(compressed_messages):
                    logger.info("Compression finished by turns with summary messages={}", len(compressed_messages))
                    return compressed_messages
                logger.warning("Summary exceeds target; retaining the complete recent suffix without it")

            logger.info("Compression finished by turns without summary messages={}", len(retained_messages))
            return retained_messages

        # 三级：直接将整个会话压缩为摘要；摘要长度由 summary_tokens 控制。
        compressed_messages = await self._summarize_messages(messages)
        logger.info("Compression finished by fallback summary messages={}", len(compressed_messages))
        return compressed_messages
