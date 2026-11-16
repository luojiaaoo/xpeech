from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from loguru import logger

from ..provider.schema import LLMParameters, LLMResponse
from ..utils.helper import token_counter
from .helper import is_timestamped_user_message
from .prompt.consolidation import SUMMARY_PROMPT, SUMMARY_TOKENS
from .prompt.helper import prepend_system_messages, split_system_messages

Message = dict[str, Any]

AUTOCOMPACT_BUFFER_TOKENS = 13_000


class ConversationCompressor:
    """负责执行对话压缩策略，不处理长期记忆。"""

    def __init__(
        self,
        *,
        chat: Callable[..., Awaitable[LLMResponse]],
        max_accept_tokens: int,
        target_tokens: int,
        token_counter: Callable[..., Awaitable[int]] = token_counter,
        recent_turns_to_keep: int = 20,
    ) -> None:
        """初始化压缩预算；摘要上限独立于模型的正常输出预算。"""
        self._chat = chat
        self._max_accept_tokens = max_accept_tokens
        self._target_tokens = target_tokens
        self._retained_tokens = target_tokens - SUMMARY_TOKENS
        self._token_counter = token_counter
        self._recent_turns_to_keep = recent_turns_to_keep

    async def should_compress(self, messages: list[Message]) -> bool:
        """判断当前消息的上下文是否达到压缩阈值。"""
        total_tokens = await self._token_counter(messages=messages)
        return total_tokens >= self._max_accept_tokens

    async def _is_within_target(self, messages: list[Message], *, with_summary: bool = False) -> bool:
        """判断当前消息的上下文是否已低于目标阈值。"""
        total_tokens = await self._token_counter(messages=messages)
        budget = self._retained_tokens if with_summary else self._target_tokens
        return total_tokens < budget

    async def _summarize_messages(self, messages: list[Message]) -> list[Message]:
        """调用模型总结对话历史，并保留系统提示词与动态前置上下文。"""
        system_messages, history_messages = split_system_messages(messages)
        messages_for_summary = [
            {"role": "system", "content": SUMMARY_PROMPT},
            *history_messages,
            {"role": "user", "content": "Please summarize the history messages."},
        ]
        try:
            response = await self._chat(
                messages=messages_for_summary,
                parameters=LLMParameters(max_tokens=SUMMARY_TOKENS),
                remove_all_tools=True,
            )
            await response.flush()
            summary = response.content
            if response.finish_reason == "length" or not summary or not summary.strip():
                raise ValueError("压缩摘要为空或被截断，保留原始上下文，请重试。")
        except Exception:
            logger.exception("Failed to summarize history")
            raise
        return prepend_system_messages([{"role": "assistant", "content": summary}], system_messages)

    @staticmethod
    def _keep_messages_for_days(days: int, messages: list[Message]) -> tuple[list[Message], list[Message]]:
        """按完整用户回合保留最近几天的原文。"""
        last_timestamp = None
        split_index = 0
        for index in range(len(messages) - 1, -1, -1):
            if not is_timestamped_user_message(messages[index]):
                continue
            timestamp = messages[index]["timestamp"]
            if last_timestamp is None:
                last_timestamp = timestamp
            if last_timestamp - timestamp > timedelta(days=days).total_seconds():
                break
            split_index = index
        return messages[:split_index], messages[split_index:]

    @staticmethod
    def _turn_start_indexes(messages: list[Message]) -> list[int]:
        """返回可安全作为保留后缀起点的索引。

        一个原始用户消息及其后直到下一个原始用户消息前的所有内容视为一个
        完整回合。只从这种边界截断，避免留下没有对应 tool call 的 tool 消息。
        """
        return [index for index, message in enumerate(messages) if is_timestamped_user_message(message)]

    async def compress(self, messages: list[Message]) -> list[Message]:
        """依次按天、按完整回合、整体摘要压缩，共用预先算好的预算。"""
        logger.info("Compressing messages messages={}", len(messages))
        system_messages, history_messages = split_system_messages(messages)

        if await self._is_within_target(messages):
            return messages
        if not await self._is_within_target(system_messages, with_summary=True):
            raise ValueError("固定上下文和工具定义过大，无法为摘要及对话保留空间，请减少预加载内容或工具。")

        # 一级：优先保留最近 7 到 2 天的原文。
        split_index = len(history_messages)
        for days in range(7, 1, -1):
            historical, recent = self._keep_messages_for_days(days, history_messages)
            if not historical:
                continue
            retained = prepend_system_messages(recent, system_messages)
            if await self._is_within_target(retained, with_summary=True):
                split_index = len(historical)
                break
        else:
            # 二级：按完整回合缩小保留范围，最多保留最近 20 回合。
            turn_start_indexes = self._turn_start_indexes(history_messages)
            for index in turn_start_indexes[-self._recent_turns_to_keep :]:
                if index == 0:
                    continue
                retained = prepend_system_messages(history_messages[index:], system_messages)
                if await self._is_within_target(retained, with_summary=True):
                    split_index = index
                    break

        # 三级：没有可保留的后缀时，整体摘要；三层都只调用一次模型。
        summarized = await self._summarize_messages(
            prepend_system_messages(history_messages[:split_index], system_messages)
        )
        compressed = summarized + history_messages[split_index:]
        # 超出压缩目标仅记录警告，继续使用压缩结果。
        if not await self._is_within_target(compressed):
            logger.warning("压缩后的上下文仍超过目标预算，继续使用压缩结果。")
        logger.info("Compression finished messages={} -> {}", len(messages), len(compressed))
        return compressed
