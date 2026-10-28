from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from ..provider.schema import LLMParameters, LLMResponse, ToolCallRequest
from ..utils.file_io import LockedFile, ensure_path
from .prompt.helper import remove_system_messages
from .tool_executor import ToolExecutionResult

ChatCallable = Callable[..., Awaitable[LLMResponse]]
ToolExecutorCallable = Callable[
    [list[ToolCallRequest], dict[str, Any]],
    Awaitable[list[ToolExecutionResult]],
]


class MemoryArgs(BaseModel):
    """定义模型保存记忆时必须提交的参数。"""

    history_entry: Annotated[
        str,
        Field(
            description=(
                "A paragraph summarizing key events/decisions/topics. Start with [YYYY-MM-DD HH:MM]. "
                "Include detail useful for grep search."
            )
        ),
    ]
    memory_update: Annotated[
        str,
        Field(
            description=(
                "Full updated long-term memory as markdown. Include all existing facts plus new ones. "
                "Return unchanged if nothing new."
            )
        ),
    ]


class MemoryStore:
    """管理工作区内的长期记忆与历史摘要文件。"""

    def __init__(self, workspace: Path):
        """初始化记忆目录及相关文件路径。"""
        self.memory_dir = ensure_path(workspace / "memory")
        self.memory_file = LockedFile(self.memory_dir / "MEMORY.md")
        self.history_file = LockedFile(self.memory_dir / "HISTORY.md")

    async def read_long_term(self) -> str:
        """读取长期记忆；文件不存在时返回空字符串。"""
        try:
            return await self.memory_file.read_text()
        except FileNotFoundError:
            return ""

    async def write_long_term(self, content: str) -> None:
        """覆盖当前存储的长期记忆。"""
        await self.memory_file.write_text(content)

    async def append_history(self, entry: str) -> None:
        """向历史摘要文件追加一条记录。"""
        await self.history_file.append_text(entry.rstrip() + "\n\n")

    async def get_memory_context(self) -> str:
        """读取长期记忆并生成系统提示词上下文。"""
        long_term = await self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def save_memory(self, args: MemoryArgs) -> str:
        """将模型整理出的历史摘要和长期记忆持久化。"""
        await self.append_history(args.history_entry.strip())
        await self.write_long_term(args.memory_update)
        return "Memory saved successfully."


@dataclass(frozen=True)
class ConsolidationResult:
    """描述一次记忆整理的状态及面向用户的说明。"""

    status: Literal["saved", "skipped", "failed"]
    message: str


class MemoryConsolidator:
    """通过模型工具调用将对话历史整理为长期记忆。"""

    def __init__(
        self,
        *,
        store: MemoryStore,
        chat: ChatCallable,
        execute_tools: ToolExecutorCallable,
        summary_tokens: int,
    ) -> None:
        """初始化记忆存储、模型调用函数和工具执行函数。"""
        self._store = store
        self._chat = chat
        self._execute_tools = execute_tools
        self._summary_tokens = summary_tokens

    async def consolidate(self, messages: list[dict[str, Any]]) -> ConsolidationResult:
        """整理对话中的重要信息，并返回结构化处理结果。"""
        clean_messages = remove_system_messages(messages)
        if not clean_messages:
            return ConsolidationResult(status="skipped", message="当前上下文为空，无需记忆")

        # 模型根据旧记忆生成完整的新版本，锁需覆盖读取、模型生成和工具保存全过程。
        async with self._store.memory_file.transaction():
            return await self._consolidate_locked(clean_messages)

    async def _consolidate_locked(
        self, clean_messages: list[dict[str, Any]]
    ) -> ConsolidationResult:
        """在已有文件锁的保护下整理并保存记忆。"""
        memory_context = await self._store.get_memory_context() or "(empty)"
        messages_for_consolidation = [
            {"role": "system", "content": "## Current Long-term Memory\n" + memory_context},
            *clean_messages,
            {
                "role": "user",
                "content": "Process this conversation and MUST call the save_memory tool with your consolidation.",
            },
        ]
        logger.info("Consolidating memory")
        response = await self._chat(
            messages=messages_for_consolidation,
            parameters=LLMParameters(max_tokens=self._summary_tokens),
            tools=[self._store.save_memory],
            remove_default_tools=True,
        )
        await response.flush()
        if not response.has_tool_calls:
            logger.info("No worth consolidating memory")
            return ConsolidationResult(status="skipped", message="未发现需要记忆的内容")

        execution_results = []
        # 同一模型响应中的多个保存调用也必须按顺序执行。
        for tool_call in response.tool_calls:
            execution_results.extend(
                await self._execute_tools([tool_call], response.mapping_tool_call_funcs)
            )
        saved = any(
            execution.call.name == "save_memory" and execution.succeeded
            for execution in execution_results
        )
        if not saved:
            logger.info("记忆整理过程中未成功调用 save_memory 工具")
            return ConsolidationResult(
                status="failed",
                message="记忆整理失败：模型未成功调用 save_memory 工具",
            )

        logger.info("Consolidating memory finished")
        return ConsolidationResult(status="saved", message="已记忆本次会话关键内容")
