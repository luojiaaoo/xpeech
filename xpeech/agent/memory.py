from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from ..provider.schema import LLMResponse, ToolCallRequest
from ..utils.helper import append_text_async, ensure_path, read_text_async, write_text_async
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
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    async def read_long_term(self) -> str:
        """读取长期记忆；文件不存在时返回空字符串。"""
        if self.memory_file.exists():
            return await read_text_async(self.memory_file)
        return ""

    async def write_long_term(self, content: str) -> None:
        """使用给定内容覆盖长期记忆文件。"""
        await write_text_async(self.memory_file, content)

    async def append_history(self, entry: str) -> None:
        """向历史摘要文件追加一条记录。"""
        await append_text_async(self.history_file, entry.rstrip() + "\n\n")

    async def get_memory_context(self) -> str:
        """生成可直接注入系统提示词的长期记忆上下文。"""
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
            max_tokens=self._summary_tokens,
            top_p=0.7,
            tools=[self._store.save_memory],
            remove_default_tools=True,
        )
        if not response.has_tool_calls:
            logger.info("No worth consolidating memory")
            return ConsolidationResult(status="skipped", message="未发现需要记忆的内容")

        execution_results = await self._execute_tools(
            response.tool_calls,
            response.mapping_tool_call_funcs,
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
