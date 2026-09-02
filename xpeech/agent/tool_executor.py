import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from ..exceptions import PathProtectionError
from ..provider.schema import ToolCallRequest
from ..utils.helper import ensure_path_async, format_exception2llm, write_text_async, format_now
from .tools.helper import get_tool_model_cls

TOOL_RESULT_DIRECTORY = "tool-results"
# read_file 是持久化结果的恢复路径。将其豁免，以避免出现“工具结果 -> read_file -> 另一个持久化工具结果”的循环。
TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS = frozenset({"read_file"})


@dataclass(frozen=True)
class ToolExecutionResult:
    """记录一次工具调用的结果、状态与耗时。"""

    call: ToolCallRequest
    value: Any
    succeeded: bool
    duration_seconds: float


class ToolExecutor:
    """解析并发执行模型请求的工具调用。"""

    def __init__(
        self,
        workspace: str | Path,
        max_result_chars: int,
    ) -> None:
        if max_result_chars < 1:
            raise ValueError("max_result_chars must be positive")
        self._workspace = Path(workspace).expanduser().resolve()
        self._max_result_chars = max_result_chars

    async def _limit_text(self, tool_call: ToolCallRequest, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        result_directory = self._workspace / TOOL_RESULT_DIRECTORY / format_now()
        await ensure_path_async(result_directory)
        result_path = result_directory / f"{tool_call.name}-{uuid4().hex[:12]}.txt"
        await write_text_async(result_path, text)
        saved_path = result_path.relative_to(self._workspace).as_posix()
        return (
            text[:max_chars] + f"\n\n... [tool result contains {len(text):,} characters; showing the first "
            f"{max_chars:,} characters]\nFull result saved to: {saved_path}"
        )

    async def execute(
        self,
        tool_calls: list[ToolCallRequest],
        mapping_tool_call_funcs: dict[str, Any],
        loop_count: int | None = None,
    ) -> list[ToolExecutionResult]:
        """并发执行全部工具调用，并按原始顺序返回结构化结果。"""

        async def execute_one(tool_call: ToolCallRequest) -> ToolExecutionResult:
            """执行单个工具调用，并将异常转换为模型可读的结果。"""
            start_time = time.perf_counter()
            tool_call_func = mapping_tool_call_funcs.get(tool_call.name)
            logger.info(
                "Tool call started loop_count={} tool_name={} args={}",
                loop_count,
                tool_call.name,
                tool_call.arguments,
            )
            try:
                if tool_call_func is None:
                    raise ValueError(f"Tool is not registered for this request: {tool_call.name}")
                model_cls = get_tool_model_cls(tool_call_func)
                if model_cls is None:
                    value = await tool_call_func()
                else:
                    value = await tool_call_func(model_cls(**tool_call.arguments))
                if (
                    isinstance(value, str)
                    and tool_call.name not in TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS
                ):
                    value = await self._limit_text(tool_call, value, self._max_result_chars)
            except PathProtectionError as exc:
                duration = time.perf_counter() - start_time
                error = format_exception2llm(exc)
                logger.warning(
                    "Tool call failed loop_count={} tool_name={} args={} exception={} duration={:.2f}s",
                    loop_count,
                    tool_call.name,
                    tool_call.arguments,
                    error,
                    duration,
                )
                return ToolExecutionResult(
                    call=tool_call,
                    value=error,
                    succeeded=False,
                    duration_seconds=duration,
                )
            except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model
                duration = time.perf_counter() - start_time
                error = format_exception2llm(exc)
                logger.exception(
                    "Tool call failed loop_count={} tool_name={} args={} duration={:.2f}s",
                    loop_count,
                    tool_call.name,
                    tool_call.arguments,
                    duration,
                )
                return ToolExecutionResult(
                    call=tool_call,
                    value=error,
                    succeeded=False,
                    duration_seconds=duration,
                )

            duration = time.perf_counter() - start_time
            logger.info(
                "Tool call successfully loop_count={} tool_name={} duration={:.2f}s",
                loop_count,
                tool_call.name,
                duration,
            )
            return ToolExecutionResult(
                call=tool_call,
                value=value,
                succeeded=True,
                duration_seconds=duration,
            )

        tasks: list[asyncio.Task[ToolExecutionResult]] = []
        async with asyncio.TaskGroup() as task_group:
            for tool_call in tool_calls:
                tasks.append(task_group.create_task(execute_one(tool_call)))
        return [task.result() for task in tasks]
