import asyncio
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from ..exceptions import PathProtectionError
from ..provider.schema import ToolCallRequest
from ..utils.helper import format_exception2llm
from .tools.helper import get_tool_model_cls


@dataclass(frozen=True)
class ToolExecutionResult:
    """记录一次工具调用的结果、状态与耗时。"""

    call: ToolCallRequest
    value: Any
    succeeded: bool
    duration_seconds: float
    error: str | None = None


class ToolExecutor:
    """解析并发执行模型请求的工具调用。"""

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
            logger.info(
                "Tool call started loop_count={} tool_name={} args={}",
                loop_count,
                tool_call.name,
                tool_call.arguments,
            )
            try:
                tool_call_func = mapping_tool_call_funcs.get(tool_call.name)
                if tool_call_func is None:
                    raise ValueError(f"Tool is not registered for this request: {tool_call.name}")
                model_cls = get_tool_model_cls(tool_call_func)
                if model_cls is None:
                    value = await tool_call_func()
                else:
                    value = await tool_call_func(model_cls(**tool_call.arguments))
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
                    error=error,
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
                    error=error,
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
