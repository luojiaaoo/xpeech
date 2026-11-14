"""通过 Starlark 实现程序化工具调用（Programmatic Tool Calling，PTC）。

本模块负责把模型生成的代码、纯计算方法和当前会话的工具执行器连接起来：
1. 校验源码和输入数据，取得当前会话的工具快照，为本次执行创建独立的解释器模块。
2. 在工作线程执行 Starlark；普通文本与 JSON 处理留在解释器内完成。
3. 脚本通过 call_tool 请求外部操作时，将调用交回主事件循环中的工具执行器。
4. 返回最终结果及调用记录；工具失败、交互请求、超时或取消都会中止后续执行。

脚本不能自行导入 Python 或访问文件、网络和进程。额外的 Python 方法来自
patch.py，是开发者维护的受信任代码；这些方法的纯计算约束需要由开发者保证。
这里不启动操作系统沙盒，计算预算与取消检查也不等同于 CPU、内存的硬隔离。
"""

import asyncio
import inspect
import json
import keyword
import re
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import starlark
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....provider.schema import ToolCallRequest
from ....utils.file_io import LockedFile
from ..helper import safe_resolve_workspace_path
from ..interactions import InteractionRequest
from . import patch

# PTC 始终注册，无需配置开关；以下默认限制集中在代码中维护。
DEFAULT_PTC_TIMEOUT_MS = 120_000  # 未指定时的总时限，与 Shell 工具一致。
MAX_PTC_TIMEOUT_MS = 240_000  # 模型可请求的最大总时限，包含等待工具返回的时间。
PTC_MAX_TOOL_CALLS = 32  # 单次流程最多派发的工具数量，不包含纯计算方法。
# 解释器每约 1000 条字节码调用一次检查；调用工具或注入方法时也会主动检查。
# 因此 ticks 是协作式检查次数，不是精确的指令数，也不能约束单条原生操作的内存分配。
PTC_MAX_TICKS = 1_000
PTC_MAX_CALLSTACK_SIZE = 64  # Starlark 函数调用栈的最大深度。


class StarlarkArgs(BaseModel):
    """模型可传入的 PTC 参数；执行器和会话身份由宿主注入，不由模型提供。"""

    model_config = ConfigDict(extra="forbid")

    code: str | None = Field(
        default=None, min_length=1, description="Starlark 源码，定义 main(inputs)；与 path 二选一。"
    )
    path: str | None = Field(
        default=None,
        min_length=1,
        description="工作区内的 UTF-8 Starlark 脚本文件路径，支持相对路径或工作区内绝对路径；与 code 二选一。",
    )
    inputs: dict[str, Any] = Field(default_factory=dict, description="传给 main(inputs) 的 JSON 对象。")
    timeout_ms: int = Field(
        default=DEFAULT_PTC_TIMEOUT_MS,
        gt=0,
        le=MAX_PTC_TIMEOUT_MS,
        description=(
            f"本次 PTC 流程的总超时毫秒数，由模型根据任务指定；默认 {DEFAULT_PTC_TIMEOUT_MS}，"
            f"最大 {MAX_PTC_TIMEOUT_MS}，包含等待工具的时间，且受外层工具执行器超时上限约束。"
        ),
    )

    @model_validator(mode="after")
    def validate_source(self):
        """源码与文件路径只接受一个，避免调用方不清楚实际执行哪份代码。"""
        if (self.code is None) == (self.path is None):
            raise ValueError("Provide exactly one of code or path")
        return self


def _json_data(value: Any) -> Any:
    """将跨边界的值复制为普通 JSON 数据，不限制其序列化长度。

    序列化后再反序列化，可以断开列表、字典与宿主对象之间的可变引用，避免
    脚本修改输入副本时影响宿主保存的原始数据。自定义宿主对象、循环引用、
    NaN 和无穷值等无法通过此边界。
    """
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return json.loads(text)


def build_ptc_tool(*, workspace: Path):
    """构造可注册的 run_ptc 工具，并固定本次注册使用的注入方法表。

    返回的异步函数由普通工具注册流程暴露给模型。其 docstring 也是模型看到的
    使用说明，因此下方会追加注入方法的名称、签名及文档字符串。
    每次执行都单独创建解释器、预算和调用记录，不在脚本之间共享模块状态。
    """
    workspace = workspace.resolve()
    # 只复制显式登记的方法表，不扫描模块中的其他函数，也不执行脚本指定的导入。
    # 这是映射的浅拷贝：函数本身仍是宿主函数，不能依靠此处复制来隔离其闭包或全局状态。
    methods = dict(patch.INJECTED_METHODS)
    # patch 是受信任的开发者注入入口，不维护内置函数名单，也不检查名称是否覆盖已有接口。
    for name, method in methods.items():
        # 在注册阶段尽早拒绝非法名称和异步方法，避免到执行脚本时才发现注入表错误。
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or keyword.iskeyword(name):
            raise ValueError(f"Invalid PTC injected method name: {name!r}")
        if not callable(method) or inspect.iscoroutinefunction(method) or inspect.iscoroutinefunction(method.__call__):
            raise TypeError(f"PTC injected method must be synchronous and callable: {name}")

    async def run_ptc(
        args: StarlarkArgs,
        tool_executor,
        session_metadata: dict[str, str] | None = None,
        user_id: str | None = None,
        sender_name: str | None = None,
        channel: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """执行 PTC（Programmatic Tool Calling），使用 Starlark 串联工具调用和文本/JSON 处理。

        code 与 path 二选一：code 直接传源码，
        path 读取工作区内的 UTF-8 脚本。脚本定义 main(inputs)，用 return 返回 JSON 兼容结果。
        通过 call_tool("工具名", 参数字典) 调用当前会话的工具，参数遵循对应工具定义。
        工具返回值可供后续语句使用；json.decode/encode 用于已确认格式的 JSON 文本。
        不支持 import、load、print 或直接访问文件、网络、进程；额外可用方法见下方列表。

        返回 JSON 报告：ok=true 时读取 result；ok=false 时读取 error 和 tool_calls。
        工具失败或遇到需要确认、提问、文件投递的操作时，本次脚本停止，后续语句不再执行。
        已完成的操作不会撤销，继续任务前先核对调用记录，避免重复写入或提交。
        报告过长时会返回预览和完整报告路径，先用 read_file 读取完整内容再解析。

        最小 code 示例（inputs 传入 {"text": " hello "}）：
            def main(inputs):
                return inputs["text"].strip()

        注入方法直接按名称调用，无需导入。只返回需要的摘要或结构化数据。
        """
        # 文件读取和解释器执行共用总时限，等待文件锁的时间也计入本次预算。
        deadline = time.monotonic() + min(args.timeout_ms / 1000, tool_executor.max_tool_timeout)
        filename = "ptc.star"
        if args.path is not None:
            source_path = safe_resolve_workspace_path(Path(args.path), workspace)
            if not source_path.is_file():
                raise ValueError("PTC script path must be an existing regular file")
            filename = source_path.relative_to(workspace).as_posix()
            # 共用工作区文件锁，读取原始源码而非 read_file 工具带行号或截断提示的正文。
            async with (
                asyncio.timeout(max(0.001, deadline - time.monotonic())),
                LockedFile(source_path).transaction(),
                await anyio.open_file(source_path, "r", encoding="utf-8-sig") as script,
            ):
                source = await script.read()
        else:
            source = args.code
        # 直接使用当前会话已注册的全部工具，不要求模型另行填写白名单。
        # 既有的工具包装器继续负责权限与确认检查，不因 PTC 调用而绕过。
        # 执行中才完成发现的 MCP 工具不加入这份快照，可在下一次 PTC 调用中使用。
        functions = dict(tool_executor.mapping_tool_call_funcs)
        inputs = _json_data(args.inputs)
        loop = asyncio.get_running_loop()
        # 解释器线程与事件循环共享停止信号；线程不能直接操作 asyncio.Task。
        stopped = threading.Event()
        # tasks 仅由事件循环访问，保存正在执行的工具任务，便于流程退出时统一取消。
        # trace 在工具派发前写入，即使调用被取消，也能保留已经尝试执行的操作记录。
        tasks: set[asyncio.Task] = set()
        trace: list[dict[str, Any]] = []
        ticks = 0

        def check_cancelled():
            """供解释器和宿主回调共同使用的协作式预算检查。

            检查失败时抛出异常，使脚本立即停止；正常返回 False 表示继续执行。
            回调期间不能重新访问正被解释器锁定的模块，否则可能造成重入死锁。
            """
            nonlocal ticks
            ticks += 1
            if stopped.is_set():
                raise RuntimeError("PTC execution cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("PTC execution timed out")
            if ticks > PTC_MAX_TICKS:
                raise RuntimeError("PTC computation budget exceeded")
            return False

        async def invoke(name: str, arguments: dict):
            """在原事件循环中执行一个已注册工具，保留现有执行链的检查与上下文。"""
            task = asyncio.current_task()
            tasks.add(task)
            entry = None
            try:
                # 调用从工作线程排队到事件循环后，流程可能已经停止，派发前必须再次检查。
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("PTC execution stopped before tool dispatch")
                entry = {"name": name, "succeeded": False, "status": "running", "interaction_type": None}
                trace.append(entry)
                # 使用已注册的函数（包括确认包装器），不访问或调用其底层原始执行函数。
                # 中间结果需要供脚本完整处理，因此关闭执行器的自动截断和落盘。
                execution = await tool_executor.execute_function(
                    ToolCallRequest(id=f"ptc-{uuid4().hex}", name=name, arguments=arguments),
                    functions[name],
                    arguments,
                    session_metadata=session_metadata,
                    user_id=user_id,
                    sender_name=sender_name,
                    channel=channel,
                    session_id=session_id,
                    skip_limit_text=True,
                )
                # 必须先按类型判断交互请求，再转换为 JSON；包括未来新增的所有子类。
                # 这里只中止流程，不把交互请求交给 Agent，也不执行确认后的原始函数。
                interaction = isinstance(execution.value, InteractionRequest)
                entry.update(
                    succeeded=execution.succeeded and not interaction,
                    status="interaction" if interaction else ("succeeded" if execution.succeeded else "failed"),
                    duration_seconds=execution.duration_seconds,
                    interaction_type=type(execution.value).__name__ if interaction else None,
                )
                if interaction:
                    raise RuntimeError(
                        f"PTC terminated: {name} returned InteractionRequest ({type(execution.value).__name__}); "
                        "interactive tools cannot run inside PTC"
                    )
                if not execution.succeeded:
                    raise RuntimeError(f"PTC tool {name} failed: {execution.value}")
                return _json_data(execution.value)
            except asyncio.CancelledError:
                # 取消不等于撤销外部效果；状态只说明宿主停止等待和继续派发该流程。
                if entry is not None:
                    entry.update(status="cancelled", succeeded=False)
                raise
            finally:
                tasks.discard(task)

        def call_tool(name, arguments):
            """供 Starlark 同步调用的工具桥接函数，运行在解释器工作线程。"""
            check_cancelled()
            if name == "run_ptc":
                raise PermissionError("Recursive PTC execution is forbidden")
            if not isinstance(name, str) or name not in functions:
                raise ValueError(f"PTC tool is not registered: {name}")
            if not isinstance(arguments, dict):
                raise TypeError("call_tool arguments must be a JSON object")
            if len(trace) >= PTC_MAX_TOOL_CALLS:
                raise RuntimeError("PTC tool call budget exceeded")
            arguments = _json_data(arguments)
            # Starlark 的调用接口是同步的，项目工具是异步的：先把协程提交到原事件循环，
            # 再由当前工作线程等待结果。不能在事件循环线程中调用 pending.result()。
            pending = asyncio.run_coroutine_threadsafe(invoke(name, arguments), loop)
            try:
                return pending.result(timeout=max(0.001, deadline - time.monotonic()))
            except FutureTimeoutError:
                # 工作线程的等待超时后也要取消提交的协程，避免工具脱离 PTC 继续等待。
                pending.cancel()
                raise TimeoutError("PTC execution timed out during tool call") from None

        def evaluate():
            """创建独立解释器，先执行源码中的定义，再调用冻结模块的 main(inputs)。"""
            dialect = starlark.Dialect.standard()
            dialect.enable_load = False
            ast = starlark.parse(filename, source, dialect)
            module = starlark.Module()
            # 模块中仅登记桥接函数和受信任的纯计算方法，不暴露执行器或会话对象。
            module.add_callable("call_tool", call_tool)
            for name, method in methods.items():
                module.add_callable(name, bind_method(name, method))
            globals_ = starlark.Globals.extended_by([starlark.LibraryExtension.Json])
            options = starlark.EvalOptions(check_cancelled=check_cancelled, max_callstack_size=PTC_MAX_CALLSTACK_SIZE)
            starlark.eval_with(options, module, ast, globals_)
            check_cancelled()
            # 冻结模块后再调用入口，避免入口修改模块级可变状态；入口自己的局部数据可修改。
            result = module.freeze().call_with(options, "main", inputs).value
            return _json_data(result)

        def bind_method(name, method):
            """为一个注入方法绑定独立闭包，避免循环注册时所有回调都引用最后一个方法。"""

            def invoke_method(*args, **kwargs):
                """调用纯计算方法；入参和返回值均通过 JSON 复制与类型校验。"""
                check_cancelled()
                data = _json_data({"args": args, "kwargs": kwargs})
                value = method(*data["args"], **data["kwargs"])
                if isinstance(value, InteractionRequest):
                    raise TypeError(f"PTC terminated: injected method {name} returned InteractionRequest")
                if inspect.isawaitable(value):
                    # 防御同步包装函数返回协程的情况；关闭未等待的协程，避免资源警告。
                    if inspect.iscoroutine(value):
                        value.close()
                    raise TypeError(f"PTC injected method must return synchronous JSON data: {name}")
                check_cancelled()
                return _json_data(value)

            return invoke_method

        # 报告只保留最终结果和调用状态，不重复附带源码或中间大数据。
        report = {
            "tool_calls": trace,
        }
        try:
            async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
                # 解释器和同步宿主回调放到工作线程；异步工具仍在上方保存的事件循环中运行。
                report["result"] = await asyncio.to_thread(evaluate)
            report["ok"] = True
        except Exception as exc:  # noqa: BLE001 - 将脚本或工具异常转为报告，并保留已发生的调用记录
            report.update(ok=False, error=str(exc) or "PTC execution timed out")
        finally:
            # asyncio 取消等待不会强制终止工作线程，因此必须先撤销它继续派发工具的能力。
            # 随后取消当前工具任务并等待清理；已经被外部服务接受的操作无法在这里回滚。
            # 原生长操作只能在返回解释器或宿主检查点后观察到停止信号。
            stopped.set()
            pending_tasks = list(tasks)
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
        return json.dumps(report, ensure_ascii=False, allow_nan=False)

    if methods:
        # 直接将实际注入表的说明提供给模型，新增方法时不必再手动维护一份提示词名单。
        run_ptc.__doc__ += "\n注入方法（可直接调用）：\n" + "\n".join(
            f"- {name}{inspect.signature(method)}: {inspect.getdoc(method) or ''}" for name, method in methods.items()
        )
    return run_ptc
