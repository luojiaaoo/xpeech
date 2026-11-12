"""按需请求执行确认的工具包装器。"""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from enum import StrEnum
from typing import Any

from pydantic import Field, create_model

from ...utils.helper import ensure_async
from .helper import get_tool_model_cls
from .interactions import ExecutionRequest

EXECUTION_DESCRIPTION = "execution_description"
EXECUTION_TITLE = "execution_title"
# 需要用户确认时返回的标记，Agent 主循环识别后询问用户再执行原函数。
EXECUTION_REQUEST = ExecutionRequest()
_DESCRIPTION = "说明本次执行具体要做什么以及目的。"
_TITLE = "根据本次操作生成简短的确认卡片标题。"
EXECUTION_PARAMETERS = {EXECUTION_DESCRIPTION, EXECUTION_TITLE}


class ExecutionDecision(StrEnum):
    """工具执行检查的结果。"""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


def build_execution_request_tool(
    function: Callable,
    requires_confirmation: Callable[[str], ExecutionDecision | Awaitable[ExecutionDecision]] | None = None,
    *,
    not_interactive: bool = False,
) -> Callable:
    """创建同名工具，默认返回标记，交由 Agent 询问用户后执行。

    requires_confirmation 接收包含执行说明、确认文案和原工具参数的 JSON
    字符串（仅包含调用时提供的字段），支持同步或异步函数。必须返回 ExecutionDecision：
    CONFIRM 时询问用户，ALLOW 时直接执行，DENY 时直接拒绝高危命令，不询问用户。
    异常或其他返回值按工具错误处理，不执行原函数。
    not_interactive=True 时必须提供 requires_confirmation；无回调的工具应在注册时过滤。
    回调判断需要确认的调用直接拒绝，不询问用户。
    """
    if not_interactive and requires_confirmation is None:
        raise ValueError(
            "not_interactive=True 时必须提供 requires_confirmation；请在注册的时候就屏蔽掉，而不是在运行时屏蔽"
        )
    model_info = get_tool_model_cls(function)
    for name in EXECUTION_PARAMETERS:
        if name in model_info.model_cls.model_fields:
            raise ValueError(f"Tool already defines reserved parameter: {name}")
    model_cls = create_model(
        f"{function.__name__}ExecutionArgs",
        __base__=model_info.model_cls,
        execution_description=(str, Field(description=_DESCRIPTION, min_length=1, pattern=r"\S")),
        execution_title=(str, Field(default="执行确认", description=_TITLE, min_length=1, pattern=r"\S")),
    )

    execution_function = ensure_async(function)
    check_confirmation = ensure_async(requires_confirmation) if requires_confirmation is not None else None

    async def execution_request(args) -> Any:
        if check_confirmation is None:
            return EXECUTION_REQUEST

        decision = await check_confirmation(args.model_dump_json(exclude_unset=True))
        if not isinstance(decision, ExecutionDecision):
            raise TypeError("requires_confirmation must return an ExecutionDecision")
        if decision is ExecutionDecision.DENY:
            raise PermissionError("发现高危命令，直接拒绝执行。")
        elif decision is ExecutionDecision.CONFIRM:
            if not_interactive:
                raise PermissionError("非交互模式下无法获取用户确认，拒绝执行该工具。")
            return EXECUTION_REQUEST
        elif decision is ExecutionDecision.ALLOW:
            arguments = args.model_dump(exclude=EXECUTION_PARAMETERS, exclude_unset=True)
            original_args = (model_info.model_cls(**arguments),) if model_info.has_pydantic_param else ()
            return await execution_function(*original_args)

    execution_request.__name__ = function.__name__
    execution_request.__annotations__ = {"args": model_cls, "return": Any}
    execution_request.__doc__ = f"{function.__doc__ or function.__name__}\n\n{_DESCRIPTION}"
    execution_request.execution_function = execution_function
    return execution_request


def execution_request_schema(tool_json: dict[str, Any]) -> dict[str, Any]:
    """给 MCP 原始 schema 增加执行说明和确认文案字段，保留其余约束且不修改服务器缓存。"""
    schema = deepcopy(tool_json)
    function = schema["function"]
    parameters = function.setdefault("parameters", {"type": "object", "properties": {}})
    properties = parameters.setdefault("properties", {})
    for name in EXECUTION_PARAMETERS:
        if name in properties:
            raise ValueError(f"Tool already defines reserved parameter: {name}")
    properties[EXECUTION_DESCRIPTION] = {
        "type": "string",
        "description": _DESCRIPTION,
        "minLength": 1,
        "pattern": r"\S",
    }
    properties[EXECUTION_TITLE] = {
        "type": "string",
        "description": _TITLE,
        "default": "执行确认",
        "minLength": 1,
        "pattern": r"\S",
    }
    parameters.setdefault("required", []).append(EXECUTION_DESCRIPTION)
    function["description"] = f"{function.get('description', '')}\n\n{_DESCRIPTION}".strip()
    return schema
