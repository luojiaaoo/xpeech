"""按需请求执行确认的工具包装器。"""

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field, create_model

from ...utils.helper import ensure_async
from .helper import get_tool_model_cls
from .interactions import ExecutionRequest

EXECUTION_DESCRIPTION = "execution_description"
EXECUTION_TITLE = "execution_title"
REQUEST_USER_CONFIRMATION = "request_user_confirmation"
# 需要用户确认时返回的标记，Agent 主循环识别后询问用户再执行原函数。
EXECUTION_REQUEST = ExecutionRequest()
_DESCRIPTION = "说明本次执行具体要做什么以及目的。"
_TITLE = "根据本次操作生成简短的确认卡片标题。"
_REQUEST_USER_CONFIRMATION_DESCRIPTION = "根据工具说明中的判断规则决定是否需要用户确认；需要时为 true，否则为 false，默认 false。"
EXECUTION_PARAMETERS = {EXECUTION_DESCRIPTION, EXECUTION_TITLE, REQUEST_USER_CONFIRMATION}


def _confirmation_prompt(prompt: str) -> str:
    return (
        f"请根据以下规则判断本次调用是否需要用户确认，并设置 {REQUEST_USER_CONFIRMATION}：\n"
        f"{prompt}\n{_REQUEST_USER_CONFIRMATION_DESCRIPTION}"
    )


class ExecutionDecision(StrEnum):
    """工具执行检查的结果。"""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class ExecutionConfirmation:
    """执行确认规则：回调与模型判断任一要求确认即询问用户，DENY 优先。"""

    callback: Callable[[str], ExecutionDecision | Awaitable[ExecutionDecision]] | None = None
    prompt: str | None = None


def build_execution_request_tool(
    function: Callable,
    requires_confirmation: ExecutionConfirmation | None = None,
    *,
    not_interactive: bool = False,
) -> Callable:
    """创建同名工具，默认返回标记，交由 Agent 询问用户后执行。

    requires_confirmation.callback 接收包含执行说明、确认文案和原工具参数的 JSON 字符串
    （仅包含调用时提供的字段），支持同步或异步函数，必须返回 ExecutionDecision。
    requires_confirmation.prompt 注入模型判断提示词，并增加 request_user_confirmation: bool = False 参数。
    回调返回 CONFIRM 或模型要求确认时询问用户；回调返回 DENY 时始终拒绝，不询问用户。
    回调异常或返回其他类型时按工具错误处理，不执行原函数。两种规则均未提供时每次询问。
    not_interactive=True 时必须提供至少一种规则，否则应在注册时过滤；需要确认的调用直接拒绝。
    """
    confirmation = requires_confirmation or ExecutionConfirmation()
    if not_interactive and confirmation.callback is None and confirmation.prompt is None:
        raise ValueError(
            "not_interactive=True 时必须提供 requires_confirmation；请在注册的时候就屏蔽掉，而不是在运行时屏蔽"
        )
    model_info = get_tool_model_cls(function)
    for name in EXECUTION_PARAMETERS:
        if name in model_info.model_cls.model_fields:
            raise ValueError(f"Tool already defines reserved parameter: {name}")
    extra_fields = {}
    if confirmation.prompt is not None:
        extra_fields[REQUEST_USER_CONFIRMATION] = (
            bool, Field(default=False, description=_REQUEST_USER_CONFIRMATION_DESCRIPTION),
        )
    model_cls = create_model(
        f"{function.__name__}ExecutionArgs",
        __base__=model_info.model_cls,
        execution_description=(str, Field(description=_DESCRIPTION, min_length=1, pattern=r"\S")),
        execution_title=(str, Field(default="执行确认", description=_TITLE, min_length=1, pattern=r"\S")),
        **extra_fields,
    )

    execution_function = ensure_async(function)
    check_confirmation = ensure_async(confirmation.callback) if confirmation.callback is not None else None

    async def execution_request(args) -> Any:
        if check_confirmation is None:
            decision = ExecutionDecision.CONFIRM if confirmation.prompt is None else ExecutionDecision.ALLOW
        else:
            decision = await check_confirmation(args.model_dump_json(exclude_unset=True))
        if not isinstance(decision, ExecutionDecision):
            raise TypeError("requires_confirmation.callback must return an ExecutionDecision")
        if decision is ExecutionDecision.DENY:
            raise PermissionError("发现高危命令，直接拒绝执行。")
        if decision is ExecutionDecision.CONFIRM or (
            confirmation.prompt is not None and args.request_user_confirmation
        ):
            if not_interactive:
                raise PermissionError("非交互模式下无法获取用户确认，拒绝执行该工具。")
            return EXECUTION_REQUEST
        arguments = args.model_dump(exclude=EXECUTION_PARAMETERS, exclude_unset=True)
        original_args = (model_info.model_cls(**arguments),) if model_info.has_pydantic_param else ()
        return await execution_function(*original_args)

    execution_request.__name__ = function.__name__
    execution_request.__annotations__ = {"args": model_cls, "return": Any}
    execution_request.__doc__ = f"{function.__doc__ or function.__name__}\n\n{_DESCRIPTION}"
    if confirmation.prompt is not None:
        execution_request.__doc__ += f"\n\n{_confirmation_prompt(confirmation.prompt)}"
    execution_request.execution_function = execution_function
    return execution_request


def execution_request_schema(
    tool_json: dict[str, Any], *, requires_confirmation: ExecutionConfirmation | None = None,
) -> dict[str, Any]:
    """给 MCP schema 增加确认字段和可选的模型判断提示词，保留约束且不修改服务器缓存。"""
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
    if requires_confirmation is not None and requires_confirmation.prompt is not None:
        properties[REQUEST_USER_CONFIRMATION] = {
            "type": "boolean",
            "description": _REQUEST_USER_CONFIRMATION_DESCRIPTION,
            "default": False,
        }
        function["description"] += f"\n\n{_confirmation_prompt(requires_confirmation.prompt)}"
    return schema
