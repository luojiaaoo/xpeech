"""按需请求执行确认的工具包装器，以及执行确认表单。"""

import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from pydantic import Field, create_model

from ...utils.helper import ensure_async
from .helper import get_tool_model_cls
from .question import USER_TIMEOUT, QuestionForm, QuestionOption, SelectField

EXECUTION_DESCRIPTION = "execution_description"
EXECUTION_REQUEST = "__xpeech_execution_request_7f3b9a62_d8e4_4c15_a0f7_6e92b1d83c54__"
_DESCRIPTION = "说明本次执行具体要做什么以及目的。"


def build_execution_question(tool_name: str, arguments: dict[str, Any], description: str) -> str:
    """通过现有 question 表单展示执行说明、参数和确认 / 取消选项。"""
    return QuestionForm(
        title=f"执行确认：{tool_name}",
        subtitle=(
            f"执行说明：{description}\n\n"
            f"执行参数：\n{json.dumps(arguments, ensure_ascii=False, indent=2)}\n\n"
            f"请在 {USER_TIMEOUT} 秒内确认；取消或超时均不会执行。"
        ),
        fields=[
            SelectField(
                name="decision",
                label="是否执行？",
                options=[
                    QuestionOption(label="确认", value="confirm"),
                    QuestionOption(label="取消", value="cancel"),
                ],
            )
        ],
    ).model_dump_json()


def build_execution_request_tool(
    function: Callable,
    requires_confirmation: Callable[[str], bool | Awaitable[bool]] | None = None,
    *,
    not_interactive: bool = False,
) -> Callable:
    """创建同名工具，默认返回标记，交由 Agent 询问用户后执行。

    requires_confirmation 接收包含 execution_description 和原工具参数的 JSON
    字符串（仅包含调用时提供的字段），支持同步或异步函数。返回 True 时询问用户，
    False 时直接执行原函数；异常或非 bool 返回值按工具错误处理，不执行原函数。
    not_interactive=True 时必须提供 requires_confirmation；无回调的工具应在注册时过滤。
    回调判断需要确认的调用直接拒绝，不询问用户。
    """
    if not_interactive and requires_confirmation is None:
        raise ValueError(
            "not_interactive=True 时必须提供 requires_confirmation；"
            "请在注册的时候就屏蔽掉，而不是在运行时屏蔽"
        )
    model_info = get_tool_model_cls(function)
    if EXECUTION_DESCRIPTION in model_info.model_cls.model_fields:
        raise ValueError(f"Tool already defines reserved parameter: {EXECUTION_DESCRIPTION}")
    model_cls = create_model(
        f"{function.__name__}ExecutionArgs",
        __base__=model_info.model_cls,
        execution_description=(str, Field(description=_DESCRIPTION, min_length=1, pattern=r"\S")),
    )

    execution_function = ensure_async(function)
    check_confirmation = ensure_async(requires_confirmation) if requires_confirmation is not None else None

    async def execution_request(args) -> Any:
        needs_confirmation = True
        if check_confirmation is not None:
            needs_confirmation = await check_confirmation(args.model_dump_json(exclude_unset=True))
            if not isinstance(needs_confirmation, bool):
                raise TypeError("requires_confirmation must return a bool")
        if needs_confirmation:
            if not_interactive:
                raise PermissionError("非交互模式下无法获取用户确认，拒绝执行该工具。")
            return EXECUTION_REQUEST

        arguments = args.model_dump(exclude={EXECUTION_DESCRIPTION}, exclude_unset=True)
        original_args = (model_info.model_cls(**arguments),) if model_info.has_pydantic_param else ()
        return await execution_function(*original_args)

    execution_request.__name__ = function.__name__
    execution_request.__annotations__ = {"args": model_cls, "return": Any}
    execution_request.__doc__ = f"{function.__doc__ or function.__name__}\n\n{_DESCRIPTION}"
    execution_request.execution_function = execution_function
    return execution_request


def execution_request_schema(tool_json: dict[str, Any]) -> dict[str, Any]:
    """给 MCP 原始 schema 增加说明字段，保留其余约束且不修改服务器缓存。"""
    schema = deepcopy(tool_json)
    function = schema["function"]
    parameters = function.setdefault("parameters", {"type": "object", "properties": {}})
    properties = parameters.setdefault("properties", {})
    if EXECUTION_DESCRIPTION in properties:
        raise ValueError(f"Tool already defines reserved parameter: {EXECUTION_DESCRIPTION}")
    properties[EXECUTION_DESCRIPTION] = {
        "type": "string",
        "description": _DESCRIPTION,
        "minLength": 1,
        "pattern": r"\S",
    }
    parameters.setdefault("required", []).append(EXECUTION_DESCRIPTION)
    function["description"] = f"{function.get('description', '')}\n\n{_DESCRIPTION}".strip()
    return schema
