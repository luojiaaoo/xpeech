import inspect
from typing import Callable, get_type_hints
from pydantic import BaseModel


def as_tool(func: Callable, name_suffix: str = "") -> dict:
    """
    将函数转换为 OpenAI Tool Schema。

    支持两种函数签名：
    1. 无参数函数
    2. 有且仅有一个参数，且该参数类型为 BaseModel 的子类
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    params = list(sig.parameters.values())

    # 名称：函数名 + 后缀
    tool_name = func.__name__ + name_suffix

    # 描述：用函数的 docstring
    description = (func.__doc__ or "").strip()
    if not description:
        raise ValueError(f"Function '{func.__name__}' must have a docstring")

    # 情况 1：无参数
    if len(params) == 0:
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    # 情况 2：单个 BaseModel 参数
    elif len(params) == 1:
        param = params[0]
        model_cls = hints.get(param.name)

        if model_cls is None:
            raise TypeError(
                f"Parameter '{param.name}' in '{func.__name__}' must have a type annotation"
            )

        if not inspect.isclass(model_cls) or not issubclass(model_cls, BaseModel):
            raise TypeError(
                f"Parameter '{param.name}' in '{func.__name__}' must be annotated "
                f"with a BaseModel subclass, got {model_cls!r}"
            )

        schema = model_cls.model_json_schema()
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
            "additionalProperties": False,
        }

    # 其他情况：不支持
    else:
        raise ValueError(
            f"Function '{func.__name__}' must accept either 0 parameters "
            f"or exactly 1 parameter annotated with a BaseModel subclass"
        )

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": parameters,
        },
    }