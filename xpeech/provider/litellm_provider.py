import json
import litellm
from typing import Any, Callable, Type, TypedDict
import functools
from pydantic import BaseModel
from .schema import LLMResponse, ToolCallRequest
from ..agent.tools.helper import as_tool

# 禁用调试信息
litellm.suppress_debug_info = True


class LiteLLMProvider:
    """LiteLLM的接口封装。"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        default_model: str,
        default_max_tokens: int = 4096,
        default_top_p: float = 0.5,
        support_image: bool = False,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.default_top_p = default_top_p
        self.default_tool_jsons: list[dict[str, Any]] = []
        self.default_mapping_tool_calls: dict[str, Callable[[Type[BaseModel] | None], str]] = {}
        self.temp_tool_jsons: list[dict[str, Any]] = []
        self.temp_mapping_tool_calls: dict[str, Callable[[Type[BaseModel] | None], str]] = {}
        self._support_image = support_image

    @property
    def support_image(self) -> bool:
        """是否支持图片输入。"""

        return self._support_image

    def register_tool(self, is_temp: bool = False):
        """用带自定义参数的函数装饰器来注册工具。"""

        def decorator(func: Callable[[Type[BaseModel] | None], str | dict]):
            # 添加工具json
            if not is_temp:
                self.default_tool_jsons.append(as_tool(func))
            else:
                self.temp_tool_jsons.append(as_tool(func))

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> str:
                rt = func(*args, **kwargs)
                if isinstance(rt, str):
                    return rt
                elif isinstance(rt, dict):
                    return json.dumps(rt, indent=4, ensure_ascii=False)
                else:
                    raise TypeError(f"Invalid return type: {type(rt)}")

            # 注册工具函数
            if not is_temp:
                self.default_mapping_tool_calls[func.__name__] = wrapper
            else:
                self.temp_mapping_tool_calls[func.__name__] = wrapper
            return wrapper

        return decorator

    @property
    def mapping_tool_calls(self) -> dict[str, Callable[[Type[BaseModel] | None], str]]:
        """获取工具函数映射。"""

        return self.default_mapping_tool_calls | self.temp_mapping_tool_calls

    def _reset_temp_tools(self):
        self.temp_tool_jsons.clear()
        self.temp_mapping_tool_calls.clear()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[Type[BaseModel] | None, str | dict]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ):
        # 使用提供的参数或默认参数
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p or self.default_top_p

        # 添加临时工具
        self._reset_temp_tools()
        for func in tools or []:
            self.register_tool(is_temp=True)(func)
        tool_jsons = self.default_tool_jsons + self.temp_tool_jsons

        # 发起请求
        try:
            completion_kwargs = {
                "model": model,
                "api_base": self.api_base,
                "api_key": self.api_key,
                "messages": messages,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }
            # 注入工具
            if tool_jsons:
                completion_kwargs["tools"] = tool_jsons
                completion_kwargs["tool_choice"] = "auto"
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

        # 解析响应
        try:
            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(
                content=f"Error parsing LLM response: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """将 LiteLLM 的响应解析为 LLMResponse 标准格式。"""
        choice = response.choices[0]
        message = choice.message

        # 提取思考内容
        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = message.reasoning_content
        elif hasattr(message, "model_extra") and message.model_extra:
            # 某些模型可能将 reasoning_content 放在额外字段中
            reasoning_content = message.model_extra.get("reasoning_content")

        # 提取工具调用信息
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # 从 JSON 字符串解析参数如果需要
                args = tc.function.arguments
                if isinstance(args, str):
                    import json

                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                tool_calls.append(
                    ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # 提取使用情况信息
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
