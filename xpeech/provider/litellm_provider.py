import litellm
from collections.abc import Callable
from typing import Any
import functools
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
        self.tools: list[dict[str, Any]] = []
        self.support_image = support_image

    def supports_image(self) -> bool:
        """是否支持图片输入。"""
        
        return self.support_image

    def register_tool(self):
        """用带自定义参数的函数装饰器来注册工具。"""

        def decorator(func: Callable[..., Any]):
            @functools.wraps(func)
            def wrapper() -> Callable[..., Any]:
                self.tools.append(as_tool(func))

            return wrapper

        return decorator

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ):
        # 使用提供的参数或默认参数
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p or self.default_top_p
        tools = [self.tools, tools] if tools is not None else self.tools

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
            if tools:
                completion_kwargs["tools"] = tools
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

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
