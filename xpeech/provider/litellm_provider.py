import json
import litellm
from typing import Any, Callable, Type
import functools
from pydantic import BaseModel
from .schema import LLMResponse, ToolCallRequest
from ..agent.tools.helper import as_tool
import inspect
from loguru import logger

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
        default_context_token: int = 200000,
        default_top_p: float = 0.5,
        support_image: bool = False,
        support_json_output: bool = False,
        extra_headers: dict = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.default_context_token = default_context_token
        self.default_top_p = default_top_p
        self.default_tool_jsons: list[dict[str, Any]] = []
        self.default_mapping_tool_call_funcs: dict[str, Callable[[Type[BaseModel] | None], str]] = {}
        self.temp_tool_jsons: list[dict[str, Any]] = []
        self.temp_mapping_tool_call_funcs: dict[str, Callable[[Type[BaseModel] | None], str]] = {}
        self._support_image = support_image
        self._support_json_output = support_json_output
        self.extra_headers = extra_headers

    @property
    def support_image(self) -> bool:
        """是否支持图片输入。"""

        return self._support_image

    @property
    def support_json_output(self) -> bool:
        """是否支持JSON输出。"""

        return self._support_json_output

    def register_tool(self, is_temp: bool = False):
        """注册工具，统一返回 async wrapper。"""

        def decorator(func: Callable[[Type[BaseModel] | None], str | dict]):
            if not is_temp:
                self.default_tool_jsons.append(as_tool(func))
            else:
                self.temp_tool_jsons.append(as_tool(func))

            def format_result(rt) -> str:
                if isinstance(rt, str):
                    return rt
                if isinstance(rt, list) and all("type" in item for item in rt): # 必须是合法的消息JSON
                    return json.dumps(rt, indent=4, ensure_ascii=False)
                else:
                    raise TypeError(f"Invalid return type: {type(rt)}")

            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> str:
                if inspect.iscoroutinefunction(func):
                    rt = await func(*args, **kwargs)
                else:
                    rt = func(*args, **kwargs)
                return format_result(rt)

            if not is_temp:
                self.default_mapping_tool_call_funcs[func.__name__] = wrapper
            else:
                self.temp_mapping_tool_call_funcs[func.__name__] = wrapper
            return wrapper

        return decorator

    @property
    def mapping_tool_call_funcs(self) -> dict[str, Callable[[Type[BaseModel] | None], str]]:
        """获取工具函数映射。"""

        return self.default_mapping_tool_call_funcs | self.temp_mapping_tool_call_funcs

    def _reset_temp_tools(self):
        self.temp_tool_jsons.clear()
        self.temp_mapping_tool_call_funcs.clear()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Callable[Type[BaseModel] | None, str | dict]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        remove_all_tools: bool = False,
        remove_default_tools: bool = False,
        json_output: bool = False,
    ) -> LLMResponse:
        # 使用提供的参数或默认参数
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p or self.default_top_p
        if json_output:
            if self.support_json_output:
                json_output = True
            else:
                logger.warning("LLM does not support JSON output, ignoring json_output parameter.")
                json_output = False
        else:
            json_output = False

        # 添加临时工具
        self._reset_temp_tools()
        for func in tools or []:
            self.register_tool(is_temp=True)(func)

        # 确定工具列表
        if remove_default_tools and remove_all_tools:
            raise ValueError("remove_default_tools and remove_all_tools cannot be True at the same time.")
        if remove_default_tools:
            tool_jsons = self.temp_tool_jsons
        else:
            tool_jsons = self.default_tool_jsons + self.temp_tool_jsons

        if remove_all_tools:
            tool_jsons = []

        # 发起请求
        try:
            completion_kwargs = {
                "model": model,
                "api_base": self.api_base,
                "api_key": self.api_key,
                "messages": messages,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "response_format": {"type": "json_object"} if json_output else None,
                "extra_headers": self.extra_headers,
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
