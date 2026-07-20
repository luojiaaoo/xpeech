from ..agent.tools.helper import get_custom_tool_func
import litellm
from typing import Any, Callable, Literal, Type
import functools
from pydantic import BaseModel
from .schema import LLMResponse, ReasoningEffort, ToolCallRequest
from .helper import LiteLLMRetryClient
from ..agent.tools.helper import as_tool
from ..agent.tools.mcp_client import MCPServerRegistration, collect_mcp_tool
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
        default_max_tokens: int = 8192,
        default_context_token: int = 200000,
        default_top_p: float = 0.5,
        default_reasoning_effort: ReasoningEffort | None = None,
        support_image: bool = False,
        support_video: bool = False,
        support_json_output: bool = False,
        extra_headers: dict = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self._default_context_token = default_context_token
        self.default_top_p = default_top_p
        self.default_reasoning_effort = default_reasoning_effort
        self.default_tool_jsons: list[dict[str, Any]] = []
        self.default_mapping_tool_call_funcs: dict[str, Callable[[Type[BaseModel] | None], str | list]] = {}
        self._support_image = support_image
        self._support_video = support_video
        self._support_json_output = support_json_output
        self.extra_headers = extra_headers
        self._retry_client = LiteLLMRetryClient()

    @property
    def default_context_token(self) -> int:
        """默认的上下文令牌数。"""

        return self._default_context_token

    @property
    def support_image(self) -> bool:
        """是否支持图片输入。"""

        return self._support_image

    @property
    def support_video(self) -> bool:
        """是否支持视频输入。"""

        return self._support_video

    @property
    def support_json_output(self) -> bool:
        """是否支持JSON输出。"""

        return self._support_json_output

    def decorator_tool_func(
        self,
        func_: Callable[[Type[BaseModel] | None], str | list],
        register_default: bool = False,
    ):
        def format_result(rt) -> str:
            if isinstance(rt, str):
                return rt
            elif isinstance(rt, list) and all("type" in item for item in rt):  # 必须是合法的消息
                return rt
            else:
                raise TypeError(f"Invalid return type: {type(rt)}")

        @functools.wraps(func_)
        async def wrapper(*args, **kwargs) -> str:
            if inspect.iscoroutinefunction(func_):
                rt = await func_(*args, **kwargs)
            else:
                rt = func_(*args, **kwargs)
            return format_result(rt)

        if register_default:
            self.default_tool_jsons.append(as_tool(func_))
            self.default_mapping_tool_call_funcs[func_.__name__] = wrapper
            return wrapper
        else:
            return as_tool(func_), wrapper

    async def decorator_mcp_tool(self, registration: MCPServerRegistration) -> MCPServerRegistration:
        async for tool_json, tool_func, tool_func_name in collect_mcp_tool(registration):
            self.default_tool_jsons.append(tool_json)
            self.default_mapping_tool_call_funcs[tool_func_name] = tool_func
        return registration

    def register_tool(self, tool_type: Literal["function", "mcp"] = "function"):
        """注册工具，统一返回 async wrapper。"""

        if tool_type == "function":
            return functools.partial(self.decorator_tool_func, register_default=True)
        if tool_type == "mcp":
            return self.decorator_mcp_tool
        raise ValueError(f"Unsupported tool type: {tool_type}")

    async def _parse_tools(
        self, funcs: list[Callable[[Type[BaseModel] | None], str | list]]
    ) -> tuple[list[dict[str, Any]], dict[str, Callable[[Type[BaseModel] | None], str | list]]]:
        """解析普通函数工具。"""

        tool_jsons = []
        mapping_tool_call_funcs = {}
        for func in funcs:
            tool_json, tool_func = self.decorator_tool_func(func, register_default=False)
            tool_jsons.append(tool_json)
            mapping_tool_call_funcs[func.__name__] = tool_func
        return tool_jsons, mapping_tool_call_funcs

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[str | Callable[[Type[BaseModel] | None], str | list]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        remove_all_tools: bool = False,
        remove_default_tools: bool = False,
        json_output: bool = False,
    ) -> LLMResponse:
        # 使用提供的参数或默认参数
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p or self.default_top_p
        reasoning_effort = reasoning_effort or self.default_reasoning_effort
        if json_output:
            if self.support_json_output:
                json_output = True
            else:
                logger.warning("LLM does not support JSON output, ignoring json_output parameter.")
                json_output = False
        else:
            json_output = False

        # 根据工具名称获取自定义工具，或使用工具函数
        tools = [get_custom_tool_func(tool) if isinstance(tool, str) else tool for tool in tools] if tools else []
        parsed_tool_jsons, parsed_mapping_tool_call_funcs = await self._parse_tools(tools)

        # 确定工具列表
        if remove_all_tools:
            tool_jsons = []
            mapping_tool_call_funcs = {}
        else:
            if remove_default_tools:
                tool_jsons = parsed_tool_jsons
                mapping_tool_call_funcs = parsed_mapping_tool_call_funcs
            else:
                tool_jsons = self.default_tool_jsons + parsed_tool_jsons
                mapping_tool_call_funcs = self.default_mapping_tool_call_funcs | parsed_mapping_tool_call_funcs

        completion_kwargs = {
            "model": model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "response_format": {"type": "json_object"} if json_output else None,
            "extra_headers": self.extra_headers,
            "reasoning_effort": reasoning_effort,
        }

        # 注入工具
        if tool_jsons:
            completion_kwargs["tools"] = tool_jsons
            completion_kwargs["tool_choice"] = "auto"
        response = await self._retry_client.acompletion(**completion_kwargs)

        # 解析响应
        try:
            return self._parse_response(
                response,
                mapping_tool_call_funcs,
            )

        except Exception as e:
            return LLMResponse(
                content=f"Error parsing LLM response: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(
        self, response: Any, mapping_tool_call_funcs: dict[str, Callable[[Type[BaseModel] | None], str | list]]
    ) -> LLMResponse:
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
            mapping_tool_call_funcs=mapping_tool_call_funcs,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )
