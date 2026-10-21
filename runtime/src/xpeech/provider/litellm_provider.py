import functools
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

import litellm
from loguru import logger
from pydantic import BaseModel

from ..agent.tools.helper import as_tool, get_custom_tool_func
from ..agent.tools.mcp_client import MCPServerRegistration, collect_mcp_tool
from ..utils.helper import ensure_async
from .helper import LiteLLMRetryClient
from .schema import (
    LLMParameters,
    LLMResponse,
    RegisteredTool,
    StreamChunk,
    ToolCallChunk,
    ToolRegistry,
)

# 禁用调试信息
litellm.suppress_debug_info = True


class LiteLLMProvider:
    """LiteLLM的接口封装。"""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        default_model: str,
        parameters: LLMParameters,
        support_image: bool = False,
        support_video: bool = False,
        support_json_output: bool = False,
        extra_headers: dict = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.parameters = parameters
        self.default_tools: ToolRegistry = {}
        self._support_image = support_image
        self._support_video = support_video
        self._support_json_output = support_json_output
        self.extra_headers = extra_headers
        self._retry_client = LiteLLMRetryClient()

    @property
    def default_context_token(self) -> int:
        """默认的上下文大小。"""

        return self.parameters.max_context_tokens

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

    def _build_function_tool(
        self,
        func_: Callable[[type[BaseModel] | None], str | list],
        is_blocking: bool,
    ) -> RegisteredTool:
        def format_result(rt) -> str | list:
            if isinstance(rt, str) or (isinstance(rt, list) and all("type" in item for item in rt)):  # 必须是合法的消息
                return rt
            else:
                raise TypeError(f"Invalid return type: {type(rt)}")

        async_func = ensure_async(func_)

        @functools.wraps(func_)
        async def wrapper(*args, **kwargs) -> str | list:
            rt = await async_func(*args, **kwargs)
            return format_result(rt)

        return RegisteredTool(
            func=wrapper,
            tool_json=as_tool(func_),
            is_blocking=is_blocking,
        )

    def _register_function_tool(
        self,
        func_: Callable[[type[BaseModel] | None], str | list],
        is_blocking: bool = False,
    ) -> None:
        registered_tool = self._build_function_tool(func_, is_blocking=is_blocking)
        self.default_tools[func_.__name__] = registered_tool

    async def _register_mcp_tools(
        self,
        registration: MCPServerRegistration,
        is_blocking: bool = False,
    ) -> None:
        async for tool_json, tool_func, tool_func_name in collect_mcp_tool(registration):
            self.default_tools[tool_func_name] = RegisteredTool(
                func=tool_func,
                tool_json=tool_json,
                is_blocking=is_blocking,
            )

    def register_tool(
        self,
        tool_type: Literal["function", "mcp"] = "function",
    ):
        """Return a function or MCP tool registrar."""

        if tool_type == "function":
            return self._register_function_tool
        if tool_type == "mcp":
            return self._register_mcp_tools
        raise ValueError(f"Unsupported tool type: {tool_type}")

    async def _parse_tools(self, funcs: list[Callable[[type[BaseModel] | None], str | list]]) -> ToolRegistry:
        """解析普通函数工具。"""

        tools: ToolRegistry = {}
        for func in funcs:
            # 解析工具函数
            tools[func.__name__] = self._build_function_tool(func, is_blocking=False)
        return tools

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[str | Callable[[type[BaseModel] | None], str | list]] | None = None,
        parameters: LLMParameters | None = None,
        remove_all_tools: bool = False,
        remove_default_tools: bool = False,
        remove_blocking_tool: bool = False,
        json_output: bool = False,
    ) -> LLMResponse:
        parameters = self.parameters if parameters is None else self.parameters.copy_with(parameters)
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
        parsed_tools = await self._parse_tools(tools)

        # 确定工具列表
        if remove_all_tools:
            registered_tools: ToolRegistry = {}
        else:
            if remove_default_tools:
                registered_tools = parsed_tools
            else:
                registered_tools = self.default_tools | parsed_tools

        if remove_blocking_tool:
            registered_tools = {
                tool_name: tool
                for tool_name, tool in registered_tools.items()
                if not tool.is_blocking
            }

        tool_jsons = [tool.tool_json for tool in registered_tools.values()]
        mapping_tool_call_funcs = {tool_name: tool.func for tool_name, tool in registered_tools.items()}

        # 参数构建
        extra_body = {
            key: value
            for key, value in {
                "top_k": parameters.top_k,
                "min_p": parameters.min_p,
                "repetition_penalty": parameters.repetition_penalty,
            }.items()
            if value is not None
        }
        completion_kwargs = {
            "model": self.default_model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "messages": messages,
            "max_tokens": parameters.max_tokens,
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "presence_penalty": parameters.presence_penalty,
            "extra_body": extra_body or None,
            "response_format": {"type": "json_object"} if json_output else None,
            "extra_headers": self.extra_headers,
            "reasoning_effort": parameters.reasoning_effort,
        }
        completion_kwargs = {key: value for key, value in completion_kwargs.items() if value is not None}

        # 注入工具
        if tool_jsons:
            completion_kwargs["tools"] = tool_jsons
            completion_kwargs["tool_choice"] = "auto"
        response = self._retry_client.acompletion(**completion_kwargs)

        return self._parse_response(response, mapping_tool_call_funcs)

    def _parse_response(
        self,
        response: AsyncIterator[Any],
        mapping_tool_call_funcs: dict[str, Callable[[type[BaseModel] | None], str | list]],
    ) -> LLMResponse:
        """将 LiteLLM 的流式响应解析为统一的混合内容流。"""
        response_holder: dict[str, LLMResponse] = {}

        async def iter_mix_chunks() -> AsyncIterator[StreamChunk]:
            async for chunk in response:
                parsed_response = response_holder["response"]
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    parsed_response.set_usage(
                        {
                            "prompt_tokens": getattr(usage, "prompt_tokens", None),
                            "completion_tokens": getattr(usage, "completion_tokens", None),
                            "total_tokens": getattr(usage, "total_tokens", None),
                        }
                    )

                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                choice = choices[0]
                if choice.finish_reason is not None:
                    parsed_response.set_finish_reason(choice.finish_reason)

                delta = choice.delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield "reasoning_content", reasoning

                content = getattr(delta, "content", None)
                if content:
                    yield "content", content

                for tool_call in getattr(delta, "tool_calls", None) or []:
                    index = getattr(tool_call, "index", 0) or 0
                    function = getattr(tool_call, "function", None)
                    yield (
                        "tool_calls",
                        ToolCallChunk(
                            index=index,
                            id=getattr(tool_call, "id", None),
                            name=getattr(function, "name", None),
                            arguments=getattr(function, "arguments", None),
                        ),
                    )

        parsed_response = LLMResponse(
            iter_mix_chunks=iter_mix_chunks(),
            mapping_tool_call_funcs=mapping_tool_call_funcs,
        )
        response_holder["response"] = parsed_response
        return parsed_response
