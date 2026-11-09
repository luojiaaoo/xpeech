import functools
from collections.abc import AsyncIterator, Callable
from typing import Any

import litellm
from fastapi import HTTPException, status
from pydantic import BaseModel

from ..agent.tools.helper import as_tool, get_custom_tool_func
from ..agent.tools.interactions import InteractionRequest
from ..utils.helper import ensure_async, token_counter
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
        extra_headers: dict | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model
        self.parameters = parameters
        self.default_tools: ToolRegistry = {}
        # 工具名到可执行函数的映射，供 ToolExecutor 初始化；与注册/解析的工具保持同步。
        self.mapping_tool_call_funcs: dict[str, Any] = {}
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
        func_: Callable[[type[BaseModel] | None], str | list | InteractionRequest],
        *,
        tool_json: dict[str, Any] | None = None,
    ) -> RegisteredTool:
        def format_result(rt) -> str | list | InteractionRequest:
            if isinstance(rt, InteractionRequest):
                # 交互请求交由 Agent 主循环识别处理，原样透传。
                return rt
            if isinstance(rt, str) or (isinstance(rt, list) and all("type" in item for item in rt)):  # 必须是合法的消息
                return rt
            else:
                raise TypeError(f"Invalid return type: {type(rt)}")

        async_func = ensure_async(func_)

        @functools.wraps(func_)
        async def wrapper(*args, **kwargs) -> str | list | InteractionRequest:
            rt = await async_func(*args, **kwargs)
            return format_result(rt)

        return RegisteredTool(
            func=wrapper,
            tool_json=as_tool(func_) if tool_json is None else tool_json,
        )

    def _publish_tool(self, name: str, tool: RegisteredTool) -> None:
        """登记工具定义，并同步可执行函数映射。"""

        self.default_tools[name] = tool
        self.mapping_tool_call_funcs[name] = tool.func

    def register_function_tool(
        self,
        func_: Callable[[type[BaseModel] | None], str | list],
    ) -> None:
        """注册普通函数工具，从函数生成 schema。"""

        self._publish_tool(func_.__name__, self._build_function_tool(func_))

    def register_mcp_tool(
        self,
        func_: Callable[[type[BaseModel] | None], str | list],
        *,
        tool_json: dict[str, Any],
    ) -> None:
        """注册已绑定的 MCP 函数，保留服务器提供的 schema。"""

        self._publish_tool(func_.__name__, self._build_function_tool(func_, tool_json=tool_json))

    async def _parse_tools(self, funcs: list[Callable[[type[BaseModel] | None], str | list]]) -> ToolRegistry:
        """解析普通函数工具。"""

        tools: ToolRegistry = {}
        for func in funcs:
            # 解析工具函数
            tools[func.__name__] = self._build_function_tool(func)
        return tools

    async def _resolve_tools(
        self,
        tools: list[str | Callable[[type[BaseModel] | None], str | list]] | None = None,
        *,
        remove_all_tools: bool = False,
        remove_default_tools: bool = False,
    ) -> ToolRegistry:
        """统一解析请求使用的工具，供模型调用和上下文统计复用。"""
        funcs = [get_custom_tool_func(tool) if isinstance(tool, str) else tool for tool in tools] if tools else []
        parsed_tools = await self._parse_tools(funcs)
        if remove_all_tools:
            return {}
        if remove_default_tools:
            return parsed_tools
        return self.default_tools | parsed_tools

    async def chat_count_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[str | Callable[[type[BaseModel] | None], str | list]] | None = None,
    ) -> int:
        """估算消息和实际启用的工具定义占用的上下文，强制不排除默认工具。"""
        registered_tools = await self._resolve_tools(
            tools, remove_all_tools=False, remove_default_tools=False
        )
        return await token_counter(messages, tools=[tool.tool_json for tool in registered_tools.values()] or None)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[str | Callable[[type[BaseModel] | None], str | list]] | None = None,
        parameters: LLMParameters | None = None,
        remove_all_tools: bool = False,
        remove_default_tools: bool = False,
        json_output: bool = False,
    ) -> LLMResponse:
        """调用模型；启用 JSON 输出且模型支持时，仅添加 response_format，不修改提示词。"""
        if json_output and not self.support_json_output:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The configured model does not support JSON output",
            )
        parameters = self.parameters if parameters is None else self.parameters.copy_with(parameters)

        registered_tools = await self._resolve_tools(
            tools, remove_all_tools=remove_all_tools, remove_default_tools=remove_default_tools
        )
        tool_jsons = [tool.tool_json for tool in registered_tools.values()]
        # 本次请求携带的工具（含按需解析的自定义工具）同样进入可执行函数映射。
        self.mapping_tool_call_funcs.update({tool_name: tool.func for tool_name, tool in registered_tools.items()})

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

        return self._parse_response(response)

    def _parse_response(
        self,
        response: AsyncIterator[Any],
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
        )
        response_holder["response"] = parsed_response
        return parsed_response
