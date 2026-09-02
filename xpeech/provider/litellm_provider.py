from ..agent.tools.helper import get_custom_tool_func
from collections.abc import AsyncIterator, Callable

import litellm
from typing import Any, Literal
import functools
from pydantic import BaseModel
from .schema import LLMParameters, LLMResponse, StreamChunk, ToolCallChunk
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
        self.default_tool_jsons: list[dict[str, Any]] = []
        self.default_mapping_tool_call_funcs: dict[str, Callable[[type[BaseModel] | None], str | list]] = {}
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

    def decorator_tool_func(
        self,
        func_: Callable[[type[BaseModel] | None], str | list],
        register_default: bool = False,
    ):
        def format_result(rt) -> str:
            if isinstance(rt, str) or (isinstance(rt, list) and all("type" in item for item in rt)):  # 必须是合法的消息
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
        self, funcs: list[Callable[[type[BaseModel] | None], str | list]]
    ) -> tuple[list[dict[str, Any]], dict[str, Callable[[type[BaseModel] | None], str | list]]]:
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
        tools: list[str | Callable[[type[BaseModel] | None], str | list]] | None = None,
        parameters: LLMParameters | None = None,
        remove_all_tools: bool = False,
        remove_default_tools: bool = False,
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
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        completion_kwargs = {key: value for key, value in completion_kwargs.items() if value is not None}

        # 注入工具
        if tool_jsons:
            completion_kwargs["tools"] = tool_jsons
            completion_kwargs["tool_choice"] = "auto"
        response = await self._retry_client.acompletion(**completion_kwargs)

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
