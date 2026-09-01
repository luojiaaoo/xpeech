import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict
import json

from pydantic import BaseModel


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]


class Usage(TypedDict):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ToolCallRequest(BaseModel):
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallChunk:
    """A single tool-call delta from a streaming provider response."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


StreamChunk = tuple[Literal["reasoning_content", "content", "tool_calls"], str | ToolCallChunk]
EndChunk = tuple[Literal["reasoning_content_end", "content_end", "tool_calls_end"], None]
ChunkMix = StreamChunk | EndChunk


END_CHUNKS: dict[str, EndChunk] = {
    "reasoning_content": ("reasoning_content_end", None),
    "content": ("content_end", None),
    "tool_calls": ("tool_calls_end", None),
}


class LLMResponse:
    """Response from an LLM provider."""

    def __init__(
        self,
        *,
        iter_mix_chunks: AsyncIterator[StreamChunk],
        mapping_tool_call_funcs: dict[str, Callable[[type[BaseModel] | None], str | list]] | None = None,
    ) -> None:
        self._stream_done = asyncio.Event()
        self._chunks: list[ChunkMix] = []
        self._completion_callbacks: list[Callable[["LLMResponse"], None]] = []
        self.iter_mix_chunks = self._record(iter_mix_chunks)
        self.mapping_tool_call_funcs = mapping_tool_call_funcs if mapping_tool_call_funcs is not None else {}
        self.finish_reason: str | None = None
        self.usage: Usage = {}

    def _record(self, source: AsyncIterator[StreamChunk]) -> AsyncIterator[ChunkMix]:
        async def _gen() -> AsyncIterator[ChunkMix]:
            previous_kind: str | None = None
            async for item in source:
                # 没有内容的跳过去
                if isinstance(item[1], str) and not item[1].strip():
                    continue
                # 将相同类型的连续块归在一起，并在切换类型时发出对应的结束标记。
                if previous_kind is not None and previous_kind != item[0]:
                    end_chunk = END_CHUNKS[previous_kind]
                    self._chunks.append(end_chunk)
                    yield end_chunk
                self._chunks.append(item)
                yield item
                previous_kind = item[0]
            if previous_kind is not None:
                end_chunk = END_CHUNKS[previous_kind]
                self._chunks.append(end_chunk)
                yield end_chunk
            self._stream_done.set()
            for callback in self._completion_callbacks:
                callback(self)

        return _gen()

    def add_completion_callback(self, callback: Callable[["LLMResponse"], None]) -> None:
        """Register a callback to run after the response stream is consumed."""
        if self._stream_done.is_set():
            raise RuntimeError("Cannot add callback after stream is done.")
        else:
            self._completion_callbacks.append(callback)

    def set_finish_reason(self, finish_reason: str) -> None:
        """Update the finish reason reported by the provider stream."""
        self.finish_reason = finish_reason

    def set_usage(self, usage: Usage) -> None:
        """Update token usage reported by the provider stream."""
        self.usage = usage

    @property
    def content(self) -> str:
        if not self._stream_done.is_set():
            raise ValueError("Content is not available until the stream is done.")
        return "".join(
            chunk for kind, chunk in self._chunks if kind == "content" and isinstance(chunk, str)
        )

    @property
    def reasoning(self) -> str:
        if not self._stream_done.is_set():
            raise ValueError("Reasoning is not available until the stream is done.")
        return "".join(
            chunk
            for kind, chunk in self._chunks
            if kind == "reasoning_content" and isinstance(chunk, str)
        )

    @property
    def tool_calls(self) -> list[ToolCallRequest]:
        if not self._stream_done.is_set():
            raise ValueError("Tool calls are not available until the stream is done.")
        tool_call_chunks = [
            chunk
            for kind, chunk in self._chunks
            if kind == "tool_calls" and isinstance(chunk, ToolCallChunk)
        ]
        if not tool_call_chunks:
            return []

        merged_calls: dict[int, dict[str, str]] = {}
        for chunk in tool_call_chunks:
            merged_call = merged_calls.setdefault(
                chunk.index,
                {"id": "", "name": "", "arguments": ""},
            )
            if chunk.id:
                merged_call["id"] = chunk.id
            if chunk.name:
                merged_call["name"] = chunk.name
            if chunk.arguments:
                merged_call["arguments"] += chunk.arguments

        tool_calls = []
        for index in sorted(merged_calls):
            merged_call = merged_calls[index]
            arguments = merged_call["arguments"]
            try:
                parsed_arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                parsed_arguments = {"raw": arguments}
            tool_calls.append(
                ToolCallRequest(
                    id=merged_call["id"],
                    name=merged_call["name"],
                    arguments=parsed_arguments,
                )
            )
        return tool_calls

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    async def flush(self) -> "LLMResponse":
        """Drain the stream and wait until the stream has finished."""
        async for _ in self.iter_mix_chunks:
            pass
        await self._stream_done.wait()
        return self


class ProviderChatKwargs(BaseModel):
    """Keyword arguments for provider chat method."""

    model: str | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    reasoning_effort: ReasoningEffort | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with None values removed."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
