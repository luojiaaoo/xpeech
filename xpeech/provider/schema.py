from typing import Any, TypedDict, Callable, Type
from pydantic import BaseModel, Field


class Usage(TypedDict):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ToolCallRequest(BaseModel):
    """A tool call request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    mapping_tool_call_funcs: dict[str, Callable[[Type[BaseModel] | None], str | list]] = Field(default_factory=dict)
    finish_reason: str = "stop"
    usage: Usage = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class ProviderChatKwargs(BaseModel):
    """Keyword arguments for provider chat method."""

    model: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with None values removed."""
        return {k: v for k, v in self.model_dump().items() if v is not None}
