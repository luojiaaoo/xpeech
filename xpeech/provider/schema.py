from typing import Any
from pydantic import BaseModel, Field


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
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)

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
