from collections.abc import AsyncIterator
from typing import Any, Self

from ..config.settings import Settings, settings
from ..provider.litellm_provider import LiteLLMProvider
from ..utils.helper import ensure_path_async
from ..utils.session import create_workspace_templates
from .loop import AgentLoop
from .server.schema import InboundMessage
from .tools.registry import register_default_tools


class AgentRunner:
    """Prepare and run an agent for one conversation session."""

    def __init__(self, session_id: str, *, config: Settings) -> None:
        self.session_id = session_id
        self.config = config
        self.workspace = (config.path.workspace_base_path / session_id).resolve()
        self._agent_loop: AgentLoop | None = None

    @classmethod
    async def create(
        cls,
        session_id: str,
        *,
        config: Settings = settings,
    ) -> Self:
        """Create a fully initialized runner for one conversation session."""
        runner = cls(session_id, config=config)
        await runner._initialize()
        return runner

    async def _initialize(self) -> None:
        """Initialize the session workspace, provider, tools, and agent loop."""
        await create_workspace_templates(await ensure_path_async(self.workspace))

        llm_config = self.config.llm
        api_key = llm_config.api_key
        provider = LiteLLMProvider(
            api_key=api_key,
            api_base=llm_config.api_base,
            default_model=llm_config.default_model,
            parameters=llm_config.parameters,
            support_image=llm_config.support_image,
            support_video=llm_config.support_video,
            support_json_output=llm_config.support_json_output,
            extra_headers={"Authorization": "Bearer " + api_key},
        )
        agent_loop = AgentLoop(
            provider=provider,
            workspace=self.workspace,
            tools=llm_config.default_tools,
            summary_tokens=llm_config.summary_tokens,
            max_iterations=llm_config.max_iterations,
        )
        await register_default_tools(
            provider=provider,
            workspace=self.workspace,
            config=self.config.tool,
        )
        self._agent_loop = agent_loop

    async def run(
        self,
        message: InboundMessage,
        *,
        background: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one inbound message and yield the agent's events."""
        if message.session_id != self.session_id:
            raise ValueError("Inbound message session_id does not match the runner session")

        if self._agent_loop is None:
            raise RuntimeError("AgentRunner must be created with AgentRunner.create()")

        async for event in self._agent_loop.run(
            message=message,
            background=background,
        ):
            yield event

    async def run_once(
        self,
        message: InboundMessage,
        *,
        background: bool = False,
    ) -> str:
        """Consume an Agent run and return its last complete assistant segment."""
        current_segment: list[str] | None = None
        final_segment: str | None = None

        async for event in self.run(message, background=background):
            event_type = event.get("event")
            if event_type == "assistant":
                if current_segment is None:
                    current_segment = []
                context = event.get("context")
                if isinstance(context, str):
                    current_segment.append(context)
            elif event_type == "assistant_end" and current_segment is not None:
                final_segment = "".join(current_segment)
                current_segment = None

        if final_segment is None or not final_segment.strip():
            raise RuntimeError("Agent run completed without model output")
        return final_segment
