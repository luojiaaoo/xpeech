from collections.abc import AsyncIterator
from typing import Any, Self

from ..config.settings import Settings, settings
from ..provider.litellm_provider import LiteLLMProvider
from ..utils.file_io import ensure_path_async
from ..utils.session import create_workspace_templates
from .loop import AgentLoop
from .server.schema import InboundMessage
from .tools.registry import register_default_tools


class AgentRunner:
    """Prepare and run an agent in one user's workspace."""

    def __init__(self, user_id: str, *, config: Settings) -> None:
        self.user_id = user_id
        self.config = config
        self.workspace = (config.path.workspace_base_path / self.user_id).resolve()
        self._agent_loop: AgentLoop | None = None

    @classmethod
    async def create(
        cls,
        user_id: str,
        *,
        config: Settings = settings,
    ) -> Self:
        """Create a fully initialized runner for one user."""
        runner = cls(user_id, config=config)
        await runner._initialize()
        return runner

    async def _initialize(self) -> None:
        """Initialize the user workspace, provider, tools, and agent loop."""
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
            user_id=self.user_id,
            provider=provider,
            workspace=self.workspace,
            config=self.config.tool,
        )
        self._agent_loop = agent_loop

    async def run(
        self,
        message: InboundMessage,
        *,
        not_interactive: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one inbound message and yield the agent's events."""
        if message.user_id != self.user_id:
            raise ValueError("Inbound message user_id does not match the runner user")

        if self._agent_loop is None:
            raise RuntimeError("AgentRunner must be created with AgentRunner.create()")

        async for event in self._agent_loop.run(
            message=message,
            not_interactive=not_interactive,
        ):
            yield event

    async def run_not_interactive(
        self,
        message: InboundMessage,
    ) -> list[dict[str, Any]]:
        """Run without interactive tools and collect events, merging consecutive text chunks."""
        events: list[dict[str, Any]] = []
        text_chunks: list[str] = []

        async for event in self.run(message, not_interactive=True):
            context = event.get("context")
            is_text_chunk = event.get("event") in {"assistant", "thinking"} and isinstance(context, str)
            if text_chunks:
                previous_metadata = {key: value for key, value in events[-1].items() if key != "context"}
                metadata = {key: value for key, value in event.items() if key != "context"}
                if is_text_chunk and metadata == previous_metadata:
                    text_chunks.append(context)
                    continue
                events[-1]["context"] = "".join(text_chunks)
                text_chunks.clear()

            if event.get("event") in {"assistant_end", "thinking_end"}:
                continue

            events.append(event.copy())
            if is_text_chunk:
                text_chunks.append(context)

        if text_chunks:
            events[-1]["context"] = "".join(text_chunks)
        return events
