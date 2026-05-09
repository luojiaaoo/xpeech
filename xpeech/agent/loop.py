from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import ProviderChatKwargs
from pathlib import Path


class AgentLoop:
    """Agent循环处理逻辑。"""

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        provider_chat_kwargs: ProviderChatKwargs,
        max_iterations: int = 20,
    ):

        self.provider = provider
        self.workspace = workspace
        self.provider_chat_kwargs = provider_chat_kwargs.to_dict()
        self.max_iterations = max_iterations
