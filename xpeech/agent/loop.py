from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import ProviderChatKwargs
from pathlib import Path
from .tools.filesystem import build_file_tools


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
        self.register_default_tools()

    def register_default_tools(self):
        """注册默认工具。"""
        read_file, write_file, edit_file, list_dir = build_file_tools(self.workspace)
        self.provider.register_tool(read_file)
        self.provider.register_tool(write_file)
        self.provider.register_tool(edit_file)
        self.provider.register_tool(list_dir)
