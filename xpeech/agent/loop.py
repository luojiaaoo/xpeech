from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import ProviderChatKwargs
from pathlib import Path
from .tools.filesystem import build_file_tools
from itertools import count


class AgentLoop:
    """Agent循环处理逻辑。"""

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        provider_chat_kwargs: ProviderChatKwargs,
        max_iterations: int = None,
    ):

        self.provider = provider
        self.workspace = workspace
        self.provider_chat_kwargs = provider_chat_kwargs.to_dict()
        self.max_iterations = max_iterations

        # 注册默认工具
        self.register_default_tools()

        # 添加一个标志位，表示Agent是否正在运行
        self._running = False

    def register_default_tools(self):
        """注册默认工具。"""

        read_file, write_file, edit_file, list_dir = build_file_tools(self.workspace)
        self.provider.register_tool(read_file)
        self.provider.register_tool(write_file)
        self.provider.register_tool(edit_file)
        self.provider.register_tool(list_dir)

    async def save_history_jsonl(self, history):
        """保存历史记录到JSONL文件。"""
        pass

    async def load_history_jsonl(self):
        """从JSONL文件加载历史记录。"""
        pass

    async def run(self):
        """运行Agent循环。"""
        self._running = True
        for loop_count in count():
            if not self._running:
                break
            if self.max_iterations and loop_count >= self.max_iterations:
                break
