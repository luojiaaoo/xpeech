from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import ProviderChatKwargs
from pathlib import Path
from .tools.filesystem import build_file_tools
from itertools import count
from ..config.settings import settings
from typing import Any
import json
from .server.schema import InboundMessage
import aiofiles
from ..config.prompt.system import build_system_prompt
from ..config.prompt.helper import build_user_prompt


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
        self.provider.register_tool()(read_file)
        self.provider.register_tool()(write_file)
        self.provider.register_tool()(edit_file)
        self.provider.register_tool()(list_dir)

    async def save_history_json(self, session_id: str, history: list[dict[str, Any]]):
        """保存历史记录到json文件。"""
        file = settings.path.session_history_path / f"{session_id}.json"
        async with aiofiles.open(file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(history, ensure_ascii=False))

    async def load_history_json(self, session_id: str) -> list[dict[str, Any]]:
        """从json文件加载历史记录。"""
        file = settings.path.session_history_path / f"{session_id}.json"
        if not file.exists():
            return []
        async with aiofiles.open(file, "r", encoding="utf-8") as f:
            content: list[dict[str, Any]] = await f.read()
        # 剔除系统提示词
        if content[0]["role"] == "system":
            content = content[1:]
        return content

    async def run(self, message: InboundMessage):
        """运行Agent循环。"""

        messages_json = await self.load_history_json(message.session_id)
        # 拼接系统提示词
        messages_json.insert(0, build_system_prompt())
        # 拼接用户消息
        messages_json.insert(
            -1,
            build_user_prompt(
                message=message,
                workspace=self.workspace,
                support_image=self.provider.support_image,
            ),
        )

        self._running = True
        for loop_count in count():
            if not self._running:
                break
            if self.max_iterations and loop_count >= self.max_iterations:
                break
            self.provider.chat(messages=messages_json)
