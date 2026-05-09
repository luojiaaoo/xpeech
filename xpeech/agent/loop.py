from tkinter import N
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
from ..agent.tools.helper import get_tool_model_cls
from ..utils.helper import ensure_async
from ..provider.schema import ToolCallRequest


class AgentLoop:
    """Agent循环处理逻辑。"""

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        provider_chat_kwargs: ProviderChatKwargs | None = None,
        max_iterations: int | None = None,
    ):

        self.provider = provider
        self.workspace = workspace
        self.provider_chat_kwargs = {} if provider_chat_kwargs is None else provider_chat_kwargs.to_dict()
        self.max_iterations = max_iterations

        # 注册默认工具
        self.register_default_tools()

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
            await f.write(json.dumps(history, indent=4, ensure_ascii=False))

    async def load_history_json(self, session_id: str) -> list[dict[str, Any]]:
        """从json文件加载历史记录。"""

        file = settings.path.session_history_path / f"{session_id}.json"
        if not file.exists():
            return []
        async with aiofiles.open(file, "r", encoding="utf-8") as f:
            content: list[dict[str, Any]] = json.loads(await f.read())
        # 剔除系统提示词
        content = [i for i in content if i["role"] != "system"]
        return content

    async def run(self, message: InboundMessage):
        """运行一次Agent循环，处理一次用户消息。"""

        messages_json = await self.load_history_json(message.session_id)
        # 拼接系统提示词
        messages_json.insert(0, build_system_prompt(self.workspace))
        # 拼接用户消息
        messages_json.append(
            build_user_prompt(
                message=message,
                workspace=self.workspace,
                support_image=self.provider.support_image,
            ),
        )

        final_content = None
        for loop_count in count():
            if self.max_iterations is not None and loop_count >= self.max_iterations:
                final_content = (
                    f"Agent loop has reached the maximum number of iterations({self.max_iterations}) and stop."
                )
                break

            response = await self.provider.chat(
                messages=messages_json,
                tools=[],
                **self.provider_chat_kwargs,
            )
            # 输出思考内容
            yield "data: {}\n\n".format(
                json.dumps({"event": "thinking", "context": response.reasoning_content or ""}, ensure_ascii=False)
            )
            # 如果有工具调用
            if response.has_tool_calls:
                # 输出工具调用内容
                yield "data: {}\n\n".format(
                    json.dumps(
                        {"event": "assistant", "context": response.content}, ensure_ascii=False
                    )
                )
                # 输出助手消息
                yield "data: {}\n\n".format(
                    json.dumps(
                        {"event": "tool_call", "context": [i.name for i in response.tool_calls]}, ensure_ascii=False
                    )
                )
                # 还原工具调用格式
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                # 创建助手消息
                msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
                if tool_call_dicts:
                    msg["tool_calls"] = tool_call_dicts
                messages_json.append(msg)

                # 执行工具调用
                for tool_call in response.tool_calls:
                    tool_call: ToolCallRequest = tool_call
                    model_cls = get_tool_model_cls(tool_call_func := self.provider.mapping_tool_call_funcs[tool_call.name])
                    result = await tool_call_func(model_cls(**tool_call.arguments))
                    # 创建工具调用结果消息
                    messages_json.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        }
                    )

                # 即将达到最大迭代次数，添加用户消息，提示达到最大迭代次数
                if self.max_iterations is not None and loop_count == self.max_iterations - 2:
                    messages_json.append(
                        {
                            "role": "user",
                            "content": "You have reached the maximum number of iterations and must stop calling tools.",
                        }
                    )

            else:
                # 没有工具，结束循环
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 拼接助手消息
        messages_json.append({"role": "assistant", "content": final_content})
        # 输出助手消息
        yield "data: {}\n\n".format(json.dumps({"event": "assistant", "context": final_content}, ensure_ascii=False))
        # 保存历史记录
        await self.save_history_json(message.session_id, messages_json)
