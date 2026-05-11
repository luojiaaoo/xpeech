from tkinter import N
from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import ProviderChatKwargs
from pathlib import Path
from .tools.filesystem import build_file_tools
from .tools.shell import build_shell_tools
from itertools import count
from ..config.settings import settings
from typing import Any
import json
from .server.schema import InboundMessage
import aiofiles
from ..config.prompt.system import build_system_prompt
from ..config.prompt.helper import build_user_prompt
from ..agent.tools.helper import get_tool_model_cls
from ..provider.schema import ToolCallRequest
import yaml
from ..utils.helper import LiteralDumper, format_exception2llm
from ..provider.schema import LLMResponse
from litellm import token_counter
from datetime import timedelta


class AgentLoop:
    """Agent循环处理逻辑。"""

    INTERATION_STOP_PROPT = "You have reached the maximum number of iterations and MUST stop calling tools."

    def __init__(
        self,
        provider: LiteLLMProvider,
        workspace: Path,
        summary_tokens: int = 8195,
        provider_chat_kwargs: ProviderChatKwargs | None = None,
        max_iterations: int | None = None,
    ):

        self.provider = provider
        self.workspace = workspace
        self.summary_tokens = summary_tokens
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
        exec = build_shell_tools(self.workspace)
        self.provider.register_tool()(exec)

    async def save_history_yaml(self, session_id: str, history: list[dict[str, Any]]):
        """保存历史记录到yaml文件。"""

        file = settings.path.session_history_path / f"{session_id}.yaml"
        async with aiofiles.open(file, "w", encoding="utf-8") as f:
            await f.write(
                yaml.dump(
                    history,
                    Dumper=LiteralDumper,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=4,
                    sort_keys=False,
                    width=1000,
                )
            )

    async def load_history_yaml(self, session_id: str) -> list[dict[str, Any]]:
        """从yaml文件加载历史记录。"""

        file = settings.path.session_history_path / f"{session_id}.yaml"
        if not file.exists():
            return []
        async with aiofiles.open(file, "r", encoding="utf-8") as f:
            content: list[dict[str, Any]] = yaml.safe_load(await f.read()) or []
        # 剔除系统提示词
        content = [i for i in content if i.get("role") != "system"]
        return content

    async def tool_call(self, response: LLMResponse, messages_yaml: list, loop_count: int):
        # 输出工具调用内容
        if response.content and response.content.strip():
            yield "data: {}\n\n".format(
                json.dumps({"event": "assistant", "context": response.content}, ensure_ascii=False)
            )
        # 输出工具调用消息
        yield "data: {}\n\n".format(
            json.dumps(
                {
                    "event": "tool_call",
                    "context": json.dumps([(i.id, i.name, i.arguments) for i in response.tool_calls]),
                },
                ensure_ascii=False,
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
        messages_yaml.append(msg)

        # 执行工具调用
        tool_call_result = []
        for tool_call in response.tool_calls:
            tool_call: ToolCallRequest = tool_call
            model_cls = get_tool_model_cls(tool_call_func := self.provider.mapping_tool_call_funcs[tool_call.name])
            try:
                result = await tool_call_func(model_cls(**tool_call.arguments))
            except Exception as e:
                result = format_exception2llm(e)
            # 创建工具调用结果消息
            messages_yaml.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": result,
                }
            )
            tool_call_result.append((tool_call.id, tool_call.name, result))
        # 输出工具调用结果消息
        yield "data: {}\n\n".format(
            json.dumps({"event": "tool_call_result", "context": json.dumps(tool_call_result)}, ensure_ascii=False)
        )

        # 即将达到最大迭代次数，添加用户消息，提示达到最大迭代次数
        if self.max_iterations is not None and loop_count == self.max_iterations - 2:
            messages_yaml.append(
                {
                    "role": "user",
                    "content": self.INTERATION_STOP_PROPT,
                }
            )

    def need_compress(self, messages):
        totol_token = token_counter(model="gpt-4o", messages=messages)
        max_accept_token = self.provider.defaulr_context_token * 0.9 - self.summary_tokens
        return totol_token >= max_accept_token

    def is_finish_compress(self, messages):
        totol_token = token_counter(model="gpt-4o", messages=messages)
        max_accept_token = self.provider.defaulr_context_token * 0.4
        return totol_token < max_accept_token

    @classmethod
    def clear_role_user_timestamp(cls, messages: list[dict]):
        rt = []
        for i in messages:
            if i["role"] == "user" and i["content"] != cls.INTERATION_STOP_PROPT:
                # 不修改原始消息
                i = i.copy()
                # 剔除时间戳
                i.pop("timestamp", None)
            rt.append(i)
        return rt

    def compress(self, messages):
        def truncate_tool_result(messages: list, truncate_count: int):
            rt = []
            for i in messages:
                if i["role"] == "tool":
                    i["content"] = i["content"][:truncate_count]
                rt.append(i)
            return rt

        # 如果不需要压缩，返回原始消息
        if not self.need_compress(messages):
            return messages

        # 一级压缩：超过1000字的工具执行结果（保留最近4次对话消息的结果调用）
        idx_split = None
        count = 0
        for i in range(len(messages) - 1, -1, -1):
            if not (messages[i]["role"] == "user" and messages[i]["content"] != self.INTERATION_STOP_PROPT):
                continue
            count += 1
            if count == 4:  # 保留4次对话
                idx_split = i
        messages = truncate_tool_result(messages[:idx_split], 1000) + messages[idx_split:]
        if self.is_finish_compress(messages):
            return messages

        # 二级压缩：只保留最后一条消息的时间戳，往前追溯1天的消息
        last_timestamp = None
        idx_split = None
        for i in range(len(messages) - 1, -1, -1):
            if not (messages[i]["role"] == "user" and messages[i]["content"] != self.INTERATION_STOP_PROPT):
                continue
            if last_timestamp is None:
                last_timestamp = messages[i]["timestamp"]
            _timestamp = messages[i]["timestamp"]
            if last_timestamp - _timestamp > timedelta(days=1).total_seconds():
                break
            idx_split = i
        messages = messages[idx_split:]

        # 三级压缩：AI总结（保留最近4次对话消息）
        ...

    async def run(self, message: InboundMessage):
        """运行一次Agent循环，处理一次用户消息。"""

        messages_yaml: list[dict] = await self.load_history_yaml(message.session_id)
        # 拼接系统提示词
        messages_yaml.insert(0, build_system_prompt(workspace=self.workspace))
        # 拼接用户消息（给user添加时间戳字典，用于二级压缩）
        messages_yaml.append(
            build_user_prompt(
                message=message,
                workspace=self.workspace,
                support_image=self.provider.support_image,
            ),
        )
        # 进行压缩
        messages_yaml = self.compress(messages_yaml)

        final_content = None
        for loop_count in count():
            if self.max_iterations is not None and loop_count >= self.max_iterations:
                final_content = (
                    f"Agent loop has reached the maximum number of iterations({self.max_iterations}) and stop."
                )
                break

            # 清理用户消息中的时间戳，开始对话
            _clean_messages = self.clear_role_user_timestamp(messages_yaml)
            response = await self.provider.chat(
                messages=_clean_messages,
                tools=[],
                **self.provider_chat_kwargs,
            )

            # 输出思考内容
            if response.reasoning_content and response.reasoning_content.strip():
                yield "data: {}\n\n".format(
                    json.dumps({"event": "thinking", "context": response.reasoning_content}, ensure_ascii=False)
                )

            # 如果有工具调用
            if response.has_tool_calls:
                async for i in self.tool_call(response, messages_yaml, loop_count):
                    yield i
            else:
                # 没有工具，结束循环
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 拼接助手消息
        messages_yaml.append({"role": "assistant", "content": final_content})

        # 输出助手消息
        yield "data: {}\n\n".format(json.dumps({"event": "assistant", "context": final_content}, ensure_ascii=False))

        # 保存历史记录
        await self.save_history_yaml(message.session_id, messages_yaml)
