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
from .prompt.system import build_system_prompt
from .prompt.helper import build_user_prompt
from ..agent.tools.helper import get_tool_model_cls
from ..provider.schema import ToolCallRequest
import yaml
from ..utils.helper import LiteralDumper, format_exception2llm
from ..provider.schema import LLMResponse
from litellm import token_counter
from datetime import timedelta
from loguru import logger
from ..agent.server.schema import InputText
from .memory import MemoryStore
from .prompt.compress import SUMMARY_PROMPT


class AgentLoop:
    """Agent循环处理逻辑。"""

    INTERATION_STOP_PROMPT = "You have reached the maximum number of iterations and MUST stop calling tools."

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

    # ----------------- 默认工具 -----------------

    def register_default_tools(self):
        """注册默认工具。"""

        # 文件读写
        read_image, read_file, write_file, edit_file, list_dir = build_file_tools(
            workspace=self.workspace,
            restrict_tools_to_workspace=settings.path.restrict_tools_to_workspace,
        )
        if self.provider.support_image:
            self.provider.register_tool()(read_image)
        self.provider.register_tool()(read_file)
        self.provider.register_tool()(write_file)
        self.provider.register_tool()(edit_file)
        self.provider.register_tool()(list_dir)

        # shell执行
        exec = build_shell_tools(
            workspace=self.workspace,
            restrict_tools_to_workspace=settings.path.restrict_tools_to_workspace,
        )
        self.provider.register_tool()(exec)

    # ----------------- history会话读写 -----------------

    async def del_history_yaml(self, session_id: str):
        """删除历史记录文件。"""
        file = settings.path.session_history_path / f"{session_id}.yaml"
        if file.exists():
            file.unlink()
        logger.info("Session history deleted session_id={}", session_id)

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
        logger.info("Session history saved session_id={}", session_id)

    async def load_history_yaml(self, session_id: str) -> list[dict[str, Any]]:
        """从yaml文件加载历史记录。"""

        file = settings.path.session_history_path / f"{session_id}.yaml"
        if not file.exists():
            return []
        async with aiofiles.open(file, "r", encoding="utf-8") as f:
            content: list[dict[str, Any]] = yaml.safe_load(await f.read()) or []
        # 剔除系统提示词
        content = [i for i in content if i["role"] != "system"]
        logger.info("Session history loaded session_id={} messages={}", session_id, len(content))
        return content

    # ----------------- agent loop tool call -----------------

    async def tool_call(self, response: LLMResponse, messages_yaml: list, loop_count: int, session_id: str):
        logger.info(
            "Processing tool calls session_id={} loop_count={} count={}",
            session_id,
            loop_count,
            len(response.tool_calls),
        )

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
            model_cls = get_tool_model_cls(tool_call_func := response.mapping_tool_call_funcs[tool_call.name])
            try:
                result = await tool_call_func(model_cls(**tool_call.arguments))
            except Exception as e:
                logger.warning(
                    "Tool call failed session_id={} loop_count={} tool_name={} args={}",
                    session_id,
                    loop_count,
                    tool_call.name,
                    tool_call.arguments,
                )
                result = format_exception2llm(e)
            else:
                logger.info(
                    "Tool call completed session_id={} loop_count={} tool_name={}",
                    session_id,
                    loop_count,
                    tool_call.name,
                )

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
            logger.warning(
                "Approaching max iterations loop_count={} max_iterations={}",
                loop_count,
                self.max_iterations,
            )
            messages_yaml.append(
                {
                    "role": "user",
                    "content": self.INTERATION_STOP_PROMPT,
                }
            )

    # ----------------- agent loop run -----------------

    async def run(self, message: InboundMessage):
        """运行一次Agent循环，处理一次用户消息。"""

        logger.info("Agent run started session_id={} workspace={}", message.session_id, self.workspace)
        messages_yaml: list[dict] = await self.load_history_yaml(message.session_id)
        # 拼接系统提示词
        messages_yaml.insert(0, await build_system_prompt(workspace=self.workspace))
        # 拼接用户消息（给user添加时间戳字典，用于二级压缩）
        messages_yaml.append(
            await build_user_prompt(
                message=message,
                workspace=self.workspace,
                support_image=self.provider.support_image,
            ),
        )

        # 用户命令拦截器
        if (
            len(message.content) == 1
            and isinstance(message.content[0], InputText)
            and (command := message.content[0].text).startswith("/")
        ):
            command = command.strip()
            if command == "/help":
                yield "data: {}\n\n".format(
                    json.dumps({"event": "command", "context": "/new -> start a new session"}, ensure_ascii=False)
                )
                return
            elif command == "/new":
                rt = await self.consolidate_memory(messages_yaml[:-1], message.session_id)
                await self.del_history_yaml(message.session_id)
                yield "data: {}\n\n".format(
                    json.dumps({"event": "command", "context": f"NEW SESSION, {rt}"}, ensure_ascii=False)
                )
                return
            yield "data: {}\n\n".format(
                json.dumps(
                    {
                        "event": "command",
                        "context": f"Oops! I don't recognize {command}. Try entering /help for a list of commands.",
                    },
                    ensure_ascii=False,
                )
            )
            return

        # 进行压缩
        messages_yaml = await self.compress(messages_yaml, message.session_id)

        final_content = None
        for loop_count in count():
            if self.max_iterations is not None and loop_count >= self.max_iterations:
                final_content = (
                    f"Agent loop has reached the maximum number of iterations({self.max_iterations}) and stop."
                )
                logger.warning(
                    "Agent loop reached max iterations session_id={} loop_count={} max_iterations={}",
                    message.session_id,
                    loop_count,
                    self.max_iterations,
                )
                break

            # 清理用户消息中的时间戳，开始对话
            _clean_messages = self.clear_role_user_timestamp(messages_yaml)
            logger.info("Calling provider chat session_id={} loop_count={}", message.session_id, loop_count)
            try:
                response = await self.provider.chat(
                    messages=_clean_messages,
                    tools=[],
                    **self.provider_chat_kwargs,
                )
            except Exception:
                logger.exception(
                    "Provider chat failed session_id={} loop_count={}",
                    message.session_id,
                    loop_count,
                )
                raise
            logger.info(
                "Provider chat completed session_id={} loop_count={} has_tool_calls={}",
                message.session_id,
                loop_count,
                response.has_tool_calls,
            )

            # 输出思考内容
            if response.reasoning_content and response.reasoning_content.strip():
                yield "data: {}\n\n".format(
                    json.dumps({"event": "thinking", "context": response.reasoning_content}, ensure_ascii=False)
                )

            # 如果有工具调用
            if response.has_tool_calls:
                async for i in self.tool_call(response, messages_yaml, loop_count, message.session_id):
                    yield i

            else:
                # 没有工具，结束循环
                final_content = response.content
                break

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
            logger.warning("Agent loop finished without final content session_id={}", message.session_id)

        # 拼接助手消息
        messages_yaml.append({"role": "assistant", "content": final_content})

        # 输出助手消息
        yield "data: {}\n\n".format(json.dumps({"event": "assistant", "context": final_content}, ensure_ascii=False))

        # 保存历史记录
        await self.save_history_yaml(message.session_id, messages_yaml)
        logger.info("Agent run completed session_id={}", message.session_id)

    # ----------------- 压缩 -----------------

    def need_compress(self, messages):
        totol_token = token_counter(model="gpt-4o", messages=messages)
        max_accept_token = self.provider.default_context_token * 0.9 - self.summary_tokens
        return totol_token >= max_accept_token

    def is_finish_compress(self, messages):
        totol_token = token_counter(model="gpt-4o", messages=messages)
        max_accept_token = self.provider.default_context_token * 0.4
        return totol_token < max_accept_token

    @classmethod
    def is_iterations_stop_user_message(cls, message: dict):
        return message["role"] == "user" and message["content"] == cls.INTERATION_STOP_PROMPT

    @classmethod
    def clear_role_user_timestamp(cls, messages: list[dict]):
        rt = []
        for i in messages:
            if i["role"] == "user" and not cls.is_iterations_stop_user_message(i):
                # 不修改原始消息
                i = i.copy()
                # 剔除时间戳
                i.pop("timestamp", None)
            rt.append(i)
        return rt

    async def compress(self, messages, session_id):
        # 如果不需要压缩，返回原始消息
        if not self.need_compress(messages):
            return messages
        logger.info("Compressing messages session_id={} messages={}", session_id, len(messages))

        await self.consolidate_memory(messages, session_id)

        def truncate_tool_result(_messages: list, truncate_count: int):
            rt = []
            for i in _messages:
                if i["role"] == "tool":
                    i["content"] = i["content"][:truncate_count]
                rt.append(i)
            return rt

        async def summary_messages(_messages: list[dict]):
            _clean_messages = self.clear_role_user_timestamp(_messages)
            system_messages = [i for i in _clean_messages if i["role"] == "system"]
            _clean_messages = [i for i in _clean_messages if i["role"] != "system"]
            _clean_messages.insert(0, {"role": "system", "content": SUMMARY_PROMPT})
            _clean_messages.append({"role": "user", "content": "Please summarize the history messages."})
            try:
                summary = (
                    await self.provider.chat(
                        messages=_clean_messages,
                        max_tokens=self.summary_tokens,
                        top_p=0.1,
                        remove_all_tools=True,
                    )
                ).content
            except Exception:
                logger.exception("Failed to summarize history session_id={}", session_id)
                raise
            _messages = [
                *system_messages,
                {"role": "assistant", "content": summary},
            ]
            return _messages

        def keep_messages_for_day(days, _messages):
            last_timestamp = None
            idx_split = 0
            for i in range(len(_messages) - 1, -1, -1):
                if not (_messages[i]["role"] == "user" and not self.is_iterations_stop_user_message(_messages[i])):
                    continue
                if last_timestamp is None:
                    last_timestamp = _messages[i]["timestamp"]
                _timestamp = _messages[i]["timestamp"]
                if last_timestamp - _timestamp > timedelta(days=days).total_seconds():
                    break
                idx_split = i
            _messages = _messages[idx_split:]
            return _messages

        # 一级压缩：超过1000字的工具执行结果（保留最近4次对话消息的结果调用）
        keep_count = 4
        keep_count = 4
        idx_split_keep = len(messages)
        count = 0
        for i in range(len(messages) - 1, -1, -1):
            if not (messages[i]["role"] == "user" and not self.is_iterations_stop_user_message(messages[i])):
                continue
            count += 1
            if count == keep_count:  # 保留4次对话
                idx_split_keep = i
        messages = truncate_tool_result(messages[:idx_split_keep], 1000) + messages[idx_split_keep:]
        if self.is_finish_compress(messages):
            logger.info("Compression finished level=1 session_id={} messages={}", session_id, len(messages))
            return messages

        # 二级压缩：只保留最后一条消息的时间戳，往前追溯7/6/5/4/3/2天的消息
        for days in range(7, 1, -1):
            messages = keep_messages_for_day(days, messages)
            if self.is_finish_compress(messages):
                logger.info("Compression finished level=2 session_id={} messages={}", session_id, len(messages))
                return messages

        # 三级压缩：AI总结历史消息，只保留最近4次对话消息
        if self.is_finish_compress(messages[idx_split_keep:]):
            logger.info("Compression level=3 summarizing history session_id={}", session_id)
            messages = await summary_messages(messages[:idx_split_keep]) + messages[idx_split_keep:]
            logger.info("Compression finished level=3 session_id={} messages={}", session_id, len(messages))
            return messages

        # 四级压缩：如果仍然不满足要求，则进行完全压缩
        logger.info("Compression level=4 summarizing all history session_id={}", session_id)
        compressed_messages = await summary_messages(messages)
        logger.info("Compression finished level=4 session_id={} messages={}", session_id, len(compressed_messages))
        return compressed_messages

    # ----------------- 记忆和历史(在/new 或者 压缩的时候触发) -----------------

    async def consolidate_memory(self, mesages, session_id):
        memory_store = MemoryStore(workspace=self.workspace)
        _clean_messages = self.clear_role_user_timestamp(mesages)
        _clean_messages = [i for i in _clean_messages if i["role"] != "system"]
        if not _clean_messages:
            return "[INFO] No messages to consolidate"
        _system_prompt = "## Current Long-term Memory\n" + (await memory_store.get_memory_context() or "(empty)")
        _clean_messages.insert(0, {"role": "system", "content": _system_prompt})
        _clean_messages.append(
            {
                "role": "user",
                "content": "Process this conversation and MUST call the save_memory tool with your consolidation.",
            }
        )
        logger.info("Consolidating memory session_id={}", session_id)
        response = await self.provider.chat(
            messages=_clean_messages,
            max_tokens=self.summary_tokens,
            top_p=0.7,
            tools=[memory_store.save_memory],
            remove_default_tools=True,
        )
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_call: ToolCallRequest = tool_call
                model_cls = get_tool_model_cls(tool_call_func := response.mapping_tool_call_funcs[tool_call.name])
                await tool_call_func(model_cls(**tool_call.arguments))
            logger.info("Consolidating memory finished session_id={}", session_id)
            return "[INFO] Consolidating memory finished"
        else:
            logger.warning("Consolidating memory failed session_id={}", session_id)
            return "[WARNING] Consolidating memory failed"
