from ..provider.litellm_provider import LiteLLMProvider
from ..provider.schema import ProviderChatKwargs
from pathlib import Path
from .tools.filesystem import build_file_tools
from .tools.shell import build_shell_tools
from .tools.web import web_fetch, web_search
from .tools.office import office_read
from .tools.file_message import build_file_message_tools
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
from ..utils.helper import LiteralDumper, format_exception2llm, token_counter
from ..provider.schema import LLMResponse
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
        tools: list[str],
        summary_tokens: int = 8192,
        provider_chat_kwargs: ProviderChatKwargs | None = None,
        max_iterations: int | None = None,
    ):

        self.provider = provider
        self.workspace = workspace
        self.tools = tools
        self.summary_tokens = summary_tokens
        self.provider_chat_kwargs = {} if provider_chat_kwargs is None else provider_chat_kwargs.to_dict()
        self.max_iterations = max_iterations
        self.max_accept_token = int(self.provider.default_context_token * 0.9 - self.summary_tokens)

        # 注册默认工具
        self.register_default_tools()

    # ----------------- 默认工具 -----------------

    def register_default_tools(self):
        """注册默认工具。"""

        # 文件读写
        read_image, read_video, read_file, write_file, edit_file, list_dir = build_file_tools(
            workspace=self.workspace,
            restrict_tools_to_workspace=settings.tool.restrict_tools_to_workspace,
        )
        if self.provider.support_image:
            self.provider.register_tool()(read_image)
        if self.provider.support_video:
            self.provider.register_tool()(read_video)
        self.provider.register_tool()(read_file)
        self.provider.register_tool()(write_file)
        self.provider.register_tool()(edit_file)
        self.provider.register_tool()(list_dir)

        # shell执行
        exec = build_shell_tools(
            workspace=self.workspace,
            restrict_tools_to_workspace=settings.tool.restrict_tools_to_workspace,
        )
        self.provider.register_tool()(exec)

        # web fetch
        self.provider.register_tool()(web_fetch)
        self.provider.register_tool()(web_search)

        # office document reader
        self.provider.register_tool()(office_read)

        # outbound send file messages
        send_file = build_file_message_tools(
            workspace=self.workspace,
            restrict_tools_to_workspace=settings.tool.restrict_tools_to_workspace,
        )
        self.provider.register_tool()(send_file)

    # ----------------- history会话读写 -----------------

    async def del_history_yaml(self, session_id: str):
        """删除历史记录文件。"""
        file = settings.path.session_history_path / f"{session_id}.yaml"
        if file.exists():
            file.unlink()
        logger.info("Session history deleted")

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
        logger.info("Session history saved")

    async def load_history_yaml(self, session_id: str) -> list[dict[str, Any]]:
        """从yaml文件加载历史记录。"""

        file = settings.path.session_history_path / f"{session_id}.yaml"
        if not file.exists():
            return []
        async with aiofiles.open(file, "r", encoding="utf-8") as f:
            content: list[dict[str, Any]] = yaml.safe_load(await f.read()) or []
        # 剔除系统提示词
        content = [i for i in content if i["role"] != "system"]
        logger.info("Session history loaded messages={}", len(content))
        return content

    # ----------------- agent loop tool call -----------------

    async def tool_call(self, response: LLMResponse, messages_yaml: list, loop_count: int):
        logger.info(
            "Processing tool calls loop_count={} count={}",
            loop_count,
            len(response.tool_calls),
        )

        # 输出工具调用内容
        if response.content and response.content.strip():
            yield {"event": "assistant", "context": response.content}

        # 输出工具调用消息
        yield {
            "event": "tool_call",
            "context": json.dumps([(i.id, i.name, i.arguments) for i in response.tool_calls]),
        }

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
        with_metas = []
        for tool_call in response.tool_calls:
            tool_call: ToolCallRequest = tool_call
            model_cls = get_tool_model_cls(tool_call_func := response.mapping_tool_call_funcs[tool_call.name])

            # 如果没有参数，则直接调用工具函数
            try:
                if model_cls is None:
                    result = await tool_call_func()
                else:
                    result = await tool_call_func(model_cls(**tool_call.arguments))
            except Exception as e:
                logger.warning(
                    "Tool call failed loop_count={} tool_name={} args={} exception={}",
                    loop_count,
                    tool_call.name,
                    tool_call.arguments,
                    format_exception2llm(e),
                )
                result = format_exception2llm(e)
            else:
                logger.info(
                    "Tool call completed loop_count={} tool_name={}",
                    loop_count,
                    tool_call.name,
                )

            # 创建工具调用结果消息
            if isinstance(result, str):
                messages_yaml.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            elif isinstance(result, list):
                # 把带_meta属性的字典过滤出来
                others = []
                for i in result:
                    if isinstance(i, dict) and "_meta" in i:
                        # 提取_meta属性，组装成消息
                        _meta = i.pop("_meta")
                        with_metas.extend([{"type": "text", "text": _meta}, i])
                    else:
                        others.append(i)
                messages_yaml.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": others,
                    }
                )
            tool_call_result.append((tool_call.id, tool_call.name, result))

        # 如果包含_meta属性，则把这类消息转成user消息
        if with_metas:
            messages_yaml.append(
                {
                    "role": "user",
                    "content": with_metas,
                }
            )

        # 输出工具调用结果消息
        yield {"event": "tool_call_result", "context": json.dumps(tool_call_result)}

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

        logger.info("Agent run started workspace={}", self.workspace)
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
                yield {"event": "command", "context": "/new -> start a new session"}
                return
            elif command == "/new":
                rt = await self.consolidate_memory(messages_yaml[:-1])
                await self.del_history_yaml(message.session_id)
                yield {"event": "command", "context": f"NEW SESSION, {rt}"}
                return
            yield {
                "event": "command",
                "context": f"Oops! I don't recognize {command}. Try entering /help for a list of commands.",
            }
            return

        final_content = None
        for loop_count in count():
            if self.max_iterations is not None and loop_count >= self.max_iterations:
                final_content = (
                    f"Agent loop has reached the maximum number of iterations({self.max_iterations}) and stop."
                )
                logger.warning(
                    "Agent loop reached max iterations loop_count={} max_iterations={}",
                    loop_count,
                    self.max_iterations,
                )
                break

            # 清理用户消息中的时间戳，开始对话
            _clean_messages = self.clear_role_user_timestamp(messages_yaml)

            # 判断是否需要压缩
            if await self.need_compress(messages_yaml):
                messages_yaml = await self.compress(messages_yaml)

            logger.info("Calling provider chat loop_count={}", loop_count)

            try:
                response = await self.provider.chat(
                    messages=_clean_messages,
                    tools=self.tools,
                    **self.provider_chat_kwargs,
                )
            except Exception:
                logger.exception(
                    "Provider chat failed loop_count={}",
                    loop_count,
                )
                raise
            logger.info(
                "Provider chat completed loop_count={} has_tool_calls={}",
                loop_count,
                response.has_tool_calls,
            )

            # 输出思考内容
            if response.reasoning_content and response.reasoning_content.strip():
                yield {"event": "thinking", "context": response.reasoning_content}

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
            logger.warning("Agent loop finished without final content")

        # 拼接助手消息
        messages_yaml.append({"role": "assistant", "content": final_content})

        # 输出助手消息
        yield {"event": "assistant", "context": final_content}

        # 保存历史记录
        await self.save_history_yaml(message.session_id, messages_yaml)

        # 输出token使用百分比
        token_count = await token_counter(messages_yaml)
        token_percent = min(1, token_count / self.max_accept_token) * 100
        token_notify = "♻️ 即将达到最大令牌数，强制压缩" if token_percent > 90 else ""
        yield {
            "event": "token_usage",
            "context": f"{token_percent:.2f}% ({token_count // 1000}k / {self.max_accept_token // 1000}k) {token_notify}",
        }

        logger.info("Agent run completed")

    # ----------------- 压缩 -----------------

    async def need_compress(self, messages):
        totol_token = await token_counter(messages=messages)
        return totol_token >= self.max_accept_token

    async def is_finish_compress(self, messages):
        totol_token = await token_counter(messages=messages)
        max_accept_token = int(self.provider.default_context_token * 0.4)
        return totol_token < max_accept_token

    @classmethod
    def is_send_user_message(cls, message: dict):
        """判断是否是用户主动发送的消息， 主动发送的消息会打上时间戳"""
        return message["role"] == "user" and "timestamp" in message

    @classmethod
    def clear_role_user_timestamp(cls, messages: list[dict]):
        rt = []
        for i in messages:
            if i["role"] == "user" and cls.is_send_user_message(i):
                # 不修改原始消息
                i = i.copy()
                # 剔除时间戳
                i.pop("timestamp", None)
            rt.append(i)
        return rt

    async def compress(self, messages):

        logger.info("Compressing messages messages={}", len(messages))

        await self.consolidate_memory(messages)

        def truncate_tool_result(_messages: list, truncate_count: int):
            rt = []
            for i in _messages:
                if i["role"] == "tool":
                    i["content"] = i["content"][:truncate_count]
                rt.append(i)
            return rt

        def split_recent_user_messages(_messages: list[dict], keep_count: int) -> int:
            count = 0
            for i in range(len(_messages) - 1, -1, -1):
                if not (_messages[i]["role"] == "user" and self.is_send_user_message(_messages[i])):
                    continue
                count += 1
                if count == keep_count:
                    return i
            return 0

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
                logger.exception("Failed to summarize history")
                raise
            _messages = [
                *system_messages,
                {"role": "assistant", "content": summary},
            ]
            return _messages

        def keep_messages_for_day(days, _messages):
            system_messages = [i for i in _messages if i["role"] == "system"]
            _messages = [i for i in _messages if i["role"] != "system"]
            last_timestamp = None
            idx_split = 0
            for i in range(len(_messages) - 1, -1, -1):
                if not (_messages[i]["role"] == "user" and self.is_send_user_message(_messages[i])):
                    continue
                if last_timestamp is None:
                    last_timestamp = _messages[i]["timestamp"]
                _timestamp = _messages[i]["timestamp"]
                if last_timestamp - _timestamp > timedelta(days=days).total_seconds():
                    break
                idx_split = i
            _messages = [*system_messages, *_messages[idx_split:]]
            return _messages

        # 一级压缩：超过1000字的工具执行结果（保留最近4次对话消息的结果调用）
        keep_count = 4
        idx_split_keep = split_recent_user_messages(messages, keep_count)
        messages = truncate_tool_result(messages[:idx_split_keep], 1000) + messages[idx_split_keep:]
        if await self.is_finish_compress(messages):
            logger.info("Compression finished level=1 messages={}", len(messages))
            return messages

        # 二级压缩：只保留最后一条消息的时间戳，往前追溯7/6/5/4/3/2天的消息
        for days in range(7, 1, -1):
            messages = keep_messages_for_day(days, messages)
            if await self.is_finish_compress(messages):
                logger.info("Compression finished level=2 messages={}", len(messages))
                return messages

        # 三级压缩：AI总结历史消息，只保留最近4次对话消息
        for keep_count in range(4, 0, -1):
            idx_split_keep = split_recent_user_messages(messages, keep_count)
            recent_messages = messages[idx_split_keep:]
            if (idx_split_keep > 0 and await self.is_finish_compress(recent_messages)) or keep_count == 1:
                logger.info("Compression level=3 summarizing history")
                messages = await summary_messages(messages[:idx_split_keep]) + recent_messages
                logger.info("Compression finished level=3 messages={}", len(messages))
                return messages

    # ----------------- 记忆和历史(在/new 或者 压缩的时候触发) -----------------

    async def consolidate_memory(self, mesages):
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
        logger.info("Consolidating memory")
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
            logger.info("Consolidating memory finished")
            return "[INFO] Consolidating memory finished"
        else:
            logger.warning("Consolidating memory failed")
            return "[WARNING] Consolidating memory failed"
