from pathlib import Path
from textwrap import dedent
from typing import Any

from ...agent.server.schema import InboundMessage, InputText


def build_system_reminder(content: str) -> dict[str, Any]:
    """将运行时提示包装成带 system-reminder 标签的 user 消息。"""
    return {
        "role": "user",
        "content": f"<system-reminder>\n{content.strip()}\n</system-reminder>",
    }


def is_system_message(message: dict[str, Any]) -> bool:
    """识别 system 消息和带 system-reminder 标签的前置 user 上下文。"""
    if message.get("role") == "system":
        return True
    if message.get("role") != "user":
        return False
    # 原始用户请求即使包含同名标签，也不能当作运行时上下文剔除。
    if "timestamp" in message:
        return False
    content = message.get("content")
    return (
        isinstance(content, str)
        and content.strip().startswith("<system-reminder>")
        and content.strip().endswith("</system-reminder>")
    )


def split_system_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """将固定系统提示词与动态前置上下文一起从对话历史分离。"""
    system_messages = []
    other_messages = []
    for message in messages:
        if is_system_message(message):
            system_messages.append(message)
        else:
            other_messages.append(message)
    return system_messages, other_messages


def remove_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """剔除 system 消息及前置 system-reminder 上下文，保留真实对话。"""
    return split_system_messages(messages)[1]


def prepend_system_messages(
    messages: list[dict[str, Any]],
    system_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [*system_messages, *messages]


def set_system_prompt(
    messages: list[dict[str, Any]],
    system_prompt: dict[str, Any],
    context_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """用本轮的最新提示词替换旧前缀，不向对话追加上下文版本。"""
    return [system_prompt, *context_messages, *remove_system_messages(messages)]


def build_inbound_message_metadata(*metas: dict[str, str], sort: bool = False) -> str:
    """将 dict[str,str] 元属性转为适合 AI 阅读的字符串。

    Args:
        meta: 元属性字典，例如 {"author": "tom", "mime": "text/plain"}
        tag: 外层包裹标签，传空字符串 "" 则不包裹
        sort: 是否按键名排序（让输出更稳定，便于 AI 阅读）

    Returns:
        格式化后的字符串

    Examples:
        >>> build({"author": "tom", "size": "1KB"})
        'author: tom\nsize: 1KB'
    """
    if not metas:
        body = ""
    else:
        meta = {k: v for d in metas for k, v in d.items()}
        # 排序可保证输出稳定，避免字典无序导致 Prompt 抖动
        items = sorted(meta.items()) if sort else meta.items()
        # 将值中的换行转义，防止破坏整体格式
        lines = [f"{k}: {v.replace(chr(10), '\\n')}" for k, v in items]
        body = "\n".join(lines)
    return body


async def build_user_prompt(message: InboundMessage, workspace: Path):

    # 时间和元数据
    parts = [
        {
            "type": "text",
            "text": build_inbound_message_metadata(
                message.session_metadata,
                {
                    "user_id": message.user_id,
                    "session_id": message.session_id,
                    "channel": message.channel,
                },
                {"sender_name": message.sender_name},
                {"timestamp": message.timestamp.strftime("%Y-%m-%d %H:%M (%A)")},
            ),
        }
    ]

    files: list[Path] = []

    for input in message.content:
        if isinstance(input, InputText):
            parts.append({"type": "text", "text": input.text})
    files.extend(message.files)
    # 文件提示词
    file_paths: str = ""
    for file in files:
        file_paths += f"- {file.relative_to(workspace).as_posix()}"
    if file_paths:
        parts.append(
            {
                "type": "text",
                "text": dedent(
                    f"""
                        ## Attachments
                        The user has uploaded {len(files)} file(s). You may reference them:

                    """
                ).lstrip()
                + file_paths,
            }
        )
    return {
        "role": "user",
        "timestamp": message.timestamp.timestamp(),
        "content": parts,
    }
