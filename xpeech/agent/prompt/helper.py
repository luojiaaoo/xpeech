from ...agent.server.schema import InboundMessage, InputText, InputImage
from pathlib import Path
from textwrap import dedent
from ...utils.helper import save_to_workspace
import base64
from typing import Any


def build_inbound_message_metadata(*metas: dict[str, str], tag: str = "metadata", sort: bool = False) -> str:
    """将 dict[str,str] 元属性转为适合 AI 阅读的字符串。

    Args:
        meta: 元属性字典，例如 {"author": "tom", "mime": "text/plain"}
        tag: 外层包裹标签，传空字符串 "" 则不包裹
        sort: 是否按键名排序（让输出更稳定，便于 AI 阅读）

    Returns:
        格式化后的字符串

    Examples:
        >>> build({"author": "tom", "size": "1KB"})
        '<metadata>\nauthor: tom\nsize: 1KB\n</metadata>'
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
    if tag:
        return f"<{tag}>\n{body}\n</{tag}>"
    return body


async def build_user_prompt(message: InboundMessage, workspace: Path, support_image: bool):

    # 时间和元数据
    parts = [
        {
            "type": "text",
            "text": build_inbound_message_metadata(
                message.session_metadata, {"timestamp": message.timestamp.strftime("%Y-%m-%d %H:%M (%A)")}
            ),
        }
    ]

    files: list[Path] = []

    # 多模态消息
    image_idx = 1
    for input in message.content:
        if isinstance(input, InputText):
            parts.append({"type": "text", "text": input.text})
        elif isinstance(input, InputImage):
            if support_image:
                parts.append({"type": "image_url", "image_url": {"url": input.image_url, "detail": "auto"}})
            else:
                # 不支持图片以文件形式存储，再让工具去解析
                files.append(await save_to_workspace(input, workspace=workspace, idx=image_idx))
                image_idx += 1
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


def build_image_content_blocks(raw: bytes, mime: str, path: str, label: str) -> list[dict[str, Any]]:
    """Build native image blocks plus a short text label."""
    b64 = base64.b64encode(raw).decode()
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
            "_meta": {"path": path},
        },
        {"type": "text", "text": label},
    ]
