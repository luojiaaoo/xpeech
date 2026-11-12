"""Render a selected view of a conversation as a self-contained HTML file."""

import argparse
import json
from html import escape
from pathlib import Path


LEVELS = {"detailed": "详细分享", "conversation": "对话分享", "summary": "摘要分享"}
ROLES = {"user": "用户", "assistant": "助手", "tool": "工具"}

CSS = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #1e293b; background: #f1f5f9; }
body { margin: 0; padding: 32px 16px; }
main { max-width: 900px; margin: auto; }
header, article, details { background: white; padding: 20px; margin-bottom: 16px;
    border: 1px solid #cbd5e1; border-radius: 12px; }
h1 { font-size: 1.6rem; margin-top: 0; }
h2, summary { font-size: 1rem; font-weight: 600; }
h2 { margin-top: 0; }
summary { cursor: pointer; }
.content { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }
.scope, .level { color: #475569; }
.user { border-left: 4px solid #2563eb; }
.tool { border-left: 4px solid #64748b; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; }
@media (max-width: 600px) { body { padding: 16px 8px; } header, article, details { padding: 16px; } }
@media print { body { background: white; padding: 0; } }
"""


def text_field(data: dict, key: str, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return escape(value)


def tool_block(label: str, content: str) -> str:
    return f'<details class="tool"><summary>{label}</summary><pre>{content}</pre></details>'


def render_share(data: dict) -> str:
    level = data.get("level")
    if level not in LEVELS:
        raise ValueError("level must be detailed, conversation or summary")
    title = text_field(data, "title", "当前会话分享")
    scope = text_field(data, "scope_note")
    sections = []
    if level == "summary":
        summary = text_field(data, "summary")
        if not summary.strip():
            raise ValueError("summary must not be empty")
        sections.append(f'<article><h2>会话摘要</h2><div class="content">{summary}</div></article>')
    else:
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for message in messages:
            role = message.get("role")
            if role not in ROLES or (role == "tool" and level != "detailed"):
                continue
            content = text_field(message, "content")
            if role == "tool":
                label = text_field(message, "name", "工具")
                sections.append(tool_block(f"工具返回：{label}", content))
            elif content.strip():
                sections.append(
                    f'<article class="{role}"><h2>{ROLES[role]}</h2>'
                    f'<div class="content">{content}</div></article>'
                )
            if role == "assistant" and level == "detailed":
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    name = text_field(function, "name")
                    arguments = text_field(function, "arguments")
                    sections.append(tool_block(f"工具调用：{name}", arguments))
        if not sections:
            raise ValueError("No shareable messages for the selected level")
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body><main>
<header><h1>{title}</h1><p class="level">{LEVELS[level]}</p><p class="scope">{scope}</p></header>
{body}
</main></body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    html = render_share(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
