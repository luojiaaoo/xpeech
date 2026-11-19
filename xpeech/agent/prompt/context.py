"""按类型生成每轮重新读取的前置 user 上下文。"""

from pathlib import Path
from textwrap import dedent
from typing import Any

import anyio

from ...config.settings import settings
from ...utils.file_io import LockedFile
from ...utils.helper import token_counter
from ..memory import MemoryStore
from ..skills.skill import SkillsLoader
from .helper import build_system_reminder

BOOTSTRAP_FILES = ("AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md")


def _get_identity(channel: str | None = None) -> str:
    """生成身份上下文，渠道配置优先，否则使用全局配置或默认身份。"""
    channel_config = settings.jwt.channels.get(channel) if channel else None
    system_identity_prompt = (
        channel_config.system_identity_prompt.strip()
        if channel_config and channel_config.system_identity_prompt.strip()
        else settings.llm.system_identity_prompt.strip()
    )
    if system_identity_prompt:
        system_identity_prompt = (
            dedent(
                """
                {system_identity_prompt}

                You are a helpful AI assistant.
                Your name is not xpeech.
                Do not introduce yourself as xpeech, do not claim that your name is xpeech, and do not treat xpeech as your identity.
            """
            )
            .lstrip()
            .format(system_identity_prompt=system_identity_prompt)
        )
    else:
        system_identity_prompt = dedent(
            """
                # xpeech 🍑

                You are xpeech, a helpful AI assistant.
                Why xpeech?
                answer: Xpeech blends the articulation of "speech" with the power and vitality of "X+peach".

                Your github link: https://github.com/luojiaaoo/xpeech
            """
        ).lstrip()

    return system_identity_prompt


def _workspace_context(workspace: Path) -> str:
    return dedent(
        """
        ## Workspace
        Your workspace is at: {workspace}
        - Home directory: {workspace}/home (`~` resolves to this directory)
        - Agent instructions: {workspace}/AGENTS.md (workspace-specific operating rules, task guidance, and behavioral requirements to follow while working)
        - Personality settings: {workspace}/SOUL.md (the assistant's identity, personality, values, tone, and communication style)
        - User information: {workspace}/USER.md (the user's profile, language, timezone, preferences, and other user-specific context)
        - Attachments: {workspace}/attachments/YYYY-MM-DD/ (uploaded files grouped by upload date)
        - Tool results: {workspace}/tool-results/YYYY-MM-DD/ (oversized full outputs grouped by creation date and named with the tool name and a unique ID)
        - Long-term memory: {workspace}/memory/MEMORY.md (write important facts here)
        - History log: {workspace}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
        - Skills: {workspace}/skills/{{skill-name}}/SKILL.md
        - Tool notes: {workspace}/TOOLS.md. Private configurations for tools
        """
    ).strip().format(workspace=workspace.as_posix())


async def build_context_messages(
    workspace: Path,
    *,
    user_id: str | None = None,
    channel: str | None,
    user_organization: str | None,
    not_interactive: bool,
    token_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """组装动态上下文时同步统计各块占用；空内容不产生消息或统计项。"""
    messages = []

    async def add(category: str, content: str) -> None:
        if content.strip():
            messages.append(build_system_reminder(content))
            token_counts[category] = token_counts.get(category, 0) + await token_counter([messages[-1]])

    await add("用户上下文", _get_identity(channel))
    await add("用户上下文", _workspace_context(workspace))
    for filename in BOOTSTRAP_FILES:
        path = workspace / filename
        if await anyio.Path(path).is_file():
            content = await LockedFile(path).read_text()
            if content.strip():
                await add("用户上下文", f"## {filename}\n\n{content}")

    memory = MemoryStore(workspace)
    await add("长期记忆", await memory.get_memory_context())
    await add("用户上下文", await memory.get_self_improvement_context())
    await add("技能", await SkillsLoader(
        workspace, user_id=user_id, channel=channel, user_organization=user_organization,
        not_interactive=not_interactive,
    ).build_skills_prompt())
    return messages
