from textwrap import dedent
from datetime import datetime
from ...agent.memory import MemoryStore

BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]


def _load_bootstrap_files(workspace) -> str:
    """Load all bootstrap files from workspace."""
    parts = []

    for filename in BOOTSTRAP_FILES:
        file_path = workspace / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            parts.append(f"## {filename}\n\n{content}")

    return "\n\n".join(parts) if parts else ""


def _get_identity(workspace: str) -> str:
    """Get the core identity section."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

    return dedent(
        f"""
            # xpeech 🍑

            You are xpeech, a helpful AI assistant.
            Why xpeech?
            answer: Xpeech blends the articulation of "speech" with the power and vitality of "X+peach".
            
            You have access to tools that allow you to:
            - Read, write, and edit files
            - Execute shell commands
            - Send messages to users

            ## Current Time
            {now}

            ## Workspace
            Your workspace is at: {workspace}
            - Long-term memory: {workspace}/memory/MEMORY.md (write important facts here)
            - History log: {workspace}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
            - Custom skills: {workspace}/skills/{{skill-name}}/SKILL.md

            ## Guidelines
            - State intent before tool calls, but NEVER predict or claim results before receiving them.
            - Before modifying a file, read it first. Do not assume files or directories exist.
            - After writing or editing a file, re-read it if accuracy matters.
            - If a tool call fails, analyze the error before retrying with a different approach.
            - Ask for clarification when the request is ambiguous.
        """
    ).lstrip()


async def build_system_prompt(workspace: str) -> str:
    parts = []
    parts.append(_get_identity(workspace))
    parts.append(_load_bootstrap_files(workspace))
    parts.append(await MemoryStore(workspace).get_memory_context())
    return {"role": "system", "content": "\n\n".join(parts)}
