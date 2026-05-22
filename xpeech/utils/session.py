from pathlib import Path
from textwrap import dedent
import aiofiles


async def create_workspace_templates(workspace: Path):
    """Create default workspace template files."""
    templates = {
        "AGENTS.md": dedent(
            """
                # Agent Instructions

                You are a helpful AI assistant. Be concise, accurate, and friendly.

                ## Guidelines

                - Always explain what you're doing before taking actions
                - Ask for clarification when the request is ambiguous
                - Use tools to help accomplish tasks
                - Remember important information in memory/MEMORY.md; past events are logged in memory/history.jsonl
            """
        ).lstrip(),
        "SOUL.md": dedent(
            """
                # Soul

                I am a lightweight AI assistant.

                ## Personality

                - Helpful and friendly
                - Concise and to the point
                - Curious and eager to learn

                ## Values

                - Accuracy over speed
                - User privacy and safety
                - Transparency in actions
            """
        ).lstrip(),
        "USER.md": dedent(
            """
                # User

                Information about the user goes here.

                ## Preferences

                - Communication style: (casual/formal)
                - Timezone: (your timezone)
                - Language: (your preferred language)
            """
        ).lstrip(),
    }

    for filename, content in templates.items():
        file_path = workspace / filename
        if not file_path.exists():
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

    # Create memory directory and MEMORY.md
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        async with aiofiles.open(memory_file, "w", encoding="utf-8") as f:
            await f.write(
                dedent(
                    """
                        # Long-term Memory

                        This file stores important information that should persist across sessions.

                        ## User Information

                        (Important facts about the user)

                        ## Preferences

                        (User preferences learned over time)

                        ## Project Context

                        (Information about ongoing projects)

                        ## Important Notes

                        (Things to remember)

                        ---

                        *This file is automatically updated by xpeech when important information should be remembered.*
                    """
                ).lstrip()
            )
    history_file = memory_dir / "HISTORY.md"
    if not history_file.exists():
        async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
            await f.write("")

    # Create skills directory for custom user skills
    skills_dir = workspace / "skills"
    skills_dir.mkdir(exist_ok=True)
