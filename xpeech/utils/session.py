from pathlib import Path
from textwrap import dedent

from .helper import ensure_path_async, write_text_async


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
        "TOOLS.md": dedent(
            """
                # TOOLS

                Local Tools Configuration Notes.
                Skills define how tools are used. This file records your specific configuration, belonging to your environment and your settings.

                ## What to Record

                Specific parameters and preferences needed when the skill runs. For example:

                - Preferred style for image generation
                - Default voice for speech generation
                - Any unique configuration related to your environment

                ## Why It's Separated

                Skills are shared, while configurations are personal. Keeping them separate ensures that updating the skill won't lose your notes, and sharing the skill won't expose your environment.

                ---

                Record anything that helps you work. This is your memo.
            """
        ).lstrip(),
    }

    for filename, content in templates.items():
        file_path = workspace / filename
        if not file_path.exists():
            await write_text_async(file_path, content)

    # Create memory directory and MEMORY.md
    memory_dir = await ensure_path_async(workspace / "memory")
    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        await write_text_async(
            memory_file,
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
            ).lstrip(),
        )
    history_file = memory_dir / "HISTORY.md"
    if not history_file.exists():
        await write_text_async(history_file, "")

    # Create skills directory for custom user skills
    await ensure_path_async(workspace / "skills")

    # Each workspace owns an isolated HOME for sandboxed commands.
    await ensure_path_async(workspace / "home")
