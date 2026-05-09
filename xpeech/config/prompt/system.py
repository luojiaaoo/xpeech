from textwrap import dedent
from datetime import datetime


def _get_identity(workspace) -> str:
    """Get the core identity section."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

    return dedent(
        f"""# xpeech 🍑

            You are xpeech, a super peach AI assistant. You have access to tools that allow you to:
            - Read, write, and edit files
            - Send messages to users

            ## Current Time
            {now}

            ## Workspace
            Your workspace is at: {workspace}
            """
    ).lstrip()


def build_system_prompt() -> str:
    return {
        "role": "system",
        "content": dedent(
            f"""
                {_get_identity()}
            """
        ).lstrip(),
    }
