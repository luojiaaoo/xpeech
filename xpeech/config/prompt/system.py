from textwrap import dedent
from datetime import datetime


def _get_identity() -> str:
    """Get the core identity section."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

    return dedent(
        f"""# xpeech 🍑

            You are xpeech, a helpful AI assistant.
            Why xpeech?
            answer: Xpeech blends the articulation of "speech" with the power and vitality of "X+peach".
            
            You have access to tools that allow you to:
            - Read, write, and edit files
            - Send messages to users

            ## Current Time
            {now}

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
