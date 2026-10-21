"""Agent tool backed by the shared channel form protocol."""

from xpeech.protocol.questions import (  # noqa: F401
    USER_TIMEOUT,
    QuestionArgs,
    QuestionForm,
    validate_question_json,
)


def ask_user_question(args: QuestionArgs) -> str:
    """通过通用问题表单 JSON 向用户追问关键信息。"""
    return args.validate_and_format()
