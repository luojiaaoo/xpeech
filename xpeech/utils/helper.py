from asyncer import asyncify
import inspect
import tiktoken
from pydantic_ai.messages import ModelMessage


def ensure_async(func):
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return asyncify(func)


def format_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def estimate_pydantic_ai_tokens(messages: list[ModelMessage], model="gpt-4o") -> int:
    """提前估算 PydanticAI 消息的 Token 数"""
    from pydantic_core import to_jsonable_python

    messages = to_jsonable_python(messages)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    total_tokens = 0

    for msg in messages:
        if "usage" in msg and "input_tokens" in msg["usage"]:
            total_tokens += msg["usage"]["input_tokens"]
            total_tokens += msg["usage"]["output_tokens"]
        else:
            total_tokens += 3
            for part in msg["parts"]:
                total_tokens += len(enc.encode(str(part["content"])))
    return total_tokens
