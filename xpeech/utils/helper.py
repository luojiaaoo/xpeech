from asyncer import asyncify
import inspect
import tiktoken
from pydantic_ai.messages import ModelMessage
from pathlib import Path


def ensure_async(func):
    """ 函数异步化 """
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return asyncify(func)


def ensure_dirpath(path: Path) -> Path:
    """确保路径的父目录存在。若路径指向文件则创建其父目录，否则创建路径本身（视为目录）。"""
    if path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def format_exception2llm(e: Exception) -> str:
    """ 给大模型看的异常内容 """
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
