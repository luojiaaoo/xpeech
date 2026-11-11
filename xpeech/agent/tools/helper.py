import inspect
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Type, get_type_hints

import anyio
from openai import pydantic_function_tool
from pydantic import BaseModel

from ...config.settings import settings
from ...exceptions import PathProtectionError
from ...utils.file_io import is_relative_path
from ...utils.helper import dynamic_import
from ...utils.security import sandbox
from ..skills.skill import BUILTIN_SKILLS_DIR


async def get_mime_type(path: Path) -> str | None:
    """Return a best-effort MIME type, preferring file content over its name."""
    try:
        async with await anyio.open_file(path, "rb") as file:
            data = await file.read(12)
    except OSError:
        data = b""

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        return mime_type

    return None


def _expand_sandbox_home_path(user_path: Path, workspace: Path) -> Path:
    path_text = str(user_path)
    if path_text == "~":
        return sandbox.get_sandbox_home(workspace)
    if path_text.startswith("~/") or path_text.startswith("~\\"):
        return sandbox.get_sandbox_home(workspace) / path_text[2:]
    if path_text.startswith("~"):
        raise PathProtectionError("Named-user home paths are not supported; use '~/' for the sandbox HOME")
    return user_path


def safe_resolve_workspace_path(
    user_path: Path,
    workspace: Path,
    protect_read_only_file: bool = True,
) -> Path:
    """Resolve a workspace path, including read-only sandbox mount mappings."""
    workspace = workspace.resolve()
    path = _expand_sandbox_home_path(user_path, workspace)

    if path.is_absolute():
        resolved_path = path.resolve()
    else:
        resolved_path = (workspace / path).resolve()

    # 先确保用户输入没有逃离工作区。
    if not is_relative_path(path_target=resolved_path, base=workspace):
        raise PathProtectionError("Path escapes workspace")

    # shell 工具会把公共 HOME 配置映射到工作区 HOME，
    # 文件工具在沙盒外运行，需要在这里做相同的路径映射。
    workspace_home = workspace / "home"
    try:
        home_relative = resolved_path.relative_to(workspace_home)
    except ValueError:
        pass
    else:
        if home_relative.parts:
            config_root = settings.path.sandbox_home_path
            config_path = config_root.joinpath(*home_relative.parts)
            if config_path.is_file():
                if protect_read_only_file:
                    raise PathProtectionError("Sandbox HOME config files are read-only")
                return config_path.resolve()

    # 非 skills 目录下的路径无需进行内置技能转换。
    workspace_skills = workspace / "skills"
    try:
        skill_relative = resolved_path.relative_to(workspace_skills)
    except ValueError:
        return resolved_path

    # 访问 workspace/skills
    if not skill_relative.parts:
        return resolved_path

    # 内置技能路径转换
    builtin_skill = (BUILTIN_SKILLS_DIR / skill_relative.parts[0]).resolve()
    if (builtin_skill / "SKILL.md").is_file():
        if protect_read_only_file:
            raise PathProtectionError("Built-in skills are read-only")
        return builtin_skill.joinpath(*skill_relative.parts[1:]).resolve()
    else:
        return resolved_path


class EmptyToolArgs(BaseModel):
    """无参数"""


@dataclass(frozen=True)
class ToolModelInfo:
    model_cls: type[BaseModel]
    has_pydantic_param: bool
    has_session_metadata: bool
    has_user_id: bool
    has_sender_name: bool
    has_channel: bool
    has_session_id: bool
    has_tool_executor: bool


INJECTED_TOOL_PARAMETERS = frozenset({
    "session_metadata", "user_id", "sender_name", "channel", "session_id", "tool_executor",
})


def get_tool_model_cls(func: Callable) -> ToolModelInfo:
    """返回工具参数模型及参数标记；无 Pydantic 参数时使用空模型。"""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    has_session_metadata = "session_metadata" in sig.parameters
    has_user_id = "user_id" in sig.parameters
    has_sender_name = "sender_name" in sig.parameters
    has_channel = "channel" in sig.parameters
    has_session_id = "session_id" in sig.parameters
    has_tool_executor = "tool_executor" in sig.parameters
    params = [
        param
        for param in sig.parameters.values()
        if param.name not in INJECTED_TOOL_PARAMETERS
    ]
    if len(params) == 1:
        param = params[0]
        model_cls = hints.get(param.name)

        if model_cls is None:
            raise TypeError(f"Parameter '{param.name}' in '{func.__name__}' must have a type annotation")

        if not inspect.isclass(model_cls) or not issubclass(model_cls, BaseModel):
            raise TypeError(
                f"Parameter '{param.name}' in '{func.__name__}' must be annotated "
                f"with a BaseModel subclass, got {model_cls!r}"
            )
        return ToolModelInfo(
            model_cls,
            True,
            has_session_metadata,
            has_user_id,
            has_sender_name,
            has_channel,
            has_session_id,
            has_tool_executor,
        )
    elif len(params) == 0:
        return ToolModelInfo(
            EmptyToolArgs,
            False,
            has_session_metadata,
            has_user_id,
            has_sender_name,
            has_channel,
            has_session_id,
            has_tool_executor,
        )
    raise ValueError(
        f"Function '{func.__name__}' must accept either 0 parameters "
        f"or exactly 1 parameter annotated with a BaseModel subclass, "
        f"optionally with session_metadata, user_id, sender_name, channel, session_id, and tool_executor"
    )


def as_tool(func: Callable[Type[BaseModel] | None, str | dict], name_suffix: str = "") -> dict:
    """
    将函数转换为 OpenAI Tool Schema。

    支持无参数函数或一个 BaseModel 参数，两者均可额外声明
    session_metadata、user_id、sender_name、channel、session_id 和 tool_executor。这些参数由运行时注入，
    不包含在工具 schema 中。
    """
    # 名称：函数名 + 后缀
    tool_name = func.__name__ + name_suffix

    # 描述：用函数的 docstring
    description = (func.__doc__ or "").strip()
    if not description:
        raise ValueError(f"Function '{func.__name__}' must have a docstring")

    model_info = get_tool_model_cls(func)

    return pydantic_function_tool(model_info.model_cls, name=tool_name, description=description)


_MODULE_CUSTOM_CACHE = {}


def get_custom_tool_func(function_name: str) -> list[dict]:
    """
    获取自定义工具函数。
    """
    function_name_paths = function_name.split(".")
    _name = settings.llm.tools_python_package
    module = dynamic_import(_name)
    if _name not in _MODULE_CUSTOM_CACHE:
        _MODULE_CUSTOM_CACHE[_name] = module
    _temp = None
    for i in function_name_paths:
        if _temp is None:
            _temp = getattr(_MODULE_CUSTOM_CACHE[_name], i)
        else:
            _temp = getattr(_temp, i)
    return _temp
