import inspect
from pathlib import Path
from typing import Callable, get_type_hints, Type
from pydantic import BaseModel, create_model
from ...utils.helper import dynamic_import, is_relative_path
from ...config.settings import settings
from ..skills.skill import BUILTIN_SKILLS_DIR
from . import sandbox
from openai import pydantic_function_tool
import os
import re
import shlex


def _expand_sandbox_home_path(user_path: str | Path) -> Path:
    path_text = str(user_path)
    if path_text == "~":
        return sandbox.get_sandbox_home()
    if path_text.startswith("~/") or path_text.startswith("~\\"):
        return sandbox.get_sandbox_home() / path_text[2:]
    return Path(user_path).expanduser()


def safe_resolve_workspace_path(
    user_path: str | Path,
    workspace: str | Path,
    include_builtin_skills_path: bool = False,
) -> Path:
    """Resolve a user path and ensure it stays inside the allowed tool roots."""
    base = Path(workspace).expanduser().resolve()
    path = _expand_sandbox_home_path(user_path, base)

    if path.is_absolute():
        resolved_path = path.resolve()
    else:
        resolved_path = (base / path).resolve()

    if is_relative_path(path_target=resolved_path, base=base):
        return resolved_path

    if include_builtin_skills_path and is_relative_path(
        path_target=resolved_path,
        base=BUILTIN_SKILLS_DIR,
    ):
        return resolved_path

    raise PermissionError(f"Path escapes workspace: {user_path}")


def get_tool_model_cls(func: Callable[Type[BaseModel] | None, str | dict]) -> type[BaseModel]:
    """
    获取工具函数的参数类型注解。
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    params = list(sig.parameters.values())
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
        return model_cls
    elif len(params) == 0:
        return None


def as_tool(func: Callable[Type[BaseModel] | None, str | dict], name_suffix: str = "") -> dict:
    """
    将函数转换为 OpenAI Tool Schema。

    支持两种函数签名：
    1. 无参数函数
    2. 有且仅有一个参数，且该参数类型为 BaseModel 的子类
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    params = list(sig.parameters.values())

    # 名称：函数名 + 后缀
    tool_name = func.__name__ + name_suffix

    # 描述：用函数的 docstring
    description = (func.__doc__ or "").strip()
    if not description:
        raise ValueError(f"Function '{func.__name__}' must have a docstring")

    # 情况 1：无参数
    if len(params) == 0:
        model_cls = create_model(f"{tool_name}Args")

    # 情况 2：单个 BaseModel 参数
    elif len(params) == 1:
        param = params[0]
        model_cls = hints.get(param.name)

        if model_cls is None:
            raise TypeError(f"Parameter '{param.name}' in '{func.__name__}' must have a type annotation")

        if not inspect.isclass(model_cls) or not issubclass(model_cls, BaseModel):
            raise TypeError(
                f"Parameter '{param.name}' in '{func.__name__}' must be annotated "
                f"with a BaseModel subclass, got {model_cls!r}"
            )

    # 其他情况：不支持
    else:
        raise ValueError(
            f"Function '{func.__name__}' must accept either 0 parameters "
            f"or exactly 1 parameter annotated with a BaseModel subclass"
        )

    return pydantic_function_tool(model_cls, name=tool_name, description=description)


def is_direct_python_pip_exec(cmd: str) -> bool:
    """
    True  = 直接执行了 python/pip，应该拦截
    False = 没有直接执行 python/pip，或者是通过 uv 执行，允许
    拦截:
        python app.py
        python3 app.py
        pip install xxx
        /usr/bin/python3 app.py
        sudo python3 app.py
        env python3 app.py
        echo ok && python3 app.py
    放行:
        uv run python app.py
        uv run python3 app.py
        uv pip install xxx
        /usr/bin/uv run python app.py
        echo "python3"
    """
    python_pip_re = re.compile(
        r"^(?:python(?:2|3)?(?:\.\d+)?|pip(?:2|3)?(?:\.\d+)?)(?:\.exe)?$",
        re.IGNORECASE,
    )
    assign_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

    def basename(x: str) -> str:
        return os.path.basename(x.strip()).lower()

    def is_python_pip(x: str) -> bool:
        return bool(python_pip_re.match(basename(x)))

    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except Exception:
        # 解析失败时保守拦截
        return bool(
            re.search(
                r"(?:^|[;&|]\s*)"
                r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
                r"(?:sudo\s+|env\s+|command\s+|exec\s+|nohup\s+|doas\s+)*"
                r"(?:\./|\.\./|~/|/(?:[^/\s]+/)*)?"
                r"(python(?:2|3)?(?:\.\d+)?|pip(?:2|3)?(?:\.\d+)?)(?:\.exe)?"
                r"(?=$|[\s;&|<>])",
                cmd,
                re.IGNORECASE,
            )
        )
    separators = {";", "&", "&&", "|", "||"}
    parts = []
    current = []
    for tok in tokens:
        if tok in separators:
            if current:
                parts.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        parts.append(current)
    for part in parts:
        i = 0
        # 跳过开头环境变量: FOO=bar python3
        while i < len(part) and assign_re.match(part[i]):
            i += 1
        while i < len(part):
            exe = basename(part[i])
            # 通过 uv 执行，直接放行这个命令片段
            if exe == "uv":
                break
            # sudo python3 / doas python3
            if exe in {"sudo", "doas"}:
                i += 1
                while i < len(part):
                    t = part[i]
                    if t == "--":
                        i += 1
                        break
                    if t.startswith("-"):
                        i += 1
                        # sudo -u root python3
                        if t in {"-u", "--user", "-g", "--group", "-h", "--host"} and i < len(part):
                            i += 1
                        continue
                    break
                continue
            # env FOO=bar python3
            if exe == "env":
                i += 1
                while i < len(part):
                    t = part[i]
                    if t == "--":
                        i += 1
                        break
                    if t.startswith("-") or assign_re.match(t):
                        i += 1
                        continue
                    break
                continue
            # command python3 / exec python3 / nohup python3
            if exe in {"command", "exec", "nohup"}:
                i += 1
                # command -v python3 只是查询，不算执行
                if exe == "command" and i < len(part) and part[i] in {"-v", "-V"}:
                    break
                while i < len(part) and part[i].startswith("-"):
                    i += 1
                continue
            # 真正执行的命令
            if is_python_pip(part[i]):
                return True
            break
    return False


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
