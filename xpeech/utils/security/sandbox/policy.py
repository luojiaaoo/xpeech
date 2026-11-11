"""Command policy for the workspace execution environment."""

import re
import shlex
from collections.abc import Callable
from pathlib import Path


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
        return Path(x.strip()).name.lower()

    def is_python_pip(x: str) -> bool:
        return bool(python_pip_re.match(basename(x)))

    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except Exception:  # noqa: BLE001 -- malformed commands still need conservative inspection.
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


SHELL_BLOCK_RULES: dict[Callable[[str], bool], str] = {
    is_direct_python_pip_exec: (
        "run Python through 'uv run python/pip' instead of calling 'python/pip' directly."
    ),
}


def guard_command(command: str) -> str:
    """Apply explicit block rules; filesystem access is enforced by bubblewrap."""
    cmd = command.strip()

    for check, message in SHELL_BLOCK_RULES.items():
        if check(cmd):
            raise RuntimeError(f"Command blocked by safety guard: {message}")

    return cmd


