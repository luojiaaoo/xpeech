"""Bubblewrap command wrapping for shell command execution."""

import os
import shlex
from pathlib import Path
from ...utils.helper import ensure_path

from ...config.settings import settings
from ..skills.skill import BUILTIN_SKILLS_DIR

_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _add_setenv(args: list[str], key: str, value: Path | str) -> None:
    args.extend(["--setenv", key, str(value)])


def get_sandbox_home() -> Path:
    """Return the shared sandbox home for all sessions under a workspace base."""
    return ensure_path(settings.path.sandbox_home_path.resolve())


def _sandbox_path(shared_home: Path) -> str:
    inherited_path = os.environ.get("PATH") or ""
    path_entries = [
        str(shared_home / ".local" / "bin"),
        str(shared_home / ".npm-global" / "bin"),
        inherited_path,
        _DEFAULT_PATH,
    ]
    return ":".join(entry for entry in path_entries if entry)


def wrap_command(command: str, workspace: str | Path) -> list[str]:
    """Wrap a command in a bubblewrap sandbox."""
    workspace = Path(workspace).expanduser().resolve()
    workspace_python_env = workspace / ".venv"
    shared_home = get_sandbox_home()
    builtin_skills = BUILTIN_SKILLS_DIR.resolve()

    args = ["bwrap", "--new-session", "--die-with-parent"]

    for key, value in {
        "HOME": shared_home,  # 共享的用户主目录
        "PATH": _sandbox_path(shared_home),  # 共享的 PATH 环境变量
        "PIP_REQUIRE_VIRTUALENV": "true",  # 确保 pip 在虚拟环境中运行
        "UV_PROJECT_ENVIRONMENT": workspace_python_env,  # uv虚拟py环境路径
        "UV_CACHE_DIR": shared_home / ".cache" / "uv",  # 包缓存目录
        "NPM_CONFIG_PREFIX": shared_home / ".npm-global",  # npm 全局安装目录
    }.items():
        _add_setenv(args, key, value)

    args.extend(["--ro-bind", "/usr", "/usr"])
    for path in (
        "/bin",
        "/lib",
        "/lib64",
        "/opt",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/pki/tls/certs",
        "/etc/pki/ca-trust",
        "/etc/crypto-policies",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
        "/etc/hosts",
    ):
        args.extend(["--ro-bind-try", path, path])

    args.extend(
        [
            *("--proc", "/proc"),
            *("--dev", "/dev"),
            *("--tmpfs", "/tmp"),
            *("--tmpfs", str(workspace.parent)),
            *("--dir", str(workspace)),
            *("--dir", str(shared_home)),
            *("--bind", str(workspace), str(workspace)),
            *("--bind", str(shared_home), str(shared_home)),
            *("--ro-bind", str(builtin_skills), str(builtin_skills)),
            *("--chdir", str(workspace)),
            *("--", "bash", "-c", command),
        ]
    )
    return shlex.join(args)
