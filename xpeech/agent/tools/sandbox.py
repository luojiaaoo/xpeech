"""Bubblewrap command wrapping for shell command execution."""

import os
import shlex
from pathlib import Path

from ...config.settings import settings
from ...utils.helper import ensure_path
from ..skills.skill import iter_builtin_skill_dirs

_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _add_setenv(args: list[str], key: str, value: Path | str) -> None:
    args.extend(["--setenv", key, str(value)])


def get_sandbox_home(workspace: str | Path) -> Path:
    """Return the current workspace's private sandbox home, creating it if needed."""
    workspace = Path(workspace).expanduser().resolve()
    home = workspace / "home"
    if home.is_symlink():
        raise ValueError(f"Sandbox home must not be a symlink: {home}")
    return ensure_path(home).resolve()


def _sandbox_path(home: Path) -> str:
    inherited_path = os.environ.get("PATH") or ""
    path_entries = [
        str(home / ".local" / "bin"),
        str(home / ".npm-global" / "bin"),
        inherited_path,
        _DEFAULT_PATH,
    ]
    return ":".join(entry for entry in path_entries if entry)


def _iter_sandbox_home_config_files() -> list[tuple[Path, Path]]:
    """Return (source, relative path) pairs for sandbox HOME config files."""
    config_root = settings.path.sandbox_home_path.expanduser().resolve()
    if not config_root.is_dir():
        return []

    config_files: list[tuple[Path, Path]] = []
    for source in sorted(config_root.rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        resolved_source = source.resolve()
        if not resolved_source.is_relative_to(config_root):
            continue
        config_files.append((resolved_source, source.relative_to(config_root)))
    return config_files


def wrap_command(command: str, workspace: str | Path, env: dict[str, str] | None = None) -> str:
    """Wrap a command in a bubblewrap sandbox."""
    workspace = Path(workspace).expanduser().resolve()
    workspace_python_env = workspace / ".venv"
    home = get_sandbox_home(workspace)
    cache_path = settings.path.cache_path.resolve()
    workspace_skills = workspace / "skills"

    args = ["bwrap", "--new-session", "--die-with-parent"]

    sandbox_env: dict[str, str | Path] = {
        "HOME": home,
        "PATH": _sandbox_path(home),
        "PIP_REQUIRE_VIRTUALENV": "true",  # 确保 pip 在虚拟环境中运行
        "UV_PROJECT_ENVIRONMENT": workspace_python_env,  # uv虚拟py环境路径
        "UV_CACHE_DIR": home / ".cache" / "uv",  # 包缓存目录
        "NPM_CONFIG_PREFIX": home / ".npm-global",  # npm 全局安装目录
    }
    sandbox_env.update(env or {})
    for key, value in sandbox_env.items():
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
            *("--dir", str(cache_path)),
            *("--bind", str(workspace), str(workspace)),
            *("--bind", str(cache_path), str(cache_path)),
        ]
    )

    # Keep centrally managed HOME config files read-only while all writable
    # HOME state remains private to the current workspace.
    for source, relative_path in _iter_sandbox_home_config_files():
        args.extend(["--ro-bind", str(source), str(home / relative_path)])

    # Present built-in and custom skills through one workspace/skills tree.
    # Built-in directories are mounted read-only over same-named workspace
    # skills so the built-in version always has priority inside the sandbox.
    for builtin_skill in iter_builtin_skill_dirs():
        target = workspace_skills / builtin_skill.name
        args.extend(["--ro-bind", str(builtin_skill.resolve()), str(target)])

    args.extend(["--chdir", str(workspace), "--", "bash", "-c", command])
    return shlex.join(args)
