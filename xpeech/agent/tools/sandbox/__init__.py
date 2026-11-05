"""Shared bubblewrap isolation for Shell and stdio MCP processes."""

import os
import shlex
import sys
from pathlib import Path

from ....config.settings import settings
from ....utils.file_io import ensure_path
from ...server.context import get_lark_user_access_token, get_user_id
from ...skills.skill import iter_builtin_skill_dirs

_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SANDBOX_EXECUTABLES = ("bwrap", "bash")
NETWORK_SANDBOX_EXECUTABLES = (*SANDBOX_EXECUTABLES, "slirp4netns")


def _add_setenv(args: list[str], key: str, value: str) -> None:
    args.extend(["--setenv", key, value])


def get_sandbox_home(workspace: Path) -> Path:
    """Create the workspace's private home without expanding the backend's HOME."""
    workspace = workspace.resolve()
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
    config_root = settings.path.sandbox_home_path
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


def wrap_sandbox_command(
    command: str | list[str], workspace: Path, *, network_isolation: bool = True,
) -> list[str]:
    """Wrap a bash command with optional slirp4netns networking (on by default)."""
    workspace = workspace.resolve()
    workspace_python_env = workspace / ".venv"
    home = get_sandbox_home(workspace)
    user_id = get_user_id()
    workspace_skills = workspace / "skills"

    args: list[str] = []
    if network_isolation:
        # Run the standard-library launcher outside the mount namespace. It
        # configures bwrap's network before allowing the command to execute.
        args.extend([
            sys.executable, "-I", str(Path(__file__).parent / "scripts" / "sandbox_network.py"),
        ])
    args.extend([
        "bwrap",
        "--new-session",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--cap-drop", "ALL",
    ])
    if network_isolation:
        args.append("--unshare-net")

    sandbox_env: dict[str, str] = {
        "HOME": str(home),
        "PATH": _sandbox_path(home),
        "PIP_REQUIRE_VIRTUALENV": "true",  # 确保 pip 在虚拟环境中运行
        "UV_PROJECT_ENVIRONMENT": str(workspace_python_env),  # uv虚拟py环境路径
        "UV_CACHE_DIR": str(home / ".cache" / "uv"),  # 包缓存目录
        "NPM_CONFIG_PREFIX": str(home / ".npm-global"),  # npm 全局安装目录
        "SHELL_USER_ID": str(user_id), # 注入 user_id，给插件使用
    }
    lark_user_access_token = get_lark_user_access_token()
    if lark_user_access_token:
        sandbox_env["LARKSUITE_CLI_USER_ACCESS_TOKEN"] = lark_user_access_token
    else:
        sandbox_env.pop("LARKSUITE_CLI_USER_ACCESS_TOKEN", None)
        args.extend(["--unsetenv", "LARKSUITE_CLI_USER_ACCESS_TOKEN"])
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
        "/etc/ld.so.cache",
        "/etc/hosts",
    ):
        args.extend(["--ro-bind-try", path, path])
    if not network_isolation:
        args.extend(["--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf"])

    args.extend(
        [
            *("--proc", "/proc"),
            *("--dev", "/dev"),
            *("--tmpfs", "/tmp"),
            *("--tmpfs", str(workspace.parent)),
            *("--dir", str(workspace)),
            *("--bind", str(workspace), str(workspace)),
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

    shell_command = shlex.join(command) if isinstance(command, list) else command
    args.extend(["--chdir", str(workspace), "--", "bash", "-c", shell_command])
    return args
