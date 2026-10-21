"""Workspace path resolution shared by the gateway and container runtime."""

from pathlib import Path
import os

from ..config.settings import settings
from ..exceptions import PathProtectionError
from ..agent.skills.skill import BUILTIN_SKILLS_DIR


def get_sandbox_home(workspace: str | Path) -> Path:
    base = Path(workspace).expanduser().resolve()
    home = base / "home"
    if home.is_symlink():
        raise ValueError(f"Sandbox home must not be a symlink: {home}")
    home.mkdir(parents=True, exist_ok=True)
    return home.resolve()


def safe_resolve_workspace_path(
    user_path: str | Path,
    workspace: str | Path,
    protect_read_only_file: bool = True,
) -> Path:
    base = Path(workspace).expanduser().resolve()
    value = str(user_path)
    if value == "~":
        path = get_sandbox_home(base)
    elif value.startswith(("~/", "~\\")):
        path = get_sandbox_home(base) / value[2:]
    else:
        path = Path(user_path).expanduser()
    candidate = Path(os.path.abspath(path if path.is_absolute() else base / path))
    if not candidate.is_relative_to(base):
        raise PathProtectionError("Path escapes workspace")
    # Managed container links point outside the writable workspace. Resolve
    # their logical mount mapping before the general symlink escape check.
    if candidate.is_relative_to(base / "home"):
        relative = candidate.relative_to(base / "home")
        config_root = settings.path.sandbox_home_path.expanduser().resolve()
        config_path = (config_root / relative).resolve()
        if relative.parts and config_path.is_file():
            if not config_path.is_relative_to(config_root):
                raise PathProtectionError("Sandbox HOME config escapes config directory")
            if protect_read_only_file:
                raise PathProtectionError("Sandbox HOME config files are read-only")
            return config_path
    if candidate.is_relative_to(base / "skills"):
        relative = candidate.relative_to(base / "skills")
        if relative.parts:
            builtin = (BUILTIN_SKILLS_DIR / relative.parts[0]).resolve()
            if (builtin / "SKILL.md").is_file():
                if protect_read_only_file:
                    raise PathProtectionError("Built-in skills are read-only")
                mapped = (builtin / Path(*relative.parts[1:])).resolve()
                if not mapped.is_relative_to(BUILTIN_SKILLS_DIR.resolve()):
                    raise PathProtectionError("Built-in skill path escapes skills directory")
                return mapped
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise PathProtectionError("Path escapes workspace")
    return resolved
