import shlex
from pathlib import Path

import pytest

from xpeech.agent.server.context import session_id_var
from xpeech.agent.tools import sandbox, shell
from xpeech.agent.tools.filesystem import ReadFileArgs, WriteFileArgs, build_file_tools
from xpeech.agent.tools.helper import safe_resolve_workspace_path
from xpeech.exceptions import PathProtectionError


def _option_pairs(args: list[str], option: str) -> list[tuple[str, str]]:
    return [(args[index + 1], args[index + 2]) for index, value in enumerate(args) if value == option]


def test_sandbox_homes_are_created_inside_each_workspace(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    first_workspace.mkdir()
    second_workspace.mkdir()

    first_home = sandbox.get_sandbox_home(first_workspace)
    second_home = sandbox.get_sandbox_home(second_workspace)

    assert first_home == first_workspace / "home"
    assert second_home == second_workspace / "home"
    assert first_home.is_dir()
    assert second_home.is_dir()
    assert first_home != second_home


def test_sandbox_home_rejects_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "home").symlink_to(tmp_path)

    with pytest.raises(ValueError, match="must not be a symlink"):
        sandbox.get_sandbox_home(workspace)


@pytest.mark.parametrize(
    ("user_path", "relative_path"),
    [
        ("~", Path("home")),
        ("~/notes/todo.md", Path("home/notes/todo.md")),
        (r"~\notes\todo.md", Path(r"home/notes\todo.md")),
    ],
)
def test_tilde_paths_resolve_to_workspace_home(
    tmp_path: Path,
    user_path: str,
    relative_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert safe_resolve_workspace_path(user_path, workspace) == workspace / relative_path


@pytest.mark.parametrize(
    "user_path",
    [
        "~/.config/uv/uv.toml",
        "home/.config/uv/uv.toml",
    ],
)
def test_sandbox_home_config_paths_map_to_managed_files_for_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    user_path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_file = config_home / ".config/uv/uv.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("managed config", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)

    assert (
        safe_resolve_workspace_path(
            user_path,
            workspace,
            protect_builtin_skills=False,
        )
        == config_file
    )


def test_sandbox_home_config_paths_are_read_only_to_file_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_file = config_home / ".npmrc"
    config_home.mkdir()
    config_file.write_text("managed config", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)

    with pytest.raises(PathProtectionError, match="HOME config files are read-only"):
        safe_resolve_workspace_path("~/.npmrc", workspace)


def test_unmapped_sandbox_home_paths_stay_in_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_home.mkdir()
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)

    assert safe_resolve_workspace_path("~/notes/todo.md", workspace) == workspace / "home/notes/todo.md"


@pytest.mark.asyncio
async def test_file_tools_can_read_but_not_write_mapped_home_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_home.mkdir()
    config_file = config_home / ".npmrc"
    config_file.write_text("registry=https://example.invalid\n", encoding="utf-8")
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    _, _, read_file, write_file, _ = build_file_tools(workspace)

    result = await read_file(ReadFileArgs(path="~/.npmrc"))

    assert "registry=https://example.invalid" in result
    with pytest.raises(PathProtectionError, match="HOME config files are read-only"):
        await write_file(WriteFileArgs(path="~/.npmrc", content="replacement"))
    assert config_file.read_text(encoding="utf-8") == "registry=https://example.invalid\n"


def test_wrap_command_uses_private_home_and_maps_mirror_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "cache"
    config_home = tmp_path / "sandbox-home"
    config_files = (
        Path(".config/uv/uv.toml"),
        Path(".pip/pip.conf"),
        Path(".npmrc"),
        Path("nested/custom/config.ini"),
    )
    for relative_path in config_files:
        config_file = config_home / relative_path
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("mirror config", encoding="utf-8")

    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(sandbox.settings.path, "cache_path", cache)
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))
    session_id = "test-session"
    agent_browser_cache = cache / session_id / "agent-browser"
    agent_browser_cache.mkdir(parents=True)

    session_token = session_id_var.set(session_id)
    try:
        args = shlex.split(sandbox.wrap_command("pwd", workspace))
    finally:
        session_id_var.reset(session_token)
    environment = dict(_option_pairs(args, "--setenv"))
    read_only_binds = set(_option_pairs(args, "--ro-bind"))
    writable_binds = set(_option_pairs(args, "--bind"))
    home = workspace / "home"

    assert environment["HOME"] == str(home)
    assert environment["UV_CACHE_DIR"] == str(home / ".cache/uv")
    assert environment["NPM_CONFIG_PREFIX"] == str(home / ".npm-global")
    assert environment["PATH"].startswith(f"{home}/.local/bin:{home}/.npm-global/bin:")
    assert (str(agent_browser_cache), str(agent_browser_cache)) in writable_binds
    assert (str(cache), str(cache)) not in writable_binds
    assert {
        (str(config_home / relative_path), str(home / relative_path)) for relative_path in config_files
    } <= read_only_binds


def test_agent_browser_context_creates_session_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setattr(shell.settings.path, "cache_path", cache)
    monkeypatch.setenv("CDP_URL", "http://browserless:3000")

    command, environment = shell._inject_command_context(
        "agent-browser open example.com",
        session_id="session-a",
    )

    session_cache = cache / "session-a" / "agent-browser"
    assert session_cache.is_dir()
    assert environment["AGENT_BROWSER_SOCKET_DIR"] == str(session_cache)
    assert command == "agent-browser --cdp http://browserless:3000 --session session-a open example.com"


def test_agent_browser_cache_binds_are_isolated_by_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache = tmp_path / "cache"
    config_home = tmp_path / "sandbox-home"
    config_home.mkdir()
    monkeypatch.setattr(sandbox.settings.path, "cache_path", cache)
    monkeypatch.setattr(sandbox.settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(sandbox, "iter_builtin_skill_dirs", lambda: iter(()))
    monkeypatch.setenv("CDP_URL", "http://browserless:3000")

    session_binds: dict[str, set[tuple[str, str]]] = {}
    for session_id in ("session-a", "session-b"):
        command, environment = shell._inject_command_context("agent-browser open example.com", session_id=session_id)
        session_token = session_id_var.set(session_id)
        try:
            args = shlex.split(sandbox.wrap_command(command, workspace, env=environment))
        finally:
            session_id_var.reset(session_token)
        session_binds[session_id] = set(_option_pairs(args, "--bind"))

    first_cache = cache / "session-a" / "agent-browser"
    second_cache = cache / "session-b" / "agent-browser"
    assert (str(first_cache), str(first_cache)) in session_binds["session-a"]
    assert (str(second_cache), str(second_cache)) not in session_binds["session-a"]
    assert (str(second_cache), str(second_cache)) in session_binds["session-b"]
    assert (str(first_cache), str(first_cache)) not in session_binds["session-b"]
