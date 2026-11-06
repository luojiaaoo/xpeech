import asyncio
import json
import os
import platform
import shlex
import shutil
import socket
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from xpeech.agent.server.context import lark_user_access_token_var, user_id_var
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import BUILTIN_SKILLS_DIR, SkillsLoader
from xpeech.agent.tools.filesystem import ReadFileArgs, WriteFileArgs, build_file_tools
from xpeech.agent.tools.helper import safe_resolve_workspace_path
from xpeech.config.settings import settings
from xpeech.exceptions import PathProtectionError
from xpeech.utils.security import sandbox
from xpeech.utils.security.sandbox import executor as executor_module
from xpeech.utils.security.sandbox.isolation import NETWORK_SANDBOX_EXECUTABLES, SANDBOX_EXECUTABLES
from xpeech.utils.security.sandbox.session import SandboxManager


@pytest_asyncio.fixture(autouse=True)
async def shell_sessions(monkeypatch):
    manager = SandboxManager()
    monkeypatch.setattr(executor_module, "sandbox_manager", manager)
    try:
        yield manager
    finally:
        await manager.close()


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


@pytest.mark.asyncio
async def test_sandbox_uses_private_network_with_outbound_nat(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))

    args = await sandbox.wrap_sandbox_command("true", tmp_path, skills_loader=SkillsLoader(tmp_path))

    assert args[:4] == [
        sys.executable, "-I", str(Path(sandbox.__file__).parent / "scripts" / "sandbox_network.py"), "bwrap",
    ]
    assert "--unshare-ipc" in args
    assert "--unshare-net" in args
    assert "rootlesskit" not in args
    assert ("/etc/resolv.conf", "/etc/resolv.conf") not in _option_pairs(args, "--ro-bind-try")


@pytest.mark.asyncio
async def test_sandbox_can_share_host_network_without_rootlesskit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))

    args = await sandbox.wrap_sandbox_command("true", tmp_path, network_isolation=False, skills_loader=SkillsLoader(tmp_path))

    assert args[0] == "bwrap"
    assert "rootlesskit" not in args
    assert not any(arg.startswith("--net=") for arg in args)
    assert "--unshare-net" not in args
    assert "--unshare-pid" in args
    assert "--unshare-ipc" in args
    assert ("/etc/resolv.conf", "/etc/resolv.conf") in _option_pairs(args, "--ro-bind-try")
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--cap-add") + 1] == "CAP_DAC_OVERRIDE"
    assert _option_pairs(args, "--bind") == [(str(tmp_path), str(tmp_path))]
    assert dict(_option_pairs(args, "--setenv"))["HOME"] == str(tmp_path / "home")


@pytest.mark.asyncio
async def test_sandbox_workspace_does_not_expand_the_backend_home(tmp_path: Path, monkeypatch) -> None:
    backend_home = tmp_path / "backend-home"
    workspace = tmp_path / "~" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(backend_home))
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))

    relative_workspace = Path("~/workspace")
    assert sandbox.get_sandbox_home(relative_workspace) == workspace / "home"
    args = await sandbox.wrap_sandbox_command("pwd", relative_workspace, skills_loader=SkillsLoader(relative_workspace))
    environment = dict(_option_pairs(args, "--setenv"))
    assert environment["HOME"] == str(workspace / "home")
    assert _option_pairs(args, "--bind") == [(str(workspace), str(workspace))]
    assert args[args.index("--chdir") + 1] == str(workspace)
    assert not backend_home.exists()


@pytest.mark.parametrize("workspace_kind", ["relative", "symlink"])
def test_resolver_normalizes_workspace(tmp_path, monkeypatch, workspace_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    supplied_workspace = Path("workspace") if workspace_kind == "relative" else alias

    assert safe_resolve_workspace_path(Path("~/notes.txt"), supplied_workspace) == workspace / "home/notes.txt"
    assert safe_resolve_workspace_path(Path("notes.txt"), supplied_workspace) == workspace / "notes.txt"
    with pytest.raises(PathProtectionError, match="Path escapes workspace"):
        safe_resolve_workspace_path(Path("../outside.txt"), supplied_workspace)


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

    assert safe_resolve_workspace_path(Path(user_path), workspace) == workspace / relative_path


@pytest.mark.parametrize("user_path", ["~root", "~root/secret.txt"])
def test_named_user_homes_are_never_expanded_by_backend(tmp_path, user_path):
    with pytest.raises(PathProtectionError, match="Named-user home paths"):
        safe_resolve_workspace_path(Path(user_path), tmp_path)


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
    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)

    assert (
        safe_resolve_workspace_path(
            Path(user_path),
            workspace,
            protect_read_only_file=False,
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
    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)

    with pytest.raises(PathProtectionError, match="HOME config files are read-only"):
        safe_resolve_workspace_path(Path("~/.npmrc"), workspace)


def test_unmapped_sandbox_home_paths_stay_in_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_home.mkdir()
    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)

    assert safe_resolve_workspace_path(Path("~/notes/todo.md"), workspace) == workspace / "home/notes/todo.md"


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
    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)
    _, _, read_file, write_file, _ = build_file_tools(workspace)

    result = await read_file(ReadFileArgs(path="~/.npmrc"))

    assert "registry=https://example.invalid" in result
    with pytest.raises(PathProtectionError, match="HOME config files are read-only"):
        await write_file(WriteFileArgs(path="~/.npmrc", content="replacement"))
    assert config_file.read_text(encoding="utf-8") == "registry=https://example.invalid\n"


@pytest.mark.asyncio
async def test_sandbox_command_uses_private_home_and_maps_mirror_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_home = tmp_path / "sandbox-home"
    config_files = (
        Path(".config/uv/uv.toml"),
        Path(".pip/pip.conf"),
        Path(".npmrc"),
        Path(".config/go/env"),
        Path("nested/custom/config.ini"),
    )
    for relative_path in config_files:
        config_file = config_home / relative_path
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("mirror config", encoding="utf-8")

    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))
    for key in ("GOPATH", "GOMODCACHE", "GOCACHE", "GOENV"):
        monkeypatch.setenv(key, str(tmp_path / "backend-go" / key))
    user_id = "test-session"

    session_token = user_id_var.set(user_id)
    try:
        args = await sandbox.wrap_sandbox_command("pwd", workspace, skills_loader=SkillsLoader(workspace))
    finally:
        user_id_var.reset(session_token)
    environment = dict(_option_pairs(args, "--setenv"))
    read_only_binds = set(_option_pairs(args, "--ro-bind"))
    writable_binds = set(_option_pairs(args, "--bind"))
    home = workspace / "home"

    assert environment["HOME"] == str(home)
    assert environment["UV_CACHE_DIR"] == str(home / ".cache/uv")
    assert environment["NPM_CONFIG_PREFIX"] == str(home / ".npm-global")
    assert environment["GOPATH"] == str(home / "go")
    assert environment["GOMODCACHE"] == str(home / "go/pkg/mod")
    assert environment["GOCACHE"] == str(home / ".cache/go-build")
    assert environment["GOENV"] == str(home / ".config/go/env")
    assert str(home / "go/bin") in environment["PATH"].split(":")
    assert "/usr/local/go/bin" in environment["PATH"].split(":")
    assert environment["SHELL_USER_ID"] == user_id
    assert environment["PATH"].startswith(f"{home / '.local/bin'}:{home / '.npm-global/bin'}:")
    assert writable_binds == {(str(workspace), str(workspace))}
    assert {
        (str(config_home / relative_path), str(home / relative_path)) for relative_path in config_files
    } <= read_only_binds


@pytest.mark.asyncio
async def test_sandbox_command_selects_lark_skills_by_injected_token(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def wrapped(token: str | None) -> tuple[dict[str, str], set[tuple[str, str]]]:
        context = lark_user_access_token_var.set(token)
        try:
            args = await sandbox.wrap_sandbox_command("true", workspace, skills_loader=SkillsLoader(workspace))
        finally:
            lark_user_access_token_var.reset(context)
        return dict(_option_pairs(args, "--setenv")), set(_option_pairs(args, "--ro-bind"))

    plain_environment, plain_binds = await wrapped(None)
    token_environment, token_binds = await wrapped("u-request-token")
    skills_target = workspace / "skills"
    builtin_root = BUILTIN_SKILLS_DIR.resolve()

    assert "LARKSUITE_CLI_USER_ACCESS_TOKEN" not in plain_environment
    assert (str(builtin_root / "lark-skills"), str(skills_target / "lark-skills")) in plain_binds
    assert (str(builtin_root / "lark-cli-auth"), str(skills_target / "lark-cli-auth")) in plain_binds

    assert token_environment["LARKSUITE_CLI_USER_ACCESS_TOKEN"] == "u-request-token"
    assert (
        str(builtin_root / "lark-skills-token"),
        str(skills_target / "lark-skills-token"),
    ) in token_binds
    assert not any(target == str(skills_target / "lark-cli-auth") for _, target in token_binds)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel_tag", "organization", "expected"),
    [
        (None, None, {"public"}),
        ("crm", None, {"public", "tagged"}),
        (None, "公司/研发/平台", {"public", "organization"}),
        ("crm", "公司/研发/平台", {"public", "tagged", "organization", "both"}),
        ("other", "公司/销售", {"public"}),
    ],
)
async def test_sandbox_only_mounts_builtin_skills_visible_to_loader(
    tmp_path, monkeypatch, channel_tag, organization, expected,
):
    builtin_root = tmp_path / "builtin"
    for name, restrictions in {
        "public": "",
        "tagged": "channel_tags: [crm]\n",
        "organization": "organization_prefixes: [公司/研发]\n",
        "both": "channel_tags: [crm]\norganization_prefixes: [公司/研发]\n",
        "invalid": "channel_tags: invalid\n",
    }.items():
        directory = builtin_root / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(f"---\nname: metadata-{name}\n{restrictions}---\nInstructions")
    monkeypatch.setattr(skill_module, "BUILTIN_SKILLS_DIR", builtin_root)
    workspace = tmp_path / "workspace"
    custom_skill = workspace / "skills/custom/SKILL.md"
    custom_skill.parent.mkdir(parents=True)
    custom_skill.write_text("---\nname: custom\n---\nCustom instructions")
    loader = SkillsLoader(workspace, channel_tag=channel_tag, user_organization=organization)
    monkeypatch.setattr(loader, "list_skill_info", AsyncMock(side_effect=AssertionError("Workspace skills requested")))

    args = await sandbox.wrap_sandbox_command("true", workspace, skills_loader=loader)

    builtin_mounts = {
        (source, target) for source, target in _option_pairs(args, "--ro-bind")
        if Path(source).is_relative_to(builtin_root)
    }
    assert builtin_mounts == {(str(builtin_root / name), str(workspace / "skills" / name)) for name in expected}


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*SANDBOX_EXECUTABLES, "fc-list")
    ),
    reason="Requires Linux sandbox dependencies and Fontconfig",
)
@pytest.mark.asyncio
async def test_sandbox_discovers_system_chinese_fonts(tmp_path, monkeypatch):
    font_command = ["fc-list", ":lang=zh", "file", "family"]
    host_fonts = subprocess.run(font_command, capture_output=True, check=True, timeout=10)
    assert b"Fontconfig error" not in host_fonts.stderr
    if not host_fonts.stdout.strip():
        pytest.skip("Requires system Chinese fonts (e.g. fonts-noto-cjk)")
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)

    command = await sandbox.wrap_sandbox_command(
        font_command, tmp_path, skills_loader=SkillsLoader(tmp_path), network_isolation=False,
    )
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 20)

    assert process.returncode == 0, stderr.decode(errors="replace")
    assert b"Fontconfig error" not in stderr
    assert set(stdout.splitlines()) == set(host_fonts.stdout.splitlines())


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*NETWORK_SANDBOX_EXECUTABLES, "tshark")
    ),
    reason="Requires Linux sandbox dependencies and TShark",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("network_isolation", [False, True])
async def test_sandbox_analyzes_ethernet_capture_with_tshark(tmp_path, monkeypatch, network_isolation):
    workspace = tmp_path / "capture-workspace"
    workspace.mkdir()
    # Ethernet broadcast containing an ARP request: who has 192.0.2.2?
    frame = bytes.fromhex(
        "ffffffffffff0200000000010806"
        "0001080006040001"
        "020000000001c0000201"
        "000000000000c0000202"
    ).ljust(60, b"\0")
    capture = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    capture += struct.pack("<IIII", 1, 0, len(frame), len(frame)) + frame
    (workspace / "sample.cap").write_bytes(capture)
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)

    command = await sandbox.wrap_sandbox_command(
        ["tshark", "-n", "-r", "sample.cap", "-Y", "arp", "-T", "json"],
        workspace, skills_loader=SkillsLoader(workspace), network_isolation=network_isolation,
    )
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 30)

    assert process.returncode == 0, stderr.decode(errors="replace")
    packets = json.loads(stdout)
    assert len(packets) == 1
    layers = packets[0]["_source"]["layers"]
    assert layers["eth"]["eth.src"] == "02:00:00:00:00:01"
    assert layers["eth"]["eth.dst"] == "ff:ff:ff:ff:ff:ff"
    assert layers["arp"]["arp.opcode"] == "1"
    assert layers["arp"]["arp.src.proto_ipv4"] == "192.0.2.1"
    assert layers["arp"]["arp.dst.proto_ipv4"] == "192.0.2.2"


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*NETWORK_SANDBOX_EXECUTABLES, "libreoffice", "pdftotext")
    ),
    reason="Requires Linux sandbox dependencies, LibreOffice and Poppler",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("network_isolation", [False, True])
async def test_sandbox_converts_document_with_libreoffice(tmp_path, monkeypatch, network_isolation):
    workspace = tmp_path / "office-workspace"
    workspace.mkdir()
    source = workspace / "sample.fodt"
    source.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 office:version="1.2" office:mimetype="application/vnd.oasis.opendocument.text">
 <office:body><office:text>
  <text:p>LibreOffice sandbox conversion</text:p><text:p>文档转换测试</text:p>
 </office:text></office:body>
</office:document>
''', encoding="utf-8")
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)

    command = await sandbox.wrap_sandbox_command(
        [
            "libreoffice", "--headless", f"-env:UserInstallation={(workspace / 'lo-profile').as_uri()}",
            "--convert-to", "pdf", "--outdir", str(workspace), str(source),
        ],
        workspace, skills_loader=SkillsLoader(workspace), network_isolation=network_isolation,
    )
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 60)

    assert process.returncode == 0, (stdout + stderr).decode(errors="replace")
    pdf = workspace / "sample.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    extracted = subprocess.run(["pdftotext", str(pdf), "-"], check=True, capture_output=True, timeout=10)
    text = extracted.stdout.decode("utf-8")
    assert "LibreOffice sandbox conversion" in text
    assert "文档转换测试" in "".join(text.split())


@pytest.mark.skipif(
    platform.system() != "Linux" or any(shutil.which(name) is None for name in NETWORK_SANDBOX_EXECUTABLES),
    reason="Requires Linux sandbox dependencies",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("network_isolation", [False, True])
async def test_root_can_read_restricted_builtin_permissions_but_not_write_read_only_mounts(
    tmp_path, monkeypatch, network_isolation,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builtin_root = tmp_path / "builtin"
    skill_dir = builtin_root / "weather"
    resources = skill_dir / "resources"
    resources.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: weather\n---\nWeather instructions")
    resource_file = resources / "data.txt"
    resource_file.write_text("resource-content\n")
    resource_file.chmod(0o000)
    skill_dir.chmod(0o111)
    monkeypatch.setattr(skill_module, "BUILTIN_SKILLS_DIR", builtin_root)
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")
    try:
        command = await sandbox.wrap_sandbox_command(
            "set -e; test $(id -u) = 0; ls skills/weather; cat skills/weather/resources/data.txt; "
            "if echo changed > skills/weather/resources/data.txt 2>/dev/null; then exit 1; fi; "
            "if touch /usr/xpeech-should-stay-read-only 2>/dev/null; then exit 1; fi; "
            "awk '/^CapEff:/ {print $2}' /proc/self/status",
            workspace, skills_loader=SkillsLoader(workspace), network_isolation=network_isolation,
        )
        # Exercise UID 0 even when the test runner is an ordinary host user.
        if os.geteuid() != 0:
            index = command.index("bwrap") + 1
            command[index:index] = ["--unshare-user", "--uid", "0", "--gid", "0"]
        process = await asyncio.create_subprocess_exec(
            *command, cwd=workspace, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        assert process.returncode == 0, stderr.decode()
        lines = stdout.decode().splitlines()
        assert "SKILL.md" in lines and "resources" in lines
        assert "resource-content" in lines
        assert int(lines[-1], 16) == 1 << 1  # Only CAP_DAC_OVERRIDE survives.
    finally:
        skill_dir.chmod(0o755)
        resource_file.chmod(0o644)
    assert resource_file.read_text() == "resource-content\n"


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*NETWORK_SANDBOX_EXECUTABLES, "node", "npm", "npx")
    ),
    reason="Requires Linux sandbox dependencies and Node.js/npm/npx",
)
@pytest.mark.asyncio
async def test_node_npm_and_npx_work_with_only_file_access_capability(tmp_path, monkeypatch):
    workspace = tmp_path / "node-workspace"
    workspace.mkdir()
    dependency = workspace / "local-dependency"
    dependency.mkdir()
    (dependency / "package.json").write_text(json.dumps({
        "name": "sandbox-local-bin", "version": "1.0.0", "bin": {"sandbox-local-bin": "bin.js"},
    }))
    executable = dependency / "bin.js"
    executable.write_text("#!/usr/bin/env node\nconsole.log('local-bin-ok');\n")
    executable.chmod(0o755)
    (workspace / "package.json").write_text(json.dumps({
        "name": "sandbox-node-test", "version": "1.0.0", "private": True, "type": "module",
        "scripts": {"build": "node build.js"},
        "dependencies": {"sandbox-local-bin": "file:./local-dependency"},
    }))
    (workspace / "build.js").write_text('''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import {execFileSync} from 'node:child_process';
import {Worker} from 'node:worker_threads';
fs.writeFileSync('build-result.txt', 'build-ok');
assert.equal(fs.readFileSync('build-result.txt', 'utf8'), 'build-ok');
assert.equal(execFileSync(process.execPath, ['-e', 'process.stdout.write("child-ok")']).toString(), 'child-ok');
await new Promise((resolve, reject) => {
    const worker = new Worker('require("node:worker_threads").parentPort.postMessage("worker-ok")', {eval: true});
    worker.once('message', value => { assert.equal(value, 'worker-ok'); resolve(); });
    worker.once('error', reject);
});
const server = http.createServer((request, response) => response.end('http-ok'));
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
try {
    const body = await new Promise((resolve, reject) => {
        http.get(`http://127.0.0.1:${server.address().port}`, response => {
            let body = ''; response.on('data', chunk => { body += chunk; }); response.on('end', () => resolve(body));
        }).on('error', reject);
    });
    assert.equal(body, 'http-ok');
    console.log('node-runtime-ok');
} finally {
    await new Promise(resolve => server.close(resolve));
}
''')
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "managed-home")

    result = await executor_module.sandbox_manager.run(
        "npm install --offline --ignore-scripts --no-audit --no-fund && npm run build && "
        "npx --offline --no-install sandbox-local-bin",
        workspace, timeout_ms=20_000, skills_loader=SkillsLoader(workspace),
    )

    assert result.returncode == 0, result.stderr.decode()
    assert not result.timed_out
    assert b"node-runtime-ok" in result.stdout
    assert b"local-bin-ok" in result.stdout
    assert (workspace / "build-result.txt").read_text() == "build-ok"


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*NETWORK_SANDBOX_EXECUTABLES, "go")
    ),
    reason="Requires Linux sandbox dependencies and Go installed under /usr or /opt",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("network_isolation", [False, True])
@pytest.mark.parametrize("third_party", [
    pytest.param(False, id="stdlib"),
    pytest.param(True, id="google-uuid", marks=pytest.mark.skipif(
        os.environ.get("XPEECH_TEST_GO_NETWORK") != "1",
        reason="Set XPEECH_TEST_GO_NETWORK=1 to download a third-party Go module",
    )),
])
async def test_go_cross_compiles_windows_and_linux_amd64(tmp_path, monkeypatch, network_isolation, third_party):
    workspace = tmp_path / "go-workspace"
    workspace.mkdir()
    (workspace / "go.mod").write_text("module example.com/crosshello\n\ngo 1.23\n")
    (workspace / "main.go").write_text(
        'package main\nimport "fmt"\nfunc main() { fmt.Println("go-sandbox-ok") }\n'
    )
    expected_output = "go-sandbox-ok"
    download_command = ""
    if third_party:
        with (workspace / "go.mod").open("a") as module_file:
            module_file.write("\nrequire github.com/google/uuid v1.6.0\n")
        (workspace / "main.go").write_text(
            'package main\nimport ("fmt"; "github.com/google/uuid")\n'
            'func main() {\n'
            '  id, err := uuid.Parse("123e4567-e89b-12d3-a456-426614174000")\n'
            '  if err != nil { panic(err) }\n'
            '  fmt.Println("go-sandbox-ok", id.String())\n'
            '}\n'
        )
        expected_output += " 123e4567-e89b-12d3-a456-426614174000"
        # Download and verify through the sandbox network, then build solely
        # from the workspace's module cache for both target platforms.
        download_command = "go mod download all; go mod verify >&2; export GOPROXY=off; "
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))
    monkeypatch.setattr(settings.path, "sandbox_home_path", Path("assets/sandbox-home").resolve())

    command = await sandbox.wrap_sandbox_command(
        f"set -eu; go env -json GOPATH GOMODCACHE GOCACHE GOPROXY; {download_command}"
        "CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o app.exe .; "
        "CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o app-linux .; "
        "CGO_ENABLED=0 go install .; crosshello",
        workspace, skills_loader=SkillsLoader(workspace), network_isolation=network_isolation,
    )
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), 180)
    assert process.returncode == 0, stderr.decode()
    environment_json, output = stdout.decode().split("}\n", 1)
    environment = json.loads(environment_json + "}")
    assert environment["GOPATH"] == str(workspace / "home/go")
    assert environment["GOMODCACHE"] == str(workspace / "home/go/pkg/mod")
    assert environment["GOCACHE"] == str(workspace / "home/.cache/go-build")
    assert environment["GOPROXY"] == "https://goproxy.cn,direct"
    assert output.strip() == expected_output
    assert any((workspace / "home/.cache/go-build").iterdir())
    if third_party:
        module_cache = workspace / "home/go/pkg/mod"
        assert (module_cache / "github.com/google/uuid@v1.6.0/uuid.go").is_file()
        assert (module_cache / "cache/download/github.com/google/uuid/@v/v1.6.0.zip").is_file()
        checksums = (workspace / "go.sum").read_text()
        assert "github.com/google/uuid v1.6.0 h1:" in checksums
        assert "github.com/google/uuid v1.6.0/go.mod h1:" in checksums
        assert b"all modules verified" in stderr

    windows = (workspace / "app.exe").read_bytes()
    assert windows[:2] == b"MZ"
    pe_offset = struct.unpack_from("<I", windows, 0x3C)[0]
    assert windows[pe_offset:pe_offset + 4] == b"PE\0\0"
    assert struct.unpack_from("<H", windows, pe_offset + 4)[0] == 0x8664  # AMD64
    linux = (workspace / "app-linux").read_bytes()
    assert linux[:6] == b"\x7fELF\x02\x01"  # 64-bit little-endian ELF
    assert struct.unpack_from("<H", linux, 18)[0] == 62  # x86-64


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in (*NETWORK_SANDBOX_EXECUTABLES, "setsid")
    ),
    reason="Requires Linux with bubblewrap, slirp4netns, bash and setsid",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["exit", "timeout", "cancel"])
async def test_shell_sandbox_isolates_processes_and_cleans_up_background_jobs_on_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
    shell_sessions,
) -> None:
    workspace = tmp_path / "workspaces" / "current"
    workspace.mkdir(parents=True)
    config_home = tmp_path / "managed-home"
    config_home.mkdir()
    managed_config = config_home / ".audit-config"
    managed_config.write_text("read-only config", encoding="utf-8")
    monkeypatch.setattr(settings.path, "sandbox_home_path", config_home)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)
    heartbeat = workspace / "heartbeat"
    worker = "for ((i=0; i<40; i++)); do printf x >> heartbeat; sleep 0.05; done"
    command = (
        "readlink /proc/self/ns/pid > isolation.txt; "
        "readlink /proc/self/ns/net >> isolation.txt; "
        "readlink /proc/self/ns/ipc >> isolation.txt; "
        "awk '/^CapEff:/ {print $2}' /proc/self/status >> isolation.txt; "
        "if (printf modified > home/.audit-config) 2>/dev/null; then exit 1; fi; "
        f"setsid bash -c {shlex.quote(worker)} & "
        "while [ ! -s heartbeat ]; do sleep 0.01; done; "
        "echo partial-output; "
        + ("true" if termination == "exit" else "sleep 30")
    )
    task = asyncio.create_task(
        executor_module.sandbox_manager.run(command, workspace, timeout_ms=1_000 if termination == "timeout" else 10_000, skills_loader=SkillsLoader(workspace))
    )
    if termination == "cancel":
        try:
            async with asyncio.timeout(5):
                while not heartbeat.exists() or heartbeat.stat().st_size == 0:
                    if task.done():
                        pytest.fail(f"Sandbox exited before starting background job: {task.result()}")
                    await asyncio.sleep(0.01)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    else:
        result = await task
        assert result.timed_out is (termination == "timeout")
        assert b"partial-output" in result.stdout
        if termination == "exit":
            assert result.returncode == 0

    await shell_sessions.close()

    pid_namespace, net_namespace, ipc_namespace, capabilities = (workspace / "isolation.txt").read_text().splitlines()
    assert pid_namespace != os.readlink("/proc/self/ns/pid")
    assert net_namespace != os.readlink("/proc/self/ns/net")
    assert ipc_namespace != os.readlink("/proc/self/ns/ipc")
    assert int(capabilities, 16) == 1 << 1  # CAP_DAC_OVERRIDE
    assert managed_config.read_text(encoding="utf-8") == "read-only config"
    await asyncio.sleep(0.1)
    stopped_heartbeat = heartbeat.read_bytes()
    await asyncio.sleep(0.2)
    assert heartbeat.read_bytes() == stopped_heartbeat


@pytest.mark.skipif(
    platform.system() != "Linux" or any(
        shutil.which(name) is None for name in NETWORK_SANDBOX_EXECUTABLES
    ) or not Path("/usr/bin/python3").exists(),
    reason="Requires Linux sandbox dependencies and system Python",
)
@pytest.mark.asyncio
async def test_concurrent_sandboxes_have_private_ports_and_ipc(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.path, "sandbox_home_path", tmp_path / "config-home")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: iter(()))
    program = """
import json, os, socket, sys
tcp = socket.socket()
tcp.bind(('0.0.0.0', 43821))
tcp.listen()
abstract = socket.socket(socket.AF_UNIX)
abstract.bind(b'\\0xpeech-private-network')
probe = socket.socket()
probe.settimeout(0.5)
assert probe.connect_ex(('10.0.2.2', int(sys.argv[1]))) != 0
print(json.dumps({name: os.readlink('/proc/self/ns/' + name) for name in ('net', 'ipc')}), flush=True)
sys.stdin.readline()
"""
    processes = []
    namespaces = []
    with socket.socket() as host_listener:
        host_listener.bind(("127.0.0.1", 0))
        host_listener.listen()
        host_port = host_listener.getsockname()[1]
        try:
            for name in ("first", "second"):
                workspace = tmp_path / name
                workspace.mkdir()
                process = await asyncio.create_subprocess_exec(
                    *(await sandbox.wrap_sandbox_command(["/usr/bin/python3", "-c", program, str(host_port)], workspace, skills_loader=SkillsLoader(workspace))),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                processes.append(process)
                line = await asyncio.wait_for(process.stdout.readline(), timeout=20)
                if not line:
                    _, stderr = await process.communicate()
                    pytest.fail(f"Sandbox failed to start: {stderr.decode(errors='replace')}")
                namespaces.append(json.loads(line))
            for kind in ("net", "ipc"):
                assert namespaces[0][kind] != namespaces[1][kind]
                assert all(ns[kind] != os.readlink(f"/proc/self/ns/{kind}") for ns in namespaces)
            for process in processes:
                _, stderr = await asyncio.wait_for(process.communicate(b"\n"), timeout=10)
                assert process.returncode == 0, stderr.decode(errors="replace")
        finally:
            for process in processes:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.wait_for(process.communicate(), timeout=10)
