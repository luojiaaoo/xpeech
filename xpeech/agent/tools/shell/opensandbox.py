"""OpenSandbox shell execution; files remain on the shared Docker host."""

import asyncio
import hashlib
import ipaddress
import json
import os
import shlex
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.transport import RetryPolicy
from opensandbox.exceptions import SandboxApiException
from opensandbox.models.execd import ExecutionHandlers, RunCommandOpts
from opensandbox.models.sandboxes import Host, NetworkPolicy, NetworkRule, Volume

from ....config.settings import OpenSandboxConfig, settings
from ...server.context import get_lark_user_access_token, get_user_id
from ...skills.skill import iter_builtin_skill_dirs
from ..sandbox import _iter_sandbox_home_config_files, get_sandbox_home
from .schema import ShellCommandResult
from .shared_slots import acquire_lock, acquire_resident, acquire_slot, read_record, write_record

TIMEOUT_MARKER = b"\n__XPEECH_COMMAND_TIMEOUT__\n"
VENV_NAME = ".venv-opensandbox"
CLI_BIN_DIR = Path(__file__).resolve().parents[4] / "docker_data/lark-cli"


class OpenSandboxExecutor:
    def __init__(self, config: OpenSandboxConfig):
        self.config = config

    def _connection(self) -> ConnectionConfig:
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(f"Missing OpenSandbox API key environment variable: {self.config.api_key_env}")
        return ConnectionConfig(
            domain=self.config.domain,
            protocol=self.config.protocol,
            api_key=key,
            use_server_proxy=True,
            disable_metrics=True,
            retry_policy=RetryPolicy.disabled(),
            request_timeout=timedelta(seconds=self.config.startup_timeout_seconds),
        )

    def _volumes(self, workspace: Path) -> list[Volume]:
        relative = workspace.relative_to(settings.path.workspace_base_path)
        if len(relative.parts) != 1:
            raise ValueError("OpenSandbox workspace must be a direct child of workspace_base_path")
        home = get_sandbox_home(workspace)
        volumes = [Volume(
            name="workspace", host=Host(path=str(workspace)),
            mountPath=str(workspace), readOnly=False,
        )]
        for binary in ("lark-cli", "lark-cli-auth"):
            volumes.append(Volume(
                name=binary, host=Host(path=str(CLI_BIN_DIR / binary)),
                mountPath=f"/usr/local/bin/{binary}", readOnly=True,
            ))
        # Same sources and targets as bwrap at 79e0baf: use the selected
        # built-in directories and actual settings-backed HOME config files.
        for index, skill in enumerate(iter_builtin_skill_dirs()):
            volumes.append(Volume(
                name=f"skill-{index}",
                host=Host(path=str(skill.resolve())),
                mountPath=str(workspace / "skills" / skill.name), readOnly=True,
            ))
        for index, (source, relative_path) in enumerate(_iter_sandbox_home_config_files()):
            volumes.append(Volume(
                name=f"home-config-{index}",
                host=Host(path=str(source)),
                mountPath=str(home / relative_path), readOnly=True,
            ))
        return volumes

    def _environment(self, workspace: Path) -> dict[str, str]:
        home = get_sandbox_home(workspace)
        venv = workspace / VENV_NAME
        return {
            "HOME": str(home),
            "PATH": (
                f"{venv}/bin:{home}/.local/bin:{home}/.npm-global/bin:"
                "/opt/miniforge/envs/runtime/bin:/opt/miniforge/condabin:/opt/miniforge/bin:"
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "PIP_REQUIRE_VIRTUALENV": "true",
            "UV_PYTHON": "/usr/local/bin/python-runtime",
            "UV_PROJECT_ENVIRONMENT": str(workspace / VENV_NAME),
            "VIRTUAL_ENV": str(workspace / VENV_NAME),
            "UV_CACHE_DIR": str(home / ".cache" / "uv"),
            "NPM_CONFIG_PREFIX": str(home / ".npm-global"),
            "CONDA_PKGS_DIRS": str(home / ".cache" / "conda"),
            "CONDA_ENVS_PATH": str(home / ".conda" / "envs"),
            "SHELL_USER_ID": get_user_id() or workspace.name,
            "XPEECH_TOKEN_MANAGER_URL": self.config.token_manager_url,
            # Always set, including an empty value, to avoid stale credentials.
            "LARKSUITE_CLI_USER_ACCESS_TOKEN": get_lark_user_access_token() or "",
        }

    def _network_policy(self) -> NetworkPolicy:
        endpoint = urlsplit(self.config.token_manager_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("OpenSandbox token_manager_url must be an absolute HTTP(S) URL")
        # Port-qualified allow rules require a literal address. This also
        # avoids granting every port on an internal DNS-resolved service.
        ipaddress.ip_address(endpoint.hostname)
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        return NetworkPolicy(defaultAction="allow", egress=[
            NetworkRule(action="allow", target=endpoint.hostname, ports=[port]),
            *[NetworkRule(action="deny", target=cidr) for cidr in (
                "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
                "100.64.0.0/10", "fc00::/7", "fe80::/10",
            )],
        ])

    async def _connect(self, sandbox_id: str) -> Sandbox | None:
        try:
            return await Sandbox.connect(
                sandbox_id, connection_config=self._connection(), skip_health_check=True,
                connect_timeout=timedelta(seconds=self.config.startup_timeout_seconds),
            )
        except SandboxApiException as exc:
            if exc.status_code == 404:
                return None
            raise

    async def _destroy(self, sandbox: Sandbox, path: Path) -> None:
        # Never free a slot when deletion has an unknown outcome.
        async with asyncio.timeout(10):
            try:
                await sandbox.kill()
                # DELETE may only acknowledge deletion. Hold capacity until
                # the management service confirms the old sandbox is gone.
                while True:
                    await sandbox.get_info()
                    await asyncio.sleep(0.1)
            except SandboxApiException as exc:
                if exc.status_code != 404:
                    raise
        write_record(path, {})

    async def _run(self, sandbox: Sandbox, command: str, workspace: Path, timeout_ms: int) -> ShellCommandResult:
        execution_id = None

        async def on_init(event):
            nonlocal execution_id
            execution_id = event.id

        # The image's supervisor bounds output and reaps even detached children.
        wrapped = shlex.join([
            "/usr/local/bin/python-runtime", "/opt/xpeech/run_command.py",
            str(timeout_ms / 1000), command,
        ])
        try:
            async with asyncio.timeout(timeout_ms / 1000 + 10):
                result = await sandbox.commands.run(
                    wrapped,
                    opts=RunCommandOpts(
                        working_directory=str(workspace),
                        envs=self._environment(workspace),
                        timeout=timedelta(milliseconds=timeout_ms + 5000),
                    ),
                    handlers=ExecutionHandlers(on_init=on_init),
                )
        except BaseException:
            if execution_id:
                try:
                    await asyncio.wait_for(sandbox.commands.interrupt(execution_id), timeout=5)
                except Exception:
                    logger.warning("Could not interrupt remote command; destroying its sandbox")
            raise
        stdout = "".join(message.text for message in result.logs.stdout).encode("utf-8")
        stderr = "".join(message.text for message in result.logs.stderr).encode("utf-8")
        if result.complete is None and result.error is None:
            raise RuntimeError("OpenSandbox command stream ended without completion")
        code = result.exit_code
        if code is None and result.error is not None:
            code = result.error.exit_code
        if code is None:
            # Do not report an SDK/runtime error as a successful shell command.
            raise RuntimeError("OpenSandbox did not return a command exit code")
        timed_out = code == 124 and TIMEOUT_MARKER in stderr
        if result.error is not None and not stderr:
            stderr = f"Remote command failed: {result.error.name}".encode()
        return ShellCommandResult(stdout, stderr.replace(TIMEOUT_MARKER, b""), code, timed_out)

    async def run(self, command: str, workspace: Path, timeout_ms: int) -> ShellCommandResult:
        config = self.config
        workspace = workspace.resolve()
        root = config.state_path
        if root.is_relative_to(settings.path.workspace_base_path):
            raise ValueError("OpenSandbox state_path must be outside user workspaces")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not os.environ.get(config.api_key_env):
            raise RuntimeError(f"Missing OpenSandbox API key environment variable: {config.api_key_env}")
        volumes = self._volumes(workspace)
        key = hashlib.sha256(str(workspace).encode()).hexdigest()
        spec = {
            "domain": config.domain, "image": config.image, "cpu": config.cpu, "memory": config.memory,
            "volumes": [volume.model_dump(mode="json") for volume in volumes],
            "network": self._network_policy().model_dump(mode="json"),
        }
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        deadline = time.monotonic() + config.queue_timeout_seconds
        async with acquire_lock(root / "topology.lock", deadline):
            topology_path = root / "topology.json"
            topology = {
                "domain": config.domain, "protocol": config.protocol,
                "max_sandboxes": config.max_sandboxes, "max_commands": config.max_commands,
                "max_creating": config.max_creating,
            }
            stored = read_record(topology_path)
            if stored and stored != topology:
                raise RuntimeError("OpenSandbox replicas must share the same topology configuration; drain before changing limits")
            if not stored:
                write_record(topology_path, topology)
        async with acquire_lock(root / f"user-{key}.lock", deadline):
            async with acquire_slot(root, "command", config.max_commands, deadline):
                async with acquire_resident(root, key, config.max_sandboxes, deadline) as (path, record):
                    sandbox = None
                    try:
                        if record.get("sandbox_id"):
                            sandbox = await self._connect(record["sandbox_id"])
                            if sandbox is not None and (
                                record.get("key") != key or record.get("fingerprint") != fingerprint or record.get("busy")
                            ):
                                await self._destroy(sandbox, path)
                                await sandbox.close()
                                sandbox = None
                        lifetime = config.idle_seconds + config.startup_timeout_seconds + timeout_ms / 1000 + 30
                        if sandbox is None:
                            async with acquire_slot(root, "create", config.max_creating, deadline):
                                record = {
                                    "key": key, "fingerprint": fingerprint, "sandbox_id": None,
                                    "expires_at": time.time() + lifetime + 2 * config.startup_timeout_seconds,
                                    "last_used": time.time(),
                                }
                                write_record(path, record)
                                sandbox = await Sandbox.create(
                                    config.image, connection_config=self._connection(), volumes=volumes,
                                    resource={"cpu": config.cpu, "memory": config.memory},
                                    network_policy=self._network_policy(),
                                    metadata={"xpeech_workspace": key},
                                    timeout=timedelta(seconds=lifetime),
                                    ready_timeout=timedelta(seconds=config.startup_timeout_seconds),
                                )
                                record["sandbox_id"] = sandbox.id
                                write_record(path, record)
                        record["busy"] = True
                        write_record(path, record)
                        await sandbox.renew(timedelta(seconds=lifetime))
                        # A separate environment avoids reusing host/bwrap Python symlinks.
                        venv = shlex.quote(str(workspace / VENV_NAME))
                        init = await self._run(
                            sandbox, f"test -x {venv}/bin/python || /usr/local/bin/python-runtime -m venv {venv}",
                            workspace, min(timeout_ms, 60_000),
                        )
                        if init.returncode != 0:
                            raise RuntimeError("Could not initialize the OpenSandbox Python environment")
                        result = await self._run(sandbox, command, workspace, timeout_ms)
                        await sandbox.renew(timedelta(seconds=config.idle_seconds))
                        record.update(busy=False, last_used=time.time(), expires_at=time.time() + config.idle_seconds)
                        write_record(path, record)
                        return result
                    except BaseException:
                        if sandbox is not None:
                            try:
                                await asyncio.wait_for(self._destroy(sandbox, path), timeout=10)
                            except Exception:
                                logger.exception("Sandbox cleanup failed; retaining its capacity slot until expiration")
                        raise
                    finally:
                        if sandbox is not None:
                            await sandbox.close()
