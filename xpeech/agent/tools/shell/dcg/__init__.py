"""Classify shell commands with the bundled dcg policy."""

import asyncio
import json
import os
from pathlib import Path

from loguru import logger

from ... import sandbox
from ...execution_request import ExecutionDecision

SHELL_CHECK_TIMEOUT = 5.0
DCG_POLICY_DIR = Path(__file__).parent


async def shell_requires_confirmation(payload: str, *, workspace: str | Path) -> ExecutionDecision:
    """一次 dcg 检查返回 ALLOW、CONFIRM 或 DENY；检查失败报工具错误。"""
    command = json.loads(payload).get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Shell command must be a non-empty string")

    # Use the bundled policy, not ambient DCG_* bypasses or policy overrides.
    env = {key: value for key, value in os.environ.items() if not key.startswith("DCG_")}
    env["DCG_CUSTOM_PATHS"] = str(DCG_POLICY_DIR / "agent.yaml")

    process = None
    try:
        workspace = Path(workspace).resolve()
        if not workspace.is_dir():
            raise ValueError(f"Invalid workspace: {workspace}")
        env["HOME"] = str(sandbox.get_sandbox_home(workspace))
        env["PWD"] = str(workspace)
        if not all((DCG_POLICY_DIR / name).is_file() for name in ("config.toml", "agent.yaml")):
            raise ValueError("Bundled dcg policy is missing")
        async with asyncio.timeout(SHELL_CHECK_TIMEOUT):
            process = await asyncio.create_subprocess_exec(
                "dcg",
                "test",
                "--stdin",
                "--robot",
                "--config",
                str(DCG_POLICY_DIR / "config.toml"),
                "--dialect",
                "posix",
                "--format",
                "json",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=env,
                cwd=workspace,
            )
            stdout, _ = await process.communicate(command.encode("utf-8"))
        result = json.loads(stdout)
        if not isinstance(result, dict) or process.returncode not in (0, 1):
            raise ValueError("Invalid dcg check result")
        decision = result.get("decision")
        if decision == "allow" and process.returncode == 0:
            return ExecutionDecision.ALLOW
        if decision == "ask":
            return ExecutionDecision.CONFIRM
        if decision == "deny":
            logger.warning("Shell high-risk command rejected: {}", result.get("rule_id"))
            return ExecutionDecision.DENY
        raise ValueError(f"dcg could not decide safely: {decision!r}")
    except (OSError, TimeoutError, ValueError) as exc:
        logger.warning("Shell dcg check failed; refusing execution: {}", exc)
        raise RuntimeError("危险命令检查失败，拒绝执行，请检查 dcg 是否可用。") from exc
    finally:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
