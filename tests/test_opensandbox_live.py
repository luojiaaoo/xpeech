"""Opt-in smoke test against a real Docker-backed OpenSandbox deployment."""

import hashlib
import os
import shutil
from uuid import uuid4

import pytest

from xpeech.agent.tools.filesystem import ReadFileArgs, WriteFileArgs, build_file_tools
from xpeech.agent.tools.shell.opensandbox import OpenSandboxExecutor
from xpeech.agent.tools.shell.shared_slots import acquire_lock, read_record
from xpeech.config.settings import settings

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_OPENSANDBOX_TESTS") != "1", reason="Requires a real OpenSandbox Docker deployment"),
]


async def test_shared_files_python_node_and_mounted_cli():
    workspace = settings.path.workspace_base_path / f"opensandbox-smoke-{uuid4().hex}"
    workspace.mkdir(parents=True)
    executor = OpenSandboxExecutor(settings.tool.shell)
    _, _, read_file, write_file, _ = build_file_tools(workspace)
    try:
        await write_file(WriteFileArgs(path="input.txt", content="written-by-local-tool"))
        command = (
            "set -eu; test \"$(cat input.txt)\" = written-by-local-tool; "
            "python -c \"from pathlib import Path; Path('output.txt').write_text('written-by-sandbox')\"; "
            "node --version; python --version; pip --version; uv --version; conda --version; lark-cli --version; "
            "test \"$HOME\" = \"$PWD/home\"; "
            "test -x /opt/xpeech/bin/lark-cli-auth; "
            "if touch /opt/xpeech/bin/should-be-readonly 2>/dev/null; then exit 1; fi"
        )
        result = await executor.run(command, workspace, timeout_ms=60_000)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "written-by-sandbox" in await read_file(ReadFileArgs(path="output.txt"))
        assert b"Python 3.12" in result.stdout + result.stderr
        # Keep the same user container but verify real timeout/partial output.
        result = await executor.run("echo partial; sleep 30", workspace, timeout_ms=500)
        assert result.timed_out and b"partial" in result.stdout
    finally:
        import time

        key = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()
        root = settings.tool.shell.state_path
        if root.exists():
            async with acquire_lock(root / f"user-{key}.lock", time.monotonic() + 10):
                for path in root.glob("resident-*.json"):
                    record = read_record(path)
                    if record.get("key") == key and record.get("sandbox_id"):
                        sandbox = await executor._connect(record["sandbox_id"])
                        if sandbox is not None:
                            try:
                                await executor._destroy(sandbox, path)
                            finally:
                                await sandbox.close()
        shutil.rmtree(workspace)
