"""Exercise the bundled policy with dcg; commands are only inspected, never run."""

import json
import shutil

import pytest

from xpeech.agent.tools.execution_request import ExecutionDecision
from xpeech.utils.security.dcg import shell_requires_confirmation

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("dcg") is None, reason="dcg is not installed"),
]


@pytest.mark.parametrize(("command", "expected"), [
    ("pwd", ExecutionDecision.ALLOW),
    ("git init", ExecutionDecision.ALLOW),
    ("echo shutdown", ExecutionDecision.ALLOW),
    ('echo "init 6"', ExecutionDecision.ALLOW),
    ("rg shutdown .", ExecutionDecision.ALLOW),
    ("command -v reboot", ExecutionDecision.ALLOW),
    ("systemctl status nginx", ExecutionDecision.ALLOW),
    ("dd if=input.bin of=output.bin", ExecutionDecision.CONFIRM),
    ("rm -rf ./build", ExecutionDecision.CONFIRM),
    ("rm -fr ./build", ExecutionDecision.CONFIRM),
    ("rm -r -f ./build", ExecutionDecision.CONFIRM),
    ("rm -f -R ./build", ExecutionDecision.CONFIRM),
    ("rm --recursive --force ./build", ExecutionDecision.CONFIRM),
    ("rm --force --recursive ./build", ExecutionDecision.CONFIRM),
    ("rm -rf /tmp/xpeech-cache", ExecutionDecision.CONFIRM),
    ("rm -r -f /tmp/xpeech-cache", ExecutionDecision.CONFIRM),
    ("rm --recursive --force /tmp/xpeech-cache", ExecutionDecision.CONFIRM),
    ("/bin/rm -fr /tmp/xpeech-cache", ExecutionDecision.CONFIRM),
    ("sudo -n rm -rf /tmp/xpeech-cache", ExecutionDecision.CONFIRM),
    ("git reset --hard", ExecutionDecision.CONFIRM),
    ("git clean -fdx", ExecutionDecision.CONFIRM),
    ("shutdown -h now", ExecutionDecision.DENY),
    ("reboot", ExecutionDecision.DENY),
    ("poweroff", ExecutionDecision.DENY),
    ("halt", ExecutionDecision.DENY),
    ("init 0", ExecutionDecision.DENY),
    ("init 6", ExecutionDecision.DENY),
    ("telinit 0", ExecutionDecision.DENY),
    ("telinit 6", ExecutionDecision.DENY),
    ("/sbin/telinit 0", ExecutionDecision.DENY),
    ("sudo -u root init 6", ExecutionDecision.DENY),
    ("env MODE=test telinit 6", ExecutionDecision.DENY),
    ("busybox poweroff", ExecutionDecision.DENY),
    ("systemctl reboot", ExecutionDecision.DENY),
    ("systemctl isolate rescue.target", ExecutionDecision.DENY),
    ("loginctl poweroff", ExecutionDecision.DENY),
    ("dd if=/dev/zero of=/dev/sda", ExecutionDecision.DENY),
    ("mkfs.ext4 /dev/sda", ExecutionDecision.DENY),
    ("rm -rf /", ExecutionDecision.DENY),
    ("rm -r -f /", ExecutionDecision.DENY),
    ("rm --recursive --force /", ExecutionDecision.DENY),
    ("rm -rf ./build && init 6", ExecutionDecision.DENY),
    ("init 6; rm -rf ./build", ExecutionDecision.DENY),
    ("rm -rf ./build\ntelinit 6", ExecutionDecision.DENY),
    ("rm -rf ./build && dd if=/dev/zero of=/dev/sda", ExecutionDecision.DENY),
    ("git reset --hard && systemctl reboot", ExecutionDecision.DENY),
    ("bash -c 'telinit 6'", ExecutionDecision.DENY),
    ('echo "$(telinit 6)"', ExecutionDecision.DENY),
    ('uv run python -c "import os; os.system(\'init 6\')"', ExecutionDecision.DENY),
])
async def test_shell_policy_decision(tmp_path, command, expected):
    assert await shell_requires_confirmation(json.dumps({"command": command}), workspace=tmp_path) is expected


async def test_shell_policy_reads_the_workspace_home(tmp_path):
    payload = json.dumps({"command": "printf data > ~/report.md"})
    target = tmp_path / "home" / "report.md"
    assert await shell_requires_confirmation(payload, workspace=tmp_path) is ExecutionDecision.ALLOW
    assert not target.exists()  # Inspection must not execute the command.

    target.write_text("existing workspace data")
    assert await shell_requires_confirmation(payload, workspace=tmp_path) is ExecutionDecision.DENY
    assert target.read_text() == "existing workspace data"
