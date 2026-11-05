"""Exercise network startup failures and helper lifetime with real namespaces."""

import os
import platform
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from xpeech.agent.tools.sandbox.scripts import sandbox_network

pytestmark = pytest.mark.skipif(
    platform.system() != "Linux" or any(shutil.which(name) is None for name in ("bwrap", "slirp4netns", "bash")),
    reason="Requires Linux with bubblewrap, slirp4netns and bash",
)


def _command(workspace: Path, script: str, *extra_mounts: str) -> list[str]:
    return [
        sys.executable, "-I", sandbox_network.__file__, "bwrap",
        "--unshare-net", "--unshare-pid", "--die-with-parent", "--new-session",
        "--cap-drop", "ALL", "--ro-bind", "/usr", "/usr",
        "--ro-bind-try", "/bin", "/bin", "--ro-bind-try", "/lib", "/lib",
        "--ro-bind-try", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
        "--bind", str(workspace), str(workspace), *extra_mounts,
        "--chdir", str(workspace), "--", "bash", "-c", script,
    ]


def test_network_failure_never_runs_the_command(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    slirp = fake_bin / "slirp4netns"
    slirp.write_text("#!/bin/sh\necho deliberate-network-failure >&2\nexit 42\n")
    slirp.chmod(0o755)

    result = subprocess.run(
        _command(tmp_path, "echo executed > should-not-exist"),
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True, timeout=10,
    )

    assert result.returncode != 0
    assert b"deliberate-network-failure" in result.stderr
    assert not (tmp_path / "should-not-exist").exists()


def test_concurrent_network_startup_preserves_namespace_permissions(tmp_path):
    processes = []
    try:
        for _ in range(16):
            processes.append(subprocess.Popen(
                _command(tmp_path, "printf ready"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ))
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr.decode()
            assert stdout == b"ready"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


@pytest.mark.parametrize("termination", ["exit", "terminate", "kill"])
def test_network_helper_stops_with_the_sandbox(tmp_path, termination):
    real_slirp = shutil.which("slirp4netns")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    helper_pid = tmp_path / "helper-pid"
    slirp = fake_bin / "slirp4netns"
    slirp.write_text(
        f"#!/bin/sh\necho $$ > {shlex.quote(str(helper_pid))}\n"
        f"exec {shlex.quote(real_slirp)} \"$@\"\n"
    )
    slirp.chmod(0o755)
    script = (
        "cat /etc/resolv.conf; readlink /proc/self/ns/net; "
        "read -r finish; exit 7"
    )
    process = subprocess.Popen(
        _command(tmp_path, script),
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0], "Sandbox did not start"
        assert process.stdout.readline() == b"nameserver 10.0.2.3\n"
        namespace = process.stdout.readline().decode().strip()
        assert namespace != os.readlink("/proc/self/ns/net")
        pid = int(helper_pid.read_text())
        if termination == "exit":
            process.communicate(b"done\n", timeout=10)
            assert process.returncode == 7
        else:
            process.send_signal(signal.SIGTERM if termination == "terminate" else signal.SIGKILL)
            process.communicate(timeout=10)
        deadline = time.monotonic() + 5
        while True:
            try:
                status = Path(f"/proc/{pid}/stat").read_text()
            except FileNotFoundError:
                break
            # An orphan may briefly remain a zombie awaiting the host's reaper.
            if status.split()[2] == "Z":
                break
            assert time.monotonic() < deadline, "slirp4netns survived sandbox shutdown"
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


def test_sandbox_can_send_outbound_tcp_and_udp(tmp_path):
    # Use the host's routed address so this test needs no public Internet
    # service and does not rely on the deliberately disabled host loopback.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        try:
            route.connect(("192.0.2.1", 9))
        except OSError:
            pytest.skip("Requires an IPv4 route")
        address = route.getsockname()[0]
    with socket.socket() as tcp, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        tcp.bind((address, 0))
        tcp.listen()
        tcp.settimeout(5)
        udp.bind((address, 0))
        udp.settimeout(5)
        result = subprocess.run(
            _command(
                tmp_path,
                f"printf tcp > /dev/tcp/{address}/{tcp.getsockname()[1]} && "
                f"printf udp > /dev/udp/{address}/{udp.getsockname()[1]}",
            ),
            capture_output=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr.decode()
        connection, _ = tcp.accept()
        with connection:
            connection.settimeout(5)
            assert connection.recv(3) == b"tcp"
        assert udp.recv(3) == b"udp"


@pytest.mark.skipif(os.geteuid() != 0, reason="Root mount regression")
def test_root_can_mount_a_regular_directory_below_a_private_host_parent(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    ordinary = private / "ordinary"
    ordinary.mkdir(mode=0o755)
    (ordinary / "config").write_text("managed config")
    os.chown(ordinary, 1000, 1000)
    result = subprocess.run(
        _command(tmp_path, "test \"$(id -u)\" = 0 && cat /managed/config", "--ro-bind", str(ordinary), "/managed"),
        capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"managed config"
