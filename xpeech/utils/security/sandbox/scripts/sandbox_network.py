"""Standard-library-only bwrap/slirp4netns launcher, run outside the sandbox."""

import fcntl
import json
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

_NS_GET_USERNS = 0xB701
_STARTUP_TIMEOUT = 20
_SIOCGIFFLAGS = 0x8913
_IFF_UP = 0x1


def _enter_network_userns(net_fd: int) -> None:
    """Join the owner of the net namespace, even if bwrap nests its userns."""
    user_fd = fcntl.ioctl(net_fd, _NS_GET_USERNS)
    try:
        # setns cannot re-enter the caller's current user namespace. Root
        # normally remains in that namespace, keeping host mount access.
        if not os.path.samestat(os.fstat(user_fd), Path("/proc/self/ns/user").stat()):
            os.setns(user_fd, 0)
    finally:
        os.close(user_fd)


def _read_startup(fd: int, deadline: float) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
        raise RuntimeError("Timed out starting sandbox networking")
    data = os.read(fd, 1)
    if not data:
        raise RuntimeError("Sandbox networking exited before becoming ready")
    return data


def _wait_for_loopback(net_fd: int, deadline: float) -> None:
    """Probe in a separate process so the supervisor keeps its host network."""
    _enter_network_userns(net_fd)
    os.setns(net_fd, os.CLONE_NEWNET)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        request = struct.pack("16s24x", b"lo")
        while True:
            response = fcntl.ioctl(probe.fileno(), _SIOCGIFFLAGS, request)
            flags = struct.unpack_from("H", response, 16)[0]
            if flags & _IFF_UP:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for bwrap's loopback setup")
            time.sleep(0.001)


def _child_pid(fd: int, deadline: float) -> int:
    line = bytearray()
    while True:
        char = _read_startup(fd, deadline)
        if char == b"\n":
            info = json.loads(line)
            if "child-pid" in info:
                return int(info["child-pid"])
            line.clear()
        else:
            line.extend(char)


def _stop(process: subprocess.Popen | None) -> None:
    if process is not None:
        if process.poll() is None:
            process.kill()
        process.wait()


def _terminate(signum: int, _frame) -> None:
    raise SystemExit(128 + signum)


def run(command: list[str]) -> int:
    """Keep the sandbox blocked until NAT is ready and own both lifetimes."""
    bwrap = slirp = probe = None
    pid = None
    # This launcher is single-threaded, so joining a namespace in the slirp
    # child's preexec_fn is safe and never changes the backend's namespace.
    with ExitStack() as stack:
        pipes = {}

        def pipe() -> tuple[int, int]:
            descriptors = os.pipe()
            for fd in descriptors:
                pipes[fd] = stack.enter_context(os.fdopen(fd, "rb", buffering=0))
            return descriptors

        status_r, status_w = pipe()
        block_r, block_w = pipe()
        ready_r, ready_w = pipe()
        # Only the launcher keeps the write end; its death signals slirp EOF.
        exit_r, _exit_w = pipe()
        dns = stack.enter_context(tempfile.TemporaryFile())
        dns.write(b"nameserver 10.0.2.3\n")
        dns.seek(0)
        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT
            bwrap = subprocess.Popen(
                [command[0], "--json-status-fd", str(status_w),
                 "--block-fd", str(block_r),
                 "--ro-bind-data", str(dns.fileno()), "/etc/resolv.conf",
                 *command[1:]],
                pass_fds=(status_w, block_r, dns.fileno()),
            )
            # Close our copies of the child endpoints so startup failures
            # produce EOF. The file objects make repeated cleanup safe.
            for fd in (status_w, block_r):
                pipes[fd].close()
            pid = _child_pid(status_r, deadline)
            # bwrap reports its PID immediately after clone(), before the
            # child initializes its UID map. Entering and execing in an
            # unmapped userns would discard slirp's namespace capabilities.
            while not Path(f"/proc/{pid}/uid_map").read_text().strip():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Timed out waiting for bwrap's UID map")
                time.sleep(0.001)
            # Pin the network namespace before entering its owner userns.
            # Looking it up through /proc/<pid> after that transition races
            # bwrap's credential changes and can fail the ptrace access check.
            net = stack.enter_context(Path(f"/proc/{pid}/ns/net").open("rb", buffering=0))
            net_fd = net.fileno()
            # The PID/UID map does not imply that bwrap finished setting up
            # loopback (notably for root, whose UID map already exists).
            # slirp --configure also brings lo up, implicitly assigning
            # 127.0.0.1. Starting it first makes bwrap's exclusive RTM_NEWADDR
            # fail with EEXIST, often printed as "No child processes".
            probe = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()),
                 "--wait-loopback", str(net_fd), str(deadline)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                pass_fds=(net_fd,),
            )
            try:
                probe_returncode = probe.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise RuntimeError("Timed out waiting for bwrap's loopback setup") from None
            if probe_returncode != 0:
                raise RuntimeError("Failed waiting for bwrap's loopback setup")
            slirp = subprocess.Popen(
                ["slirp4netns", "--configure", "--disable-host-loopback",
                 "--cidr=10.0.2.0/24", "--userns-path=/proc/self/ns/user",
                 "--enable-sandbox", "--ready-fd", str(ready_w),
                 "--exit-fd", str(exit_r), "--netns-type=path",
                 f"/proc/self/fd/{net_fd}", "tap0"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                pass_fds=(ready_w, exit_r, net_fd),
                preexec_fn=lambda: _enter_network_userns(net_fd),
            )
            net.close()
            pipes[ready_w].close()
            if _read_startup(ready_r, deadline) != b"1" or slirp.poll() is not None:
                raise RuntimeError("slirp4netns failed to configure sandbox networking")
            os.write(block_w, b"1")
            while True:
                try:
                    returncode = bwrap.wait(timeout=0.2)
                    return returncode if returncode >= 0 else 128 - returncode
                except subprocess.TimeoutExpired:
                    if slirp.poll() is not None:
                        raise RuntimeError("slirp4netns exited while the sandbox was running")
        finally:
            # Kill bwrap before releasing --block-fd on startup failure. EOF
            # would otherwise let the command run without a configured network.
            if pid is not None and bwrap is not None and bwrap.poll() is None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            _stop(bwrap)
            _stop(probe)
            _stop(slirp)
            # --exit-fd also stops slirp if this supervisor receives SIGKILL;
            # --die-with-parent does the same for bwrap and its process tree.


if __name__ == "__main__":
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, _terminate)
    try:
        if sys.argv[1:2] == ["--wait-loopback"]:
            _wait_for_loopback(int(sys.argv[2]), float(sys.argv[3]))
        else:
            sys.exit(run(sys.argv[1:]))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Sandbox network setup failed: {error}", file=sys.stderr)
        sys.exit(1)
