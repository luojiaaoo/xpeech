"""Run one command, bound output, and reap its children before returning.

Installed in the execution image, never imported by the application. Linux's
subreaper support keeps double-forked/setsid background children owned by this
supervisor so they can be stopped even after their original shell exits.
"""

import ctypes
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_LIMIT = 128 * 1024  # per stream, before it reaches the SDK/backend
TIMEOUT_MARKER = b"\n__XPEECH_COMMAND_TIMEOUT__\n"


class CommandInterrupted(Exception):
    """Distinct from InterruptedError, which selectors intentionally swallows."""


def descendants(pid: int) -> list[int]:
    found = []
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return found
    for child in children:
        child_pid = int(child)
        found.extend(descendants(child_pid))
        found.append(child_pid)
    return found


def stop_children() -> None:
    deadline = time.monotonic() + 3
    while True:
        for pid in descendants(os.getpid()):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Could not reap all command children")
        time.sleep(0.01)


def run(timeout: float, command: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")

    def interrupted(_signum, _frame):
        raise CommandInterrupted("Remote command interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = subprocess.Popen(
        ["bash", "-c", command], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    sizes = {1: 0, 2: 0}
    timed_out = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, 1)
            selector.register(process.stderr, selectors.EVENT_READ, 2)
            while selector.get_map():
                code = process.poll()
                if code is not None:
                    stop_children()
                elif time.monotonic() >= deadline:
                    timed_out = True
                    stop_children()
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target = key.data
                    remaining = max(0, OUTPUT_LIMIT - sizes[target])
                    if remaining:
                        os.write(target, chunk[:remaining])
                    if sizes[target] <= OUTPUT_LIMIT < sizes[target] + len(chunk):
                        os.write(target, b"\n[command output truncated]\n")
                    sizes[target] += len(chunk)
            # A command can close both streams and continue running.
            remaining = deadline - time.monotonic()
            if process.returncode is None:
                try:
                    process.wait(timeout=max(0, remaining))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    stop_children()
        if timed_out:
            os.write(2, TIMEOUT_MARKER)
            return 124
        code = process.returncode
        return 128 - code if code is not None and code < 0 else (code or 0)
    finally:
        # Do not let repeated termination prevent cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop_children()
        process.stdout.close()
        process.stderr.close()


if __name__ == "__main__":
    try:
        sys.exit(run(float(sys.argv[1]), sys.argv[2]))
    except CommandInterrupted:
        sys.exit(143)
