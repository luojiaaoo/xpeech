"""Single-host admission control shared by every backend replica.

Lock files live outside user workspaces on a shared local filesystem. flock
releases locks on process death; no per-process semaphore or expiring lease can
accidentally admit a second command while the first process still owns a lock.
"""

import asyncio
import fcntl
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any


def read_record(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    os.replace(temporary, path)


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def try_acquire(self) -> bool:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return True

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@asynccontextmanager
async def acquire_lock(path: Path, deadline: float):
    lock = FileLock(path)
    try:
        while not lock.try_acquire():
            if time.monotonic() >= deadline:
                raise TimeoutError("Sandbox queue is full; please retry later")
            await asyncio.sleep(0.05)
        yield
    finally:
        lock.release()


@asynccontextmanager
async def acquire_slot(root: Path, prefix: str, count: int, deadline: float):
    lock = None
    try:
        while lock is None:
            for index in range(count):
                candidate = FileLock(root / f"{prefix}-{index}.lock")
                if candidate.try_acquire():
                    lock = candidate
                    break
            if lock is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Sandbox queue is full; please retry later")
                await asyncio.sleep(0.05)
        yield
    finally:
        if lock is not None:
            lock.release()


@asynccontextmanager
async def acquire_resident(root: Path, key: str, count: int, deadline: float):
    """Prefer this user's container, then an empty slot, then unlocked LRU.

    The caller holds the user's lock. Resident slots remain locked until the
    command has stopped, so an executing container cannot be evicted.
    """
    lock = None
    try:
        while lock is None:
            candidates = []
            for index in range(count):
                path = root / f"resident-{index}.json"
                record = read_record(path)
                priority = 0 if record.get("key") == key else (1 if not record else 2)
                candidates.append((priority, record.get("last_used", 0), index))
            for _, _, index in sorted(candidates):
                candidate = FileLock(root / f"resident-{index}.lock")
                if candidate.try_acquire():
                    lock = candidate
                    path = root / f"resident-{index}.json"
                    record = read_record(path)
                    # A killed backend may have an outstanding create request.
                    # Keep its slot until its remote TTL has elapsed.
                    if record and not record.get("sandbox_id") and record.get("expires_at", 0) > time.time():
                        lock.release()
                        lock = None
                        continue
                    break
            if lock is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("All sandbox slots are busy; please retry later")
                await asyncio.sleep(0.05)
        yield path, record
    finally:
        if lock is not None:
            lock.release()
