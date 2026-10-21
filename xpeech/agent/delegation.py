"""Run-scoped authority granted by the gateway to an agent container."""

import secrets
import asyncio
from dataclasses import dataclass, field
from typing import Any

from .server.schema import InboundMessage


@dataclass
class RuntimeLease:
    token: str
    user_id: str
    message: InboundMessage | None = None
    sandbox: Any = None
    base_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    completed_callbacks: dict[str, tuple[str, dict, Any]] = field(default_factory=dict)
    record_received: bool = False
    callback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# These capabilities live only for an active run. They are unrelated to the
# channel JWT key and cannot authorize calls for another user or another run.
ACTIVE_RUNTIME_LEASES: dict[str, RuntimeLease] = {}
ACTIVE_USER_RUNS: dict[str, RuntimeLease] = {}


def new_runtime_lease(user_id: str) -> RuntimeLease:
    lease = RuntimeLease(token=secrets.token_urlsafe(32), user_id=user_id)
    ACTIVE_RUNTIME_LEASES[lease.token] = lease
    return lease


def release_runtime_lease(lease: RuntimeLease) -> None:
    ACTIVE_RUNTIME_LEASES.pop(lease.token, None)
    if ACTIVE_USER_RUNS.get(lease.user_id) is lease:
        ACTIVE_USER_RUNS.pop(lease.user_id, None)
