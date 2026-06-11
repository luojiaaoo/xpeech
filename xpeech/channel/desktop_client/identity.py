from __future__ import annotations

from dataclasses import dataclass
import getpass
import uuid


@dataclass(frozen=True, slots=True)
class DesktopIdentity:
    machine_code: str
    session_id: str
    username: str


def get_identity() -> DesktopIdentity:
    machine_code = f"{uuid.getnode():012x}"[-12:]
    return DesktopIdentity(
        machine_code=machine_code,
        session_id=f"desktop_{machine_code}",
        username=getpass.getuser(),
    )
