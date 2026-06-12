from __future__ import annotations

from dataclasses import dataclass
import getpass
import hashlib
import uuid


@dataclass(frozen=True, slots=True)
class DesktopIdentity:
    machine_code: str
    session_id: str
    username: str


def get_identity() -> DesktopIdentity:
    machine_code = _machine_code()
    return DesktopIdentity(
        machine_code=machine_code,
        session_id=f"desktop_{machine_code}",
        username=getpass.getuser(),
    )


def _machine_code() -> str:
    username = getpass.getuser()
    mac = f"{uuid.getnode():012x}"
    fingerprint = "\n".join(["xpeech-desktop-hw", username, mac])
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
