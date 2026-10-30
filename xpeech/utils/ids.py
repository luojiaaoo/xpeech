import re
from uuid import UUID

from uuid6 import uuid7

from ..exceptions import PathProtectionError

DEFAULT_SESSION_ID = "default"
MAX_ID_LENGTH = 128
_ID_PATTERN = re.compile(rf"[A-Za-z0-9 _-]{{1,{MAX_ID_LENGTH}}}")


def _validate_id(value: str, kind: str) -> str:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise PathProtectionError(
            f"{kind.capitalize()} ID must be 1-{MAX_ID_LENGTH} characters long and contain only "
            "ASCII letters, digits, hyphens, underscores, or spaces"
        )
    return value


def validate_user_id(user_id: str) -> str:
    """Validate a user ID before using it in a filesystem path."""
    return _validate_id(user_id, "user")


def validate_session_id(session_id: str) -> str:
    """Validate the default main session or a canonical lowercase UUIDv7."""
    if session_id == DEFAULT_SESSION_ID:
        return session_id
    if not isinstance(session_id, str):
        raise PathProtectionError("Session ID must be 'default' or a canonical UUIDv7")
    try:
        parsed = UUID(session_id)
    except (ValueError, AttributeError, TypeError):
        raise PathProtectionError("Session ID must be 'default' or a canonical UUIDv7") from None
    canonical = str(parsed)
    if parsed.version != 7 or session_id != canonical:
        raise PathProtectionError("Session ID must be 'default' or a canonical UUIDv7")
    return canonical


def generate_session_id() -> str:
    """Generate a canonical UUIDv7 session ID as defined by RFC 9562."""
    return str(uuid7())
