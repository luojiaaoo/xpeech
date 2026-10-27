import re

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
    """Validate a session ID before using it in a filesystem path."""
    return _validate_id(session_id, "session")
