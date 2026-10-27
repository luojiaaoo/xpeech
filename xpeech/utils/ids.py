from ..exceptions import PathProtectionError

DEFAULT_SESSION_ID = "default"
MAX_ID_LENGTH = 128


def _validate_id(value: str, kind: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_ID_LENGTH
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\0" in value
    ):
        raise PathProtectionError(f"Invalid {kind} ID")
    return value


def validate_user_id(user_id: str) -> str:
    """Validate a user ID before using it in a filesystem path."""
    return _validate_id(user_id, "user")


def validate_session_id(session_id: str) -> str:
    """Validate a session ID before using it in a filesystem path."""
    return _validate_id(session_id, "session")
