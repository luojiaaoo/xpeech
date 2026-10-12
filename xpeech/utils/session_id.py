from ..exceptions import PathProtectionError

MAX_SESSION_ID_LENGTH = 128


def validate_session_id(session_id: str) -> str:
    """Validate a session ID before using it in a filesystem path."""
    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > MAX_SESSION_ID_LENGTH
        or session_id in {".", ".."}
        or "/" in session_id
        or "\\" in session_id
        or "\0" in session_id
    ):
        raise PathProtectionError("Invalid session ID")
    return session_id
