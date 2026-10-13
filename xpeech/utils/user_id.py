from ..exceptions import PathProtectionError

MAX_USER_ID_LENGTH = 128


def validate_user_id(user_id: str) -> str:
    """Validate a user ID before using it in a filesystem path."""
    if (
        not isinstance(user_id, str)
        or not user_id
        or len(user_id) > MAX_USER_ID_LENGTH
        or user_id in {".", ".."}
        or "/" in user_id
        or "\\" in user_id
        or "\0" in user_id
    ):
        raise PathProtectionError("Invalid user ID")
    return user_id
