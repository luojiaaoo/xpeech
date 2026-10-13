from contextvars import ContextVar

session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_access_token_var: ContextVar[str | None] = ContextVar("user_access_token", default=None)


def get_session_id() -> str | None:
    return session_id_var.get()


def get_request_id() -> str | None:
    return request_id_var.get()


def get_user_access_token() -> str | None:
    return user_access_token_var.get()
