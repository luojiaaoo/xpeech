from contextvars import ContextVar

session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_session_id() -> str | None:
    return session_id_var.get()


def get_request_id() -> str | None:
    return request_id_var.get()
