from contextvars import ContextVar

session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)


def get_session_id() -> str | None:
    return session_id_var.get()
