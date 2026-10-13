from contextvars import ContextVar

user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_user_id() -> str | None:
    return user_id_var.get()


def get_request_id() -> str | None:
    return request_id_var.get()
