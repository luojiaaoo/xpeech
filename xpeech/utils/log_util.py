def format_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"