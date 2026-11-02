from typing import Any


def configure_sqlite_connection(
    dbapi_connection: Any,
    _connection_record: Any = None,
    *,
    foreign_keys: bool = False,
    secure_delete: bool = False,
) -> None:
    """Allow concurrent readers and a writer on each SQLite connection."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA busy_timeout = 30000")
        cursor.execute("PRAGMA journal_mode = WAL")
        if foreign_keys:
            cursor.execute("PRAGMA foreign_keys = ON")
        if secure_delete:
            cursor.execute("PRAGMA secure_delete = ON")
    finally:
        cursor.close()
