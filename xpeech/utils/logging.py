import sys
from loguru import logger
from ..agent.server.context import get_session_id

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{extra[session_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _inject_context(record):
    extra: dict = record["extra"]
    # 可能logger.bind(session_id="manual-session").info("hello")，所以用setdefault
    extra.setdefault("session_id", get_session_id())


def configure_logging():
    logger.remove()
    logger.configure(patcher=_inject_context)
    logger.add(
        sys.stderr,
        format=_FORMAT,
    )
