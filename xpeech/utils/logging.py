import sys

from loguru import logger

from ..agent.server.context import get_request_id, get_user_id
from ..config.settings import settings

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>{extra[request_id]}</yellow> | "
    "<magenta>{extra[user_id]}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _inject_context(record):
    extra: dict = record["extra"]
    # 可能logger.bind(user_id="manual-session").info("hello")，所以用setdefault
    extra.setdefault("request_id", get_request_id())
    extra.setdefault("user_id", get_user_id())


def configure_logging():
    logger.remove()
    logger.configure(patcher=_inject_context)
    logger.add(
        sys.stderr,
        format=_FORMAT,
    )
    logger.add(
        settings.path.log_path / "xpeech.log",
        format=_FORMAT,
        encoding="utf-8",
        rotation=f"{settings.logging.max_file_size_mb} MB",
        retention=f"{settings.logging.retention_days} days",
        enqueue=True,
    )
