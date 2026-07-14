"""Network URL validation utilities."""

from __future__ import annotations

from urllib.parse import urlparse


def validate_url_target(url: str) -> tuple[bool, str]:
    """Validate that a URL has an HTTP(S) scheme and hostname.

    Returns (ok, error_message).  When ok is True, error_message is empty.
    """
    try:
        p = urlparse(url)
    except Exception as e:
        return False, str(e)

    if p.scheme not in ("http", "https"):
        return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
    if not p.netloc:
        return False, "Missing domain"

    hostname = p.hostname
    if not hostname:
        return False, "Missing hostname"

    return True, ""
