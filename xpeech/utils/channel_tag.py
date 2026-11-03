"""Validate a channel's single tag and match it against resource tag lists."""

import re
from collections.abc import Iterable

CHANNEL_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


def validate_channel_tag(channel_tag: str) -> str:
    """Validate a single channel tag."""

    if not isinstance(channel_tag, str) or not CHANNEL_TAG_PATTERN.fullmatch(channel_tag):
        raise ValueError(
            "channel_tag must be a 1-64 character string using letters, numbers, '.', '_', ':', or '-'"
        )
    return channel_tag


def normalize_channel_tags(channel_tags: Iterable[str]) -> list[str]:
    """Validate, de-duplicate, and preserve the order of channel tags."""

    if isinstance(channel_tags, (str, bytes)):
        raise TypeError("channel_tags must be a list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in channel_tags:
        validate_channel_tag(tag)
        if tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def channel_tag_allows_access(resource_channel_tags: Iterable[str], channel_tag: str | None) -> bool:
    """Return whether a public or channel-scoped resource is visible to a connection."""

    required = set(resource_channel_tags)
    return not required or channel_tag in required
