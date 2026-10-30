"""Shared helpers for channel-scoped resources and JWT channels."""

import re
from collections.abc import Iterable

CHANNEL_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


def normalize_channel_tags(channel_tags: Iterable[str]) -> list[str]:
    """Validate, de-duplicate, and preserve the order of channel tags."""

    if isinstance(channel_tags, (str, bytes)):
        raise TypeError("channel_tags must be a list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in channel_tags:
        if not isinstance(tag, str) or not CHANNEL_TAG_PATTERN.fullmatch(tag):
            raise ValueError(
                "channel_tags must contain only 1-64 character strings using letters, numbers, '.', '_', ':', or '-'"
            )
        if tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def channel_tags_allow_access(resource_channel_tags: Iterable[str], channel_tags: Iterable[str]) -> bool:
    """Return whether a public or channel-scoped resource is visible to a connection."""

    required = set(resource_channel_tags)
    return not required or not required.isdisjoint(channel_tags)
