"""查询用户所属组织，并匹配资源配置的组织前缀。"""

from collections.abc import Iterable

from loguru import logger

from .command_template import CommandTemplateError, execute_command_template


def normalize_organization_prefixes(prefixes: Iterable[str]) -> list[str]:
    """校验组织前缀非空且无首尾空白，按原顺序去重。"""
    if isinstance(prefixes, (str, bytes)):
        raise TypeError("organization_prefixes must be a list of non-empty strings")
    normalized = []
    for prefix in prefixes:
        if not isinstance(prefix, str) or not prefix.strip() or prefix != prefix.strip():
            raise ValueError("organization_prefixes must contain non-empty strings without surrounding whitespace")
        if prefix not in normalized:
            normalized.append(prefix)
    return normalized


def organization_allows_access(prefixes: Iterable[str], user_organization: str | None) -> bool:
    """未配置组织前缀的资源公开；已配置的资源需与用户组织做字面前缀匹配。"""
    required = tuple(prefixes)
    return not required or bool(user_organization and user_organization.startswith(required))


async def resolve_user_organization(
    user_id: str, *, command_template: str | None, timeout: float = 10.0,
) -> str | None:
    """通过服务端命令查询用户组织；查询失败时不加载有组织限制的资源。"""
    if command_template is None:
        return None
    try:
        organization = await execute_command_template(
            command_template,
            {"user_id": user_id},
            operation="organization.command_template",
            timeout=timeout,
            required_variables=("user_id",),
        )
    except CommandTemplateError as exc:
        logger.warning("Organization lookup failed: {}", exc)
        return None
    return organization or None
