"""HTTP calls between API workers and the single scheduler service."""

from typing import Any

from httpx import AsyncClient

from ....utils.internal_jwt import create_internal_access_token


async def scheduler_request(
    method: str,
    path: str,
    user_id: str | None = None,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
) -> Any:
    from ...server.server import app

    scheduler_url = app.state.scheduler_url
    headers = {"Authorization": f"Bearer {create_internal_access_token()}"}
    if user_id is not None:
        headers["x-user-id"] = user_id
    async with AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            f"{scheduler_url}{path}",
            headers=headers,
            json=payload,
        )
    if response.status_code in {400, 404}:
        raise ValueError(response.json()["detail"])
    response.raise_for_status()
    return response.json() if response.content else None
