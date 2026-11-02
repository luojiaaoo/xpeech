"""Channel-level background message polling, independent of user chat APIs."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from ...tools.schedule.client import scheduler_request
from ..auth import AuthenticatedChannel, require_channel_jwt
from ..schema import FeishuBackgroundMessage

router = APIRouter()


@router.get("/background_message", response_model=FeishuBackgroundMessage)
async def poll_background_message(
    auth: Annotated[AuthenticatedChannel, Depends(require_channel_jwt)],
) -> FeishuBackgroundMessage | Response:
    """Any backend can long-poll the shared scheduler for a channel message."""
    if auth.channel != "feishu":
        raise HTTPException(
            status_code=403,
            detail="Background messages are only available for the Feishu channel",
        )
    payload = await scheduler_request("GET", f"/background_message?channel={auth.channel}")
    if payload is None:
        return Response(status_code=204)
    return FeishuBackgroundMessage.model_validate(payload)
