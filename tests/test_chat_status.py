"""Session recovery checks the request lease using the authenticated user identity."""

import httpx
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.server.api import app
from xpeech.agent.server.schema import InboundMessage
from xpeech.agent.server.session_guard import CHAT_GUARD
from xpeech.utils.channel_jwt import UserTokenParams, create_token

SESSION_ID = "018f0000-0000-7000-8000-000000000001"
OTHER_SESSION_ID = "018f0000-0000-7000-8000-000000000002"


@pytest.mark.asyncio
async def test_chat_status_tracks_the_lease_and_isolates_users_and_sessions():
    token = create_token(UserTokenParams("web_client", user_id="status-user"))
    other_token = create_token(UserTokenParams("web_client", user_id="other-user"))
    headers = {"Authorization": f"Bearer {token}", "x-session-id": SESSION_ID}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        idle = await client.get("/chat/status", headers=headers)
        assert idle.status_code == 200
        assert idle.json() == {"active": False}
        assert idle.headers["cache-control"] == "no-store"
        inbox, owner = await CHAT_GUARD.submit(InboundMessage(
            user_id="status-user", session_id=SESSION_ID, channel="web_client", sender_name="Alice",
            content=[], session_metadata={}, files=[],
        ))
        assert owner
        try:
            busy = await client.get("/chat/status", headers=headers)
            assert busy.json() == {"active": True}
            other_user = await client.get("/chat/status", headers={
                **headers, "Authorization": f"Bearer {other_token}", "x-user-id": "status-user",
            })
            assert other_user.json() == {"active": False}
            other_session = await client.get("/chat/status", headers={**headers, "x-session-id": OTHER_SESSION_ID})
            assert other_session.json() == {"active": False}
        finally:
            await CHAT_GUARD.release("status-user", SESSION_ID, inbox)
        released = await client.get("/chat/status", headers=headers)
        assert released.json() == {"active": False}


@pytest.mark.parametrize("session_id", [None, "", "not-a-uuid"])
def test_chat_status_requires_a_valid_session_id(session_id):
    token = create_token(UserTokenParams("web_client", user_id="status-user"))
    headers = {"Authorization": f"Bearer {token}"}
    if session_id is not None:
        headers["x-session-id"] = session_id
    assert TestClient(app).get("/chat/status", headers=headers).status_code == 422


def test_chat_status_requires_authentication():
    assert TestClient(app).get("/chat/status", headers={"x-session-id": SESSION_ID}).status_code == 401
