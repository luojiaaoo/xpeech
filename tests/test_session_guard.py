import asyncio

import pytest

from xpeech.agent.server.schema import InboundMessage
from xpeech.agent.server.session_guard import SessionChatGuard, SessionInbox


def message(text="hello", *, user_id="user", session_id="session"):
    return InboundMessage(
        user_id=user_id, session_id=session_id, channel="test", sender_name="Alice",
        content=[{"text": text}], session_metadata={}, files=[],
    )


@pytest.mark.parametrize("phase", ["pending", "active", "closed"])
@pytest.mark.parametrize("field, value", [
    ("channel", "other"),
    ("sender_name", "Bob"), ("session_metadata", {"trace": "other"}),
])
def test_inbox_rejects_changed_context(phase, field, value):
    inbox = SessionInbox(message())
    if phase != "pending":
        inbox.start()
    if phase == "closed":
        inbox.close()

    with pytest.raises(ValueError, match=field):
        inbox.enqueue(message().model_copy(update={field: value}))

    assert inbox.user_id == "user"
    assert inbox.drain() == []
    followup = message("same user").model_copy(update={"json_output": True})
    assert inbox.enqueue(followup) is (phase == "active")
    assert inbox.drain() == ([followup] if phase == "active" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pending", "active", "closed"])
@pytest.mark.parametrize("field, value", [
    ("channel", "other"), ("sender_name", "Bob"),
    ("session_metadata", {"trace": "other"}),
])
async def test_submit_rejects_changed_context_before_waiting(phase, field, value):
    guard = SessionChatGuard()
    inbox, _ = await guard.submit(message())
    if phase != "pending":
        inbox.start()
    if phase == "closed":
        inbox.close()

    try:
        with pytest.raises(ValueError, match=field):
            await asyncio.wait_for(guard.submit(message().model_copy(update={field: value})), 1)
        assert inbox.drain() == []
    finally:
        await guard.release("user", "session", inbox)


@pytest.mark.asyncio
async def test_chat_can_join_active_loop_but_waits_for_cleanup_after_it_closes():
    guard = SessionChatGuard()
    inbox, owner = await guard.submit(message())
    assert owner

    followup = message("next").model_copy(update={"json_output": True})
    joining = asyncio.create_task(guard.submit(followup))
    await asyncio.sleep(0)
    assert not joining.done()

    inbox.start()
    assert await asyncio.wait_for(joining, 1) == (inbox, False)
    assert inbox.drain_or_close() == [followup]
    assert inbox.drain_or_close() == []

    waiting = asyncio.create_task(guard.submit(message("new turn")))
    await asyncio.sleep(0)
    assert not waiting.done()
    assert await guard.is_active("user", "session")

    await guard.release("user", "session", inbox)
    next_inbox, owner = await asyncio.wait_for(waiting, 1)
    assert owner and next_inbox is not inbox
    await guard.release("user", "session", inbox)
    assert await guard.is_active("user", "session")


@pytest.mark.asyncio
async def test_active_loop_is_isolated_by_user_and_session():
    guard = SessionChatGuard()
    inbox, owner = await guard.submit(message())
    assert owner
    inbox.start()

    assert (await guard.submit(message(user_id="other")))[1]
    assert (await guard.submit(message(session_id="other")))[1]
    assert inbox.drain() == []
