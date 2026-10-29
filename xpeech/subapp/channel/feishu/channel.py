from collections.abc import Callable
from typing import Any

from lark_channel import FeishuChannel


class XpeechFeishuChannel(FeishuChannel):
    """Keep card answers independent of messages waiting for those answers."""

    async def _through_action_safety(
        self,
        *,
        event_id: str,
        queue_scope: str,
        handler: Callable[[], Any],
    ) -> None:
        # SDK 1.4 shares message and card queues; preserve all other safety checks.
        if event_id.startswith("card:"):
            queue_scope = f"card-action:{queue_scope or event_id}"
        await super()._through_action_safety(
            event_id=event_id,
            queue_scope=queue_scope,
            handler=handler,
        )
