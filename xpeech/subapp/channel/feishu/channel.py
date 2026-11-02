import json
from collections.abc import Callable
from typing import Any

from lark_channel import FeishuChannel
from lark_channel.api.cardkit.v1.model.patch_card_element_request import PatchCardElementRequest
from lark_channel.api.cardkit.v1.model.patch_card_element_request_body import PatchCardElementRequestBody


class XpeechFeishuChannel(FeishuChannel):
    """Keep card answers independent of messages waiting for those answers."""

    async def patch_card_element(
        self,
        card_id: str,
        element_id: str,
        partial_element: dict[str, Any],
        sequence: int,
    ) -> None:
        """Update the tool summary without resetting the panel's expanded state."""
        body = (
            PatchCardElementRequestBody.builder()
            .partial_element(json.dumps(partial_element, ensure_ascii=False))
            .sequence(sequence)
            .build()
        )
        request = PatchCardElementRequest.builder().card_id(card_id).element_id(element_id).request_body(body).build()
        response = await self.client.cardkit.v1.card_element.apatch(request)
        if not response.success():
            raise RuntimeError(f"Feishu update card element failed: {response.code}: {response.msg}")

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
