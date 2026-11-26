import json
from collections.abc import Callable
from typing import Any

from lark_channel import FeishuChannel
from lark_channel.api.cardkit.v1.model.patch_card_element_request import PatchCardElementRequest
from lark_channel.api.cardkit.v1.model.patch_card_element_request_body import PatchCardElementRequestBody
from lark_channel.core import AccessTokenType, BaseRequest, HttpMethod


class XpeechFeishuChannel(FeishuChannel):
    """Keep card answers independent of messages waiting for those answers."""

    async def list_message_folder(self, message_id: str, file_key: str) -> list[dict[str, Any]]:
        """List the immediate children of a folder attached to a chat message."""
        request = (
            BaseRequest.builder()
            .http_method(HttpMethod.GET)
            .uri("/open-apis/im/v1/files/:file_key/folder")
            .paths({"file_key": file_key})
            .queries([("srctype", "message"), ("srcid", message_id), ("recursive", "false")])
            .token_types({AccessTokenType.TENANT})
            .build()
        )
        response = await self.client.arequest(request)
        if not response.success():
            raise RuntimeError(f"Feishu folder listing failed: {response.code}: {response.msg}")
        data = json.loads(response.raw.content)["data"]
        count = data.get("all_count")
        items = data.get("items")
        if items is None and count == 0:
            items = []
        if (
            not isinstance(items, list)
            or data.get("has_more")
            or (count is not None and count != len(items))
        ):
            raise ValueError("Feishu returned an incomplete folder listing")
        return items

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
