import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from html import unescape
from io import BytesIO
from pathlib import Path
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zipfile import ZipFile

import httpx
import pytest
from lark_channel import (
    CardActionEvent,
    CardActionPayload,
    Conversation,
    EventOperator,
    FeishuChannel,
    FileContent,
    FolderContent,
    Identity,
    ImageContent,
    InboundMessage,
    MediaContent,
    PostContent,
    SendResult,
    TextContent,
)
from lark_channel.channel.safety.pipeline import SafetyPipeline
from lark_channel.core import AccessTokenType, HttpMethod

import xpeech.subapp.channel.feishu.backend as feishu_backend
import xpeech.subapp.channel.feishu.bridge as feishu_bridge
import xpeech.subapp.channel.feishu.conversation as feishu_conversation
import xpeech.subapp.channel.feishu.delivery as feishu_delivery
import xpeech.subapp.channel.feishu.inbound as feishu_inbound
import xpeech.subapp.channel.feishu.questions as feishu_questions
from xpeech.subapp.channel.feishu.bridge import FeishuBridge
from xpeech.subapp.channel.feishu.cards import (
    build_feishu_background_task_card,
    build_feishu_question_card,
)
from xpeech.subapp.channel.feishu.channel import XpeechFeishuChannel
from xpeech.subapp.channel.feishu.config import DEFAULT_SESSION_ID
from xpeech.subapp.channel.feishu.formatting import FeishuEventFormatter
from xpeech.subapp.channel.feishu.inbound import FeishuInbound
from xpeech.subapp.channel.schema import ChatEvent, ChatEventType, FileData, Message, TextData
from xpeech.utils.question import build_execution_question


def _inbound_message(
    content,
    *,
    message_id: str = "om_message",
    chat_type: str = "p2p",
) -> InboundMessage:
    return InboundMessage(
        id=message_id,
        create_time=123_000,
        conversation=Conversation(chat_id="oc_chat", chat_type=chat_type),
        sender=Identity(open_id="ou_sender", display_name="Alice"),
        content=content,
    )


def _bridge_with_channel(
    channel,
    *,
    employee_no: str = "E1001",
    email: str | None = "alice@example.com",
) -> FeishuBridge:
    channel.on = MagicMock()
    channel.block_batch_scope = MagicMock()
    channel.unblock_batch_scope = MagicMock()
    bridge = FeishuBridge("http://backend.test", "cli_test", "secret", channel=channel)
    bridge.inbound.get_user_identity = AsyncMock(return_value=(employee_no, email))
    return bridge


def _remember_question(bridge, message_id, question):
    card = feishu_questions.QuestionCard(message_id, json.dumps(question))
    bridge.questions.question_cards[message_id] = card
    return card


def test_channel_uses_expected_compatibility_configuration():
    bridge = FeishuBridge(
        chat_url="http://backend.test",
        app_id="cli_test",
        app_secret="secret",
    )

    assert bridge.channel.config.policy.dm_policy == "open"
    assert bridge.channel.config.policy.group_policy == "open"
    assert bridge.channel.config.policy.require_mention is True
    assert bridge.channel.config.safety.dedup.ttl_seconds == 43_200
    assert bridge.channel.config.safety.chat_queue.enabled is False
    assert bridge.channel.config.outbound.retry.max_attempts == 5
    assert bridge.channel.config.security.mode == "compat"


@pytest.mark.asyncio
async def test_sdk_delivers_busy_messages_and_stop_without_waiting_for_the_original_reply(tmp_path, monkeypatch):
    bridge = FeishuBridge("http://backend.test", "cli_test", "secret")
    channel = bridge.channel
    channel._bg_loop = asyncio.get_running_loop()
    bridge.inbound.get_user_identity = AsyncMock(return_value=("E1001", "alice@example.com"))
    channel.download_resource_to_file = AsyncMock(return_value=tmp_path / "report.txt")
    channel.update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_question"))
    channel.send = AsyncMock(return_value=SendResult(success=True, message_id="om_stop_reply"))
    stop_chat = AsyncMock(return_value=httpx.Response(
        200, json={"stopped": True}, request=httpx.Request("POST", "http://backend.test/chat/stop"),
    ))
    monkeypatch.setattr(feishu_bridge, "stop_chat", stop_chat)
    question_card = _remember_question(bridge, "om_question", {
        "type": "form", "fields": [{"type": "input", "name": "answer", "label": "回答"}],
    })
    first_started = asyncio.Event()
    answered = asyncio.Event()
    batches = []
    submitted = asyncio.Event()

    async def consume(messages):
        batches.append(messages)
        if len(batches) == 4:
            submitted.set()
        if len(batches) == 1:
            first_started.set()
            await answered.wait()

    async def answer(*_args, **_kwargs):
        answered.set()

    bridge.consume = consume
    notify_question = AsyncMock(side_effect=answer)
    monkeypatch.setattr(feishu_questions, "notify_question", notify_question)
    safety = channel.config.safety
    pipeline = SafetyPipeline(
        loop=asyncio.get_running_loop(),
        on_message=bridge._on_message,
        policy=channel.config.policy,
        dedup_config=safety.dedup,
        batch_config=safety.text_batch,
        media_batch_config=safety.media_batch,
        queue_config=safety.chat_queue,
    )
    channel._safety = pipeline
    contents = [
        TextContent(text="first"),
        TextContent(text="second" * 1000),
        FileContent(file_key="file_report", file_name="report.txt"),
        TextContent(text="third"),
    ]
    messages = [_inbound_message(content, message_id=f"om_{index}") for index, content in enumerate(contents)]
    stop_message = _inbound_message(TextContent(text=" /stop\n"), message_id="om_stop")
    stop_message.create_time = int(time() * 1000)
    for message in messages:
        message.create_time = int(time() * 1000)
    action = SimpleNamespace(
        event=SimpleNamespace(
            context=SimpleNamespace(open_message_id="om_question", open_chat_id="oc_chat"),
            operator=SimpleNamespace(open_id="ou_sender"),
            action=SimpleNamespace(tag="button", value={}, form_value={"answer": "yes"}),
        )
    )
    first = asyncio.create_task(pipeline.push_message(messages[0]))
    try:
        await asyncio.wait_for(first_started.wait(), timeout=2)
        for message in messages[1:]:
            await pipeline.push_message(message)
        await asyncio.wait_for(submitted.wait(), timeout=1)
        assert not first.done()
        await asyncio.wait_for(pipeline.push_message(stop_message), timeout=1)
        await pipeline.push_message(stop_message)
        stop_chat.assert_awaited_once_with(
            "E1001", bridge.chat_url, "feishu", session_id=DEFAULT_SESSION_ID,
        )
        assert not first.done()

        # These go through the real SDK card dispatcher, including its dedup gate.
        await channel._handle_interaction_event(action)
        await channel._handle_interaction_event(action)
        await asyncio.wait_for(answered.wait(), timeout=1)
        await asyncio.wait_for(first, timeout=1)
        await asyncio.wait_for(pipeline.dispose(), timeout=2)
    finally:
        answered.set()
        await first
        await pipeline.dispose()

    assert [[message.message_id for message in batch] for batch in batches] == [
        ["om_0"],
        ["om_1"], ["om_2"], ["om_3"],
    ]
    assert batches[1][0].content == [TextData(text="second" * 1000)]
    assert batches[2][0].content == [FileData(file=tmp_path / "report.txt")]
    assert batches[3][0].content == [TextData(text="third")]
    notify_question.assert_awaited_once_with(
        "E1001",
        {"answer": "yes"},
        bridge.chat_url,
        "feishu",
        session_id=DEFAULT_SESSION_ID,
    )
    channel.update_card.assert_awaited_once_with(
        "om_question", build_feishu_question_card(question_card.context, disabled=True, answers={"answer": "yes"}),
    )


@pytest.mark.parametrize(
    ("event_id", "scope", "expected"),
    [
        ("card:answer", "oc_chat", "card-action:oc_chat"),
        ("card:answer", "", "card-action:card:answer"),
        ("comment:doc:comment", "doc", "doc"),
    ],
)
@pytest.mark.asyncio
async def test_card_queue_override_preserves_parent_safety(monkeypatch, event_id, scope, expected):
    parent_safety = AsyncMock()
    monkeypatch.setattr(FeishuChannel, "_through_action_safety", parent_safety)
    channel = XpeechFeishuChannel(app_id="cli_test", app_secret="secret")
    handler = AsyncMock()

    await channel._through_action_safety(event_id=event_id, queue_scope=scope, handler=handler)

    parent_safety.assert_awaited_once_with(
        event_id=event_id,
        queue_scope=expected,
        handler=handler,
    )


def test_background_task_card_contains_result_content():
    assert build_feishu_background_task_card(
        "**scheduled result**",
        schedule_type="cron",
        schedule_value="0 9 * * *",
    ) == {
        "schema": "2.0",
        "header": {
            "title": {"tag": "plain_text", "content": "定时任务执行结果"},
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": "**调度类型**：cron\n**调度表达式**：`0 9 * * *`",
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                },
                {
                    "tag": "markdown",
                    "content": "**scheduled result**",
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                },
            ]
        },
    }


def test_background_task_card_escapes_dynamic_markup():
    raw = '**结果**\n[链接](https://example.test)\n<rss>甲 &amp; 乙</rss>'
    card = build_feishu_background_task_card(raw, schedule_type="<cron>", schedule_value="<expression>")
    metadata, result = card["body"]["elements"]
    for element in (metadata, result):
        assert "<" not in element["content"] and ">" not in element["content"]
    assert unescape(result["content"]) == raw
    assert unescape(metadata["content"]) == "**调度类型**：<cron>\n**调度表达式**：`<expression>`"
    assert "**结果**" in result["content"] and "[链接](https://example.test)" in result["content"]










@pytest.mark.asyncio
async def test_get_user_identity_queries_contact_api_and_caches_result():
    response = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(user=SimpleNamespace(employee_no="E1001", email="alice@example.com")),
    )
    get_user = AsyncMock(return_value=response)
    client = SimpleNamespace(contact=SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(aget=get_user))))
    bridge = SimpleNamespace(inbound=FeishuInbound(SimpleNamespace(client=client)))

    first = await bridge.inbound.get_user_identity("ou_sender")
    second = await bridge.inbound.get_user_identity("ou_sender")

    assert first == second == ("E1001", "alice@example.com")
    get_user.assert_awaited_once()
    request = get_user.await_args.args[0]
    assert request.user_id == "ou_sender"
    assert request.user_id_type == "open_id"


@pytest.mark.parametrize(
    ("response", "error_message"),
    [
        (
            SimpleNamespace(success=lambda: False, code=41050, msg="no user authority", data=None),
            "Failed to get Feishu user identity",
        ),
        (
            SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="success",
                data=SimpleNamespace(user=SimpleNamespace(employee_no=None, email="alice@example.com")),
            ),
            "Feishu user has no employee_no",
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_user_identity_rejects_api_failure_or_missing_fields(response, error_message: str):
    get_user = AsyncMock(return_value=response)
    client = SimpleNamespace(contact=SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(aget=get_user))))
    bridge = SimpleNamespace(inbound=FeishuInbound(SimpleNamespace(client=client)))

    with pytest.raises(RuntimeError, match=error_message):
        await FeishuInbound.get_user_identity.__wrapped__(bridge.inbound, "ou_missing")


@pytest.mark.parametrize(
    ("email", "enterprise_email", "expected_email"),
    [
        ("alice@example.com", "alice@company.example", "alice@example.com"),
        (None, "alice@company.example", "alice@company.example"),
        (None, None, None),
    ],
)
@pytest.mark.asyncio
async def test_get_user_identity_supports_enterprise_or_missing_email(email, enterprise_email, expected_email):
    response = SimpleNamespace(
        success=lambda: True,
        data=SimpleNamespace(user=SimpleNamespace(employee_no="E1001", email=email, enterprise_email=enterprise_email)),
    )
    get_user = AsyncMock(return_value=response)
    client = SimpleNamespace(contact=SimpleNamespace(v3=SimpleNamespace(user=SimpleNamespace(aget=get_user))))
    bridge = SimpleNamespace(inbound=FeishuInbound(SimpleNamespace(client=client)))

    identity = await FeishuInbound.get_user_identity.__wrapped__(bridge.inbound, "ou_sender")

    assert identity == ("E1001", expected_email)


@pytest.mark.asyncio
async def test_parse_text_message_uses_employee_number_and_feishu_user_metadata():
    bridge = _bridge_with_channel(SimpleNamespace())

    async with AsyncExitStack() as resources:
        message = await bridge.inbound.parse_message(_inbound_message(TextContent(text="hello")), resources)

    assert message.message_id == "om_message"
    assert message.chat_id == "oc_chat"
    assert message.user_id == "E1001"
    assert message.sender_name == "Alice"
    assert message.timestamp == 123
    assert message.content == [TextData(text="hello")]
    assert message.session_metadata == {
        "open_id": "ou_sender",
        "email": "alice@example.com",
    }


@pytest.mark.asyncio
async def test_parse_text_message_allows_missing_email():
    bridge = _bridge_with_channel(SimpleNamespace(), email=None)

    async with AsyncExitStack() as resources:
        message = await bridge.inbound.parse_message(_inbound_message(TextContent(text="hello")), resources)

    assert message.user_id == "E1001"
    assert message.session_metadata == {
        "open_id": "ou_sender",
    }


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.asyncio
async def test_on_message_keeps_request_cleanup_with_its_owner_and_does_not_block_sdk_scope(cancelled):
    channel = SimpleNamespace(block_batch_scope=MagicMock(), unblock_batch_scope=MagicMock())
    bridge = _bridge_with_channel(channel)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=123,
        session_metadata={},
    )
    bridge.inbound.parse_message = AsyncMock(return_value=message)
    started = asyncio.Event()
    release = asyncio.Event()

    async def consume(messages):
        assert messages == [message]
        started.set()
        await release.wait()

    bridge.consume = consume
    async with asyncio.TaskGroup() as tasks:
        task = tasks.create_task(bridge._on_message(_inbound_message(TextContent(text="hello"))))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()
        channel.block_batch_scope.assert_not_called()
        channel.unblock_batch_scope.assert_not_called()
        if cancelled:
            task.cancel()
        else:
            release.set()

    assert task.cancelled() is cancelled
    channel.unblock_batch_scope.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_ignores_unsupported_message(monkeypatch: pytest.MonkeyPatch):
    bridge = _bridge_with_channel(SimpleNamespace())
    warning = MagicMock()
    monkeypatch.setattr(feishu_bridge, "logger", SimpleNamespace(warning=warning))
    inbound_message = _inbound_message(TextContent(text="hello"), chat_type="group")

    await bridge._on_message(inbound_message)

    bridge.channel.unblock_batch_scope.assert_not_called()
    warning.assert_called_once_with(
        "Unsupported Feishu message: chat_type=group, content_type=TextContent, message_id=om_message",
    )


@pytest.mark.asyncio
async def test_channel_send_logs_message_type(monkeypatch: pytest.MonkeyPatch):
    send_result = SendResult(success=True, message_id="om_sent")
    bridge = _bridge_with_channel(SimpleNamespace())
    bridge.delivery.send = AsyncMock(return_value=send_result)
    info = MagicMock()
    monkeypatch.setattr(feishu_delivery, "logger", SimpleNamespace(info=info))

    await bridge.delivery.channel_send(
        to="oc_chat",
        message={"text": "hello"},
        opts={"reply_to": "om_parent"},
        user_id="E1001",
        sender_name="Alice",
        message_type=ChatEventType.ASSISTANT,
        iter_content=None,
    )

    bridge.delivery.send.assert_awaited_once_with(
        to="oc_chat",
        message={"text": "hello"},
        iter_content=None,
        opts={"reply_to": "om_parent"},
    )
    info.assert_called_once_with(
        "Feishu message sending user_id={} sender_name={} message_type={}",
        "E1001",
        "Alice",
        ChatEventType.ASSISTANT,
    )


@pytest.mark.parametrize(
    ("content", "resource_type", "file_name", "downloaded_name"),
    [
        (ImageContent(image_key="img_key"), "image", None, "img_key.png"),
        (FileContent(file_key="file_key", file_name="report.pdf"), "file", "report.pdf", "report.pdf"),
        (MediaContent(file_key="video_key", file_name="demo.mp4"), "video", "demo.mp4", "demo.mp4"),
    ],
)
@pytest.mark.asyncio
async def test_parse_resource_message_uses_channel_download_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content,
    resource_type: str,
    file_name: str | None,
    downloaded_name: str,
):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "E1001" / downloaded_name
    download_resource = AsyncMock(return_value=downloaded_path)
    bridge = _bridge_with_channel(SimpleNamespace(download_resource_to_file=download_resource))

    async with AsyncExitStack() as resources:
        message = await bridge.inbound.parse_message(_inbound_message(content), resources)

    expected_kwargs = {
        "resource_type": resource_type,
        "message_id": "om_message",
        "dest_dir": tmp_path / "E1001",
    }
    if file_name:
        expected_kwargs["file_name"] = file_name
    file_key = content.image_key if resource_type == "image" else content.file_key
    download_resource.assert_awaited_once_with(file_key, **expected_kwargs)
    assert message.content == [FileData(file=downloaded_path)]


@pytest.mark.asyncio
async def test_parse_post_reads_content_v2_document_and_preserves_attachment_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    downloaded_path = tmp_path / "E1001" / "img_post.png"
    download_resource = AsyncMock(return_value=downloaded_path)
    bridge = _bridge_with_channel(SimpleNamespace(download_resource_to_file=download_resource))
    post = PostContent(
        post={
            "content_v2": [
                [
                    {"tag": "text", "text": "before"},
                    {"tag": "a", "text": "link", "href": "https://example.com"},
                ],
                [{"tag": "img", "image_key": "img_post"}],
                [{"tag": "text", "text": "after"}],
            ],
        }
    )

    async with AsyncExitStack() as resources:
        message = await bridge.inbound.parse_message(_inbound_message(post), resources)

    assert message.content == [
        TextData(text="before\n[link](https://example.com)"),
        TextData(text="[Attachment: img_post.png]"),
        FileData(file=downloaded_path),
        TextData(text="after"),
    ]
    download_resource.assert_awaited_once_with(
        "img_post",
        resource_type="image",
        message_id="om_message",
        dest_dir=tmp_path / "E1001",
    )


@pytest.mark.parametrize("as_post", [False, True])
@pytest.mark.asyncio
async def test_on_message_archives_feishu_folder(tmp_path, monkeypatch, as_post):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    channel = XpeechFeishuChannel(app_id="cli_test", app_secret="secret")
    bridge = _bridge_with_channel(channel)
    folders = {
        "root": [
            {"file_key": "root_file", "name": "同名.txt", "is_folder": False},
            {"file_key": "nested", "name": "子目录", "is_folder": True},
            {"file_key": "empty", "name": "空目录", "is_folder": True},
        ],
        "nested": [{"file_key": "nested_file", "name": "同名.txt", "is_folder": False}],
        "empty": [],
    }

    async def request(req):
        assert req.http_method == HttpMethod.GET
        assert req.uri == "/open-apis/im/v1/files/:file_key/folder"
        assert req.token_types == {AccessTokenType.TENANT}
        assert dict(req.queries) == {"srctype": "message", "srcid": "om_message", "recursive": "false"}
        items = folders[req.paths["file_key"]]
        return SimpleNamespace(
            success=lambda: True,
            raw=SimpleNamespace(content=json.dumps({"data": {"items": items, "all_count": len(items)}}).encode()),
        )

    async def download(file_key, *, resource_type, message_id, dest_dir, file_name):
        assert resource_type == "file"
        assert message_id == "om_message"
        path = dest_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(file_key.encode())
        return path

    monkeypatch.setattr(channel.client, "arequest", request)
    channel.download_resource_to_file = AsyncMock(side_effect=download)
    channel.add_reaction = AsyncMock()
    channel.send = AsyncMock()

    uploads = []

    async def events():
        yield {"event": "user"}

    @asynccontextmanager
    async def open_chat_stream(_auth, _sender, content, _metadata, url, *, session_id, files):
        uploads.append((url, json.loads(content), [(field, name, file.read()) for field, (name, file, _) in files]))
        yield httpx.Response(200, request=httpx.Request("POST", url)), events()

    monkeypatch.setattr(feishu_backend, "open_chat_stream", open_chat_stream)
    content = FolderContent(file_key="root", file_name="教程")
    if as_post:
        content = PostContent(post={
            "content_v2": [[{"tag": "text", "text": "嗨嗨"}]],
            "files": [
                {"file_key": "root", "file_name": "教程", "is_folder": True},
                {"file_key": "ordinary_file", "file_name": "普通.txt"},
            ],
        })

    await bridge._on_message(_inbound_message(content))

    ((url, text, files),) = uploads
    assert url == "http://backend.test/chat"
    if as_post:
        assert text == [{"text": "嗨嗨"}]
        assert files[1:] == [("files", "普通.txt", b"ordinary_file")]
    else:
        assert text == []
        assert len(files) == 1
    field, name, data = files[0]
    assert (field, name) == ("files", "教程.zip")
    with ZipFile(BytesIO(data)) as archive:
        assert set(archive.namelist()) == {
            "教程/", "教程/同名.txt", "教程/子目录/", "教程/子目录/同名.txt", "教程/空目录/",
        }
        assert archive.read("教程/同名.txt") == b"root_file"
        assert archive.read("教程/子目录/同名.txt") == b"nested_file"
    channel.send.assert_not_awaited()
    assert not list(tmp_path.glob("xpeech-feishu-folder-*"))


@pytest.mark.asyncio
async def test_list_message_folder_rejects_truncated_listing(monkeypatch):
    channel = XpeechFeishuChannel(app_id="cli_test", app_secret="secret")
    request = AsyncMock(return_value=SimpleNamespace(
        success=lambda: True,
        raw=SimpleNamespace(content=b'{"data": {"items": [], "all_count": 1}}'),
    ))
    monkeypatch.setattr(channel.client, "arequest", request)

    with pytest.raises(ValueError, match="incomplete folder listing"):
        await channel.list_message_folder("om_message", "folder_key")


@pytest.mark.parametrize("as_post", [False, True])
@pytest.mark.asyncio
async def test_on_message_reports_folder_download_failure(tmp_path, monkeypatch, as_post):
    monkeypatch.setattr(feishu_inbound, "FEISHU_CACHE_DIR", tmp_path)
    channel = SimpleNamespace(
        list_message_folder=AsyncMock(return_value=[{"file_key": "child", "name": "child.txt"}]),
        download_resource_to_file=AsyncMock(side_effect=RuntimeError("child download failed")),
        send=AsyncMock(return_value=SendResult(success=True)),
    )
    bridge = _bridge_with_channel(channel)
    bridge.consume = AsyncMock()
    content = FolderContent(file_key="root", file_name="教程")
    if as_post:
        content = PostContent(post={
            "content_v2": [[{"tag": "text", "text": "嗨嗨"}]],
            "files": [{"file_key": "root", "file_name": "教程", "is_folder": True}],
        })

    await bridge._on_message(_inbound_message(content))

    channel.send.assert_awaited_once_with(
        to="oc_chat",
        message={"text": "读取飞书文件夹失败，请将文件夹压缩为 ZIP 后重新发送。"},
        opts={"reply_to": "om_message"},
    )
    if as_post:
        assert bridge.consume.await_args.args[0][0].content == [TextData(text="嗨嗨")]
    else:
        bridge.consume.assert_not_awaited()
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_card_action_uses_typed_form_value(monkeypatch: pytest.MonkeyPatch):
    notify_question = AsyncMock()
    monkeypatch.setattr(feishu_questions, "notify_question", notify_question)
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {
        "type": "form", "fields": [
            {"type": "input", "name": "answer", "label": "回答", "default_value": "initial"},
            {"type": "select", "name": "choice", "label": "单选", "options": [{"label": "A", "value": "a"}]},
            {"type": "multi_select", "name": "tags", "label": "多选", "options": [{"label": "B", "value": "b"}]},
            {"type": "date", "name": "day", "label": "日期"},
            {"type": "datetime", "name": "when", "label": "时间"},
        ],
    })
    answers = {
        "answer": "typed", "choice": "a", "tags": ["b"],
        "day": "2026-11-12 +0800", "when": "2026-11-12 09:30 +0800",
    }
    event = CardActionEvent(
        message_id="om_message",
        chat_id="oc_chat",
        operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(form_value=answers),
        raw={"event": {"action": {"form_value": {"answer": "raw"}}}},
    )

    await bridge.questions.handle_action(event)

    notify_question.assert_awaited_once_with(
        "E1001",
        answers,
        "http://backend.test",
        "feishu",
        session_id=DEFAULT_SESSION_ID,
    )
    bridge.inbound.get_user_identity.assert_awaited_once_with("ou_sender")
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(question_card.context, disabled=True, answers=answers),
    )
    elements = update_card.await_args.args[1]["body"]["elements"][0]["elements"]
    fields = {element["name"]: element for element in elements if "name" in element}
    assert all(field["disabled"] for field in fields.values())
    assert fields["answer"]["default_value"] == "typed"
    assert fields["choice"]["initial_option"] == "a"
    assert fields["tags"]["selected_values"] == ["b"]
    assert fields["day"]["initial_date"] == "2026-11-12"
    assert fields["when"]["initial_datetime"] == "2026-11-12 09:30"
    submit = elements[-1]["columns"][0]["elements"][0]
    assert submit["disabled"] is True
    assert submit["text"]["content"] == "提交"


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["confirm", "cancel"])
async def test_confirmation_button_callback_submits_decision(monkeypatch, decision):
    notify_question = AsyncMock()
    monkeypatch.setattr(feishu_questions, "notify_question", notify_question)
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {"type": "confirm", "description": "继续执行？"})
    event = CardActionEvent(
        message_id="om_message",
        chat_id="oc_chat",
        operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"type": "confirm", "decision": decision, "tool_call_id": None}),
    )
    await bridge.questions.handle_action(event)
    notify_question.assert_awaited_once_with(
        "E1001", {"decision": decision}, "http://backend.test", "feishu", session_id=DEFAULT_SESSION_ID,
    )
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(question_card.context, disabled=True),
    )
    card = update_card.await_args.args[1]
    assert card["body"]["elements"][0]["text"]["content"] == "继续执行？"
    assert all(column["elements"][0]["disabled"] for column in card["body"]["elements"][-1]["columns"])
    await bridge.questions.handle_action(event)
    assert notify_question.await_count == 1


@pytest.mark.parametrize("state", ["pending", "submitted", "finished"])
def test_execution_confirmation_uses_custom_copy_and_folds_function_and_arguments(state):
    context = build_execution_question(
        "shell", {"command": "rm -rf test", "timeout_ms": 10000},
        "先确认 test 文件夹存在，然后删除并验证。", tool_call_id="delete-test",
        title="删除 test 文件夹",
    )
    result = {"success": True, "result": "已删除"} if state == "finished" else None
    card = build_feishu_question_card(context, disabled=state == "submitted", tool_result=result)
    assert card["header"]["title"]["content"] == "删除 test 文件夹"
    assert card["header"]["template"] == ("green" if state == "finished" else "blue")
    elements = card["body"]["elements"]
    assert elements[0]["text"]["content"] == "先确认 test 文件夹存在，然后删除并验证。"
    details = elements[1]
    assert details["tag"] == "collapsible_panel" and details["expanded"] is False
    assert details["header"]["title"]["content"] == "> 工具与参数"
    assert details["header"]["expanded_title"]["content"] == "∨ 工具与参数"
    assert "shell" in details["elements"][0]["content"]
    assert "rm -rf test" in details["elements"][0]["content"]
    buttons = [column["elements"][0] for column in elements[-1]["columns"]]
    assert [button["text"]["content"] for button in buttons] == ["确认", "取消"]
    assert all(button["disabled"] == (state != "pending") for button in buttons)
    assert ("请在 300 秒内确认" in json.dumps(elements, ensure_ascii=False)) == (state == "pending")
    if result:
        assert elements[-2]["header"]["title"]["content"] == "> 结果"
        assert elements[-2]["header"]["expanded_title"]["content"] == "∨ 结果"


@pytest.mark.parametrize("description", [None, "执行参数：\n这只是说明，不是参数 JSON"])
def test_structured_tool_details_do_not_depend_on_description_format(description):
    card = build_feishu_question_card(json.dumps({
        "type": "confirm", "tool_name": "shell", "description": description, "arguments": {},
    }))
    elements = card["body"]["elements"]
    if description:
        assert elements.pop(0)["text"]["content"] == description
    details = elements[0]
    assert details["tag"] == "collapsible_panel" and details["expanded"] is False
    assert "shell" in details["elements"][0]["content"]
    assert "{}" in details["elements"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [SendResult(success=False), RuntimeError("Feishu unavailable")])
async def test_card_update_failure_does_not_block_answer_submission(monkeypatch, failure):
    notify = AsyncMock()
    monkeypatch.setattr(feishu_questions, "notify_question", notify)

    async def update(*_args, **_kwargs):
        notify.assert_awaited_once()
        if isinstance(failure, Exception):
            raise failure
        return failure

    update_card = AsyncMock(side_effect=update)
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    card = _remember_question(bridge, "om_message", {"type": "confirm"})
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"decision": "confirm"}),
    )

    await bridge.questions.handle_action(event)
    await bridge.questions.handle_action(event)

    notify.assert_awaited_once_with(
        "E1001", {"decision": "confirm"}, bridge.chat_url, "feishu", session_id=DEFAULT_SESSION_ID,
    )
    assert card.disabled is True
    assert update_card.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_answer_does_not_disable_card_and_allows_retry(monkeypatch, confirmation, error_type):
    update_card = AsyncMock(return_value=SendResult(success=True))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question = {"type": "confirm"} if confirmation else {
        "type": "form", "fields": [{"type": "input", "name": "answer", "label": "回答"}],
    }
    card = _remember_question(bridge, "om_message", question)
    answers = {"decision": "confirm"} if confirmation else {"answer": "保留我的回答"}
    error = error_type("submission interrupted")
    notify = AsyncMock(side_effect=[error, None])
    monkeypatch.setattr(feishu_questions, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(form_value=answers),
    )

    with pytest.raises(error_type) as exc:
        await bridge.questions.handle_action(event)
    assert exc.value is error
    assert card.disabled is False
    assert card.submitting is False
    update_card.assert_not_awaited()

    await bridge.questions.handle_action(event)
    await bridge.questions.handle_action(event)
    assert card.disabled is True
    assert notify.await_count == 2
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(card.context, disabled=True, answers=answers),
    )


@pytest.mark.asyncio
async def test_pending_submission_ignores_duplicate_clicks_without_disabling_card(monkeypatch):
    update_card = AsyncMock(return_value=SendResult(success=True))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    card = _remember_question(bridge, "om_message", {"type": "confirm"})
    started = asyncio.Event()
    release = asyncio.Event()

    async def submit(*_args, **_kwargs):
        update_card.assert_not_awaited()
        started.set()
        await release.wait()

    notify = AsyncMock(side_effect=submit)
    monkeypatch.setattr(feishu_questions, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"decision": "confirm"}),
    )
    callback = asyncio.create_task(bridge.questions.handle_action(event))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert card.submitting is True
        assert card.disabled is False
        await asyncio.wait_for(bridge.questions.handle_action(event), timeout=1)
        notify.assert_awaited_once()
        update_card.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(callback, timeout=1)
    assert card.submitting is False
    assert card.disabled is True
    update_card.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_success", [False, True])
async def test_failed_submission_does_not_change_a_finished_confirmation(monkeypatch, finish_success):
    update_card = AsyncMock(return_value=SendResult(success=finish_success))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    card = _remember_question(bridge, "om_message", {"type": "confirm", "tool_call_id": "call-1"})
    tool_result = {"success": True, "result": "已执行"}

    async def notify(*_args, **_kwargs):
        await bridge.questions.finish_confirmation(card, tool_result)
        raise RuntimeError("submission response lost")

    monkeypatch.setattr(feishu_questions, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"decision": "confirm"}),
    )

    with pytest.raises(RuntimeError):
        await bridge.questions.handle_action(event)
    assert card.disabled is True
    assert card.submitting is False
    assert update_card.await_count == 1
    assert update_card.await_args.args == (
        "om_message", build_feishu_question_card(card.context, tool_result=tool_result),
    )


@pytest.mark.parametrize("event_type", [ChatEventType.ASSISTANT, ChatEventType.COMMAND, ChatEventType.TOKEN_USAGE])
@pytest.mark.asyncio
async def test_channel_send_reports_delivery_failure(event_type):
    send = AsyncMock(return_value=SendResult(success=False))
    bridge = _bridge_with_channel(SimpleNamespace(send=send))

    with pytest.raises(RuntimeError, match="Feishu send message"):
        await bridge.delivery.channel_send(
            to="oc_chat",
            message={"card": {}},
            opts=None,
            user_id="feishu_oc_chat",
            sender_name="Alice",
            message_type=event_type,
            iter_content=None,
        )


@pytest.mark.asyncio
async def test_channel_send_creates_independent_messages_for_the_same_user():
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_sent"))
    update_card = AsyncMock()
    bridge = _bridge_with_channel(SimpleNamespace(send=send, update_card=update_card))

    messages = [{"card": {"body": {"elements": [{"tag": "markdown", "content": text}]}}}
                for text in ("first", "second")]
    for message in messages:
        await bridge.delivery.channel_send(
            to="oc_chat",
            message=message,
            opts=None,
            user_id="E1001",
            sender_name="Alice",
            message_type=ChatEventType.COMMAND,
            iter_content=None,
        )

    assert [call.kwargs["message"] for call in send.await_args_list] == messages
    update_card.assert_not_awaited()


def test_thinking_card_wraps_content_in_collapsible_panel():
    async def chunks():
        yield "thinking"

    content = chunks()
    bridge = SimpleNamespace(formatter=FeishuEventFormatter())

    message, iter_content = bridge.formatter._format_chat_event(ChatEvent(event=ChatEventType.THINKING, context=content))

    assert iter_content is content
    panel = message["card"]["body"]["elements"][0]
    assert panel == {
        "tag": "collapsible_panel",
        "element_id": "panel",
        "expanded": False,
        "header": {
            "title": {"tag": "plain_text", "content": "> 查看思考过程"},
            "expanded_title": {"tag": "plain_text", "content": "∨ 查看思考过程"},
        },
        "elements": [
            {
                "tag": "markdown",
                "margin": "0px 0px 0px 0px",
                "content": "思考中...",
                "text_size": "notation",
                "text_align": "left",
                "icon": {"tag": "standard_icon", "token": "tab-more_outlined", "color": "green"},
                "element_id": "main",
            }
        ],
    }


@pytest.mark.asyncio
async def test_streaming_thinking_uses_custom_card(monkeypatch: pytest.MonkeyPatch):
    async def chunks():
        yield "thinking"

    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    card = {"schema": "2.0", "body": {"elements": [{"tag": "markdown", "element_id": "main"}]}}
    create_card_instance = AsyncMock(return_value="card_thinking")
    send_card_by_reference = AsyncMock(return_value=SendResult(success=True, message_id="om_stream"))
    update_card_element_content = AsyncMock()
    finish_streaming_card = AsyncMock()
    update_card = AsyncMock()
    bridge = _bridge_with_channel(
        SimpleNamespace(
            create_card_instance=create_card_instance,
            send_card_by_reference=send_card_by_reference,
            update_card_element_content=update_card_element_content,
            finish_streaming_card=finish_streaming_card,
            update_card=update_card,
        )
    )
    await bridge.delivery.channel_send(
        to="oc_chat",
        message={"card": card},
        opts=None,
        user_id="feishu_oc_chat",
        sender_name="Alice",
        message_type=ChatEventType.THINKING,
        iter_content=chunks(),
    )

    create_card_instance.assert_awaited_once_with(card)
    send_card_by_reference.assert_awaited_once_with("oc_chat", "card_thinking")
    update_card_element_content.assert_awaited_once_with(
        "card_thinking",
        "main",
        "thinking",
        sequence=1,
    )
    finish_streaming_card.assert_awaited_once_with("card_thinking", sequence=2)
    update_card.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", [ChatEventType.THINKING, ChatEventType.ASSISTANT])
@pytest.mark.parametrize("interval", [0, 60])
async def test_streaming_cards_escape_split_markup_in_updates_and_final_flush(monkeypatch, event_type, interval):
    tokens = ['**结果**\n[链接](https://example.test)\n<', 'rss>甲 &', 'amp; 乙</rss>']

    async def chunks():
        for token in tokens:
            yield token

    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", interval)
    monkeypatch.setattr(feishu_delivery.asyncio, "sleep", AsyncMock())
    channel = SimpleNamespace(
        create_card_instance=AsyncMock(return_value="stream-card"),
        send_card_by_reference=AsyncMock(return_value=SendResult(success=True, message_id="stream-message")),
        update_card_element_content=AsyncMock(),
        finish_streaming_card=AsyncMock(),
    )
    bridge = _bridge_with_channel(channel)
    message, iter_content = bridge.formatter._format_chat_event(ChatEvent(event=event_type, context=chunks()))
    await bridge.delivery.send("oc_chat", message, iter_content)

    updates = [call.args[2] for call in channel.update_card_element_content.await_args_list]
    assert len(updates) == (len(tokens) if interval == 0 else 1)
    assert all("<" not in content and ">" not in content for content in updates)
    assert unescape(updates[-1]) == "".join(tokens)
    assert "**结果**" in updates[-1] and "[链接](https://example.test)" in updates[-1]
    if interval == 0:
        assert [unescape(content) for content in updates] == ["".join(tokens[:i]) for i in range(1, 4)]
    channel.finish_streaming_card.assert_awaited_once_with("stream-card", sequence=len(updates) + 1)


@pytest.mark.asyncio
async def test_consume_sends_one_sdk_batch_as_one_prompt(monkeypatch):
    messages = [
        Message(
            message_id=f"om_message_{index}",
            chat_id="oc_chat",
            user_id="E1001",
            sender_name="Alice",
            content=[TextData(text=text)],
            timestamp=0,
            session_metadata={},
        )
        for index, text in enumerate(("first", "second"), start=1)
    ]
    requests = []

    async def chat_events(request_messages, chat_url, *, session_id):
        requests.append((request_messages, chat_url, session_id))
        if False:
            yield

    monkeypatch.setattr(feishu_conversation, "iter_chat_events", chat_events)
    channel = SimpleNamespace(add_reaction=AsyncMock())
    bridge = _bridge_with_channel(channel)

    await bridge.consume(messages)

    assert requests == [
        (messages, "http://backend.test/chat", DEFAULT_SESSION_ID)
    ]
    assert [call.args[0] for call in channel.add_reaction.await_args_list] == [
        "om_message_1",
        "om_message_2",
    ]


@pytest.mark.asyncio
async def test_consume_streams_assistant_chunks_as_reply(monkeypatch: pytest.MonkeyPatch):
    async def chunks():
        for chunk in ["hello", " ", "world"]:
            yield chunk

    async def streaming_chat_events(*_args, **_kwargs):
        yield ChatEvent(event=ChatEventType.ASSISTANT, context=chunks())

    monkeypatch.setattr(feishu_conversation, "iter_chat_events", streaming_chat_events)
    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)
    create_card_instance = AsyncMock(return_value="card_assistant")
    send_card_by_reference = AsyncMock(return_value=SendResult(success=True, message_id="om_stream"))
    update_card_element_content = AsyncMock()
    finish_streaming_card = AsyncMock()
    channel = SimpleNamespace(
        add_reaction=AsyncMock(),
        create_card_instance=create_card_instance,
        send_card_by_reference=send_card_by_reference,
        update_card_element_content=update_card_element_content,
        finish_streaming_card=finish_streaming_card,
    )
    bridge = _bridge_with_channel(channel)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    await bridge.consume([message])

    created_card = create_card_instance.await_args.args[0]
    assert created_card["body"]["elements"][0]["icon"] == {
        "tag": "standard_icon",
        "token": "robot_filled",
        "color": "red",
    }
    send_card_by_reference.assert_awaited_once_with(
        "oc_chat",
        "card_assistant",
        reply_to="om_message",
    )
    assert [await_call.args[2] for await_call in update_card_element_content.await_args_list] == [
        "hello",
        "hello ",
        "hello world",
    ]
    finish_streaming_card.assert_awaited_once_with("card_assistant", sequence=4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (409, "Session 'E1001' already has an active chat request"),
        (500, "Backend unavailable"),
    ],
)
async def test_consume_returns_backend_http_error_to_user(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    detail: str,
):
    request = httpx.Request("POST", "http://backend.test/chat")
    response = httpx.Response(status_code, request=request, json={"detail": detail})

    async def busy_chat_events(*_args, **_kwargs):
        raise httpx.HTTPStatusError("Conflict", request=request, response=response)
        yield

    monkeypatch.setattr(feishu_conversation, "iter_chat_events", busy_chat_events)
    send = AsyncMock(return_value=SendResult(success=True, message_id="om_busy"))
    channel = SimpleNamespace(add_reaction=AsyncMock(), send=send)
    bridge = _bridge_with_channel(channel)
    message = Message(
        message_id="om_message",
        chat_id="oc_chat",
        user_id="E1001",
        sender_name="Alice",
        content=[TextData(text="hello")],
        timestamp=0,
        session_metadata={},
    )

    await bridge.consume([message])

    send.assert_awaited_once()
    assert send.await_args.kwargs["to"] == "oc_chat"
    assert send.await_args.kwargs["opts"] == {"reply_to": "om_message"}
    sent_message = send.await_args.kwargs["message"]
    assert isinstance(sent_message, dict)
    error_element = sent_message["card"]["body"]["elements"][0]
    assert error_element["content"] == f"**[错误]** {status_code}: {detail}"
    assert error_element["icon"] == {
        "tag": "standard_icon",
        "token": "warning_outlined",
        "color": "red",
    }
    assert "streaming_mode" not in sent_message["card"].get("config", {})


@pytest.mark.asyncio
async def test_execution_result_cannot_be_overwritten_by_submitted_confirmation_state(monkeypatch):
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {"type": "confirm", "tool_call_id": "call-1"})
    tool_result = {"success": False, "result": "执行出错"}
    final_card = build_feishu_question_card(question_card.context, tool_result=tool_result)

    async def notify(*_args, **_kwargs):
        update_card.assert_not_awaited()
        await bridge.questions.finish_confirmation(question_card, tool_result)

    monkeypatch.setattr(feishu_questions, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={
            "type": "confirm", "decision": "confirm", "tool_call_id": "call-1",
        }),
    )
    await bridge.questions.handle_action(event)
    update_card.assert_awaited_once_with("om_message", final_card)


@pytest.mark.asyncio
async def test_confirmation_timeout_before_identity_lookup_finishes_keeps_final_card(monkeypatch):
    update_card = AsyncMock(return_value=SendResult(success=True, message_id="om_message"))
    bridge = _bridge_with_channel(SimpleNamespace(update_card=update_card))
    question_card = _remember_question(bridge, "om_message", {"type": "confirm", "tool_call_id": "call-1"})
    identity_started = asyncio.Event()
    release_identity = asyncio.Event()

    async def identity(_open_id):
        identity_started.set()
        await release_identity.wait()
        return "E1001", None

    bridge.inbound.get_user_identity = identity
    notify = AsyncMock()
    monkeypatch.setattr(feishu_questions, "notify_question", notify)
    event = CardActionEvent(
        message_id="om_message", chat_id="oc_chat", operator=EventOperator(open_id="ou_sender"),
        action=CardActionPayload(tag="button", value={"type": "confirm", "decision": "confirm"}),
    )
    callback = asyncio.create_task(bridge.questions.handle_action(event))
    try:
        await asyncio.wait_for(identity_started.wait(), timeout=1)
        tool_result = {"success": False, "result": "用户确认超时，工具未执行。"}
        await bridge.questions.finish_confirmation(question_card, tool_result)
    finally:
        release_identity.set()
        await asyncio.wait_for(callback, timeout=1)
    update_card.assert_awaited_once_with(
        "om_message", build_feishu_question_card(question_card.context, tool_result=tool_result),
    )
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_lifecycle_disconnects_on_cancellation():
    started = asyncio.Event()
    async def connect():
        started.set()
        await asyncio.Event().wait()
    bridge = _bridge_with_channel(SimpleNamespace(connect=connect, disconnect=AsyncMock()))
    task = asyncio.create_task(bridge.start())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    bridge.channel.disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, {"stopped": True}, "已停止当前会话。"),
        (200, {"stopped": False}, "当前没有正在运行的会话。"),
        (503, {"detail": "unavailable"}, "停止失败，请稍后重试。"),
        (200, {}, "停止失败，请稍后重试。"),
    ],
)
async def test_stop_is_a_direct_channel_api_call_without_parsing_an_agent_message(monkeypatch, status, payload, expected):
    channel = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True, message_id="reply")))
    bridge = _bridge_with_channel(channel)
    bridge.inbound.parse_message = AsyncMock()
    bridge.consume = AsyncMock()
    request = httpx.Request("POST", "http://backend.test/chat/stop")
    stop_chat = AsyncMock(return_value=httpx.Response(status, request=request, json=payload))
    monkeypatch.setattr(feishu_bridge, "stop_chat", stop_chat)

    await bridge._on_message(_inbound_message(TextContent(text="  /stop\n")))

    stop_chat.assert_awaited_once_with("E1001", bridge.chat_url, "feishu", session_id=DEFAULT_SESSION_ID)
    bridge.inbound.parse_message.assert_not_awaited()
    bridge.consume.assert_not_awaited()
    command_message, _ = bridge.formatter._format_chat_event(
        ChatEvent(event=ChatEventType.COMMAND, context=expected),
    )
    channel.send.assert_awaited_once_with(
        to="oc_chat", message=command_message, opts={"reply_to": "om_message"},
    )
    element = channel.send.await_args.kwargs["message"]["card"]["body"]["elements"][0]
    assert element["content"] == f"**[命令]** {expected}"
    assert element["icon"]["token"] == "command_outlined"


@pytest.mark.asyncio
async def test_stop_identity_lookup_failure_never_calls_the_backend(monkeypatch):
    channel = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True, message_id="reply")))
    bridge = _bridge_with_channel(channel)
    bridge.inbound.get_user_identity.side_effect = RuntimeError("identity unavailable")
    stop_chat = AsyncMock()
    monkeypatch.setattr(feishu_bridge, "stop_chat", stop_chat)
    await bridge._on_message(_inbound_message(TextContent(text="/stop")))
    stop_chat.assert_not_awaited()
    element = channel.send.await_args.kwargs["message"]["card"]["body"]["elements"][0]
    assert element["content"] == "**[命令]** 停止失败，请稍后重试。"


@pytest.mark.asyncio
async def test_stop_requires_an_exact_standalone_command(monkeypatch):
    bridge = _bridge_with_channel(SimpleNamespace())
    bridge.consume = AsyncMock()
    stop_chat = AsyncMock()
    monkeypatch.setattr(feishu_bridge, "stop_chat", stop_chat)
    await bridge._on_message(_inbound_message(TextContent(text="请解释 /stop")))
    stop_chat.assert_not_awaited()
    assert bridge.consume.await_args.args[0][0].content == [TextData(text="请解释 /stop")]


@pytest.mark.asyncio
async def test_same_user_commands_and_streams_send_without_waiting_for_an_active_stream(monkeypatch):
    channel = SimpleNamespace(
        add_reaction=AsyncMock(),
        send=AsyncMock(return_value=SendResult(success=True, message_id="om_command")),
        create_card_instance=AsyncMock(side_effect=["card_first", "card_later"]),
        send_card_by_reference=AsyncMock(return_value=SendResult(success=True, message_id="om_stream")),
        update_card_element_content=AsyncMock(),
        finish_streaming_card=AsyncMock(),
    )
    bridge = _bridge_with_channel(channel)
    started = asyncio.Event()
    release = asyncio.Event()
    bridge.questions.send_question = AsyncMock(side_effect=lambda _to, context:
        _remember_question(bridge, "question", json.loads(context)))

    async def chunks(name):
        if name == "first":
            started.set()
            await release.wait()
        yield f"{name} reply"

    async def events(messages, *_args, **_kwargs):
        name = messages[0].content[0].text
        if name == "first":
            yield ChatEvent(event=ChatEventType.QUESTION, context='{"fields":[]}')
        if name == "queued":
            yield ChatEvent(event=ChatEventType.COMMAND, context="这条消息将用于引导后续回复：「queued」")
        else:
            yield ChatEvent(event=ChatEventType.ASSISTANT, context=chunks(name))

    monkeypatch.setattr(feishu_conversation, "iter_chat_events", events)
    monkeypatch.setattr(feishu_delivery, "FEISHU_STREAM_UPDATE_INTERVAL_SECONDS", 0)

    def message(text):
        return Message(message_id=text, chat_id="chat", user_id="E1001", sender_name="Alice",
                       content=[TextData(text=text)], timestamp=0, session_metadata={})

    first = asyncio.create_task(bridge.consume([message("first")]))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.wait_for(bridge.consume([message("queued")]), timeout=1)
        element = channel.send.await_args.kwargs["message"]["card"]["body"]["elements"][0]
        assert element["content"] == "**[命令]** 这条消息将用于引导后续回复：「queued」"
        assert not first.done()
        assert "question" in bridge.questions.question_cards

        await asyncio.wait_for(bridge.consume([message("later")]), timeout=1)
        assert not first.done()
        assert "question" in bridge.questions.question_cards
        channel.update_card_element_content.assert_awaited_once_with(
            "card_later", "main", "later reply", sequence=1,
        )
        channel.finish_streaming_card.assert_awaited_once_with("card_later", sequence=2)
    finally:
        release.set()
        await asyncio.wait_for(first, timeout=1)

    assert [call.args for call in channel.send_card_by_reference.await_args_list] == [
        ("chat", "card_first"), ("chat", "card_later"),
    ]
    assert [call.kwargs for call in channel.send_card_by_reference.await_args_list] == [
        {"reply_to": "first"}, {"reply_to": "later"},
    ]
    assert [call.args for call in channel.update_card_element_content.await_args_list] == [
        ("card_later", "main", "later reply"), ("card_first", "main", "first reply"),
    ]
    assert [call.kwargs["sequence"] for call in channel.update_card_element_content.await_args_list] == [1, 1]
    assert [call.args for call in channel.finish_streaming_card.await_args_list] == [
        ("card_later",), ("card_first",),
    ]
    assert [call.kwargs["sequence"] for call in channel.finish_streaming_card.await_args_list] == [2, 2]
    assert "question" not in bridge.questions.question_cards
