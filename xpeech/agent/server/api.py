from .server import app
from .schema import OutboundMessage
from fastapi import Depends
from typing import Annotated
from datetime import datetime
from fastapi import File, Form, UploadFile, HTTPException, status
from .schema import InputContent, InboundMessage
import json
from ...config.settings import settings
from ...utils.helper import save_to_workspace, ensure_path
from ...utils.session import create_workspace_templates
from ...provider.litellm_provider import LiteLLMProvider
from ..loop import AgentLoop
from fastapi.responses import StreamingResponse


def session_metadata(
    session_metadata: Annotated[
        str,
        Form(
            description='会话的元数据，JSON 格式字符串，如：{"sender_id": "xxxx","channel": "feishu"}',
        ),
    ] = "{}",
):
    """将会话元数据的 JSON 字符串解析为字典。"""
    try:
        parsed_dict = json.loads(session_metadata)
        return parsed_dict
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


def content(
    content: Annotated[
        str,
        Form(
            description='消息内容列表，JSON 格式字符串。支持文本和图片：[{"text": "你好"}, {"image_url": "data:image/png;base64,iVBOR..."}]',
        ),
    ],
):
    """将消息内容列表的 JSON 字符串解析为 InputContent 列表。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON format")


@app.post("/chat", response_model=OutboundMessage)
async def chat(
    session_id: Annotated[str, Form(description="会话的ID")],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[InputContent, Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile], File(default_factory=list, description="消息附件列表")],
):
    """Receive a message and return a response."""

    workspace = settings.path.workspace_base_path / session_id

    # 创建工作目录
    if not workspace.exists():
        await create_workspace_templates(ensure_path(workspace))

    # 把files都保存到工作目录
    files_ = []
    for file in files:
        file_path = save_to_workspace(file=file, workspace=workspace)
        files_.append(file_path)

    # 创建消息对象
    message = InboundMessage(
        session_id=session_id,
        session_metadata=session_metadata,
        content=content,
        timestamp=timestamp,
        files=files_,
    )

    # 开启Agent Loop
    provider = LiteLLMProvider(
        api_key="1cda8xxx5985dbf",
        api_base="https://ark.cn-beijing.volces.com/api/coding/v1",
        default_model="zai/glm-5.1",
    )
    return StreamingResponse(
        AgentLoop(
            provider=provider,
            workspace=workspace,
            max_iterations=30,
        ).run(message=message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
