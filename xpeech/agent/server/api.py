from datetime import datetime
from typing import Annotated, TypeAlias
from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class InputText(BaseModel):
    """文本输入内容块。"""

    text: Annotated[str, Field(description="发送给模型的文本内容")]


class InputImage(BaseModel):
    """图片输入内容块。"""

    image_url: Annotated[str, Field(description="图片 URL，或 data URL 格式的 base64 图片")]


InputContent: TypeAlias = InputText | InputImage


class InboundMessage(BaseModel):
    """收到的消息 schema。"""

    session_id: Annotated[str, Field(description="会话的ID")]
    session_metadata: Annotated[dict[str, str], Field(default_factory=dict, description="会话的元数据")]
    content: Annotated[list[InputContent], Field(description="消息内容列表")]
    files: Annotated[list[UploadFile], File(description="消息附件列表")]
    timestamp: Annotated[datetime, Field(default_factory=datetime.now, description="消息的时间戳")]
