from datetime import datetime
from typing import Annotated, TypeAlias
from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class InputText(BaseModel):
    """文本输入内容块。"""

    text: Annotated[str, Field(description="发送给模型的文本内容")]


class InputImage(BaseModel):
    """图片输入内容块。"""

    image_url: Annotated[str, Field(description="base64 格式图片")]


InputContent: TypeAlias = InputText | InputImage


class InboundMessage(BaseModel):
    """收到的消息 schema。"""

    session_id: Annotated[str, Field(description="会话的ID")]
    session_metadata: Annotated[dict[str, str], Field(default_factory=dict, description="会话的元数据")]
    content: Annotated[list[InputContent], Field(description="消息内容列表")]
    files: Annotated[list[UploadFile], File(description="消息附件列表")]
    timestamp: Annotated[datetime, Field(default_factory=datetime.now, description="消息的时间戳")]


class OutputText(BaseModel):
    """文本输出内容块。"""

    text: Annotated[str, Field(description="模型生成的文本内容")]


class OutputImage(BaseModel):
    """图片输出内容块。"""

    image_url: Annotated[str, Field(description="base64 格式图片")]


class OutputFile(BaseModel):
    """文件输出内容块。"""

    file_name: Annotated[str, Field(description="文件名")]
    file_content: Annotated[UploadFile, File(description="base64 格式文件")]


OutputContent: TypeAlias = OutputText | OutputImage | OutputFile


class OutboundMessage(BaseModel):
    """回复的消息迭代器的 schema。"""

    content: Annotated[OutputContent, Field(description="消息内容")]
