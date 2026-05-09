from typing import Annotated, TypeAlias
from pydantic import BaseModel, Field
from datetime import datetime
from fastapi import UploadFile


# ------------------ 输入内容块 ------------------
class InputText(BaseModel):
    """文本输入内容块。"""

    text: Annotated[str, Field(description="发送给模型的文本内容")]


class InputImage(BaseModel):
    """图片输入内容块。"""

    image_url: Annotated[str, Field(description="base64 格式图片")]


class InputContent(BaseModel):
    """输入内容块。"""

    content: Annotated[list[InputText | InputImage], Field(description="消息内容")]


class InboundMessage(InputContent):
    """请求的消息的 schema。"""

    session_id: Annotated[str, Field(description="会话ID")]
    session_metadata: Annotated[dict[str, str], Field(description="会话元数据")]
    timestamp: Annotated[datetime, Field(description="消息时间戳")]
    files: Annotated[list[UploadFile], Field(description="消息附件")]


# ------------------ 输出内容块 ------------------
class OutputText(BaseModel):
    """文本输出内容块。"""

    text: Annotated[str, Field(description="模型生成的文本内容")]


class OutputImage(BaseModel):
    """图片输出内容块。"""

    image_url: Annotated[str, Field(description="base64 格式图片")]


class OutputFile(BaseModel):
    """文件输出内容块。"""

    file_name: Annotated[str, Field(description="文件名")]
    file_content: Annotated[str, Field(description="base64 格式文件")]


OutputContent: TypeAlias = OutputText | OutputImage | OutputFile


class OutboundMessage(BaseModel):
    """回复的消息迭代器的 schema。"""

    content: Annotated[OutputContent, Field(description="消息内容")]
