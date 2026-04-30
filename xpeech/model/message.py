from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from model.enum import ChannelType, FormType
from datetime import datetime
from pydantic_ai import (
    BinaryContent,
    AudioMediaType,
    ImageMediaType,
    VideoMediaType,
    DocumentMediaType,
    TextContent,
)
from typing import Any


class TextMessage(BaseModel):
    """文本消息。"""

    text: Annotated[TextContent, Field(description='文本内容')]

    @field_validator('text')
    def not_empty(cls, v):
        if not v.content:
            raise ValueError('Text message cannot be empty')
        return v


class ImageMessage(BaseModel):
    """图片消息。"""

    image_file: Annotated[BinaryContent, Field(description='图片文件内容')]

    @field_validator('image_file')
    def not_empty(cls, v):
        media_type: ImageMediaType = v.media_type
        if media_type != 'image/png':
            raise ValueError('Image message must be PNG')
        return v


class AudioMessage(BaseModel):
    """音频消息。"""

    audio_file: Annotated[BinaryContent, Field(description='音频文件内容')]

    @field_validator('audio_file')
    def not_empty(cls, v):
        media_type: AudioMediaType = v.media_type
        if media_type != 'audio/mpeg':
            raise ValueError('Audio message must be MP3')
        return v


class VideoMessage(BaseModel):
    """视频消息。"""

    video_file: Annotated[BinaryContent, Field(description='视频文件内容')]

    @field_validator('video_file')
    def not_empty(cls, v):
        media_type: VideoMediaType = v.media_type
        if media_type != 'video/mp4':
            raise ValueError('Video message must be MP4')
        return v


class DocumentMessage(BaseModel):
    """
    文档消息。
    仅支持 PDF 格式，请在上传前将其他文件类型转换为 PDF。
    """

    document_file: Annotated[BinaryContent, Field(description='文档文件内容')]

    @field_validator('document_file')
    def not_empty(cls, v):
        media_type: DocumentMediaType = v.media_type
        if media_type != 'application/pdf':
            raise ValueError('Document message must be PDF')
        return v


class FileMessage(BaseModel):
    """
    文件消息。
    """

    file_path: Annotated[str, Field(description='文件内容')]


class FormMessage(BaseModel):
    """表单互动。"""

    type_form: Annotated[FormType, Field(description='Form type')]
    title: Annotated[str, Field(description='Form title')]
    option_names: Annotated[list[str], Field(description='Option names')]
    option_values: Annotated[list[str] | None, Field(description='Option values. if input type, leave empty')]


class InboundMessage(BaseModel):
    """消息接收自聊天通道。"""

    channel: ChannelType  # 通道类型
    sender_id: str  # 发送者ID
    chat_id: str  # 通道ID
    content: list[TextMessage | ImageMessage | AudioMessage | VideoMessage | DocumentMessage]  # 消息列表
    file: list[FileMessage]  # 文件列表
    timestamp: Annotated[datetime, Field(default_factory=datetime.now, description='消息时间戳')]  # 消息时间戳
    metadata: Annotated[dict[str, Any], Field(default_factory=dict, description='通道特定数据')]  # 通道特定数据，比如发送者昵称/部门

    @property
    def session_key(self) -> str:
        """根据这个字段进行隔离，包括工作区/会话"""
        return f'{self.channel}:{self.chat_id}:{self.sender_id}'


class OutboundMessage(BaseModel):
    """消息发送至聊天通道。"""

    channel: ChannelType  # 通道类型
    chat_id: str  # 通道ID
    content: list[TextMessage | ImageMessage | VideoMessage | FileMessage]  # 消息列表
    reply_to: Annotated[str | None, Field(description='@对象')] = None  # @对象
    metadata: Annotated[dict[str, Any], Field(default_factory=dict, description='通道特定数据')]  # 通道特定数据
    form: Annotated[list[FormMessage], Field(default_factory=list, description='交互表达')]  # 交互表达

