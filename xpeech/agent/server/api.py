from .server import app
from .model import OutboundMessage
from fastapi import Depends
from typing import Annotated
from datetime import datetime
from fastapi import File, Form, UploadFile
from .model import session_metadata, content, InputContent


@app.post("/chat", response_model=OutboundMessage)
def chat(
    session_id: Annotated[str, Form(description="会话的ID")],
    session_metadata: Annotated[dict[str, str], Depends(session_metadata)],
    content: Annotated[list[InputContent], Depends(content)],
    timestamp: Annotated[datetime, Form(default_factory=datetime.now, description="消息的时间戳")],
    files: Annotated[list[UploadFile] | None, File(description="消息附件列表")] = None,
):
    """Receive a message and return a response."""
    return OutboundMessage(content=[])
