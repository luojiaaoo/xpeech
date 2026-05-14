from pydantic import BaseModel, Field
from typing import Annotated


def hello():
    """This is a test function"""
    return "hello"


class Message(BaseModel):
    content: Annotated[str, Field(description="The content of the message")]


def echo(message: Message):
    """Echo the message content"""
    return message.content
