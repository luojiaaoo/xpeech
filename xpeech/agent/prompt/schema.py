from typing import Any

from pydantic import BaseModel, Field, model_validator


class VideoMetadata(BaseModel):
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class VideoContentBlocks(BaseModel):
    blocks: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_blocks(self):
        if len(self.blocks) != 2:
            raise ValueError("video content blocks must contain exactly two blocks")

        video_block, label_block = self.blocks
        if set(video_block) != {"type", "video_url", "_meta"}:
            raise ValueError("invalid video_url block keys")
        if video_block["type"] != "video_url":
            raise ValueError("first video content block must be video_url")
        if set(video_block["video_url"]) != {"url"} or not isinstance(video_block["video_url"]["url"], str):
            raise ValueError("invalid video_url payload")
        if set(video_block["_meta"]) != {"path"} or not isinstance(video_block["_meta"]["path"], str):
            raise ValueError("invalid video metadata payload")

        if set(label_block) != {"type", "text"}:
            raise ValueError("invalid video label block keys")
        if label_block["type"] != "text":
            raise ValueError("second video content block must be text")
        if not isinstance(label_block["text"], str):
            raise ValueError("video label text must be a string")
        return self
