from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
