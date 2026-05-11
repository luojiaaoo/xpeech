from pathlib import Path
from ..utils.helper import ensure_path
from pydantic import BaseModel, Field
from typing import Annotated
import aiofiles


class MemoryArgs(BaseModel):
    history_entry: Annotated[
        str,
        Field(
            description="A paragraph summarizing key events/decisions/topics. Start with [YYYY-MM-DD HH:MM]. Include detail useful for grep search."
        ),
    ]
    memory_update: Annotated[
        str,
        Field(
            description="Full updated long-term memory as markdown. Include all existing facts plus new ones. Return unchanged if nothing new."
        ),
    ]


class MemoryStore:
    def __init__(self, workspace: Path):
        self.memory_dir = ensure_path(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    async def read_long_term(self) -> str:
        if self.memory_file.exists():
            async with aiofiles.open(self.memory_file, "r", encoding="utf-8") as f:
                return await f.read()
        return ""

    async def write_long_term(self, content: str) -> None:
        async with aiofiles.open(self.memory_file, "w", encoding="utf-8") as f:
            await f.write(content)

    async def append_history(self, entry: str) -> None:
        async with aiofiles.open(self.history_file, "a", encoding="utf-8") as f:
            await f.write(entry.rstrip() + "\n\n")

    async def get_memory_context(self) -> str:
        long_term = await self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    async def save_memory(self, args: MemoryArgs):
        """Save the memory consolidation result to persistent storage."""
        await self.append_history(args.history_entry.strip())
        await self.write_long_term(args.memory_update)
