"""统一管理文件路径、异步读写、原子保存和进程内路径锁。"""

import asyncio
import stat
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from weakref import WeakKeyDictionary, WeakValueDictionary

import anyio
from asyncer import asyncify
from charset_normalizer import from_bytes
from starlette.datastructures import UploadFile

from .helper import ensure_finish_async, stack_ensure_aclosing


_PATH_LOCKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, WeakValueDictionary[Path, asyncio.Lock]
] = WeakKeyDictionary()


@dataclass
class _Transaction:
    """标记事务有效期，防止子任务沿用已结束事务的持锁状态。"""

    active: bool = True


class LockedFile:
    """单次操作自动加锁；transaction 可将同一把锁提升到多步上下文。"""

    def __init__(self, path: Path):
        """规范化文件路径。"""
        self.path = path.resolve()
        self._transaction_state: ContextVar[_Transaction | None] = ContextVar(
            "locked_file_transaction", default=None,
        )

    @asynccontextmanager
    async def _acquire_lock(self) -> AsyncIterator[None]:
        """获取当前事件循环中该路径唯一的操作锁。"""
        locks = _PATH_LOCKS.setdefault(asyncio.get_running_loop(), WeakValueDictionary())
        lock = locks.get(self.path)
        if lock is None:
            lock = asyncio.Lock()
            locks[self.path] = lock
        async with lock:
            yield

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator["LockedFile"]:
        """持续持有路径锁并返回自身，块内通过同一对象操作时复用锁。"""
        current = self._transaction_state.get()
        if current is not None and current.active:
            yield self
            return
        async with self._acquire_lock():
            transaction = _Transaction()
            token = self._transaction_state.set(transaction)
            try:
                yield self
            finally:
                transaction.active = False
                self._transaction_state.reset(token)

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        """事务内复用已有锁，普通调用仍单独获取路径锁。"""
        current = self._transaction_state.get()
        if current is not None and current.active:
            yield
            return
        async with self._acquire_lock():
            yield

    async def read_text(self, encoding: str | None = "utf-8") -> str:
        """加锁读取文本。"""
        async with self._operation():
            return await ensure_finish_async(lambda: anyio.Path(self.path).read_text(encoding=encoding))

    async def read_bytes(self) -> bytes:
        """加锁读取二进制内容。"""
        async with self._operation():
            return await ensure_finish_async(anyio.Path(self.path).read_bytes)

    async def write_text(self, content: str, encoding: str | None = "utf-8") -> None:
        """加锁原子替换文本文件。"""
        async with self._operation():
            await self._atomic_write(content, encoding)

    async def write_bytes(self, content: bytes) -> None:
        """加锁原子替换二进制文件。"""
        async with self._operation():
            await self._atomic_write(content)

    async def create_text(self, content: str, encoding: str | None = "utf-8") -> bool:
        """加锁创建文本文件；目标已存在时不覆盖。"""
        async with self._operation():
            if self.path.exists():
                return False
            return await self._atomic_write(content, encoding, overwrite=False)

    async def append_text(self, content: str, encoding: str | None = "utf-8") -> None:
        """加锁向文件末尾追加文本。"""
        async def append() -> None:
            async with await anyio.open_file(self.path, "a", encoding=encoding) as stream:
                await stream.write(content)

        async with self._operation():
            await ensure_finish_async(append)

    async def _atomic_write(
        self,
        content: str | bytes,
        encoding: str | None = "utf-8",
        *,
        overwrite: bool = True,
    ) -> bool:
        """写完同文件系统中的临时文件后原子发布，调用方须已持有路径锁。

        overwrite 为 False 时仅在目标不存在时创建，并返回是否创建成功。
        """
        path = anyio.Path(self.path)

        async def write() -> bool:
            async with anyio.TemporaryDirectory(dir=path.parent, prefix=".xpeech-") as directory:
                temporary = anyio.Path(directory) / "content.tmp"
                try:
                    mode = stat.S_IMODE((await path.stat()).st_mode)
                except FileNotFoundError:
                    mode = None
                if isinstance(content, bytes):
                    async with await temporary.open("xb") as stream:
                        await stream.write(content)
                else:
                    async with await temporary.open("x", encoding=encoding) as stream:
                        await stream.write(content)
                if mode is not None:
                    await temporary.chmod(mode)
                if overwrite:
                    await temporary.replace(path)
                else:
                    try:
                        await path.hardlink_to(temporary)
                    except FileExistsError:
                        return False
                return True

        return await ensure_finish_async(write)


def ensure_path(path_: Path) -> Path:
    path_.mkdir(parents=True, exist_ok=True)
    return path_


async def ensure_path_async(path_: Path) -> Path:
    await anyio.Path(path_).mkdir(parents=True, exist_ok=True)
    return path_


def is_relative_path(path_target: Path, base: Path):
    if not base.is_absolute():
        raise ValueError("Base path must be absolute")
    if not path_target.is_absolute():
        raise ValueError("Path target must be absolute")
    try:
        path_target.relative_to(base)
    except ValueError:
        return False
    return True


async def save_to_workspace(file: UploadFile, workspace: Path):
    attachments_dir = workspace / "attachments" / datetime.now().astimezone().strftime("%Y-%m-%d")
    await anyio.Path(attachments_dir).mkdir(parents=True, exist_ok=True)
    original_path = attachments_dir / Path(file.filename).name

    async with stack_ensure_aclosing() as stack:
        directory = await stack.enter_async_context(
            anyio.TemporaryDirectory(dir=attachments_dir, prefix=".xpeech-upload-"),
        )
        temporary = anyio.Path(directory) / "upload.tmp"

        async def publish() -> Path:
            # 用硬链接同时占用文件名并发布完整内容，避免先检查存在、再创建的竞争。
            index = 0
            while True:
                candidate = original_path if index == 0 else original_path.with_name(
                    f"{original_path.stem}_{index}{original_path.suffix}"
                )
                try:
                    await anyio.Path(candidate).hardlink_to(temporary)
                except FileExistsError:
                    index += 1
                else:
                    return candidate

        out_file = await stack.enter_async_context(await temporary.open("xb"))
        while chunk := await file.read(1024 * 1024):
            await ensure_finish_async(lambda: out_file.write(chunk))
        await ensure_finish_async(out_file.aclose)
        return await ensure_finish_async(publish)


async def super_read_text(file_path: Path | None = None, file_bytes: bytes | None = None) -> tuple[str | None, str | None]:
    if not (file_path or file_bytes):
        raise ValueError("Internal error: No file path or bytes provided")
    if file_path:
        file_bytes = await LockedFile(file_path).read_bytes()
    content = (await asyncify(from_bytes)(file_bytes)).best()
    if content is not None:
        return content.__str__(), content.encoding
    else:
        return None, None
