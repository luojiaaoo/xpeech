from pathlib import Path
import base64
import mimetypes
import re
from typing import Union
import aiohttp
import aiofiles
from fastapi import UploadFile
from ..agent.server.schema import InputImage
import inspect
from asyncer import asyncify
import yaml
from charset_normalizer import from_bytes, from_path


def ensure_async(func):
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return asyncify(func)


def ensure_path(path_: Path):
    path_.mkdir(parents=True, exist_ok=True)
    return path_


def is_relative_path(path_target: Path, base: Path):
    path_target = path_target.resolve()
    base = base.resolve()
    try:
        path_target.relative_to(base)
    except ValueError:
        return False
    return True


def format_exception2llm(e: Exception) -> str:
    """给大模型看的异常内容"""
    return f"{type(e).__name__}: {e}"


def msys_to_win(msys_path: str) -> str:
    """将 MSYS2 路径转换为 Windows 路径
    例: /c/Users/Test -> C:/Users/Test
       /d/code        -> D:/code
    """
    # 匹配 /c/ 开头的挂载路径
    match = re.match(r"^/([A-Za-z])/(.*)", msys_path.replace("\\", "/"))
    if match:
        drive = match.group(1).upper()  # 盘符转大写 (Windows惯例)
        rest = match.group(2)  # 斜杠转反斜杠
        return f"{drive}:/{rest}"

    # 如果不符合 MSYS2 挂载格式，直接返回
    return msys_path


async def save_to_workspace(file: UploadFile | InputImage, workspace: Path, idx: int | None = None):
    if isinstance(file, UploadFile):
        file_path = workspace / Path(file.filename).name
        async with aiofiles.open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
        return file_path
    elif isinstance(file, InputImage):
        return await _save_image_url(file, workspace, str(idx))


def detect_image_mime(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes, ignoring file extension."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def super_read_text(file_path: Path = None, file_bytes: bytes = None) -> str | None:
    if not (file_path or file_bytes):
        raise ValueError("Internal error: No file path or bytes provided")
    if file_path:
        return (await asyncify(from_path)(file_path)).best().__str__()
    elif file_bytes:
        return (await asyncify(from_bytes)(file_bytes)).best().__str__()


async def _save_image_url(file: InputImage, output_dir: Union[str, Path], stem: str) -> Path:
    """
    将 image_url（base64 data URI 或 http(s)）保存为本地文件。
    返回最终保存的 Path。
    """
    image_url = file.image_url
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # ═════════════════════════════════════════════════════
    # 情况 A：Base64 Data URI
    # ═════════════════════════════════════════════════════
    if image_url.startswith("data:"):
        match = re.match(r"data:(?P<mime>[\w/+-]+);base64,(?P<data>.+)", image_url)
        if not match:
            raise ValueError("Invalid base64 data URI format")
        mime = match.group("mime")
        image_bytes = base64.b64decode(match.group("data"))
        ext = mimetypes.guess_extension(mime) or ".png"
        filename = f"{stem}{ext}"
        file_path = output_dir / filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(image_bytes)
        return file_path
    # ═════════════════════════════════════════════════════
    # 情况 B：HTTP(S) 外链
    # ═════════════════════════════════════════════════════
    elif image_url.startswith(("http://", "https://")):
        url_path = Path(image_url.split("?")[0].split("#")[0])
        filename = stem + "_" + url_path.name if url_path.suffix else f"{stem}.png"
        file_path = output_dir / filename
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                image_bytes = await resp.read()
                # 如果 URL 里没有扩展名，用 Content-Type 补一个
                if not file_path.suffix:
                    ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
                    ext = mimetypes.guess_extension(ct) or ".png"
                    file_path = file_path.with_suffix(ext)
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(image_bytes)
        return file_path
    else:
        raise ValueError(f"Unsupported image_url scheme: {image_url[:60]}...")


class LiteralDumper(yaml.SafeDumper):
    pass


def literal_str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


LiteralDumper.add_representer(str, literal_str_representer)


# async def save_llm_images(
#     message_content: list[dict],
#     output_dir: Union[str, Path]
# ) -> list[Path]:
#     """
#     从 LLM message.content 列表中提取所有 image_url 并保存。
#     例如 OpenAI / Claude 返回的：
#     [
#       {"type": "text", "text": "..."},
#       {"type": "image_url", "image_url": {"url": "data:image/..."}}
#     ]
#     """
#     saved: list[Path] = []
#     output_dir = Path(output_dir)
#     for part in message_content:
#         if part.get("type") == "image_url":
#             url = part["image_url"]["url"]
#             path = await save_image_url(url, output_dir)
#             saved.append(path)
#     return saved
