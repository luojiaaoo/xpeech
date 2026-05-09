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


def ensure_async(func):
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return asyncify(func)


def ensure_path(path_: Path):
    path_.mkdir(parents=True, exist_ok=True)
    return path_


async def save_to_workspace(file: UploadFile | InputImage, workspace: Path, idx: int | None = None):
    if isinstance(file, UploadFile):
        file_path = workspace / Path(file.filename).name
        async with aiofiles.open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
        return file_path
    elif isinstance(file, InputImage):
        return await _save_image_url(file, workspace, str(idx))


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
