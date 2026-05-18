from pathlib import Path
import base64
import math
import mimetypes
import re
from typing import Union
from uuid import uuid4
import aiohttp
import aiofiles
from ..agent.server.schema import InputImage
import inspect
from asyncer import asyncify
import yaml
from starlette.datastructures import UploadFile
from charset_normalizer import from_bytes, from_path
from importlib.util import spec_from_file_location, module_from_spec
import sys
from litellm import token_counter as _token_counter
from io import BytesIO
from PIL import Image
import tempfile
from functools import cache

# PIL.Image.ANTIALIAS is deprecated, use PIL.Image.Resampling.LANCZOS instead
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import VideoFileClip


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


def dynamic_import(path: str, module_name: str | None = None):
    path_obj = Path(path).resolve()
    if path_obj.is_dir():
        init_file = path_obj / "__init__.py"
        if not init_file.exists():
            raise ImportError(f"目录不是 package: {path_obj}")
        name = module_name or path_obj.name
        spec = spec_from_file_location(
            name,
            init_file,
            submodule_search_locations=[str(path_obj)],
        )
    elif path_obj.is_file():
        name = module_name or path_obj.stem
        spec = spec_from_file_location(name, path_obj)
    else:
        raise FileNotFoundError(path_obj)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入: {path_obj}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


async def token_counter(messages: list[dict]):
    cleaned_messages, video_tokens = await _strip_video_blocks_and_count_tokens(messages)
    return _token_counter(model="gpt-4o", messages=cleaned_messages) + video_tokens


def _parse_base64_url(url: str):
    match = re.match(r"data:(?P<mime>[\w/+-]+);base64,(?P<data>.+)", url)
    if not match:
        raise ValueError("Invalid base64 data URI format")
    return match.group("mime"), base64.b64decode(match.group("data"))


async def _strip_video_blocks_and_count_tokens(messages: list[dict]) -> tuple[list[dict], int]:
    cleaned_messages = []
    video_tokens = 0

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            cleaned_messages.append(message)
            continue

        cleaned_content = []
        has_video = False
        for block in content:
            if block["type"] != "video_url":
                cleaned_content.append(block)
                continue
            _, _bytes = _parse_base64_url(block["video_url"]["url"])
            duration = (await read_video_metadata_by_bytes(_bytes))["duration"]
            video_tokens += math.ceil(duration * 100)
            has_video = True

        if has_video:
            cleaned_messages.append({**message, "content": cleaned_content})
        else:
            cleaned_messages.append(message)

    return cleaned_messages, video_tokens


async def super_read_text(file_path: Path = None, file_bytes: bytes = None) -> tuple[str | None, str | None]:
    if not (file_path or file_bytes):
        raise ValueError("Internal error: No file path or bytes provided")
    if file_path:
        content = (await asyncify(from_path)(file_path)).best()
    elif file_bytes:
        content = (await asyncify(from_bytes)(file_bytes)).best()
    if content is not None:
        return content.__str__(), content.encoding
    else:
        return None, None


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
        mime, image_bytes = _parse_base64_url(url=image_url)
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


def compress_image_bytes_to_jpg(
    input_bytes: bytes,
    target_kb: int = 500,
    min_quality: int = 10,
    max_quality: int = 95,
) -> bytes:
    target_bytes = target_kb * 1024
    img = Image.open(BytesIO(input_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    low, high = min_quality, max_quality
    best_data = None
    while low <= high:
        quality = (low + high) // 2
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= target_bytes:
            best_data = data
            low = quality + 1
        else:
            high = quality - 1
    if best_data is None:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=min_quality, optimize=True)
        best_data = buffer.getvalue()
    return best_data


async def compress_video_to_mp4(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float | None = None,
    end_time: float | None = None,
    fps: int | None = None,
    bitrate: str = "2000k",
    max_width: int | None = 1920,
    max_height: int | None = 1080,
) -> Path:
    """
    压缩视频文件并转换为 MP4 格式，支持剪辑和时间段选择。

    Args:
        input_path: 输入视频文件路径
        output_path: 输出视频文件路径
        start_time: 起始时间（秒），如果为 None 则从视频开头开始
        end_time: 结束时间（秒），如果为 None 则到视频结尾
        fps: 目标帧率，如果为 None 则保持原帧率
        bitrate: 视频比特率，默认 "2000k"
        max_width: 最大宽度，如果视频超过此宽度则会缩放，默认 1920
        max_height: 最大高度，如果视频超过此高度则会缩放，默认 1080

    Returns:
        输出视频文件的 Path 对象
    """
    input_path = Path(input_path)
    output_path = Path(output_path).with_suffix(".mp4")
    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    def _process_video():
        # 加载视频
        clip = VideoFileClip(str(input_path))

        # 剪辑时间段
        if start_time is not None or end_time is not None:
            clip = clip.subclip(0 if start_time is None else start_time, end_time)

        # 调整尺寸（如果需要）
        if max_width or max_height:
            original_width, original_height = clip.size
            if max_width and original_width > max_width:
                ratio = max_width / original_width
                new_width = max_width
                new_height = int(original_height * ratio)
                clip = clip.resize((new_width, new_height))
            elif max_height and original_height > max_height:
                ratio = max_height / original_height
                new_height = max_height
                new_width = int(original_width * ratio)
                clip = clip.resize((new_width, new_height))

        # 设置帧率（如果需要）
        output_fps = fps or getattr(clip, "fps", None) or getattr(getattr(clip, "reader", None), "fps", None) or 24
        clip = clip.set_fps(output_fps)

        # 写入输出文件
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            bitrate=bitrate,
            preset="medium",
            fps=output_fps,
            audio_codec="aac" if clip.audio is not None else None,
            audio=clip.audio is not None,
            logger=None,  # 不显示进度日志
        )

        clip.close()

    # 异步执行视频处理
    await asyncify(_process_video)()

    return output_path


@cache
async def read_video_metadata_by_bytes(raw: bytes) -> dict[str, object]:
    def _get(_path):
        clip = VideoFileClip(str(_path))
        width, height = clip.size
        clip.close()
        return {"duration": clip.duration, "width": width, "height": height}

    with tempfile.TemporaryDirectory(prefix="xpeech-video-") as temp_dir:
        output_path = Path(temp_dir) / "video.mp4"
        async with aiofiles.open(output_path, "wb") as out_file:
            await out_file.write(raw)
        return await asyncify(_get)(output_path)


async def read_video_metadata(input_path: Union[str, Path]) -> dict[str, object]:
    """Read basic video metadata without loading the video into the LLM context."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_path}")
    async with aiofiles.open(input_path, "rb") as f:
        raw = await f.read()
    return await read_video_metadata_by_bytes(raw)


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
