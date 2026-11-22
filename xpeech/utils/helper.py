from pathlib import Path
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
import inspect
from asyncer import asyncify
import anyio
import httpx
import yaml
from importlib.util import spec_from_file_location, module_from_spec
import sys
from io import BytesIO
from PIL import Image
from async_lru import alru_cache

# PIL.Image.ANTIALIAS is deprecated, use PIL.Image.Resampling.LANCZOS instead
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import VideoFileClip


def ensure_async(func):
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return asyncify(func)


async def ensure_finish_async[T](operation: Callable[[], Awaitable[T]]) -> T:
    """等待异步操作结束；调用方被取消时，先等待操作完成，再传递取消异常。

    取消异步任务不会停止已运行的线程，因此必须等待线程完成，
    避免提前释放锁或清理仍在使用的临时文件。
    """
    async def run() -> T:
        with anyio.CancelScope(shield=True):
            return await operation()

    task = asyncio.create_task(run())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
        raise


@asynccontextmanager
async def stack_ensure_aclosing() -> AsyncIterator[AsyncExitStack]:
    """取消时仍等待栈内资源关闭和清理完成。"""
    stack = AsyncExitStack()
    try:
        yield stack
    finally:
        await ensure_finish_async(stack.aclose)


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


def format_now(format_string: str = "%Y-%m-%d") -> str:
    return datetime.now().astimezone().strftime(format_string)


async def _vllm_tokenize(payload: dict) -> dict:
    """调用独立 vLLM 服务，不在本地估算或降级。"""
    from ..config.settings import settings

    tokenizer = settings.llm.tokenizer
    async with httpx.AsyncClient(base_url=tokenizer.api_base, timeout=tokenizer.timeout) as client:
        response = await client.post("/tokenize", json=payload)
        response.raise_for_status()
        return response.json()


async def token_counter(messages: list[dict] | None = None, tools: list[dict] | None = None) -> int:
    """由 vLLM 对完整消息、工具和多模态内容执行聊天模板分词。"""
    from ..config.settings import settings

    messages = messages or []
    # Qwen3.8 的聊天模板要求有用户消息，系统提示词和工具分项需要占位。
    if not any(message.get("role") == "user" for message in messages):
        messages = [*messages, {"role": "user", "content": ""}]

    payload = {
        "model": settings.llm.tokenizer.model,
        "messages": messages,
        "add_generation_prompt": True,
        "return_token_strs": False,
    }
    if tools is not None:
        payload["tools"] = tools
    result = await _vllm_tokenize(payload)
    count = result.get("count")
    if type(count) is not int or count < 0:
        raise RuntimeError("vLLM /tokenize returned an invalid token count")
    return count


def compress_image(
    input_bytes: bytes,
    target_kb: int = 300,
    min_quality: int = 10,
    max_quality: int = 95,
) -> tuple[bytes, str]:
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
    return best_data, "image/jpeg"


async def compress_video(
    input_path: Path,
    output_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
    fps: int | None = None,
    bitrate: str = "2000k",
    max_width: int | None = 1920,
    max_height: int | None = 1080,
) -> tuple[bytes, str]:
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
        压缩后的视频字节和 MIME 类型
    """
    from .file_io import LockedFile

    output_path = output_path.with_suffix(".mp4")
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

    return await LockedFile(output_path).read_bytes(), "video/mp4"


@alru_cache(maxsize=1000)
async def read_video_metadata(input_path: Path) -> dict[str, object]:
    """Read basic video metadata without loading the video into the LLM context."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    def _get() -> dict[str, object]:
        clip = VideoFileClip(str(input_path))
        try:
            width, height = clip.size
            return {"duration": clip.duration, "width": width, "height": height}
        finally:
            clip.close()

    return await asyncify(_get)()


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
