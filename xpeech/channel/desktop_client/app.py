from __future__ import annotations

import asyncio
import base64
import mimetypes
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from .config import DesktopConfig, load_config, save_api_base_url
from .identity import get_identity
from .xpeech_client import XpeechDesktopClient, serialize_event_for_js


TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "xpeech_desktop_client"


class DesktopApi:
    def __init__(self):
        self._window: Any | None = None
        self.clear_browser_files()

    def _set_window(self, window: Any) -> None:
        self._window = window

    def get_config(self):
        return asdict(load_config())

    def save_api_base_url(self, api_base_url: str):
        url = api_base_url.rstrip("/")
        save_api_base_url(url)
        return asdict(DesktopConfig(api_base_url=url))

    def get_identity(self):
        return asdict(get_identity())

    def save_browser_files(self, files: list[dict[str, str]]):
        TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for file_info in files:
            name = Path(file_info.get("name") or "attachment").name
            data_url = file_info.get("data_url") or ""
            if "," not in data_url:
                continue

            header, encoded = data_url.split(",", 1)
            mime_type = ""
            if header.startswith("data:") and ";" in header:
                mime_type = header[5:].split(";", 1)[0]

            suffix = Path(name).suffix
            if not suffix and mime_type:
                suffix = mimetypes.guess_extension(mime_type) or ""

            stem = Path(name).stem or "attachment"
            save_path = TEMP_UPLOAD_DIR / f"{stem}_{uuid4().hex}{suffix}"
            save_path.write_bytes(base64.b64decode(encoded))
            saved_files.append(self._file_payload(save_path))

        return saved_files

    def clear_browser_files(self) -> None:
        shutil.rmtree(TEMP_UPLOAD_DIR, ignore_errors=True)

    def send_message(self, content: str, files: list[dict[str, str]]):
        asyncio.run(self._send_message(content, files))
        return {"message": "completed"}

    def answer_question(self, answer: Any):
        config = load_config()
        identity = get_identity()
        client = XpeechDesktopClient(config.api_base_url)
        asyncio.run(client.answer_question(identity.session_id, answer))
        return {"message": "Answer received"}

    def auto_download_file(self, remote_path: str) -> str | None:
        """Automatically download a remote file to the user's Downloads folder."""
        try:
            downloads_dir = Path.home() / "Downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)

            file_name = Path(remote_path).name
            save_path = downloads_dir / file_name
            # Avoid overwriting existing files
            if save_path.exists():
                stem = save_path.stem
                suffix = save_path.suffix
                counter = 1
                while save_path.exists():
                    save_path = downloads_dir / f"{stem} ({counter}){suffix}"
                    counter += 1

            config = load_config()
            identity = get_identity()
            client = XpeechDesktopClient(config.api_base_url)
            asyncio.run(client.download_file(identity.session_id, remote_path, save_path))
            return str(save_path)
        except Exception:
            return None

    def reveal_file(self, file_path: str) -> bool:
        """Open the system file explorer and highlight the given file."""
        path = Path(file_path)
        if not path.exists():
            return False

        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            # Linux: open the parent directory
            subprocess.Popen(["xdg-open", str(path.parent)])
        return True

    def _file_payload(self, path: Path) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        mime_type, _ = mimetypes.guess_type(resolved.name)
        payload: dict[str, Any] = {
            "path": str(resolved),
            "name": resolved.name,
            "mime_type": mime_type or "application/octet-stream",
            "size": resolved.stat().st_size if resolved.exists() else 0,
        }
        if mime_type and mime_type.startswith("image/") and resolved.exists():
            encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
            payload["data_url"] = f"data:{mime_type};base64,{encoded}"
        return payload

    async def _send_message(self, content: str, files: list[dict[str, str]]) -> None:
        config = load_config()
        identity = get_identity()
        file_paths = [Path(file_info["path"]).expanduser().resolve() for file_info in files]
        client = XpeechDesktopClient(config.api_base_url)
        await client.send_message(
            identity=identity,
            content=content,
            files=file_paths,
            on_event=self._emit_event,
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self._window is None:
            return
        script = f"window.__xpeechDesktopEvent && window.__xpeechDesktopEvent({serialize_event_for_js(event)})"
        self._window.evaluate_js(script)


def run(dev: bool = False) -> None:
    index_html = Path(__file__).with_name("frontend").joinpath("dist", "index.html").resolve()
    url = "http://localhost:5173" if dev else index_html.as_uri()
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError("pywebview is required to run the desktop client. Install dependencies with `uv sync`.") from exc

    api = DesktopApi()
    config = load_config()
    window_kwargs: dict = {
        "js_api": api,
        "width": 1180,
        "height": 760,
        "min_size": (900, 620),
    }
    if config.app_icon and Path(config.app_icon).exists():
        try:
            window_kwargs["icon"] = str(Path(config.app_icon).resolve())
        except Exception:
            pass
    window = webview.create_window(config.app_name, url, **window_kwargs)
    api._set_window(window)
    try:
        webview.start()
    finally:
        api.clear_browser_files()
