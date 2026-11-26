from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import anyio

from ....utils.helper import ensure_finish_async
from .channel import XpeechFeishuChannel


def _path_component(name: str) -> str:
    if (
        not isinstance(name, str)
        or name.rstrip(" .") == ""
        or any(char in name for char in "/\\:\x00")
    ):
        raise ValueError(f"Invalid Feishu folder entry name: {name!r}")
    return name


async def archive_message_folder(
    channel: XpeechFeishuChannel,
    message_id: str,
    file_key: str | None,
    folder_name: str | None,
    directory: Path,
) -> Path:
    """Build a ZIP in a request-owned directory from one Feishu message folder."""
    if not message_id or not isinstance(file_key, str) or not file_key:
        raise ValueError("Missing Feishu message or folder key")

    root = PurePosixPath(_path_component(folder_name or "folder"))
    archive_path = directory / f"{root.name}.zip"
    with ZipFile(archive_path, mode="x", compression=ZIP_DEFLATED) as archive:
        pending = [(file_key, root, frozenset())]
        while pending:
            key, path, ancestors = pending.pop()
            if key in ancestors:
                raise ValueError("Circular Feishu folder reference")
            archive.mkdir(str(path))
            names: set[str] = set()
            for item in await channel.list_message_folder(message_id, key):
                name = _path_component(item["name"])
                child_key = item["file_key"]
                if not isinstance(child_key, str) or not child_key or name in names:
                    raise ValueError("Invalid or duplicate Feishu folder entry")
                names.add(name)
                child_path = path / name
                if item.get("is_folder") is True:
                    pending.append((child_key, child_path, ancestors | {key}))
                    continue

                downloaded = await channel.download_resource_to_file(
                    child_key,
                    resource_type="file",
                    message_id=message_id,
                    dest_dir=directory,
                    file_name="resource",
                )
                await ensure_finish_async(
                    lambda: anyio.to_thread.run_sync(archive.write, downloaded, str(child_path)),
                )
                downloaded.unlink()

    return archive_path
