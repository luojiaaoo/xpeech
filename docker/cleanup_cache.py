import errno
import stat
import tomllib
from contextlib import suppress
from pathlib import Path
from time import time


def clean_cache(cache_dir: Path) -> None:
    cutoff = time() - 24 * 60 * 60
    for directory, _, files in cache_dir.walk(top_down=False):
        remove_folder = False
        if directory.parent == cache_dir and directory.name.startswith("xpeech-feishu-folder-"):
            with suppress(FileNotFoundError):
                remove_folder = directory.lstat().st_mtime < cutoff

        for name in files:
            path = directory / name
            with suppress(FileNotFoundError):
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                    path.unlink()

        if remove_folder:
            try:
                directory.rmdir()
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.ENOTEMPTY):
                    raise


if __name__ == "__main__":
    with Path("conf.toml").open("rb") as config:
        cache_dir = Path(tomllib.load(config)["path"]["cache_path"]).expanduser().resolve()
    clean_cache(cache_dir)
