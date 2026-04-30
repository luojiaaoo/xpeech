from pathlib import Path


class FilesystemTools:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        """只接受相对路径，解析后必须位于 workspace 内。"""
        if Path(path).is_absolute():
            raise ValueError(f"Only relative paths allowed: {path}")
        resolved = (self.workspace / path).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def read_file(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int | None = 200,
    ) -> str:
        """
        Read a file (text).
        Text output format: LINE_NUM|CONTENT.
        Use offset and limit for large text files.
        Reads exceeding ~128K chars are truncated.
        """
        fp = self._resolve(path)
        if not fp.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        text = fp.read_text(encoding="utf-8").replace("\r\n", "\n")
        lines = text.splitlines()
        total = len(lines)
        start = max(0, offset - 1)
        if start >= total:
            return f"Error: offset {offset} is beyond end of file ({total} lines)"
        end = min(start + limit, total)
        out = "\n".join(f"{i + 1}| {lines[i]}" for i in range(start, end))
        if end < total:
            out += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
        else:
            out += f"\n\n(End of file — {total} lines total)"
        return out

    def write_file(self, path: str, content: str) -> str:
        """
        Write content to a file. Overwrites if the file already exists;
        For partial edits, prefer rearch_replace instead.
        """
        fp = self._resolve(path)
        if not fp.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        fp.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"

    def create_file(self, path: str) -> str:
        """
        Create a new file.
        """
        fp = self._resolve(path)
        if fp.exists():
            raise FileExistsError(f"File already exists: {path}")
        fp.touch()
        return f"Successfully created {path}"

    def delete_file(self, path: str) -> str:
        """
        Delete a file.
        """
        fp = self._resolve(path)
        if not fp.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        fp.unlink()
        return f"Successfully deleted {path}"
    
    def move_file(self, src: str, dst: str) -> str:
        """
        Move or rename a file from src to dst.
        """
        src_fp = self._resolve(src)
        dst_fp = self._resolve(dst)
        if not src_fp.is_file():
            raise FileNotFoundError(f"Source not found: {src}")
        if dst_fp.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        dst_fp.parent.mkdir(parents=True, exist_ok=True)
        src_fp.rename(dst_fp)
        return f"Successfully moved {src} to {dst}"
    
    def copy_file(self, src: str, dst: str) -> str:
        """
        Copy a file from src to dst.
        """
        src_fp = self._resolve(src)
        dst_fp = self._resolve(dst)
        if not src_fp.is_file():
            raise FileNotFoundError(f"Source not found: {src}")
        if dst_fp.exists():
            raise FileExistsError(f"Destination already exists: {dst}")
        dst_fp.parent.mkdir(parents=True, exist_ok=True)
        dst_fp.write_bytes(src_fp.read_bytes())
        return f"Successfully copied {src} to {dst}"

    def search_files(self, pattern: str) -> str:
        """
        Search for files matching a pattern.
        Glob pattern to match (example: '**/*.txt' to find all txt files).
        """
        matches = list(self.workspace.rglob(pattern))
        if not matches:
            return f"No files found matching: {pattern}"
        return "\n".join(str(m.relative_to(self.workspace)) for m in matches)

    def rearch_replace(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> str:
        """
        Edit a file by replacing old_text with new_text.
        If old_text matches multiple times, you must provide more context
        or set replace_all=true. Shows a diff of the closest match on failure.
        """
        fp = self._resolve(path)
        if not fp.is_file():
            raise FileNotFoundError(f"Not a file: {path}")

        text = fp.read_text(encoding="utf-8").replace("\r\n", "\n")
        old = old_text.replace("\r\n", "\n")
        new = new_text.replace("\r\n", "\n")

        count = text.count(old)
        if count == 0:
            raise ValueError(f"old_text not found in {path}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_text appears {count} times in {path}. "
                "Provide more context or set replace_all=True."
            )

        result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        fp.write_text(result, encoding="utf-8")
        return f"Successfully edited {path}"

    def list_dir(
        self, path: str = ".", recursive: bool = False, max_entries: int = 200
    ) -> str:
        """
        List the contents of a directory.
        Set recursive=true to explore nested structure.
        Common noise directories (.git, node_modules, __pycache__, etc.) are auto-ignored.
        """
        fp = self._resolve(path)
        if not fp.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        ignore = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
        }
        items = []
        total = 0

        if recursive:
            for item in sorted(fp.rglob("*")):
                if any(p in ignore for p in item.parts):
                    continue
                total += 1
                if len(items) < max_entries:
                    rel = item.relative_to(fp)
                    items.append(f"{rel}/" if item.is_dir() else str(rel))
        else:
            for item in sorted(fp.iterdir()):
                if item.name in ignore:
                    continue
                total += 1
                if len(items) < max_entries:
                    prefix = "📁 " if item.is_dir() else "📄 "
                    items.append(f"{prefix}{item.name}")

        if not items:
            return f"Directory {path} is empty"

        result = "\n".join(items)
        if total > max_entries:
            result += f"\n\n(truncated, showing first {max_entries} of {total} entries)"
        return result
