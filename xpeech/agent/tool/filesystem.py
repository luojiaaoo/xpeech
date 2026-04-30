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
        读取文本文件内容，按行编号输出。

        Args:
            path: 相对于 workspace 的文件路径。
            offset: 起始行号（从 1 开始），用于分页读取大文件。
            limit: 最多读取的行数，默认 200 行。

        Returns:
            格式为 "行号| 内容" 的文本，末尾附带总行数提示。
            若 offset 超出文件末尾，返回错误提示。

        Raises:
            FileNotFoundError: 路径不是文件或文件不存在。
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
        将内容完整写入已有文件（覆盖写入）。

        Args:
            path: 相对于 workspace 的文件路径，文件必须已存在。
            content: 要写入的文本内容，编码为 UTF-8。

        Returns:
            操作结果提示，包含写入的字符数。

        Raises:
            FileNotFoundError: 文件不存在（需要先 create_file）。
        """
        fp = self._resolve(path)
        if not fp.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        fp.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"

    def create_file(self, path: str) -> str:
        """
        创建一个空文件。若父目录不存在会自动递归创建。

        Args:
            path: 相对于 workspace 的文件路径，文件必须不存在。

        Returns:
            操作结果提示。

        Raises:
            FileExistsError: 文件已存在。
        """
        fp = self._resolve(path)
        if fp.exists():
            raise FileExistsError(f"File already exists: {path}")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        return f"Successfully created {path}"

    def delete_file(self, path: str) -> str:
        """
        删除一个文件。

        Args:
            path: 相对于 workspace 的文件路径，必须是文件（不能是目录）。

        Returns:
            操作结果提示。

        Raises:
            FileNotFoundError: 路径不是文件或文件不存在。
        """
        fp = self._resolve(path)
        if not fp.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        fp.unlink()
        return f"Successfully deleted {path}"
    
    def move_file(self, src: str, dst: str) -> str:
        """
        移动或重命名文件。若目标父目录不存在会自动递归创建。

        Args:
            src: 源文件路径（相对于 workspace）。
            dst: 目标文件路径（相对于 workspace），目标文件必须不存在。

        Returns:
            操作结果提示。

        Raises:
            FileNotFoundError: 源文件不存在。
            FileExistsError: 目标文件已存在。
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
        复制文件。若目标父目录不存在会自动递归创建。

        Args:
            src: 源文件路径（相对于 workspace）。
            dst: 目标文件路径（相对于 workspace），目标文件必须不存在。

        Returns:
            操作结果提示。

        Raises:
            FileNotFoundError: 源文件不存在。
            FileExistsError: 目标文件已存在。
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
        在 workspace 中递归搜索匹配 glob 模式的文件。

        Args:
            pattern: glob 匹配模式，例如 '**/*.txt' 查找所有 txt 文件，
                     '*.py' 查找根目录下的 py 文件。

        Returns:
            匹配文件的相对路径列表（每行一个），无匹配时返回提示信息。
        """
        matches = list(self.workspace.rglob(pattern))
        if not matches:
            return f"No files found matching: {pattern}"
        return "\n".join(str(m.relative_to(self.workspace)) for m in matches)

    def rearch_replace(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> str:
        """
        通过精确匹配替换来编辑文件内容。

        Args:
            path: 相对于 workspace 的文件路径。
            old_text: 要被替换的原始文本（必须精确匹配）。
            new_text: 替换后的新文本。
            replace_all: 当 old_text 出现多次时，设为 True 替换所有匹配，
                         否则需提供更多上下文使 old_text 唯一。

        Returns:
            操作结果提示。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: old_text 在文件中未找到，或出现多次但未设置 replace_all。
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
        列出目录内容。

        Args:
            path: 相对于 workspace 的目录路径，默认为 workspace 根目录。
            recursive: 是否递归列出子目录内容，默认 False。
            max_entries: 最多返回的条目数，默认 200，超出时截断并提示。

        Returns:
            目录条目列表。非递归模式下条目前带图标前缀（📁 目录 / 📄 文件），
            递归模式下显示相对路径。自动忽略 .git、node_modules、__pycache__ 等目录。

        Raises:
            NotADirectoryError: 路径不是目录。
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
