import asyncio

import pytest

from xpeech.utils.session import create_workspace_templates


@pytest.mark.asyncio
async def test_parallel_initialization_preserves_user_files(tmp_path):
    (tmp_path / "memory").mkdir()
    user_files = {
        "AGENTS.md": "custom instructions",
        "memory/MEMORY.md": "remembered facts",
        "memory/HISTORY.md": "past events",
    }
    for relative, content in user_files.items():
        (tmp_path / relative).write_text(content, encoding="utf-8")

    await asyncio.gather(*(create_workspace_templates(tmp_path) for _ in range(8)))

    for relative, content in user_files.items():
        assert (tmp_path / relative).read_text(encoding="utf-8") == content
    for name in ("SOUL.md", "USER.md", "TOOLS.md"):
        assert (tmp_path / name).read_text(encoding="utf-8").startswith("# ")
    assert (tmp_path / "skills").is_dir()
    assert (tmp_path / "home").is_dir()
    assert not list(tmp_path.rglob("*.tmp"))
