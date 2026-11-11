from pathlib import Path

import pytest

from xpeech.agent.memory import build_memo_tree


def make_memos(root: Path, files: list[str]) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("memo body", encoding="utf-8")


@pytest.mark.asyncio
async def test_tree_is_sorted_and_must_read_is_first_without_reading_content(tmp_path, monkeypatch):
    root = tmp_path / "self-improving"
    make_memos(root, ["tools/z.md", "tools/a.md", "coding/async.md", "tools/必读.md"])

    def no_read(*_args, **_kwargs):
        pytest.fail("The tree must not read memo contents")

    monkeypatch.setattr(Path, "open", no_read)

    assert await build_memo_tree(root) == (
        "self-improving/\n"
        "├── coding/\n"
        "│   └── async.md\n"
        "└── tools/\n"
        "    ├── 必读.md\n"
        "    ├── a.md\n"
        "    └── z.md"
    )


@pytest.mark.asyncio
async def test_tree_reflects_changes_without_writing_a_manifest(tmp_path):
    root = tmp_path / "self-improving"
    make_memos(root, ["编码经验/异步编程.md"])
    first = await build_memo_tree(root)
    (root / "编码经验/异步编程.md").rename(root / "编码经验/任务取消.md")
    second = await build_memo_tree(root)

    assert "异步编程.md" in first
    assert "异步编程.md" not in second
    assert "任务取消.md" in second
    assert {path.name for path in root.iterdir()} == {"编码经验"}
    assert (root / "编码经验/任务取消.md").read_text(encoding="utf-8") == "memo body"


@pytest.mark.parametrize("state", ["missing", "empty", "empty-category"])
@pytest.mark.asyncio
async def test_no_memos_produce_no_tree(tmp_path, state):
    root = tmp_path / "self-improving"
    if state != "missing":
        root.mkdir()
    if state == "empty-category":
        (root / "empty").mkdir()

    assert await build_memo_tree(root) == ""


@pytest.mark.asyncio
async def test_only_two_visible_markdown_levels_are_shown(tmp_path):
    root = tmp_path / "self-improving"
    make_memos(root, [
        "tools/valid.md", "index.md", "tools/nested/deep.md", ".hidden/private.md", "tools/.private.md",
    ])
    (root / "tools/notes.txt").write_text("not a memo", encoding="utf-8")

    assert await build_memo_tree(root) == "self-improving/\n└── tools/\n    └── valid.md"


@pytest.mark.asyncio
async def test_symlink_files_and_categories_are_not_followed(tmp_path):
    root = tmp_path / "self-improving"
    make_memos(root, ["tools/valid.md"])
    outside = tmp_path / "other-user"
    make_memos(outside, ["private/secret.md"])
    (root / "outside").symlink_to(outside / "private", target_is_directory=True)
    (root / "tools/secret.md").symlink_to(outside / "private/secret.md")

    assert await build_memo_tree(root) == "self-improving/\n└── tools/\n    └── valid.md"


@pytest.mark.asyncio
async def test_symlink_root_is_not_followed(tmp_path):
    outside = tmp_path / "other-user"
    make_memos(outside, ["private/secret.md"])
    root = tmp_path / "self-improving"
    root.symlink_to(outside, target_is_directory=True)

    assert await build_memo_tree(root) == ""
