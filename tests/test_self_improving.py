import pytest

from xpeech.agent import memory as memory_module
from xpeech.agent.memory import MemoryArgs, MemoryStore
from xpeech.agent.prompt.system import build_system_prompt
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.utils.session import create_workspace_templates


TREE = "self-improving/\n└── tool-use/\n    ├── 必读.md\n    └── errors.md"


def write_memos(workspace):
    directory = workspace / "memory/self-improving"
    (directory / "tool-use").mkdir()
    (directory / "tool-use/必读.md").write_text("mandatory-memo-body", encoding="utf-8")
    (directory / "tool-use/errors.md").write_text("knowledge-memo-body", encoding="utf-8")
    return directory


@pytest.mark.asyncio
async def test_memos_survive_initialization_and_memory_save(tmp_path):
    await create_workspace_templates(tmp_path)
    directory = write_memos(tmp_path)
    store = MemoryStore(tmp_path)
    rule = "- [confirmed-rule] 默认中文回答"
    await store.write_long_term(rule)

    await store.save_memory(MemoryArgs(history_entry="event", memory_update=rule + "\nproject facts"))
    await create_workspace_templates(tmp_path)
    prompt = await build_system_prompt(tmp_path, not_interactive=True)

    assert f"## Long-term Memory\n{rule}\nproject facts" in prompt["content"]
    assert prompt["content"].count(rule) == 1
    assert prompt["content"].count(TREE) == 1
    assert {path.name for path in directory.iterdir()} == {"tool-use"}
    assert (directory / "tool-use/必读.md").read_text(encoding="utf-8") == "mandatory-memo-body"
    assert "mandatory-memo-body" not in prompt["content"]
    assert "knowledge-memo-body" not in prompt["content"]


@pytest.mark.asyncio
async def test_only_current_users_tree_is_injected(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    await create_workspace_templates(first)
    await create_workspace_templates(second)
    write_memos(first)
    await MemoryStore(first).write_long_term("first-user-rule")
    other_category = second / "memory/self-improving/coding"
    other_category.mkdir()
    (other_category / "async.md").write_text("other-memo-body", encoding="utf-8")
    other_tree = "self-improving/\n└── coding/\n    └── async.md"

    prompt = await build_system_prompt(first)
    other_prompt = await build_system_prompt(second)

    assert TREE in prompt["content"]
    assert TREE not in other_prompt["content"]
    assert other_tree in other_prompt["content"]
    assert other_tree not in prompt["content"]
    assert "first-user-rule" not in other_prompt["content"]
    assert "mandatory-memo-body" not in prompt["content"]
    assert "knowledge-memo-body" not in prompt["content"]
    assert "other-memo-body" not in other_prompt["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["missing", "empty", "root-files"])
async def test_missing_or_empty_memo_tree_keeps_context(tmp_path, state):
    store = MemoryStore(tmp_path)
    if state != "missing":
        directory = tmp_path / "memory/self-improving"
        directory.mkdir()
        if state == "root-files":
            (directory / "index.md").write_text("obsolete index content", encoding="utf-8")
            (directory / "notes.md").write_text("root-level note", encoding="utf-8")

    context = await store.get_self_improvement_context()

    assert "## Self-improvement Memos\nRoot: memory/self-improving/" in context
    assert "use the self-improving skill" in context
    assert "(No self-improvement memos found yet.)" in context
    assert "obsolete index content" not in context
    assert "root-level note" not in context


@pytest.mark.asyncio
async def test_unreadable_directory_does_not_break_long_term_memory(tmp_path, monkeypatch):
    await create_workspace_templates(tmp_path)
    await MemoryStore(tmp_path).write_long_term("important-fact")

    def unreadable(_root):
        raise PermissionError("unreadable directory")

    monkeypatch.setattr(memory_module, "build_memo_tree", unreadable)
    prompt = await build_system_prompt(tmp_path)

    assert "important-fact" in prompt["content"]
    assert "Not loaded: the memo directory could not be scanned" in prompt["content"]


@pytest.mark.asyncio
async def test_symlink_memo_root_is_ignored(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    await create_workspace_templates(workspace)
    outside = tmp_path / "other-user"
    (outside / "private").mkdir(parents=True)
    (outside / "private/another-user.md").write_text("private body", encoding="utf-8")
    memo_root = workspace / "memory/self-improving"
    memo_root.rmdir()
    memo_root.symlink_to(outside, target_is_directory=True)

    prompt = await build_system_prompt(workspace)

    assert "another-user.md" not in prompt["content"]
    assert "private body" not in prompt["content"]
    assert "(No self-improvement memos found yet.)" in prompt["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_self_improving_is_discoverable_on_demand_with_accessible_references(tmp_path, not_interactive):
    loader = SkillsLoader(tmp_path, not_interactive=not_interactive)
    skills = await loader.list_builtin_skill_info()
    skill = next(item for item in skills if item.metadata.name == "self-improving")

    assert skill.metadata.always is False
    assert skill.metadata.internal is True
    prompt = await loader.build_skills_prompt()
    assert "<name>self-improving</name>" in prompt
    assert "### Skill: self-improving" not in prompt
    assert skill.body not in prompt
    assert "### Skill: memory" in prompt
    for reference in ("learning.md", "storage.md", "operations.md"):
        assert (loader.builtin_skills / "self-improving/references" / reference).is_file()
