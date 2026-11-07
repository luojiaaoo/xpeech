import json
import re
from pathlib import Path
from unittest.mock import AsyncMock
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from xpeech.agent.prompt.system import build_system_prompt
from xpeech.agent.server.context import lark_user_access_token_var
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillInfo, SkillMetadata, SkillsLoader


def test_public_builtin_short_descriptions_are_present_and_readable():
    for skill_file in sorted(skill_module.BUILTIN_SKILLS_DIR.glob("*/SKILL.md")):
        content = skill_file.read_bytes().decode("utf-8")
        skill = SkillInfo(source="builtin", path=str(skill_file), content=content)
        if skill.metadata.internal:
            continue

        summary = skill.metadata.kwargs.get("short_description")
        assert isinstance(summary, str) and summary.strip(), skill_file
        assert re.search(r"[\u4e00-\u9fff]", summary), skill_file
        assert not re.search(r"\?{3,}|\ufffd|\u951f\u65a4\u62f7", content), skill_file


@pytest.mark.parametrize(
    "skill_name",
    [
        "autosar-arxml",
        "uds-diagnostics",
        "can-bus-analysis",
        "embedded-c-review",
        "embedded-systems",
        "cpp-pro",
        "linux-kernel-driver-enablement",
        "datasheets",
        "cad-code-generation",
        "cadquery-export",
        "automotive-ethernet-analysis",
        "iso26262-hara",
        "capl-writer",
        "iso26262-fmea",
        "aspice-review",
        "misra-c-review",
        "documents",
    ],
)
def test_engineering_skill_metadata_and_upstream_integrity(skill_name: str):
    root = skill_module.BUILTIN_SKILLS_DIR / skill_name
    content = (root / "SKILL.md").read_bytes().decode("utf-8")
    skill = SkillInfo(source="builtin", path=str(root / "SKILL.md"), content=content)

    assert skill.metadata.name == {"automotive-ethernet-analysis": "wireshark"}.get(skill_name, skill_name)
    assert skill.metadata.description.strip()
    assert re.search(r"[\u4e00-\u9fff]", skill.metadata.kwargs["short_description"])
    assert "???" not in content
    assert "\ufffd" not in content
    assert skill.metadata.always is False

    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    assert set(source) == {"repository", "commit", "path", "license"}
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    assert source["repository"].startswith("https://github.com/")
    assert source["path"]
    assert source["license"] in {"MIT", "Apache-2.0"}
    assert not (root / "LICENSE").exists()
    assert not (root / "UPSTREAM.md").exists()


@pytest.mark.asyncio
async def test_skills_loader_creates_workspace_when_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    workspace = tmp_path / "workspace"

    loader = SkillsLoader(workspace)

    assert not workspace.exists()
    assert loader.workspace_skills == workspace / "skills"
    assert await loader.list_skill_info() == []
    assert (workspace / "skills").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_kind", ["none", "missing", "existing"])
async def test_list_builtin_skill_info_does_not_access_workspace(tmp_path, monkeypatch, workspace_kind):
    builtin_dir = tmp_path / "builtin-folder"
    builtin_dir.mkdir()
    skill_file = builtin_dir / "SKILL.md"
    skill_file.write_text("---\nname: builtin-name\n---\nBuilt-in instructions")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [builtin_dir])
    workspace = None if workspace_kind == "none" else tmp_path / "workspace"
    if workspace_kind == "existing":
        workspace_skill = workspace / "skills/custom/SKILL.md"
        workspace_skill.parent.mkdir(parents=True)
        workspace_skill.write_text("---\nname: custom\n---\nWorkspace instructions")
    monkeypatch.setattr(skill_module, "ensure_path_async", AsyncMock(side_effect=AssertionError("Workspace accessed")))
    monkeypatch.setattr(skill_module.aiofiles.os, "scandir", AsyncMock(side_effect=AssertionError("Workspace scanned")))

    [skill] = await SkillsLoader(workspace).list_builtin_skill_info()

    assert skill.source == "builtin"
    assert skill.metadata.name == "builtin-name"
    assert skill.path == (str(workspace / "skills/builtin-folder/SKILL.md") if workspace is not None else "")
    if workspace_kind == "missing":
        assert not workspace.exists()


def test_skill_content_is_frozen_after_creation():
    content = "---\nname: original\n---\n# Original"
    skill = SkillInfo(source="workspace", path="SKILL.md", content=content)
    assert skill.metadata.name == "original"
    assert skill.body == "# Original"

    with pytest.raises(ValidationError, match="Field is frozen"):
        skill.content = "---\nname: changed\n---\n# Changed"

    assert skill.content == content
    assert skill.metadata.name == "original"
    assert skill.body == "# Original"


@pytest.mark.parametrize("skill_name", ["ask-user-question", "memory", "skill-creator"])
def test_internal_builtin_skills_are_marked(skill_name: str):
    skill_file = skill_module.BUILTIN_SKILLS_DIR / skill_name / "SKILL.md"
    skill = SkillInfo(source="builtin", path=str(skill_file), content=skill_file.read_text(encoding="utf-8"))

    assert skill.metadata.internal is True
    assert skill.metadata.kwargs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_system_prompt_filters_builtin_interactive_skill(tmp_path: Path, not_interactive: bool):
    prompt = await build_system_prompt(tmp_path, not_interactive=not_interactive)

    assert ("<name>ask-user-question</name>" in prompt["content"]) is not not_interactive
    assert ("### Skill: ask-user-question" in prompt["content"]) is not not_interactive
    assert ("通过 `ask_user_question` 向用户追问" in prompt["content"]) is not not_interactive
    assert "<name>memory</name>" in prompt["content"]
    assert "### Skill: memory" in prompt["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["builtin", "workspace"])
@pytest.mark.parametrize("always", [False, True])
@pytest.mark.parametrize("not_interactive", [False, True])
async def test_interactive_skill_filter_applies_to_list_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    always: bool,
    not_interactive: bool,
):
    skill_dir = tmp_path / "builtin" if source == "builtin" else tmp_path / "skills" / "ask-user-question"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: ask-user-question\nalways: {str(always).lower()}\n---\n"
        "# Interactive instructions\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [skill_dir] if source == "builtin" else [])
    regular_dir = tmp_path / "skills" / "regular-skill"
    regular_dir.mkdir(parents=True)
    (regular_dir / "SKILL.md").write_text(
        "---\nname: regular-skill\n---\n# Regular instructions\n", encoding="utf-8",
    )
    loader = SkillsLoader(tmp_path, not_interactive=not_interactive)

    skills = await loader.list_skill_info()
    if source == "builtin":
        assert await loader.list_builtin_skill_info() == [skill for skill in skills if skill.source == "builtin"]
    names = {skill.metadata.name for skill in skills}
    expected = {"regular-skill"}
    if not not_interactive:
        expected.add("ask-user-question")
    assert names == expected

    prompt = await loader.build_skills_prompt()
    assert ("<name>ask-user-question</name>" in prompt) is not not_interactive
    assert ("# Interactive instructions" in prompt) is (always and not not_interactive)
    assert "<name>regular-skill</name>" in prompt


@pytest.mark.parametrize(
    ("skill_name", "short_description"),
    [
        ("algorithmic-poster-philosophy", "海报设计"),
        ("chat2duckdb", "DuckDB 数据分析"),
        ("lark-skills", "飞书全能力助手"),
        ("markdown-formatter", "Markdown 格式化"),
        ("presentation-generation", "HTML 演示文稿生成"),
        ("weather", "天气查询与预报"),
    ],
)
def test_builtin_short_descriptions_match_backend_server(
    skill_name: str, short_description: str
):
    skill_file = skill_module.BUILTIN_SKILLS_DIR / skill_name / "SKILL.md"
    skill = SkillInfo(source="builtin", path=str(skill_file), content=skill_file.read_text(encoding="utf-8"))

    assert skill.metadata.kwargs["short_description"] == short_description


@pytest.mark.asyncio
async def test_injected_token_selects_dedicated_lark_skill(tmp_path: Path):
    context = lark_user_access_token_var.set("u-request-token")
    try:
        skills = await SkillsLoader(tmp_path / "workspace").list_skill_info()
    finally:
        lark_user_access_token_var.reset(context)

    lark_skills = [skill for skill in skills if skill.metadata.name == "lark-skills-token"]
    assert len(lark_skills) == 1
    assert "用户访问令牌由请求 JWT 注入" in lark_skills[0].body
    assert lark_skills[0].path == str(tmp_path / "workspace/skills/lark-skills-token/SKILL.md")
    assert all(skill.metadata.name not in {"lark-skills", "lark-cli-auth"} for skill in skills)


@pytest.mark.parametrize("content", ["# Instructions", "---\nname: example\n# Missing closing delimiter"])
def test_skill_body_requires_frontmatter(content: str):
    skill = SkillInfo(source="workspace", path="SKILL.md", content=content)

    with pytest.raises(ValueError, match="Invalid SKILL.md frontmatter"):
        _ = skill.body


@pytest.mark.asyncio
async def test_skill_metadata_drives_list_and_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "custom-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        '---\nname: custom-skill\ndescription: "Useful & focused"\nalways: true\nextra: ignored\n---\n',
        encoding="utf-8",
    )
    loader = SkillsLoader(workspace)

    assert (await loader.list_skill_info())[0].metadata == SkillMetadata(
        name="custom-skill",
        description="Useful & focused",
        always=True,
        kwargs={"extra": "ignored"},
    )
    skills_prompt = await loader.build_skills_prompt()
    assert skills_prompt.startswith("# Active Skills\n\n### Skill: custom-skill\n\n")
    assert "\n\n---\n\n# Skills\n\n" in skills_prompt
    assert "<description>Useful &amp; focused</description>" in skills_prompt
    assert "`builtin` means a shared skill built into the sandbox" in skills_prompt
    assert "`internal` marks a skill related to the agent's runtime mechanics" in skills_prompt
    summary_start = skills_prompt.index("<skills>")
    summary_end = skills_prompt.index("</skills>") + len("</skills>")
    summary = ElementTree.fromstring(skills_prompt[summary_start:summary_end])
    assert summary.findtext("skill/description") == "Useful & focused"
    assert summary.find("skill").attrib == {"source": "workspace", "internal": "false"}


@pytest.mark.asyncio
async def test_skill_name_comes_from_document_not_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    workspace = tmp_path / "workspace"
    skill_file = workspace / "skills" / "folder-name" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    content = "---\nname: document-name\ndescription: From markdown\n---\n# Instructions\n"
    skill_file.write_text(content, encoding="utf-8")
    loader = SkillsLoader(workspace)

    skills = await loader.list_skill_info()
    assert [(skill.metadata.name, skill.path) for skill in skills] == [("document-name", str(skill_file))]
    assert skills[0].metadata == SkillMetadata(name="document-name", description="From markdown")
    assert skills[0].content == content
    assert skills[0].body == "# Instructions"
    prompt = await loader.build_skills_prompt()
    assert "<name>document-name</name>" in prompt
    assert f"<location>{skill_file}</location>" in prompt


@pytest.mark.asyncio
async def test_builtin_name_takes_precedence_over_workspace_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    builtin_dir = tmp_path / "builtin-folder"
    builtin_dir.mkdir()
    (builtin_dir / "SKILL.md").write_text(
        "---\nname: shared-name\ndescription: Built-in\n---\n# Built-in\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [builtin_dir])
    workspace = tmp_path / "workspace"
    workspace_skill = workspace / "skills" / "workspace-folder" / "SKILL.md"
    workspace_skill.parent.mkdir(parents=True)
    workspace_skill.write_text(
        "---\nname: shared-name\ndescription: Workspace\n---\n# Workspace\n",
        encoding="utf-8",
    )
    loader = SkillsLoader(workspace)

    skills = await loader.list_skill_info()
    assert len(skills) == 1
    assert skills[0].metadata.name == "shared-name"
    assert skills[0].source == "builtin"
    assert skills[0].metadata.description == "Built-in"
    assert skills[0].path == str(workspace / "skills" / "builtin-folder" / "SKILL.md")


@pytest.mark.asyncio
async def test_filtered_builtin_directory_still_takes_precedence_over_workspace(tmp_path, monkeypatch):
    builtin_dir = tmp_path / "builtin/shared-folder"
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text('---\nname: shared-name\nfilter: channel("crm")\n---\nBuilt-in')
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [builtin_dir])
    workspace = tmp_path / "workspace"
    for folder in ("shared-folder", "other-folder"):
        skill_file = workspace / "skills" / folder / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text("---\nname: shared-name\n---\nWorkspace")
    loader = SkillsLoader(workspace)

    assert await loader.list_builtin_skill_info() == []
    [skill] = await loader.list_skill_info()
    assert skill.source == "workspace"
    assert skill.path == str(workspace / "skills/other-folder/SKILL.md")


@pytest.mark.asyncio
async def test_skill_metadata_defaults_and_invalid_name_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "custom-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        '---\nname: custom-skill\ndescription: []\nalways: "true"\n---\n',
        encoding="utf-8",
    )
    invalid_skill = workspace / "skills" / "missing-name" / "SKILL.md"
    invalid_skill.parent.mkdir()
    invalid_skill.write_text("---\ndescription: Missing name\n---\n", encoding="utf-8")
    loader = SkillsLoader(workspace)

    skills = await loader.list_skill_info()
    assert len(skills) == 1
    assert skills[0].metadata == SkillMetadata(
        name="custom-skill",
        description="custom-skill",
        always=False,
    )


@pytest.mark.asyncio
async def test_build_prompt_reads_each_skill_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    skill_file = tmp_path / "workspace" / "skills" / "always-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: always-skill\ndescription: Always useful\nalways: true\n---\n# Do this\n",
        encoding="utf-8",
    )
    reads = []
    parses = []
    read_text = skill_module.LockedFile.read_text
    safe_load = skill_module.yaml.safe_load

    async def count_reads(file: skill_module.LockedFile) -> str:
        reads.append(file.path)
        return await read_text(file)

    def count_parses(content: str):
        parses.append(content)
        return safe_load(content)

    monkeypatch.setattr(skill_module.LockedFile, "read_text", count_reads)
    monkeypatch.setattr(skill_module.yaml, "safe_load", count_parses)
    prompt = await SkillsLoader(tmp_path / "workspace").build_skills_prompt()

    assert reads == [skill_file]
    assert len(parses) == 1
    assert "### Skill: always-skill\n\n# Do this" in prompt


@pytest.mark.asyncio
async def test_build_prompt_marks_internal_skill_without_activating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    skill_file = tmp_path / "workspace" / "skills" / "internal-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: internal-skill\ndescription: Internal utility\ninternal: true\n---\n# Internal instructions\n",
        encoding="utf-8",
    )

    prompt = await SkillsLoader(tmp_path / "workspace").build_skills_prompt()

    assert "# Active Skills" not in prompt
    assert "# Internal instructions" not in prompt
    assert "<skill source='workspace' internal='true'>" in prompt
