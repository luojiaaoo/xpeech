from pathlib import Path
from xml.etree import ElementTree

import pytest
from pydantic import ValidationError

from xpeech.agent.server.context import lark_user_access_token_var
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillInfo, SkillMetadata, SkillsLoader


@pytest.mark.asyncio
async def test_skills_loader_creates_workspace_when_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [])
    workspace = tmp_path / "workspace"

    loader = SkillsLoader(workspace)

    assert not workspace.exists()
    assert loader.workspace_skills == workspace / "skills"
    assert await loader.list_skill_info() == []
    assert (workspace / "skills").is_dir()


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
    skill = SkillInfo(source="builtin", path=str(skill_file), content=skill_file.read_text())

    assert skill.metadata.internal is True
    assert skill.metadata.kwargs == {}


@pytest.mark.parametrize(
    ("skill_name", "short_description"),
    [
        ("algorithmic-poster-philosophy", "海报设计"),
        ("chat2duckdb", "DuckDB 数据分析"),
        ("lark-skills", "飞书全能力助手"),
        ("markdown-formatter", "Markdown 格式化"),
        ("presentation-generation", "演示文稿生成"),
        ("weather", "天气查询与预报"),
    ],
)
def test_builtin_short_descriptions_match_backend_server(
    skill_name: str, short_description: str
):
    skill_file = skill_module.BUILTIN_SKILLS_DIR / skill_name / "SKILL.md"
    skill = SkillInfo(source="builtin", path=str(skill_file), content=skill_file.read_text())

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
