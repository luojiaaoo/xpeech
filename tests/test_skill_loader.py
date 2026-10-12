from pathlib import Path
from xml.etree import ElementTree

import pytest

from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillMetadata, SkillsLoader


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

    assert await loader.get_skill_metadata("custom-skill") == SkillMetadata(
        description="Useful & focused",
        always=True,
    )
    assert (await loader.list_skill_info())[0].metadata == SkillMetadata(
        description="Useful & focused",
        always=True,
    )
    metadata_calls: list[str] = []
    get_skill_metadata = loader.get_skill_metadata

    async def count_metadata_calls(name: str) -> SkillMetadata:
        metadata_calls.append(name)
        return await get_skill_metadata(name)

    monkeypatch.setattr(loader, "get_skill_metadata", count_metadata_calls)
    skills_prompt = await loader.build_skills_prompt()
    assert metadata_calls == ["custom-skill"]
    assert skills_prompt.startswith("# Active Skills\n\n### Skill: custom-skill\n\n")
    assert "\n\n---\n\n# Skills\n\n" in skills_prompt
    assert "<description>Useful &amp; focused</description>" in skills_prompt
    summary_start = skills_prompt.index("<skills>")
    summary_end = skills_prompt.index("</skills>") + len("</skills>")
    summary = ElementTree.fromstring(skills_prompt[summary_start:summary_end])
    assert summary.findtext("skill/description") == "Useful & focused"


@pytest.mark.asyncio
async def test_skill_metadata_defaults_for_missing_or_invalid_frontmatter(tmp_path: Path):
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "custom-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        '---\ndescription: []\nalways: "true"\n---\n',
        encoding="utf-8",
    )
    loader = SkillsLoader(workspace)

    assert await loader.get_skill_metadata("custom-skill") == SkillMetadata(
        description="custom-skill",
        always=False,
    )
    assert await loader.get_skill_metadata("missing-skill") == SkillMetadata(
        description="missing-skill",
        always=False,
    )
