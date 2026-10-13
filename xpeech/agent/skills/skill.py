"""Skills loader for agent capabilities."""

import re
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal

import aiofiles.os
import yaml
from loguru import logger
from pydantic import BaseModel, Field, computed_field

from ...utils.helper import ensure_path_async, read_text_async

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent / "buildin"
FRONTMATTER_PATTERN = re.compile(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)


class SkillMetadata(BaseModel):
    """Fields read from a skill's YAML frontmatter."""

    name: str
    description: str
    always: bool = False


class SkillInfo(BaseModel):
    """Information about an available skill."""

    source: Literal["builtin", "workspace"]
    path: str
    content: Annotated[str, Field(frozen=True)]

    @computed_field
    @cached_property
    def metadata(self) -> SkillMetadata:
        """Parse and cache metadata from the SKILL.md content."""

        def read_frontmatter() -> dict | None:
            match = FRONTMATTER_PATTERN.match(self.content)
            if not match:
                return None
            try:
                parsed = yaml.safe_load(match.group(1))
            except yaml.YAMLError:
                return None
            return parsed if isinstance(parsed, dict) else None

        def metadata_from_frontmatter(frontmatter: dict, name: str) -> SkillMetadata:
            description = frontmatter.get("description")
            always = frontmatter.get("always")
            return SkillMetadata(
                name=name,
                description=description.strip() if isinstance(description, str) and description.strip() else name,
                always=always if isinstance(always, bool) else False,
            )

        frontmatter = read_frontmatter()
        if frontmatter is None:
            raise ValueError("Invalid SKILL.md frontmatter")
        name = frontmatter.get("name")
        return metadata_from_frontmatter(frontmatter, name)

    @computed_field
    @cached_property
    def body(self) -> str:
        """Return and cache the Markdown body without YAML frontmatter."""
        match = FRONTMATTER_PATTERN.match(self.content)
        if not match:
            raise ValueError("Invalid SKILL.md frontmatter")
        return self.content[match.end() :].strip()


def iter_builtin_skill_dirs() -> list[Path]:
    """Return valid built-in skill directories in a stable order."""
    return sorted(
        (
            skill_dir
            for skill_dir in BUILTIN_SKILLS_DIR.iterdir()
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").is_file()
        ),
        key=lambda skill_dir: skill_dir.name,
    )


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(self, workspace: Path | None):
        self.workspace_skills = workspace / "skills" if workspace is not None else None
        self.builtin_skills = BUILTIN_SKILLS_DIR

    async def _skill_files(self) -> list[tuple[Path, Literal["builtin", "workspace"]]]:
        """Return skill documents, with built-ins taking precedence."""
        builtin_dirs = iter_builtin_skill_dirs()
        builtin_dir_names = {skill_dir.name for skill_dir in builtin_dirs}
        files: list[tuple[Path, Literal["builtin", "workspace"]]] = [
            (skill_dir / "SKILL.md", "builtin") for skill_dir in builtin_dirs
        ]
        if self.workspace_skills is not None:
            await ensure_path_async(self.workspace_skills)
            entries = await aiofiles.os.scandir(self.workspace_skills)
            for entry in sorted(entries, key=lambda item: item.name):
                skill_dir = self.workspace_skills / entry.name
                skill_file = skill_dir / "SKILL.md"
                if (
                    entry.name not in builtin_dir_names
                    and await aiofiles.os.path.isdir(skill_dir)
                    and await aiofiles.os.path.isfile(skill_file)
                ):
                    files.append((skill_file, "workspace"))
        return files

    async def list_skill_info(self) -> list[SkillInfo]:
        """Return valid skills with metadata and full SKILL.md content."""
        skills = []
        seen_names = set()
        for skill_file, source in await self._skill_files():
            content = await read_text_async(skill_file)
            # Built-in skills are mounted into the workspace by the shell sandbox.
            if source == "builtin":
                visible_path = (
                    str(self.workspace_skills / skill_file.parent.name / "SKILL.md")
                    if self.workspace_skills is not None
                    else ""
                )
            else:
                visible_path = str(skill_file)
            skill = SkillInfo(source=source, path=visible_path, content=content)
            try:
                metadata = skill.metadata
            except ValueError as exc:
                logger.warning("Skipping invalid skill '{}': {}", skill_file, exc)
                continue
            if metadata.name in seen_names:
                logger.warning("Skipping duplicate skill '{}' at '{}'", metadata.name, skill_file)
                continue
            seen_names.add(metadata.name)
            skills.append(skill)
        return skills

    async def build_skills_prompt(self) -> str:
        """Build the complete skills section for the system prompt."""

        def format_skills_summary(skills: list[SkillInfo]) -> str:
            """Format already loaded skill information as XML."""
            if not skills:
                return ""

            def escape_xml(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            lines = ["<skills>"]
            for skill in skills:
                lines.append(f"  <skill source='{skill.source}'>")
                lines.append(f"    <name>{escape_xml(skill.metadata.name)}</name>")
                lines.append(f"    <description>{escape_xml(skill.metadata.description)}</description>")
                lines.append(f"    <location>{escape_xml(skill.path)}</location>")
                lines.append("  </skill>")
            lines.append("</skills>")

            return "\n".join(lines)

        skills = await self.list_skill_info()
        if not skills:
            return ""

        parts = []
        active_skills = [
            f"### Skill: {skill.metadata.name}\n\n{skill.body}"
            for skill in skills
            if skill.metadata.always
        ]
        if active_skills:
            active_content = "\n\n---\n\n".join(active_skills)
            parts.append(f"# Active Skills\n\n{active_content}")

        skills_summary = format_skills_summary(skills)
        if skills_summary:
            parts.append(
                "# Skills\n\n"
                "The following skills extend your capabilities. To use a skill, "
                "read its SKILL.md file using the read_file tool.\n\n"
                f"{skills_summary}\n"
            )

        return "\n\n---\n\n".join(parts)
