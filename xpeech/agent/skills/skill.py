"""Skills loader for agent capabilities."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from ...utils.helper import read_text_async

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent / "buildin"


class SkillMetadata(BaseModel):
    """Fields read from a skill's YAML frontmatter."""

    description: str
    always: bool = False


class SkillInfo(BaseModel):
    """Information about an available skill."""

    name: str
    metadata: SkillMetadata
    source: Literal["builtin", "workspace"]
    path: str


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

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = BUILTIN_SKILLS_DIR

    def list_skills(self) -> list[dict[str, str]]:
        """
        List all available skills.

        Args:
            filter_unavailable: If True, filter out skills with unmet requirements.

        Returns:
            List of skill info dicts with 'name', 'path', 'source'.
        """
        skills = []

        # Built-in skills (highest priority)
        for skill_dir in iter_builtin_skill_dirs():
            # Built-in skills are mounted into the workspace skills directory
            # by the shell sandbox. Expose that unified path to the agent.
            skill_file = self.workspace_skills / skill_dir.name / "SKILL.md"
            skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})

        # Workspace skills cannot override a built-in skill with the same name.
        if self.workspace_skills.exists():
            builtin_names = {skill["name"] for skill in skills}
            for skill_dir in sorted(self.workspace_skills.iterdir(), key=lambda path: path.name):
                if skill_dir.is_dir() and skill_dir.name not in builtin_names:
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        return skills

    async def list_skill_info(self) -> list[SkillInfo]:
        """Return all available skills with their display metadata."""
        result = []
        for skill in self.list_skills():
            result.append(
                SkillInfo(
                    name=skill["name"],
                    metadata=await self.get_skill_metadata(skill["name"]),
                    source=skill["source"],
                    path=skill["path"],
                )
            )
        return result

    async def load_skill(self, name: str) -> str | None:
        """
        Load a skill by name.

        Args:
            name: Skill name (directory name).

        Returns:
            Skill content or None if not found.
        """
        # Check built-in first
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return await read_text_async(builtin_skill)

        # Fall back to workspace skills
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return await read_text_async(workspace_skill)

        return None

    async def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        Load specific skills for inclusion in agent context.

        Args:
            skill_names: List of skill names to load.

        Returns:
            Formatted skills content.
        """
        parts = []
        for name in skill_names:
            content = await self.load_skill(name)
            if content:
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    @staticmethod
    def _format_skills_summary(skills: list[SkillInfo]) -> str:
        """Format already loaded skill information as XML."""
        if not skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for skill in skills:
            lines.append(f"  <skill source='{skill.source}'>")
            lines.append(f"    <name>{escape_xml(skill.name)}</name>")
            lines.append(f"    <description>{escape_xml(skill.metadata.description)}</description>")
            lines.append(f"    <location>{escape_xml(skill.path)}</location>")
            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    async def build_skills_prompt(self) -> str:
        """Build the complete skills section for the system prompt."""
        skills = await self.list_skill_info()
        if not skills:
            return ""

        parts = []
        always_skills = [skill.name for skill in skills if skill.metadata.always]
        if always_skills:
            always_content = await self.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        skills_summary = self._format_skills_summary(skills)
        if skills_summary:
            parts.append(
                "# Skills\n\n"
                "The following skills extend your capabilities. To use a skill, "
                "read its SKILL.md file using the read_file tool.\n\n"
                f"{skills_summary}\n"
            )

        return "\n\n---\n\n".join(parts)

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
        return content

    async def get_skill_metadata(self, name: str) -> SkillMetadata:
        """
        Extract description and always from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Typed metadata, with the skill name and always=False as fallbacks.
        """
        content = await self.load_skill(name)
        if content is None:
            return SkillMetadata(description=name)

        parsed: dict = {}
        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                try:
                    metadata = yaml.safe_load(match.group(1))
                except yaml.YAMLError:
                    metadata = None
                if isinstance(metadata, dict):
                    parsed = metadata

        description = parsed.get("description")
        always = parsed.get("always")
        return SkillMetadata(
            description=description.strip() if isinstance(description, str) and description.strip() else name,
            always=always if isinstance(always, bool) else False,
        )
