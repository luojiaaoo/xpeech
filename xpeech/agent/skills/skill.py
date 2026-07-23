"""Skills loader for agent capabilities."""

import re
from pathlib import Path
import aiofiles

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent / "buildin"


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
            for skill_dir in self.workspace_skills.iterdir():
                if skill_dir.is_dir() and skill_dir.name not in builtin_names:
                    skill_file = skill_dir / "SKILL.md"
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        return skills

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
                async with aiofiles.open(builtin_skill, mode="r", encoding="utf-8") as f:
                    return await f.read()

        # Fall back to workspace skills
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            async with aiofiles.open(workspace_skill, mode="r", encoding="utf-8") as f:
                return await f.read()

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

    async def build_skills_summary(self) -> str:
        """
        Build a summary of all skills (name, description, path, availability).

        This is used for progressive loading - the agent can read the full
        skill content using read_file when needed.

        Returns:
            XML-formatted skills summary.
        """
        all_skills = self.list_skills()
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            name = escape_xml(s["name"])
            path = s["path"]
            source = s["source"]
            desc = escape_xml(await self._get_skill_description(s["name"]))

            lines.append(f"  <skill source='{source}'>")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")
            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    async def _get_skill_description(self, name: str) -> str:
        """Get the description of a skill from its frontmatter."""
        meta = await self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
        return content

    async def get_always_skills(self) -> list[str]:
        """Get skills marked as always=true that meet requirements."""
        result = []
        for s in self.list_skills():
            meta = await self.get_skill_metadata(s["name"]) or {}
            if meta.get("always"):
                result.append(s["name"])
        return result

    async def get_skill_metadata(self, name: str) -> dict | None:
        """
        Get metadata from a skill's frontmatter.

        Args:
            name: Skill name.

        Returns:
            Metadata dict or None.
        """
        content = await self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # Simple YAML parsing
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip("\"'")
                return metadata

        return None
