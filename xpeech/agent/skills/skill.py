"""Skills loader for agent capabilities."""

import re
from collections.abc import Collection
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import yaml
from async_lru import alru_cache
from loguru import logger
from pydantic import BaseModel, Field, computed_field, field_validator

from ...utils.file_io import LockedFile, ensure_path_async
from ...utils.filter_expr import FilterContext, filter_allows_access, validate_filter_source
from ...utils.helper import resolve_path_without_symlinks_async

# Default builtin skills directory (relative to this file)
BUILTIN_SKILLS_DIR = Path(__file__).parent / "buildin"
INTERACTIVE_SKILL_NAMES = {"ask-user-question"}
FRONTMATTER_PATTERN = re.compile(r"^---\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)


class SkillMetadata(BaseModel):
    """Fields read from a skill's YAML frontmatter."""

    name: str
    description: str
    always: bool = False
    internal: bool = False
    filter: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("filter")
    @classmethod
    def validate_filter(cls, value: str | None) -> str | None:
        return validate_filter_source(value)


class SkillInfo(BaseModel):
    """Information about an available skill."""

    source: Literal["builtin", "workspace"]
    sandbox_filepath: str = Field(
        serialization_alias="path", description="Absolute path to SKILL.md inside the sandbox",
    )
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
            internal = frontmatter.get("internal")
            kwargs = {
                key: value
                for key, value in frontmatter.items()
                if key not in {"name", "description", "always", "internal", "filter"}
            }
            return SkillMetadata(
                name=name,
                description=description.strip() if isinstance(description, str) and description.strip() else name,
                always=always if isinstance(always, bool) else False,
                internal=internal if isinstance(internal, bool) else False,
                filter=frontmatter.get("filter"),
                kwargs=kwargs,
            )

        frontmatter = read_frontmatter()
        if frontmatter is None:
            raise ValueError("Invalid SKILL.md frontmatter")
        name = frontmatter.get("name")
        return metadata_from_frontmatter(frontmatter, name)

    @computed_field
    @cached_property
    def body(self) -> str:
        """Return the Markdown body verbatim after the YAML frontmatter."""
        match = FRONTMATTER_PATTERN.match(self.content)
        if not match:
            raise ValueError("Invalid SKILL.md frontmatter")
        return self.content[match.end() :]

    @property
    def instructions(self) -> str:
        """Return the base directory and full body for model context."""
        base_directory = Path(self.sandbox_filepath).parent
        body = self.body
        separator = "\n" if body.startswith("\n") else "\n\n"
        return f"Base directory for this skill: {base_directory}{separator}{body}"


@alru_cache(maxsize=1024)
async def _read_skill_cached(
    skill_file: Path, source: Literal["builtin", "workspace"], sandbox_filepath: str, version: tuple[int, ...],
) -> SkillInfo | None:
    file = LockedFile(skill_file)
    try:
        skill = SkillInfo(source=source, sandbox_filepath=sandbox_filepath, content=await file.read_text())
        # Validate frontmatter while loading so invalid skills are skipped below.
        _ = skill.metadata
    except (FileNotFoundError, IsADirectoryError):
        return None
    except ValueError as exc:
        logger.warning("Skipping invalid skill '{}': {}", skill_file, exc)
        return None
    return skill


async def _build_skill_reader(
    skill_file: Path, source: Literal["builtin", "workspace"], sandbox_filepath: str,
) -> SkillInfo | None:
    """Check the current file version before looking up the cached SkillInfo."""
    try:
        skill_file = await resolve_path_without_symlinks_async(skill_file)
        stat = await anyio.Path(skill_file).stat()
    except FileNotFoundError:
        return None
    version = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_dev, stat.st_ino
    return await _read_skill_cached(skill_file, source, sandbox_filepath, version)


async def _iter_skill_files(directory: Path, excluded_names: Collection[str] = ()) -> list[Path]:
    """List SKILL.md files in direct children without following symbolic links."""
    root = await resolve_path_without_symlinks_async(directory)
    entries = sorted([entry async for entry in anyio.Path(root).iterdir()])
    files = []
    for entry in entries:
        if entry.name in excluded_names:
            continue
        skill_file = await resolve_path_without_symlinks_async(Path(entry) / "SKILL.md")
        if await anyio.Path(skill_file).is_file():
            files.append(skill_file)
    return files


class SkillsLoader:
    """
    Loader for agent skills.

    Skills are markdown files (SKILL.md) that teach the agent how to use
    specific tools or perform certain tasks.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        user_id: str,
        channel: str,
        user_organization: str | None = None,
        not_interactive: bool = False,
    ):
        self.workspace_skills = workspace.resolve() / "skills"
        self.builtin_skills = BUILTIN_SKILLS_DIR
        self.channel = channel
        self.user_id = user_id
        self.user_organization = user_organization
        self.not_interactive = not_interactive

    async def _load_skill_info(
        self,
        skill_files: list[tuple[Path, Literal["builtin", "workspace"]]],
        seen_names: set[str],
    ) -> list[SkillInfo]:
        """Read, filter and de-duplicate skill documents."""
        skills = []
        for skill_file, source in skill_files:
            # Built-in skills are mounted into the workspace by the shell sandbox.
            if source == "builtin":
                sandbox_filepath = str(self.workspace_skills / skill_file.parent.name / "SKILL.md")
            else:
                sandbox_filepath = str(skill_file)
            skill = await _build_skill_reader(skill_file, source, sandbox_filepath)
            if skill is None:
                continue
            metadata = skill.metadata
            if self.not_interactive and metadata.name in INTERACTIVE_SKILL_NAMES:
                continue
            # Workspace skills are placed by the user and are not subject to access filters.
            if source == "builtin" and not filter_allows_access(
                metadata.filter, FilterContext(channel=self.channel, org=self.user_organization, user_id=self.user_id)
            ):
                continue
            if metadata.name in seen_names:
                logger.warning("Skipping duplicate skill '{}' at '{}'", metadata.name, skill_file)
                continue
            seen_names.add(metadata.name)
            # Callers may edit metadata or paths without changing the shared cached snapshot.
            skills.append(skill.model_copy(deep=True))
        return skills

    async def list_skill_info(self, *, builtin_only: bool = False) -> list[SkillInfo]:
        """List available skills, optionally skipping all workspace access."""
        async def _builtin_skill_dirs() -> list[Path]:
            """Return valid built-in skill directories in a stable order."""
            return [skill_file.parent for skill_file in await _iter_skill_files(self.builtin_skills)]

        builtin_dirs = await _builtin_skill_dirs()
        builtin_skills = await self._load_skill_info(
            [(directory / "SKILL.md", "builtin") for directory in builtin_dirs],
            set(),
        )
        if builtin_only:
            return builtin_skills
        await ensure_path_async(self.workspace_skills)
        workspace_files = await _iter_skill_files(
            self.workspace_skills, {directory.name for directory in builtin_dirs},
        )
        workspace_skills = await self._load_skill_info(
            [(file, "workspace") for file in workspace_files],
            {skill.metadata.name for skill in builtin_skills},
        )
        # 内置技能已完成 filter 筛选：匹配的内置技能 → 用户技能 → 公共内置技能。
        # 各组保持原有顺序；用户技能的 filter 不参与筛选或排序。
        return [
            *[skill for skill in builtin_skills if skill.metadata.filter is not None],
            *workspace_skills,
            *[skill for skill in builtin_skills if skill.metadata.filter is None],
        ]

    async def get_skill(self, name: str) -> SkillInfo:
        """Look up a catalog skill by its frontmatter name."""
        skills = await self.list_skill_info()
        for skill in skills:
            if skill.metadata.name == name:
                return skill

        raise LookupError(f"Skill {name!r} is not available in this session")

    async def build_skills_prompt(self) -> str:
        """Build the skills catalog and always-loaded bodies for user-role context."""

        def format_skills_summary(skills: list[SkillInfo]) -> str:
            """Format already loaded skill information as XML."""
            if not skills:
                return ""

            def escape_xml(s: str) -> str:
                return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            lines = ["<skills>"]
            for skill in skills:
                lines.append(
                    f"  <skill source='{skill.source}' "
                    f"internal='{str(skill.metadata.internal).lower()}'>"
                )
                lines.append(f"    <name>{escape_xml(skill.metadata.name)}</name>")
                lines.append(f"    <description>{escape_xml(skill.metadata.description)}</description>")
                lines.append("  </skill>")
            lines.append("</skills>")

            return "\n".join(lines)

        skills = await self.list_skill_info()
        if not skills:
            return ""

        parts = []
        active_skills = [
            f"### Skill: {skill.metadata.name}\n\n{skill.instructions}"
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
                "Use the read_skill tool with the exact <name> to load a skill's instructions. "
                "Skills in Active Skills are already loaded.\n"
                "`builtin` means a shared skill built into the sandbox, `workspace` means a skill "
                "placed in the workspace by the user, and `internal` marks a skill related to the "
                "agent's runtime mechanics.\n\n"
                f"{skills_summary}\n"
            )

        return "\n\n---\n\n".join(parts)
