from pydantic import BaseModel, Field

from ..skills.skill import SkillsLoader


class ReadSkillArgs(BaseModel):
    name: str = Field(min_length=1, description="Exact skill name from the skills catalog")


def build_read_skill_tool(*, skills_loader: SkillsLoader):
    async def read_skill(args: ReadSkillArgs) -> str:
        """Return the skill's base directory and full Markdown body by name."""
        skill = await skills_loader.get_skill(args.name)
        return skill.instructions

    return read_skill
