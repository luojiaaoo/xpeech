from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from ..provider.schema import ReasoningEffort
from ..utils.helper import ensure_path


conf_toml_path = "conf.toml"  # Configuration TOML file path
conf_env_path = ".env"  # Configuration environment file path

env_conf = dict(
    env_prefix="",
    env_nested_delimiter="__",
    env_file=conf_env_path,
    extra="ignore",
)


class PathConfig(BaseModel):
    """Path configuration settings."""

    session_path: Path
    session_history_path: Path
    workspace_base_path: Path
    restrict_tools_to_workspace: bool = True


class LLMConfig(BaseModel):
    """LLM provider configuration settings."""

    api_key: str
    api_base: str
    default_model: str
    default_context_token: int
    default_top_p: float
    tools_python_package: str
    default_tools: list[str]
    default_reasoning_effort: ReasoningEffort | None = None
    support_image: bool = False
    support_json_output: bool = False


class FeishuConfig(BaseModel):
    """Feishu channel configuration settings."""

    app_id: str
    app_secret: str
    idle_timeout: int = 5
    parallel: int = 4


class Settings(BaseSettings):
    path: PathConfig
    llm: LLMConfig
    feishu: FeishuConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            dotenv_settings,
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=conf_toml_path,
            ),
        )

    model_config = SettingsConfigDict(**env_conf)


settings = Settings()
ensure_path(settings.path.session_path)
ensure_path(settings.path.session_history_path)
ensure_path(settings.path.workspace_base_path)
