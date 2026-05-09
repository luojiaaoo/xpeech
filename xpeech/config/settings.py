from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from .helper import TomlConfigSettingsSource
from pathlib import Path
from ..utils.helper import ensure_path

env_conf = dict(
    env_prefix="XPEECH_",
    env_nested_delimiter="__",
    extra="ignore",
)


conf_toml_path = "conf.toml"  # Configuration TOML file path
conf_env_path = ".env"  # Configuration environment file path


class PathConfig(BaseModel):
    session_path: Path
    session_history_path: Path
    workspace_base_path: Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(**env_conf)

    path: PathConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(
                settings_cls,
                conf_toml_path,
                env_files=[conf_env_path],
            ),
            file_secret_settings,
        )


settings = Settings()
ensure_path(settings.path.session_path)
ensure_path(settings.path.session_history_path)
ensure_path(settings.path.workspace_base_path)
