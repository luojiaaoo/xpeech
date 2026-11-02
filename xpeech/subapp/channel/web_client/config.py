import re
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

from ....config import settings as config_module


class EmbedConfig(BaseModel):
    allowed_origins: list[str] = Field(default_factory=list)

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, origins: list[str]) -> list[str]:
        normalized: list[str] = []
        for origin in origins:
            origin = origin.strip().lower().rstrip("/")
            if not re.fullmatch(r"https?://(?:[a-z0-9.-]+|\[[0-9a-f:]+\])(?::[0-9]+)?", origin):
                raise ValueError("embed.allowed_origins requires exact HTTP(S) origins without paths or wildcards")
            parsed = urlsplit(origin)
            if parsed.port == (443 if parsed.scheme == "https" else 80):
                origin = origin.rsplit(":", 1)[0]
            if origin not in normalized:
                normalized.append(origin)
        return normalized


class _WebClientSettings(BaseModel):
    embed: EmbedConfig = Field(default_factory=EmbedConfig)


class EmbedSettings(BaseSettings):
    """Read web-only embed options without extending shared backend settings."""

    web_client: _WebClientSettings = Field(default_factory=_WebClientSettings)
    model_config = SettingsConfigDict(
        env_prefix="XPEECH_", env_nested_delimiter="__", extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=config_module.conf_toml_path),
        )
