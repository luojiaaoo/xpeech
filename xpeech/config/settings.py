from pathlib import Path
from threading import Lock

from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import field_validator
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


class _RoundRobinApiKeySelector:
    """Select LLM API keys in round-robin order within this server process."""

    def __init__(self) -> None:
        self._index = 0
        self._lock = Lock()

    def next(self, api_keys: list[str]) -> str:
        if len(api_keys) == 1:
            return api_keys[0]

        with self._lock:
            api_key = api_keys[self._index % len(api_keys)]
            self._index += 1
            return api_key


class PathConfig(BaseModel):
    """Path configuration settings."""

    session_path: Path
    session_history_path: Path
    workspace_base_path: Path


class ToolConfig(BaseModel):
    """Tool safety configuration settings."""

    restrict_tools_to_workspace: bool = True
    allowed_networks: list[str] = Field(default_factory=list)


class LLMConfig(BaseModel):
    """LLM provider configuration settings."""

    _api_key_selector: _RoundRobinApiKeySelector = PrivateAttr(default_factory=_RoundRobinApiKeySelector)
    _api_keys: list[str] = PrivateAttr(default_factory=list)
    api_key_config: str = Field(validation_alias="api_key", exclude=True, repr=False)

    api_base: str
    default_model: str
    default_context_token: int
    default_top_p: float
    tools_python_package: str
    default_tools: list[str]
    system_name: str = "assistant"
    custom_system_prompt: str = ""
    default_reasoning_effort: ReasoningEffort | None = None
    support_image: bool = False
    support_video: bool = False
    support_json_output: bool = False
    parallel: int = Field(default=4)

    @field_validator("api_key_config")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        api_keys = [key.strip() for key in value.split(",") if key.strip()]
        if not api_keys:
            raise ValueError("LLM api_key cannot be empty")
        return value.strip()

    def model_post_init(self, __context: object) -> None:
        self._api_keys = [key.strip() for key in self.api_key_config.split(",") if key.strip()]

    @property
    def api_key(self) -> str:
        """Return the next API key for an LLM request."""

        return self._api_key_selector.next(self._api_keys)


class FeishuConfig(BaseModel):
    """Feishu channel configuration settings."""

    app_id: str
    app_secret: str
    idle_timeout: int = 5


class Settings(BaseSettings):
    path: PathConfig
    tool: ToolConfig = ToolConfig()
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
