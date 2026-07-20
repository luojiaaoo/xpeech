from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from ..provider.schema import ReasoningEffort
from ..utils.helper import ensure_path

if Path(".env").exists():
    load_dotenv(".env")

conf_toml_path = "conf.toml"  # Configuration TOML file path


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
    sandbox_home_path: Path
    cache_path: Path
    log_path: Path


class BrowserPreviewConfig(BaseModel):
    """Browser preview URL and file storage settings."""

    browser_preview_base_url: str = "http://127.0.0.1:7878/browser_preview"
    browser_preview_path: Path = Path("data/browser_preview")

    @field_validator("browser_preview_base_url")
    @classmethod
    def validate_browser_preview_base_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("browser_preview_base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("browser_preview_base_url cannot contain a query or fragment")
        if not parsed.path or parsed.path == "/":
            raise ValueError("browser_preview_base_url must contain a route path")
        return value

    @property
    def route_path(self) -> str:
        return urlsplit(self.browser_preview_base_url).path.rstrip("/")


class ToolConfig(BaseModel):
    """Tool safety configuration settings."""

    browser_preview: BrowserPreviewConfig = Field(default_factory=BrowserPreviewConfig)
    mcp_servers: dict[str, "MCPServerSettings"] = Field(default_factory=dict, validation_alias="mcpServers")


class MCPServerSettings(BaseModel):
    """MCP server configuration."""

    model_config = ConfigDict(extra="forbid")

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"], validation_alias="enabled_tools")
    tool_timeout: float = Field(default=30.0, validation_alias="tool_timeout")
    max_result_chars: int = Field(default=50_000, ge=1_000, validation_alias="max_result_chars")


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
    system_name: str = ""
    custom_system_prompt: str = ""
    default_reasoning_effort: ReasoningEffort | None = None
    support_image: bool = False
    support_video: bool = False
    support_json_output: bool = False
    parallel: int = Field(default=4)
    max_iterations: int = Field(default=40, ge=1)

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


class LoggingConfig(BaseModel):
    """Runtime log file settings."""

    retention_days: int = Field(default=7, ge=1)
    max_file_size_mb: int = Field(default=10, ge=1)


class Settings(BaseSettings):
    path: PathConfig
    tool: ToolConfig = ToolConfig()
    llm: LLMConfig
    feishu: FeishuConfig
    logging: LoggingConfig = LoggingConfig()

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
            env_settings,
            TomlConfigSettingsSource(
                settings_cls,
                toml_file=conf_toml_path,
            ),
        )

    model_config = SettingsConfigDict(
        env_prefix="XPEECH_",
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = Settings()
ensure_path(settings.path.session_path)
ensure_path(settings.path.session_history_path)
ensure_path(settings.path.workspace_base_path)
ensure_path(settings.path.cache_path)
ensure_path(settings.path.log_path)
ensure_path(settings.tool.browser_preview.browser_preview_path)
