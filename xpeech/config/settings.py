from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from ..provider.schema import LLMParameters
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
    session_record_path: Path


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

    max_result_chars: int = Field(default=10_000, ge=1_000, validation_alias="max_result_chars")
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


class LLMConfig(BaseModel):
    """LLM provider configuration settings."""

    model_config = ConfigDict(extra="forbid")

    _api_key_selector: _RoundRobinApiKeySelector = PrivateAttr(default_factory=_RoundRobinApiKeySelector)
    _api_keys: list[str] = PrivateAttr(default_factory=list)
    api_key_config: str = Field(validation_alias="api_key", exclude=True, repr=False)

    api_base: str
    default_model: str
    parameters: LLMParameters
    tools_python_package: str
    default_tools: list[str]
    system_name: str = ""
    system_identity_prompt: str = ""
    custom_system_prompt: str = ""
    support_image: bool = False
    support_video: bool = False
    support_json_output: bool = False
    parallel: int = Field(default=4)
    summary_tokens: int = Field(default=8192, gt=0)
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


class JWTConfig(BaseModel):
    """Shared JWT settings for channel-to-API authentication."""

    secret_key: str = Field(min_length=32, repr=False)
    algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_seconds: int = Field(default=60, ge=1, le=60)


class OAuth2Config(BaseModel):
    """Optional OAuth2 authorization-code login settings for the web client."""

    enabled: bool = False
    provider_name: str = Field(default="OAuth2", min_length=1, max_length=32)
    display_type: Literal["qrcode", "link"] = "qrcode"
    client_id: str = ""
    client_secret: str = Field(default="", repr=False)
    authorization_url: str = ""
    token_url: str = ""
    userinfo_url: str = ""
    redirect_uri: str | None = None
    scopes: list[str] = Field(default_factory=lambda: ["openid", "profile"])
    # jq queries against the userinfo JSON, e.g. ".data.employee_no".
    session_id_claim: str = Field(default=".sub", min_length=1)
    username_claim: str = Field(default=".name", min_length=1)
    auto_create_users: bool = False
    use_pkce: bool = True
    token_auth_method: Literal["client_secret_post", "client_secret_basic"] = (
        "client_secret_post"
    )
    extra_authorization_params: dict[str, str] = Field(default_factory=dict)

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OAuth2 provider_name cannot be empty")
        return value

    @field_validator(
        "authorization_url",
        "token_url",
        "userinfo_url",
        "redirect_uri",
    )
    @classmethod
    def validate_http_url(cls, value: str | None) -> str | None:
        if not value:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OAuth2 endpoints must be absolute HTTP(S) URLs")
        return value

    @model_validator(mode="after")
    def validate_enabled_settings(self) -> "OAuth2Config":
        if self.enabled:
            required = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "authorization_url": self.authorization_url,
                "token_url": self.token_url,
                "userinfo_url": self.userinfo_url,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    f"enabled OAuth2 login requires: {', '.join(missing)}"
                )
        return self


class InjectPromptConfig(BaseModel):
    """Resolve a one-shot user-message prefix by invoking a configured command."""

    enabled: bool = False
    command_template: str = ""

    @model_validator(mode="after")
    def validate_enabled_settings(self) -> "InjectPromptConfig":
        self.command_template = self.command_template.strip()
        if self.enabled and not self.command_template:
            raise ValueError("enabled inject_prompt requires command_template")
        if (
            self.enabled
            and "${state}" not in self.command_template
            and "$state" not in self.command_template
        ):
            raise ValueError(
                "inject_prompt.command_template requires ${state} or $state"
            )
        return self


class WebClientConfig(BaseModel):
    """Web client storage settings."""

    database_path: Path
    cookie_name: str = Field(
        default="xpeech_session",
        min_length=1,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    oauth2: list[OAuth2Config] = Field(default_factory=list)
    inject_prompt: InjectPromptConfig = Field(default_factory=InjectPromptConfig)

    @model_validator(mode="after")
    def validate_oauth2_provider_names(self) -> "WebClientConfig":
        provider_names: set[str] = set()
        for oauth2 in self.oauth2:
            normalized_name = oauth2.provider_name.casefold()
            if normalized_name in provider_names:
                raise ValueError(
                    f"OAuth2 provider_name must be unique: {oauth2.provider_name}"
                )
            provider_names.add(normalized_name)
        return self


class Settings(BaseSettings):
    path: PathConfig
    tool: ToolConfig = ToolConfig()
    llm: LLMConfig
    feishu: FeishuConfig
    logging: LoggingConfig = LoggingConfig()
    jwt: JWTConfig
    web_client: WebClientConfig = Field(default_factory=WebClientConfig)

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
ensure_path(settings.path.session_record_path.expanduser().parent)
ensure_path(settings.path.workspace_base_path)
ensure_path(settings.path.cache_path)
ensure_path(settings.path.log_path)
ensure_path(settings.tool.browser_preview.browser_preview_path)
