from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import sys
import os

sys.path.append(os.getcwd())

from xpeech.config.helper import (
    MissingEnvVarError,
    TomlConfigSettingsSource,
    load_env_files,
    load_toml_file,
    load_toml_files,
    toml_config_settings_source,
)


CONFIG_DIR = Path(__file__).parent / "config"
DEFAULT_TOML = CONFIG_DIR / "default.toml"
LOCAL_TOML = CONFIG_DIR / "local.toml"
DOTENV = CONFIG_DIR / ".env"


def test_loads_dotenv_file_with_special_characters() -> None:
    env = load_env_files([DOTENV])

    assert env["OPENAI_API_KEY"] == "sk-test+/=:with$symbols"
    assert env["SPECIAL_CHARS_SECRET"] == (
        'quote " double, dollar $, slash /, equals =, colon :'
    )


def test_loads_toml_and_expands_values_from_dotenv_file() -> None:
    config = load_toml_file(DEFAULT_TOML, env_files=[DOTENV])

    assert config["llm"] == {
        "model": "default-model",
        "timeout": 30,
        "api_key": "sk-test+/=:with$symbols",
        "base_url": "https://api.openai.com/v1",
        "special_chars": 'quote " double, dollar $, slash /, equals =, colon :',
        "literal_placeholder": "${OPENAI_API_KEY}",
        "tags": ["agent", "local"],
    }
    assert config["agent"]["tools"] == ["filesystem", "shell"]


def test_merges_default_and_local_toml_from_left_to_right() -> None:
    config = load_toml_files([DEFAULT_TOML, LOCAL_TOML], env_files=[DOTENV])

    assert config["llm"]["model"] == "local-model"
    assert config["llm"]["timeout"] == 60
    assert config["llm"]["api_key"] == "sk-test+/=:with$symbols"
    assert config["history"] == {"enabled": True, "path": "history.jsonl"}
    assert config["server"] == {"host": "127.0.0.1", "port": 9000}


def test_missing_required_env_placeholder_raises() -> None:
    with pytest.raises(MissingEnvVarError) as exc_info:
        load_toml_file(DEFAULT_TOML, env={})

    assert exc_info.value.name == "OPENAI_API_KEY"


def test_missing_toml_file_returns_empty_dict_by_default(tmp_path) -> None:
    assert load_toml_file(tmp_path / "missing.toml") == {}


def test_missing_toml_file_can_be_required(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_toml_file(tmp_path / "missing.toml", missing_ok=False)


def test_plain_callable_source_reads_real_toml_files() -> None:
    source = toml_config_settings_source(
        DEFAULT_TOML,
        LOCAL_TOML,
        env_files=[DOTENV],
    )

    config = source()

    assert config["llm"]["model"] == "local-model"
    assert config["llm"]["api_key"] == "sk-test+/=:with$symbols"


def test_pydantic_settings_reads_toml_and_dotenv_sources() -> None:
    class LLMConfig(BaseModel):
        model: str
        timeout: int
        api_key: str
        base_url: str
        special_chars: str
        literal_placeholder: str
        tags: list[str]

    class AgentConfig(BaseModel):
        max_steps: int
        tools: list[str]

    class HistoryConfig(BaseModel):
        enabled: bool
        path: str

    class ServerConfig(BaseModel):
        host: str
        port: int

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix="XPEECH_TEST_",
            env_file=DOTENV,
            env_nested_delimiter="__",
            extra="ignore",
        )

        llm: LLMConfig
        agent: AgentConfig
        history: HistoryConfig
        server: ServerConfig

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
                    DEFAULT_TOML,
                    LOCAL_TOML,
                    env_files=[DOTENV],
                ),
                file_secret_settings,
            )

    settings = Settings()

    assert settings.llm.model == "from-dotenv"
    assert settings.llm.timeout == 60
    assert settings.llm.api_key == "sk-test+/=:with$symbols"
    assert settings.llm.literal_placeholder == "${OPENAI_API_KEY}"
    assert settings.agent.tools == ["filesystem", "shell"]
    assert settings.server.port == 9000
