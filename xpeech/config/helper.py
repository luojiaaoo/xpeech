#### AI生成代码，请勿修改

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


EnvMapping = Mapping[str, str]
ConfigDict = dict[str, Any]
EnvFile = str | Path

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
_ESCAPED_ENV_PREFIX = "\0XPEECH_ESCAPED_ENV_PREFIX\0"


class MissingEnvVarError(ValueError):
    """Raised when a TOML env placeholder references a missing env var."""

    def __init__(self, name: str):
        super().__init__(f"Missing required environment variable: {name}")
        self.name = name


def load_env_files(paths: Sequence[EnvFile]) -> dict[str, str]:
    """Load dotenv files without mutating os.environ."""
    values: dict[str, str] = {}

    for path in paths:
        env_path = Path(path)
        if not env_path.exists():
            continue

        for key, value in dotenv_values(env_path).items():
            if value is not None:
                values[key] = value

    return values


def build_env(
    *,
    env_files: Sequence[EnvFile] | None = None,
    env: EnvMapping | None = None,
) -> dict[str, str]:
    """Build the env used for TOML placeholder expansion."""
    values = load_env_files(env_files or ())
    values.update(os.environ)
    if env is not None:
        values.update(env)

    return values


def deep_merge(base: ConfigDict, override: Mapping[str, Any]) -> ConfigDict:
    """Recursively merge two config dicts and return a new dict."""
    merged = dict(base)

    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value

    return merged


def expand_env_vars(
    value: Any,
    *,
    env: EnvMapping | None = None,
    strict: bool = True,
) -> Any:
    """Expand ${ENV_NAME} and ${ENV_NAME:-default} inside TOML values."""
    env = os.environ if env is None else env

    if isinstance(value, Mapping):
        return {
            key: expand_env_vars(item, env=env, strict=strict)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [expand_env_vars(item, env=env, strict=strict) for item in value]

    if not isinstance(value, str):
        return value

    text = value.replace("$${", _ESCAPED_ENV_PREFIX)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)

        if name in env:
            return env[name]
        if default is not None:
            return default
        if strict:
            raise MissingEnvVarError(name)
        return match.group(0)

    return _ENV_PATTERN.sub(replace, text).replace(_ESCAPED_ENV_PREFIX, "${")


def load_toml_file(
    path: str | Path,
    *,
    expand_env: bool = True,
    env_files: Sequence[EnvFile] | None = None,
    env: EnvMapping | None = None,
    strict_env: bool = True,
    missing_ok: bool = True,
) -> ConfigDict:
    """Load one TOML file and optionally expand env placeholders."""
    toml_path = Path(path)
    if not toml_path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(toml_path)

    with toml_path.open("rb") as file:
        data = tomllib.load(file)

    if expand_env:
        data = expand_env_vars(
            data,
            env=build_env(env_files=env_files, env=env),
            strict=strict_env,
        )

    return data


def load_toml_files(
    paths: Sequence[str | Path],
    *,
    expand_env: bool = True,
    env_files: Sequence[EnvFile] | None = None,
    env: EnvMapping | None = None,
    strict_env: bool = True,
    missing_ok: bool = True,
) -> ConfigDict:
    """Load and deep-merge TOML files from left to right."""
    config: ConfigDict = {}

    for path in paths:
        data = load_toml_file(
            path,
            expand_env=expand_env,
            env_files=env_files,
            env=env,
            strict_env=strict_env,
            missing_ok=missing_ok,
        )
        config = deep_merge(config, data)

    return config


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """A pydantic-settings source backed by one or more TOML files."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        *paths: str | Path,
        expand_env: bool = True,
        env_files: Sequence[EnvFile] | None = None,
        env: EnvMapping | None = None,
        strict_env: bool = True,
        missing_ok: bool = True,
    ):
        super().__init__(settings_cls)
        self.paths = paths
        self.expand_env = expand_env
        self.env_files = env_files
        self.env = env
        self.strict_env = strict_env
        self.missing_ok = missing_ok

    def __call__(self) -> ConfigDict:
        return load_toml_files(
            self.paths,
            expand_env=self.expand_env,
            env_files=self.env_files,
            env=self.env,
            strict_env=self.strict_env,
            missing_ok=self.missing_ok,
        )

    def get_field_value(
        self,
        field: FieldInfo,
        field_name: str,
    ) -> tuple[Any, str, bool]:
        return None, field_name, False


def toml_config_settings_source(
    *paths: str | Path,
    expand_env: bool = True,
    env_files: Sequence[EnvFile] | None = None,
    env: EnvMapping | None = None,
    strict_env: bool = True,
    missing_ok: bool = True,
) -> Callable[[], ConfigDict]:
    """Build a plain callable source for pydantic-settings."""

    def source() -> ConfigDict:
        return load_toml_files(
            paths,
            expand_env=expand_env,
            env_files=env_files,
            env=env,
            strict_env=strict_env,
            missing_ok=missing_ok,
        )

    return source
