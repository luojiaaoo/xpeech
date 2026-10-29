#!/usr/bin/env python3
"""Render lark-cli sources from Xpeech's build-time TOML configuration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

APP_ID_MARKER = "{{APP_ID}}"


class ConfigError(ValueError):
    """Raised when the lark-cli build configuration is incomplete or invalid."""


def _required_string(mapping: dict[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def load_build_config(config_path: Path) -> str:
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    feishu = config.get("feishu")
    if not isinstance(feishu, dict):
        raise ConfigError("missing [feishu] configuration")

    return _required_string(feishu, "app_id", "feishu")


def render_source(config_path: Path, template_path: Path) -> str:
    app_id = load_build_config(config_path)
    source = template_path.read_text(encoding="utf-8")
    replacements = {
        APP_ID_MARKER: json.dumps(app_id),
    }
    for marker, value in replacements.items():
        marker_count = source.count(marker)
        if marker_count > 1:
            raise RuntimeError(f"template must contain at most one {marker} marker")
        if marker_count == 1:
            source = source.replace(marker, value)
    return source


def write_source(output_path: Path, source: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        dir=output_path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(source)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject [feishu] values into Go source.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_source(args.output, render_source(args.config, args.template))


if __name__ == "__main__":
    main()
