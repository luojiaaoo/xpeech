from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomlkit


DEFAULT_API_BASE_URL = "http://127.0.0.1:7878"
DEFAULT_APP_NAME = "Xpeech Desktop"
DEFAULT_APP_ICON = ""
CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(slots=True)
class DesktopConfig:
    api_base_url: str
    app_name: str
    app_icon: str


def load_config(config_path: Path = CONFIG_PATH) -> DesktopConfig:
    if not config_path.exists():
        data = dict()
    else:
        with config_path.open("r", encoding="utf-8") as file:
            data = tomlkit.parse(file.read())
    return DesktopConfig(
        api_base_url=str(data.get("api_base_url", DEFAULT_API_BASE_URL)).rstrip("/"),
        app_name=str(data.get("app_name", DEFAULT_APP_NAME)),
        app_icon=str(data.get("app_icon", DEFAULT_APP_ICON)),
    )


def save_api_base_url(api_base_url: str, config_path: Path = CONFIG_PATH) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            doc = tomlkit.parse(file.read())
    else:
        doc = tomlkit.document()
        doc.add(tomlkit.comment("Desktop client configuration"))
    doc["api_base_url"] = api_base_url.rstrip("/")
    config_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
