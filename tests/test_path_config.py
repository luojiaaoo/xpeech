import pytest

from xpeech.config.settings import BrowserPreviewConfig, LarkCLIConfig, PathConfig, WebClientConfig


def test_config_paths_expand_using_backend_home(tmp_path, monkeypatch):
    backend_home = tmp_path / "backend-home"
    monkeypatch.setenv("HOME", str(backend_home))
    config = PathConfig.model_validate({name: f"~/data/{name}" for name in PathConfig.model_fields})

    for name in PathConfig.model_fields:
        assert getattr(config, name) == backend_home / "data" / name


@pytest.mark.parametrize(("model", "field"), [
    (BrowserPreviewConfig, "browser_preview_path"),
    (LarkCLIConfig, "database_path"),
    (WebClientConfig, "database_path"),
])
def test_other_config_paths_are_absolute_at_load(tmp_path, monkeypatch, model, field):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "backend-home"))

    assert getattr(model.model_validate({field: "~/store"}), field) == tmp_path / "backend-home/store"
    assert getattr(model.model_validate({field: "relative/store"}), field) == tmp_path / "relative/store"


def test_default_config_paths_are_also_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert BrowserPreviewConfig().browser_preview_path == tmp_path / "docker_data/browser_preview"
    assert LarkCLIConfig().database_path == tmp_path / "docker_data/lark-token-manager/tokens.db"
