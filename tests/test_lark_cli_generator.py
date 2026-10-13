from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "feishu.lark-cli" / "generate.py"
MYCRED_TEMPLATE_PATH = PROJECT_ROOT / "feishu.lark-cli" / "mycred" / "mycred.go.tmpl"

SPEC = spec_from_file_location("lark_cli_generator", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
lark_cli_generator = module_from_spec(SPEC)
SPEC.loader.exec_module(lark_cli_generator)


def test_render_user_token_provider_injects_app_id_without_app_secret(tmp_path: Path):
    config_path = tmp_path / "conf.toml"
    config_path.write_text('[feishu]\napp_id = "cli_test"\n', encoding="utf-8")

    source = lark_cli_generator.render_source(config_path, MYCRED_TEMPLATE_PATH)

    assert 'appID = "cli_test"' in source
    assert "{{APP_ID}}" not in source
    assert '"LARKSUITE_CLI_USER_ACCESS_TOKEN"' in source
    assert "credential.SupportsUser" in source
    assert "http" not in source
    assert "managerURL" not in source


@pytest.mark.parametrize(
    ("feishu_config", "message"),
    [
        ("", r"missing \[feishu\]"),
        ('[feishu]\napp_secret = "secret"', "feishu.app_id"),
    ],
)
def test_invalid_lark_cli_build_config_is_rejected(
    tmp_path: Path,
    feishu_config: str,
    message: str,
):
    config_path = tmp_path / "conf.toml"
    config_path.write_text(feishu_config, encoding="utf-8")

    with pytest.raises(lark_cli_generator.ConfigError, match=message):
        lark_cli_generator.render_source(config_path, MYCRED_TEMPLATE_PATH)
