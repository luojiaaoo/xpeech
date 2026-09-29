from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "feishu.lark-cli" / "generate.py"
MYCRED_TEMPLATE_PATH = PROJECT_ROOT / "feishu.lark-cli" / "mycred" / "mycred.go.tmpl"
OAUTH_TEMPLATE_PATH = PROJECT_ROOT / "feishu.lark-cli" / "oauth" / "main.go.tmpl"
OAUTH_GO_TEST_PATH = PROJECT_ROOT / "feishu.lark-cli" / "oauth" / "main_test.go"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"

SPEC = spec_from_file_location("lark_cli_generator", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
lark_cli_generator = module_from_spec(SPEC)
SPEC.loader.exec_module(lark_cli_generator)


def _write_config(path: Path, app_id: str = "cli_test", app_secret: str = "secret"):
    path.write_text(
        f"""
[feishu]
app_id = {app_id!r}
app_secret = {app_secret!r}
""",
        encoding="utf-8",
    )


def test_render_oauth_source_uses_non_blocking_manager_client(tmp_path: Path):
    config_path = tmp_path / "conf.toml"
    _write_config(config_path, app_secret='secret-with-"-quote')

    source = lark_cli_generator.render_source(config_path, OAUTH_TEMPLATE_PATH)

    assert 'managerURL = "http://token-manager:7883"' in source
    assert '"/v1/authorization/start"' in source
    assert '"/v1/status?session_id="' in source
    assert '"/v1/logout"' in source
    assert '"domain"' in source
    assert "scopesForDomains" in source
    assert "device_code" not in source
    assert "{{APP_ID}}" not in source
    assert "{{APP_SECRET}}" not in source


def test_render_mycred_source_delegates_user_auth_to_executable(tmp_path: Path):
    config_path = tmp_path / "conf.toml"
    _write_config(config_path)

    source = lark_cli_generator.render_source(config_path, MYCRED_TEMPLATE_PATH)

    assert 'managerURL = "http://token-manager:7883"' in source
    assert "token manager" in source
    assert "oauthExecutable" not in source
    assert "XPEECH_TOKEN_MANAGER_URL" not in source


def test_refresh_flow_requires_rotation_and_classifies_terminal_errors(
    tmp_path: Path,
):
    config_path = tmp_path / "conf.toml"
    _write_config(config_path)

    source = lark_cli_generator.render_source(config_path, MYCRED_TEMPLATE_PATH)

    assert '"/v1/token/resolve"' in source
    assert "SHELL_SESSION_ID" in source
    assert "access_token" in source
    assert '"/v1/token/invalidate"' in source
    assert "sha256.Sum256" in source
    assert "99991668" in source


def test_docker_builder_runs_behavioral_oauth_tests():
    go_tests = OAUTH_GO_TEST_PATH.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "TestScopesFlagSplitsAndPreservesValues" in go_tests
    assert "TestLoginRequiresScopeOrDomain" in go_tests
    assert "COPY feishu.lark-cli/oauth/main_test.go" in dockerfile
    assert "CGO_ENABLED=0 go test ./cmd/xpeech-lark-cli-auth" in dockerfile


@pytest.mark.parametrize(
    ("feishu_config", "message"),
    [
        ("", "missing \\[feishu\\]"),
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
        lark_cli_generator.render_source(config_path, OAUTH_TEMPLATE_PATH)
