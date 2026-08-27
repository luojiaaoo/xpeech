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


def test_render_oauth_source_injects_credentials_and_fixed_scopes(tmp_path: Path):
    config_path = tmp_path / "conf.toml"
    _write_config(config_path, app_secret='secret-with-"-quote')

    source = lark_cli_generator.render_source(config_path, OAUTH_TEMPLATE_PATH)

    assert 'appID     = "cli_test"' in source
    assert 'appSecret = "secret-with-\\"-quote"' in source
    assert '\t"offline_access",' in source
    assert '\t"contact:user.base:readonly",' in source
    assert '\t"contact:user.employee:readonly",' in source
    assert "larkauth.RequestDeviceAuthorization(" in source
    assert "larkauth.PollDeviceToken(" in source
    assert '"github.com/larksuite/cli/internal/auth"' in source
    assert "devicePollTimeout          = 60 * time.Second" in source
    assert "maxDevicePolls             = 2" in source
    assert '`json:"poll_attempts,omitempty"`' in source
    assert "context.WithTimeout(ctx, time.Duration(pollSeconds)*time.Second)" in source
    assert 'errors.New("用户未授权' in source
    assert 'if device.VerificationUriComplete == "" {' in source
    assert "result.UserCode = device.UserCode" in source
    assert "UserCode:         device.UserCode" not in source
    assert "redirectURI" not in source
    assert "authorizationEndpoint" not in source
    assert "codeVerifier" not in source
    assert "{{APP_ID}}" not in source
    assert "{{APP_SECRET}}" not in source


def test_render_mycred_source_delegates_user_auth_to_executable(tmp_path: Path):
    config_path = tmp_path / "conf.toml"
    _write_config(config_path)

    source = lark_cli_generator.render_source(config_path, MYCRED_TEMPLATE_PATH)

    assert 'oauthExecutable     = "lark-oauth"' in source
    assert "deviceAuthorization" not in source
    assert "oauthTokenEndpoint" not in source


def test_refresh_flow_requires_rotation_and_classifies_terminal_errors(
    tmp_path: Path,
):
    config_path = tmp_path / "conf.toml"
    _write_config(config_path)

    source = lark_cli_generator.render_source(config_path, OAUTH_TEMPLATE_PATH)

    assert "shouldRefreshUserToken(cached, desiredScopes)" in source
    assert "probePrivateJSONWrite(cachePath)" in source
    assert "saveUserTokenReliably(cachePath, refreshed)" in source
    assert "refreshFailureRequiresAuthorization(refreshErr)" in source
    assert "refreshToken := firstNonEmpty" not in source
    assert "successful response did not rotate refresh_token" in source
    assert "feishuRefreshInvalid     = 20026" in source
    assert "feishuRefreshExpired     = 20037" in source
    assert "feishuRefreshRevoked     = 20064" in source
    assert "feishuRefreshAlreadyUsed = 20073" in source


def test_docker_builder_runs_behavioral_oauth_tests():
    go_tests = OAUTH_GO_TEST_PATH.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "TestRefreshUserTokenRotatesAndValidatesResponse" in go_tests
    assert "TestRefreshFailureClassification" in go_tests
    assert "TestShouldRefreshOnlyForExpiredAccessWithExistingScopes" in go_tests
    assert "TestSaveUserTokenReliablyUsesPrivateAtomicCache" in go_tests
    assert "COPY feishu.lark-cli/oauth/main_test.go" in dockerfile
    assert "CGO_ENABLED=0 go test ./cmd/xpeech-lark-oauth" in dockerfile


@pytest.mark.parametrize(
    ("feishu_config", "message"),
    [
        ("", "missing \\[feishu\\]"),
        ('[feishu]\napp_secret = "secret"', "feishu.app_id"),
        ('[feishu]\napp_id = "cli_test"', "feishu.app_secret"),
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
