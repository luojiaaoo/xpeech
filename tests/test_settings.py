import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("with_environment", [False, True])
def test_settings_read_toml_without_loading_environment_or_dotenv(tmp_path, with_environment):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "conf.toml").write_text(
        (root / "conf.toml.example").read_text(encoding="utf-8"), encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "XPEECH_DATABASE__HOST=dotenv-host\nXPEECH_DOTENV_ONLY=loaded\n",
        encoding="utf-8",
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("XPEECH_")}
    if with_environment:
        environment.update({
            "XPEECH_DATABASE": "invalid-json-must-not-be-parsed",
            "XPEECH_DATABASE__HOST": "environment-host",
            "DATABASE": "invalid-json-must-not-be-parsed",
        })
    result = subprocess.run(
        [sys.executable, "-c", """
import json
import os
import sys
sys.path.insert(0, sys.argv[1])
from xpeech.config.settings import Settings, settings
print(json.dumps({
    "host": settings.database.host,
    "reloaded_host": Settings().database.host,
    "constructor_host": Settings(database={"host": "constructor-host"}).database.host,
    "dotenv_loaded": "XPEECH_DOTENV_ONLY" in os.environ,
}))
""", str(root)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == {
        "host": "postgres", "reloaded_host": "postgres",
        "constructor_host": "postgres", "dotenv_loaded": False,
    }
