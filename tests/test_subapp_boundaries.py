"""Keep supporting applications downstream of the reusable core."""

import ast
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path
from unittest.mock import Mock

import pytest

XPEECH_ROOT = Path(__file__).resolve().parents[1] / "xpeech"


def test_core_does_not_import_subapps():
    violations = []
    for path in XPEECH_ROOT.rglob("*.py"):
        relative = path.relative_to(XPEECH_ROOT)
        if (
            relative == Path("__main__.py")
            or relative.parts[0] == "subapp"
            or relative.parts[:3] == ("agent", "skills", "buildin")
        ):
            continue
        package = ".".join(("xpeech", *relative.parts[:-1]))
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    module = resolve_name("." * node.level + module, package)
                modules = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            else:
                continue
            if any(module == "xpeech.subapp" or module.startswith("xpeech.subapp.") for module in modules):
                violations.append(f"{relative}:{node.lineno}")
    assert not violations, f"Core imports supporting applications: {violations}"


@pytest.mark.parametrize(
    "entrypoint, arguments, target, expected",
    [
        ("xpeech.__main__", [], "xpeech.agent.server.app", {"host": "0.0.0.0", "port": 7878}),
        ("xpeech.__main__", ["api", "--port", "8080"], "xpeech.agent.server.app", {"host": "0.0.0.0", "port": 8080}),
        ("xpeech.__main__", ["scheduler"], "xpeech.subapp.scheduler.app", {"host": "0.0.0.0", "port": 8090}),
        ("xpeech.__main__", ["token_manager"], "xpeech.subapp.lark_token_manager.app", {"host": "0.0.0.0", "port": 7883}),
        ("xpeech.__main__", ["feishu"], "xpeech.subapp.channel.feishu.bridge", {"chat_url": "http://127.0.0.1:7878"}),
        ("xpeech.__main__", ["web_client"], "xpeech.subapp.channel.web_client.app", {"host": "0.0.0.0", "port": 7939, "backend_url": "http://127.0.0.1:7878"}),
    ],
)
def test_service_entrypoints_dispatch_to_their_own_apps(monkeypatch, entrypoint, arguments, target, expected):
    run = Mock()
    monkeypatch.setattr(import_module(target), "run", run)
    monkeypatch.setattr("sys.argv", ["xpeech", *arguments])
    import_module(entrypoint).main()
    run.assert_called_once_with(**expected)
