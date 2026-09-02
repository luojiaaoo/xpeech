from html import unescape
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from xpeech.agent.server.api import app
from xpeech.agent.server.routes.public_file import preview_file
from xpeech.agent.tools.browser_preview import BrowserPreviewArgs, build_browser_preview_tool
from xpeech.agent.tools.helper import as_tool
from xpeech.config.settings import settings
from xpeech.utils.jwt_auth import create_access_token


@pytest.fixture
def file_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    workspace_root = tmp_path / "workspaces"
    preview_root = tmp_path / "previews"
    workspace_root.mkdir()
    preview_root.mkdir()
    monkeypatch.setattr(settings.path, "workspace_base_path", workspace_root)
    monkeypatch.setattr(settings.tool.browser_preview, "browser_preview_path", preview_root)
    return workspace_root, preview_root


def test_download_session_file_requires_valid_jwt(file_roots: tuple[Path, Path]):
    workspace_root, _ = file_roots
    session_id = "session-1"
    session_workspace = workspace_root / session_id
    session_workspace.mkdir()
    (session_workspace / "result.txt").write_text("downloaded", encoding="utf-8")
    client = TestClient(app)
    url = f"/sessions/{session_id}/files"
    params = {"path": "result.txt"}

    missing_token = client.get(url, params=params)
    invalid_token = client.get(
        url,
        params=params,
        headers={"Authorization": "Bearer invalid-token"},
    )
    valid_token = client.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {create_access_token()}"},
    )

    assert missing_token.status_code == 401
    assert invalid_token.status_code == 401
    assert valid_token.status_code == 200
    assert valid_token.content == b"downloaded"


def test_preview_directory_is_public_and_recursively_navigable(
    file_roots: tuple[Path, Path],
):
    _, preview_root = file_roots
    preview_id = uuid4()
    directory = preview_root / str(preview_id)
    nested_name = "nested & 目录"
    nested = directory / nested_name
    nested.mkdir(parents=True)
    (directory / "empty").mkdir()
    (directory / "z.txt").write_text("root file", encoding="utf-8")
    special_file_name = "页面 & one.html"
    (directory / special_file_name).write_text("<h1>preview</h1>", encoding="utf-8")
    (nested / "child.txt").write_text("nested file", encoding="utf-8")

    client = TestClient(app)
    root_url = f"{settings.tool.browser_preview.route_path}/{preview_id}/"
    response = client.get(root_url)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html; charset=utf-8")
    assert response.headers["cache-control"] == "no-store"
    assert "../" not in response.text
    assert response.text.index("empty/") < response.text.index("z.txt")
    assert "页面 &amp; one.html" in response.text

    nested_url = root_url + quote(nested_name, safe="") + "/"
    special_file_url = root_url + quote(special_file_name, safe="")
    assert f'href="{nested_url}"' in unescape(response.text)
    assert f'href="{special_file_url}"' in unescape(response.text)

    nested_response = client.get(nested_url)
    assert nested_response.status_code == 200
    assert f'href="{root_url}"' in unescape(nested_response.text)
    assert "child.txt" in nested_response.text

    file_response = client.get(nested_url + "child.txt")
    assert file_response.status_code == 200
    assert file_response.content == b"nested file"
    assert file_response.headers["cache-control"] == "no-store"

    empty_response = client.get(root_url + "empty/")
    assert empty_response.status_code == 200
    assert "This directory is empty." in empty_response.text
    assert f'href="{root_url}"' in unescape(empty_response.text)


@pytest.mark.asyncio
async def test_preview_rejects_parent_path(file_roots: tuple[Path, Path]):
    with pytest.raises(HTTPException) as exc_info:
        await preview_file(uuid4(), "../outside.txt")

    assert exc_info.value.status_code == 403


def test_preview_rejects_symlink_outside_uuid_and_missing_paths(
    file_roots: tuple[Path, Path],
    tmp_path: Path,
):
    _, preview_root = file_roots
    preview_id = uuid4()
    directory = preview_root / str(preview_id)
    directory.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (directory / "escape.txt").symlink_to(outside_file)

    client = TestClient(app)
    root_url = f"{settings.tool.browser_preview.route_path}/{preview_id}/"

    assert client.get(root_url + "escape.txt").status_code == 403
    assert client.get(root_url + "missing.txt").status_code == 404


def test_browser_preview_tool_returns_directory_usage_guidance(tmp_path: Path):
    workspace = tmp_path / "workspace"
    previews = tmp_path / "previews"
    workspace.mkdir()
    site = workspace / "site"
    (site / "pages").mkdir(parents=True)
    (site / "pages" / "index.html").write_text("<h1>Preview</h1>", encoding="utf-8")
    tool = build_browser_preview_tool(
        workspace=workspace,
        browser_preview_path=previews,
        browser_preview_base_url="http://localhost/browser_preview",
    )

    schema = as_tool(tool)["function"]
    description = schema["description"]
    path_description = schema["parameters"]["properties"]["path"]["description"]
    result = tool(BrowserPreviewArgs(path="site"))

    assert "opens navigation for the copied directory" in description
    assert "relative to the copied directory" in description
    assert "<preview_url>pages/index.html" in description
    assert path_description == "Workspace directory or HTML file to copy for browser preview"
    assert result.startswith("Directory preview URL: http://localhost/browser_preview/")
    assert "This URL opens navigation for the copied directory." in result
    assert "append its URL-encoded path relative to the copied directory" in result
    assert result.endswith("pages/index.html")
