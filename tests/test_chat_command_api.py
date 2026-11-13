from unittest.mock import AsyncMock

import httpx
import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import xpeech.agent.skills.skill as skill_module
from xpeech.agent.server.api import app
from xpeech.config.settings import settings
from xpeech.subapp.channel.web_client.dao import User
from xpeech.subapp.channel.web_client.history import JsonlEventHistoryRepository
from xpeech.subapp.channel.web_client.routes import proxy
from xpeech.utils.channel_jwt import UserJWTAuth, UserTokenParams, create_token


@pytest.fixture
def chat_command_client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", AsyncMock(return_value=[]))
    workspace = tmp_path / "alice"
    workspace.mkdir()
    token = create_token(UserTokenParams("web_client", user_id="alice"))
    return TestClient(app), workspace, {"Authorization": f"Bearer {token}"}


def test_catalog_returns_public_skills_and_real_commands(chat_command_client):
    client, workspace, headers = chat_command_client
    for name, internal in (("demo", False), ("private", True)):
        path = workspace / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {name}\ndescription: Describe {name}\ninternal: {str(internal).lower()}\n---\nInstructions")
    response = client.get("/chat_command", headers=headers, params={"user_id": "bob"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert {item["name"] for item in response.json()["commands"]} == {"help", "new", "clear"}
    assert response.json()["skills"] == [{
        "name": "demo", "description": "Describe demo", "path": str(workspace / "skills/demo/SKILL.md"),
    }]
    assert client.get("/chat_command").status_code == 401


def test_skill_catalog_orders_filtered_builtins_then_workspace_then_public_builtins(chat_command_client, tmp_path, monkeypatch):
    client, workspace, headers = chat_command_client
    builtins = []
    for name, expression in (
        ("z-public", None), ("channel-match", 'channel("web_client")'),
        ("a-public", None), ("org-match", 'org("公司/研发/")'),
        ("denied", 'channel("crm")'),
    ):
        directory = tmp_path / "builtins" / name
        directory.mkdir(parents=True)
        filter_line = f"filter: {expression}\n" if expression else ""
        (directory / "SKILL.md").write_text(f"---\nname: {name}\n{filter_line}---\nBody")
        builtins.append(directory)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", AsyncMock(return_value=builtins))
    monkeypatch.setattr(type(settings.extra_user_info), "resolve_user_organization", AsyncMock(return_value="公司/研发/平台"))
    for name, expression in (("workspace-denied", "False"), ("workspace-match", "True"), ("workspace-public", None)):
        path = workspace / "skills" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        filter_line = f"filter: '{expression}'\n" if expression else ""
        path.write_text(f"---\nname: {name}\n{filter_line}---\nBody")
    expected = ["channel-match", "org-match", "workspace-denied", "workspace-match", "workspace-public", "z-public", "a-public"]
    assert [item["metadata"]["name"] for item in client.get("/skills", headers=headers).json()] == expected
    assert [item["name"] for item in client.get("/chat_command", headers=headers).json()["skills"]] == expected


def test_file_choices_use_name_prefixes_and_stay_in_the_users_workspace(chat_command_client, tmp_path):
    client, workspace, headers = chat_command_client
    (workspace / "Report.txt").write_text("report")
    (workspace / "old-report.txt").write_text("old report")
    (workspace / "Reports").mkdir()
    (workspace / "Reports" / "中文 文件.txt").write_text("nested")
    outside = tmp_path / "bob"
    outside.mkdir()
    (outside / "secret.txt").write_text("private")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    (workspace / "secret-link.txt").symlink_to(outside / "secret.txt")
    (workspace / "loop").symlink_to(workspace / "loop")
    response = client.get("/files/list", params={"query": "rep", "user_id": "bob"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"entries": [
        {"name": "Reports", "path": "Reports", "is_directory": True},
        {"name": "Report.txt", "path": "Report.txt", "is_directory": False},
    ], "truncated": False}
    entries = client.get("/files/list", headers=headers).json()["entries"]
    assert {item["name"] for item in entries} == {"Report.txt", "old-report.txt", "Reports"}
    nested = client.get("/files/list", params={"path": "Reports", "query": "中文"}, headers=headers)
    assert nested.json()["entries"][0]["path"] == "Reports/中文 文件.txt"
    for path in ("../bob", str(outside), "escape"):
        assert client.get("/files/list", params={"path": path}, headers=headers).status_code == 403
    assert client.get("/files/list", params={"path": "missing"}, headers=headers).status_code == 404
    assert client.get("/files/list").status_code == 401


def test_file_choices_limit_results_and_handle_new_workspaces(chat_command_client):
    client, workspace, headers = chat_command_client
    for index in range(105):
        (workspace / f"file-{index:03}.txt").touch()
    result = client.get("/files/list", headers=headers).json()
    assert len(result["entries"]) == 100
    assert result["truncated"] is True
    matches = client.get("/files/search", headers=headers, params={"query": "file-*"}).json()
    assert len(matches["entries"]) == 100
    assert matches["truncated"] is True
    second_page = client.get("/files/list", headers=headers, params={"offset": 100}).json()
    assert len(second_page["entries"]) == 5
    assert second_page["truncated"] is False
    assert not {item["path"] for item in result["entries"]} & {item["path"] for item in second_page["entries"]}
    assert client.get("/files/list", headers=headers, params={"offset": -1}).status_code == 422
    token = create_token(UserTokenParams("web_client", user_id="new-user"))
    assert client.get("/files/list", headers={"Authorization": f"Bearer {token}"}).json() == {
        "entries": [], "truncated": False,
    }


def test_file_routes_reject_workspace_roots_linked_to_another_user(chat_command_client, tmp_path):
    client, _, _ = chat_command_client
    other = tmp_path / "bob"
    other.mkdir()
    (other / "private.txt").write_text("private")
    (tmp_path / "alias").symlink_to(other, target_is_directory=True)
    token = create_token(UserTokenParams("web_client", user_id="alias"))
    headers = {"Authorization": f"Bearer {token}"}
    for endpoint, params in (
        ("/files/list", {}), ("/files/search", {"query": "*"}), ("/files", {"path": "private.txt"}),
        ("/skills", {}), ("/chat_command", {}),
    ):
        assert client.get(endpoint, headers=headers, params=params).status_code == 403


def test_file_search_matches_names_recursively_and_stays_in_user_workspace(chat_command_client, tmp_path):
    client, workspace, headers = chat_command_client
    (workspace / "docs" / "nested").mkdir(parents=True)
    (workspace / "Report.txt").touch()
    (workspace / "docs" / "nested" / "report-final.txt").touch()
    (workspace / "docs" / "old-report.txt").touch()
    outside = tmp_path / "bob"
    outside.mkdir()
    (outside / "report-secret.txt").touch()
    (workspace / "report-escape").symlink_to(outside, target_is_directory=True)
    (workspace / "report-secret.txt").symlink_to(outside / "report-secret.txt")
    (workspace / "report-loop").symlink_to(workspace / "report-loop")
    response = client.get("/files/search", headers=headers, params={"query": "rep*", "user_id": "bob"})
    assert response.status_code == 200
    assert {item["path"] for item in response.json()["entries"]} == {"Report.txt", "docs/nested/report-final.txt"}
    folders = client.get("/files/search", headers=headers, params={"query": "docs"}).json()["entries"]
    assert [(item["path"], item["is_directory"]) for item in folders] == [("docs", True)]
    assert client.get("/files/search", headers=headers, params={"query": "../bob/rep"}).status_code == 403
    assert client.get("/files/search", headers=headers, params={"query": ""}).status_code == 422
    assert client.get("/files/search", headers=headers).status_code == 422
    assert client.get("/files/search", params={"query": "rep"}).status_code == 401


@pytest.mark.parametrize(("query", "expected"), [
    ("*.PY", {"main.py", "src/a.py", "src/nested/b.py", "docs/aabbcc.py"}),
    ("src/*.py", set()),
    ("**/*.py", set()),
    ("[ab].py", {"src/a.py", "src/nested/b.py"}),
    ("report?.txt", {"report1.txt"}),
    ("main.py", {"main.py"}),
    ("main", set()),
    ("*bb*", {"docs/aabbcc.py"}),
    ("src", {"src"}),
    ("*src*", {"src"}),
    ("a**.py", {"src/a.py", "docs/aabbcc.py"}),
    ("[", set()),
])
def test_file_search_fnmatch_patterns_match_only_names(chat_command_client, query, expected):
    client, workspace, headers = chat_command_client
    for name in ("main.py", "main.py.bak", "src/a.py", "src/nested/b.py", "report1.txt", "report12.txt", "docs/aabbcc.py"):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    response = client.get("/files/search", headers=headers, params={"query": query})
    assert response.status_code == 200
    assert {item["path"] for item in response.json()["entries"]} == expected


def test_search_prunes_internal_directories_but_tree_can_browse_them(chat_command_client, monkeypatch):
    client, workspace, headers = chat_command_client
    excluded = {".git", ".venv", ".local_lib", "node_modules", "__pycache__", ".cache"}
    for directory in sorted(excluded):
        path = workspace / directory / "internal.txt"
        path.parent.mkdir(parents=True)
        path.touch()
    for name in ("docs/report.md", "docs/node_modules/internal.txt", ".notes/readme.md", "build/result.pdf", "dist/report.html"):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (workspace / "git-alias").symlink_to(workspace / ".git", target_is_directory=True)
    (workspace / "internal-link.txt").symlink_to(workspace / ".git" / "internal.txt")
    walk = anyio.Path.walk
    visited = []

    async def track_walk(path, **kwargs):
        async for directory, dirs, files in walk(path, **kwargs):
            visited.append(directory.relative_to(workspace).parts)
            yield directory, dirs, files

    monkeypatch.setattr(anyio.Path, "walk", track_walk)
    response = client.get("/files/search", headers=headers, params={"query": "*"})
    assert response.status_code == 200
    paths = {item["path"] for item in response.json()["entries"]}
    assert {"docs/report.md", ".notes/readme.md", "build/result.pdf", "dist/report.html"} <= paths
    assert all(not excluded.intersection(path.split("/")) for path in paths)
    assert "git-alias" not in paths
    assert "internal-link.txt" not in paths
    assert ("docs",) in visited
    assert all(not excluded.intersection(parts) for parts in visited)
    assert ("git-alias",) not in visited
    for query in (".git", "node_modules", "git-alias", "internal-link.txt"):
        assert client.get("/files/search", headers=headers, params={"query": query}).json() == {
            "entries": [], "truncated": False,
        }
    listed = client.get("/files/list", headers=headers).json()["entries"]
    assert excluded <= {item["name"] for item in listed}
    for directory in sorted(excluded):
        entries = client.get("/files/list", headers=headers, params={"path": directory}).json()["entries"]
        assert [item["name"] for item in entries] == ["internal.txt"]


def test_search_stops_walking_after_enough_matches(chat_command_client, monkeypatch):
    client, workspace, headers = chat_command_client
    names = [f"file-{index}.txt" for index in range(120)]
    for name in names:
        (workspace / name).touch()

    async def bounded_walk(path, **kwargs):
        yield path, [], names
        raise AssertionError("Search must stop walking after finding 101 matches")

    monkeypatch.setattr(anyio.Path, "walk", bounded_walk)
    result = client.get("/files/search", headers=headers, params={"query": "*.txt"}).json()
    assert len(result["entries"]) == 100
    assert result["truncated"] is True


@pytest.mark.parametrize("identity", ["cookie", "jwt"])
@pytest.mark.parametrize("endpoint", ["chat_command", "files/list", "files/search"])
@pytest.mark.parametrize("status_code", [200, 403])
def test_chat_command_proxy_preserves_auth_query_and_errors(tmp_path, monkeypatch, identity, endpoint, status_code):
    token = create_token(UserTokenParams("web_client", user_id="alice"))
    user = UserJWTAuth(token) if identity == "jwt" else User(user_id="alice", username="Alice", password_hash="unused")

    def current_user():
        return user

    web_app = FastAPI()
    web_app.include_router(proxy.create_proxy_router(
        "http://backend.test", current_user, current_user, JsonlEventHistoryRepository(tmp_path), "test_cookie",
    ))

    def respond(request):
        assert request.url.path == f"/{endpoint}"
        assert request.url.params["query"] == "中文"
        forwarded = request.headers["authorization"].removeprefix("Bearer ")
        assert UserJWTAuth(forwarded).user_id == "alice"
        if identity == "jwt":
            assert forwarded == token
        return httpx.Response(status_code, json={"result": "choices"})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    response = TestClient(web_app).get(f"/api/{endpoint}", params={"query": "中文"})
    assert response.status_code == status_code
    assert response.json() == {"result": "choices"}
    assert response.headers["cache-control"] == "no-store"
