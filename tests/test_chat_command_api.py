from unittest.mock import AsyncMock

import httpx
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
    matches = client.get("/files/search", headers=headers, params={"query": "file-"}).json()
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


def test_file_search_returns_only_recursive_prefix_matches(chat_command_client, tmp_path):
    client, workspace, headers = chat_command_client
    (workspace / "docs" / "nested").mkdir(parents=True)
    (workspace / "Report.txt").touch()
    (workspace / "docs" / "nested" / "report-final.txt").touch()
    (workspace / "docs" / "old-report.txt").touch()
    outside = tmp_path / "bob"
    outside.mkdir()
    (outside / "report-secret.txt").touch()
    (workspace / "report-escape").symlink_to(outside, target_is_directory=True)
    response = client.get("/files/search", headers=headers, params={"query": "rep", "user_id": "bob"})
    assert response.status_code == 200
    assert {item["path"] for item in response.json()["entries"]} == {"Report.txt", "docs/nested/report-final.txt"}
    nested = client.get("/files/search", headers=headers, params={"query": "docs/rep"}).json()
    assert [item["path"] for item in nested["entries"]] == ["docs/nested/report-final.txt"]
    assert client.get("/files/search", headers=headers, params={"query": "docs/"}).json() == {"entries": [], "truncated": False}
    assert client.get("/files/search", headers=headers, params={"query": "../bob/rep"}).status_code == 403
    assert client.get("/files/search", headers=headers, params={"query": ""}).status_code == 422
    assert client.get("/files/search", headers=headers).status_code == 422
    assert client.get("/files/search", params={"query": "rep"}).status_code == 401


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
