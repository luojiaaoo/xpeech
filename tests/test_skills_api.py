from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xpeech.agent.server.api import app
from xpeech.config.settings import settings
from xpeech.utils.jwt_auth import create_access_token


@pytest.fixture
def skill_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path, dict[str, str]]:
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()
    monkeypatch.setattr(settings.path, "workspace_base_path", workspace_root)
    headers = {"Authorization": f"Bearer {create_access_token()}"}
    return TestClient(app), workspace_root, headers


def _skill_file(
    name: str = "hello-skill",
    *,
    description: str = "Say hello",
    always: bool = False,
) -> bytes:
    always_line = "always: true\n" if always else ""
    return f"---\nname: {name}\ndescription: {description}\n{always_line}---\n\n# Instructions\n".encode("utf-8")


@pytest.mark.parametrize("name", ["hello-skill", "Hello_Skill", "中文 技能", "s" * 80])
def test_import_and_list_session_skill(skill_api: tuple[TestClient, Path, dict[str, str]], name: str):
    client, workspace_root, headers = skill_api

    imported = client.post(
        "/skills/import",
        params={"user_id": "session-1"},
        files={"file": ("SKILL.md", _skill_file(name, always=True), "text/markdown")},
        headers=headers,
    )

    assert imported.status_code == 201
    assert imported.json() == {
        "metadata": {
            "name": name,
            "description": "Say hello",
            "always": True,
            "internal": False,
            "kwargs": {},
        },
        "source": "workspace",
        "path": str(workspace_root / "session-1" / "skills" / name / "SKILL.md"),
        "content": _skill_file(name, always=True).decode("utf-8"),
        "body": "# Instructions",
    }
    assert (
        workspace_root / "session-1" / "skills" / name / "SKILL.md"
    ).read_bytes() == _skill_file(name, always=True)

    response = client.get("/skills", params={"user_id": "session-1"}, headers=headers)

    assert response.status_code == 200
    user_skill = next(skill for skill in response.json() if skill["metadata"]["name"] == name)
    assert user_skill == imported.json()
    assert any(skill["source"] == "builtin" for skill in response.json())


def test_skill_list_is_isolated_by_session(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, workspace_root, headers = skill_api
    imported = client.post(
        "/skills/import",
        params={"user_id": "session-a"},
        files={"file": ("SKILL.md", _skill_file(), "text/markdown")},
        headers=headers,
    )
    assert imported.status_code == 201

    other_session = client.get("/skills", params={"user_id": "session-b"}, headers=headers)

    assert other_session.status_code == 200
    assert {skill["source"] for skill in other_session.json()} == {"builtin"}
    assert all(
        Path(skill["path"]).is_relative_to(workspace_root / "session-b" / "skills")
        for skill in other_session.json()
    )
    assert "hello-skill" not in {skill["metadata"]["name"] for skill in other_session.json()}
    assert (workspace_root / "session-b").is_dir()


@pytest.mark.parametrize("params", [None, {"user_id": ""}])
def test_skill_list_without_session_returns_only_builtin_skills(
    skill_api: tuple[TestClient, Path, dict[str, str]],
    params: dict[str, str] | None,
):
    client, _, headers = skill_api
    imported = client.post(
        "/skills/import",
        params={"user_id": "session-a"},
        files={"file": ("SKILL.md", _skill_file(), "text/markdown")},
        headers=headers,
    )
    assert imported.status_code == 201

    response = client.get("/skills", params=params, headers=headers)

    assert response.status_code == 200
    assert response.json()
    assert {skill["source"] for skill in response.json()} == {"builtin"}
    assert {skill["path"] for skill in response.json()} == {""}
    assert "hello-skill" not in {skill["metadata"]["name"] for skill in response.json()}


def test_import_rejects_duplicate_skill(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, _, headers = skill_api
    request = {
        "params": {"user_id": "session-1"},
        "files": {"file": ("SKILL.md", _skill_file(), "text/markdown")},
        "headers": headers,
    }

    assert client.post("/skills/import", **request).status_code == 201
    duplicate = client.post("/skills/import", **request)

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "Skill 'hello-skill' already exists"


@pytest.mark.parametrize("name", ["../escaped", "..", ".", "/tmp/escaped", "nested/skill"])
def test_import_rejects_unsafe_skill_destination(
    skill_api: tuple[TestClient, Path, dict[str, str]],
    name: str,
):
    client, workspace_root, headers = skill_api

    response = client.post(
        "/skills/import",
        params={"user_id": "session-1"},
        files={"file": ("SKILL.md", _skill_file(name=name), "text/markdown")},
        headers=headers,
    )

    assert response.status_code == 422
    assert not (workspace_root / "session-1").exists()


def test_import_uses_loader_metadata_fallback(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, _, headers = skill_api

    response = client.post(
        "/skills/import",
        params={"user_id": "session-1"},
        files={"file": ("SKILL.md", b"---\nname: minimal-skill\n---\n# Instructions\n", "text/markdown")},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["metadata"] == {
        "name": "minimal-skill",
        "description": "minimal-skill",
        "always": False,
        "internal": False,
        "kwargs": {},
    }


def test_import_rejects_unloadable_skill(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, _, headers = skill_api

    response = client.post(
        "/skills/import",
        params={"user_id": "session-1"},
        files={"file": ("SKILL.md", b"# Missing frontmatter\n", "text/markdown")},
        headers=headers,
    )

    assert response.status_code == 422


def test_import_requires_skill_md_filename(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, _, headers = skill_api

    response = client.post(
        "/skills/import",
        params={"user_id": "session-1"},
        files={"file": ("hello-skill.skill", _skill_file(), "application/zip")},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Uploaded file must be named SKILL.md"


@pytest.mark.parametrize("user_id", ["..", "../escape", r"..\escape", "x" * 129])
def test_skill_endpoints_reject_unsafe_user_id(
    skill_api: tuple[TestClient, Path, dict[str, str]],
    user_id: str,
):
    client, _, headers = skill_api

    response = client.get("/skills", params={"user_id": user_id}, headers=headers)

    assert response.status_code == 422

    import_response = client.post(
        "/skills/import",
        params={"user_id": user_id},
        files={"file": ("SKILL.md", _skill_file(), "text/markdown")},
        headers=headers,
    )
    assert import_response.status_code == 422


def test_other_session_routes_reject_unsafe_id(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, _, headers = skill_api

    assert client.get(
        "/sessions/%2E%2E/files",
        params={"path": "example.txt"},
        headers=headers,
    ).status_code == 422
    assert client.post(
        "/answer_question",
        data={"answer": "yes"},
        headers={**headers, "x-user-id": "../escape"},
    ).status_code == 422


def test_skill_endpoints_require_jwt(skill_api: tuple[TestClient, Path, dict[str, str]]):
    client, _, _ = skill_api

    assert client.get("/skills", params={"user_id": "session-1"}).status_code == 401
