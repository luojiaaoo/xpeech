from importlib import import_module
from pathlib import Path

from fastapi.testclient import TestClient

web_client_app = import_module("xpeech.channel.web_client.app")
WebConfig = web_client_app.WebConfig
create_app = web_client_app.create_app


def test_admin_can_delete_a_user_but_not_themselves(tmp_path: Path):
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_users",
        )
    )

    with TestClient(app) as client:
        assert client.delete("/api/admin/users/1").status_code == 401

        login = client.post(
            "/api/auth/login",
            json={"session_id": "admin", "password": "admin123456"},
        )
        admin_id = login.json()["id"]
        assert client.delete(f"/api/admin/users/{admin_id}").status_code == 400

        created = client.post(
            "/api/admin/users",
            json={
                "username": "测试Viewer",
                "session_id": "viewer-session",
                "password": "viewer123456",
                "is_admin": False,
            },
        )
        user_id = created.json()["id"]
        deleted = client.delete(f"/api/admin/users/{user_id}")

        assert deleted.status_code == 204
        assert client.delete(f"/api/admin/users/{user_id}").status_code == 404
        assert [user["id"] for user in client.get("/api/admin/users").json()] == [
            admin_id
        ]


def test_regular_user_can_change_their_own_password(tmp_path: Path):
    app = create_app(
        WebConfig(
            backend_url="http://backend.test",
            database_path=tmp_path / "password-users.db",
            static_dir=tmp_path / "missing-static",
            system_name="Test Assistant",
            cookie_name="xpeech_session_password",
        )
    )

    with TestClient(app) as client:
        assert client.patch(
            "/api/auth/password",
            json={"current_password": "old-password", "new_password": "new-password"},
        ).status_code == 401
        client.post(
            "/api/auth/login",
            json={"session_id": "admin", "password": "admin123456"},
        )
        client.post(
            "/api/admin/users",
            json={
                "username": "普通用户",
                "session_id": "regular-user",
                "password": "old-password",
                "is_admin": False,
            },
        )
        client.post("/api/auth/logout")
        login = client.post(
            "/api/auth/login",
            json={"session_id": "regular-user", "password": "old-password"},
        )
        wrong_password = client.patch(
            "/api/auth/password",
            json={"current_password": "wrong-password", "new_password": "new-password"},
        )
        changed = client.patch(
            "/api/auth/password",
            json={"current_password": "old-password", "new_password": "new-password"},
        )

        assert login.status_code == 200
        assert login.json()["is_admin"] is False
        assert wrong_password.status_code == 400
        assert wrong_password.json()["detail"] == "当前密码错误"
        assert changed.status_code == 204
        assert client.get("/api/auth/me").status_code == 200

        client.post("/api/auth/logout")
        assert client.post(
            "/api/auth/login",
            json={"session_id": "regular-user", "password": "old-password"},
        ).status_code == 401
        assert client.post(
            "/api/auth/login",
            json={"session_id": "regular-user", "password": "new-password"},
        ).status_code == 200
