import re
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from xpeech.subapp.channel.web_client.models import OAuth2WebConfig, WebConfig
from xpeech.subapp.channel.web_client.routes.auth import create_auth_router


@pytest.mark.parametrize("entry_state", [None, "state-token-1234567890"])
def test_oauth2_state_is_independent_and_unique_for_each_login(entry_state):
    oauth = OAuth2WebConfig(
        provider_name="test", client_id="client", client_secret="secret",
        authorization_url="https://oauth.test/authorize", token_url="https://oauth.test/token",
        userinfo_url="https://oauth.test/userinfo", redirect_uri="http://testserver/api/auth/oauth2/callback",
        scopes=(), user_id_claim=".sub", use_pkce=True,
        token_auth_method="client_secret_post", extra_authorization_params={},
    )
    config = WebConfig(backend_url="http://backend", static_dir=Path("."), system_name="Test", oauth2=(oauth,))

    def current_user():
        return MagicMock()

    app = FastAPI()
    app.include_router(create_auth_router(config, MagicMock(), current_user, MagicMock(), MagicMock()))
    body = {"provider_name": "test"}
    if entry_state is not None:
        body["state"] = entry_state
    with TestClient(app) as api:
        states = []
        for _ in range(2):
            response = api.post("/api/auth/oauth2/qr", json=body)
            assert response.status_code == 200
            state = parse_qs(urlsplit(response.json()["authorization_url"]).query)["state"][0]
            assert re.fullmatch(r"[A-Za-z0-9_-]{43}", state)
            if entry_state is not None:
                assert entry_state not in state
            states.append(state)
        assert states[0] != states[1]
