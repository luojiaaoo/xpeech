"""Custom JWT channels select public and matching Skill/MCP resources."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.api import app
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent.tools import mcp_client, registry
from xpeech.config.settings import JWTChannelConfig, JWTConfig, MCPServerSettings, ToolConfig, settings

CRM_KEY = "crm-channel-key-that-is-at-least-32-characters"
ERP_KEY = "erp-channel-key-that-is-at-least-32-characters"


@pytest.fixture
def channels(monkeypatch):
    channels = {
        "crm": JWTChannelConfig(secret_key=CRM_KEY, channel_tags=["crm", "sales"]),
        "erp": JWTChannelConfig(secret_key=ERP_KEY, channel_tags=["erp"]),
    }
    monkeypatch.setattr(settings.jwt, "channels", channels)
    return channels


@pytest.mark.parametrize("name", ["internal", "web_client", "feishu", "", "bad name"])
def test_custom_channels_reject_reserved_or_invalid_names(name):
    config = settings.jwt.model_dump()
    config["channels"] = {name: {"secret_key": CRM_KEY, "channel_tags": ["crm"]}}

    with pytest.raises(ValueError, match="channel names"):
        JWTConfig.model_validate(config)


@pytest.mark.parametrize("key_source", ["secret_key", "web_client_secret_key", "feishu_secret_key"])
def test_custom_channel_cannot_reuse_builtin_key(key_source):
    config = settings.jwt.model_dump()
    config["channels"] = {"crm": {"secret_key": config[key_source], "channel_tags": ["crm"]}}

    with pytest.raises(ValueError, match="distinct values"):
        JWTConfig.model_validate(config)


def test_custom_channels_cannot_share_keys():
    config = settings.jwt.model_dump()
    config["channels"] = {name: {"secret_key": CRM_KEY, "channel_tags": [name]} for name in ["crm", "erp"]}
    with pytest.raises(ValueError, match="distinct values"):
        JWTConfig.model_validate(config)


@pytest.mark.parametrize("channel_tags", ["crm", None, [1], [""], ["bad tag"], ["*"]])
def test_resource_tag_validation_is_consistent(channel_tags):
    with pytest.raises(ValueError):
        JWTChannelConfig(secret_key=CRM_KEY, channel_tags=channel_tags)
    with pytest.raises(ValueError):
        MCPServerSettings(command="demo", channel_tags=channel_tags)
    with pytest.raises(ValueError):
        skill_module.SkillMetadata(name="demo", description="Demo", channel_tags=channel_tags)


def test_custom_channel_can_get_token_and_use_business_endpoints(channels):
    client = TestClient(app)
    response = client.post("/token", data={"username": "crm", "password": CRM_KEY})
    assert response.status_code == 200
    token = response.json()["access_token"]
    assert jwt.get_unverified_header(token)["kid"] == "crm"
    headers = {"Authorization": f"Bearer {token}", "x-user-id": "tag-test"}
    assert client.post("/answer_question", data={"answer": "ok"}, headers=headers).status_code == 200
    assert client.get("/statistics", headers=headers).status_code == 403
    assert client.post("/internal/scheduled-run", headers=headers).status_code == 401


@pytest.mark.parametrize("claims", [{}, {"lark_user_access_token": "u-forbidden"}])
def test_custom_channel_rejects_wrong_key_or_feishu_claim(channels, claims):
    key = CRM_KEY if claims else ERP_KEY
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60), **claims},
        key,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "crm"},
    )
    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_skill_api_filters_using_server_channel_tags(
    channels,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path)
    builtin_dirs = []
    for name, channel_tags in [
        ("public", []),
        ("crm-skill", ["crm"]),
        ("erp-skill", ["erp"]),
        ("shared", ["sales", "erp"]),
    ]:
        skill_file = tmp_path / "builtin" / name / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text(f"---\nname: {name}\nchannel_tags: {channel_tags!r}\n---\n# {name}\n")
        builtin_dirs.append(skill_file.parent)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: builtin_dirs)
    user_skill_names = {"user-public", "user-crm", "user-erp"}
    for name, channel_tags in [("user-public", []), ("user-crm", ["crm"]), ("user-erp", ["erp"])]:
        skill_file = tmp_path / "tag-user" / "skills" / name / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        skill_file.write_text(f"---\nname: {name}\nchannel_tags: {channel_tags!r}\n---\n# {name}\n")

    client = TestClient(app)

    def visible_skills(channel, key, **extra_claims):
        token = jwt.encode(
            {"exp": datetime.now(UTC) + timedelta(seconds=60), **extra_claims},
            key,
            algorithm=settings.jwt.algorithm,
            headers={"kid": channel},
        )
        response = client.get(
            "/skills",
            params={"user_id": "tag-user"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        return {skill["metadata"]["name"] for skill in response.json()}

    assert visible_skills("crm", CRM_KEY, channel_tags=["erp"]) == user_skill_names | {"public", "crm-skill", "shared"}
    assert visible_skills("erp", ERP_KEY) == user_skill_names | {"public", "erp-skill", "shared"}
    assert visible_skills(
        "web_client", settings.jwt.web_client_secret_key, channel_tags=["crm"]
    ) == user_skill_names | {"public"}


@pytest.mark.asyncio
async def test_generic_skill_tags_do_not_restrict_channel_access(monkeypatch, tmp_path):
    skill_dir = tmp_path / "builtin" / "classified"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: classified\ntags: [finance, 'Long Form']\n---\n# Instructions\n")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [skill_dir])

    [skill] = await SkillsLoader(tmp_path).list_skill_info()
    assert skill.metadata.channel_tags == []
    assert skill.metadata.kwargs["tags"] == ["finance", "Long Form"]


@pytest.mark.parametrize("path, not_interactive", [("/chat", False), ("/chat_sync", True)])
def test_chat_passes_verified_channel_tags_to_registration(
    channels,
    monkeypatch,
    tmp_path,
    path,
    not_interactive,
):
    created = []

    async def run(message):
        yield {"event": "assistant", "context": "ok"}

    async def run_not_interactive(message):
        return [event async for event in run(message)]

    async def create(user_id, **kwargs):
        created.append((user_id, kwargs))
        return SimpleNamespace(workspace=tmp_path, run=run, run_not_interactive=run_not_interactive)

    monkeypatch.setattr(AgentRunner, "create", create)
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60), "channel_tags": ["erp"]},
        CRM_KEY,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "crm"},
    )
    response = TestClient(app).post(
        path,
        headers={"Authorization": f"Bearer {token}", "x-user-id": "tag-user", "x-sender-name": "test"},
        data={"content": '[{"text":"hello"}]', "session_metadata": '{"channel":"erp","channel_tags":"erp"}'},
    )
    assert response.status_code == 200
    assert created == [("tag-user", {"not_interactive": not_interactive, "channel_tags": ["crm", "sales"]})]


@pytest.mark.parametrize("source", ["builtin", "workspace"])
@pytest.mark.asyncio
async def test_always_skill_channel_tags_only_restrict_builtins(monkeypatch, tmp_path, source):
    skill_root = tmp_path / "builtin" if source == "builtin" else tmp_path / "skills"
    skill_file = skill_root / "dedicated" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: dedicated\nchannel_tags: [crm, crm]\nalways: true\n---\n# CRM instructions")
    builtin_dirs = [skill_file.parent] if source == "builtin" else []
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: builtin_dirs)
    loader = SkillsLoader(tmp_path)
    prompt = await loader.build_skills_prompt()
    if source == "builtin":
        assert prompt == ""
    else:
        assert "# CRM instructions" in prompt
    loader = SkillsLoader(tmp_path, channel_tags=["crm"])
    prompt = await loader.build_skills_prompt()
    assert "# CRM instructions" in prompt
    assert "channel_tags='crm'" in prompt
    skill = (await loader.list_skill_info())[0]
    assert skill.metadata.channel_tags == ["crm"]
    assert "channel_tags" not in skill.metadata.kwargs


@pytest.mark.parametrize("channel_tags_yaml", ["crm", "null", "[1]", "['']"])
@pytest.mark.asyncio
async def test_invalid_skill_channel_tags_are_skipped(monkeypatch, tmp_path, channel_tags_yaml):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)
    skill_file = tmp_path / "skills" / "invalid" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(f"---\nname: invalid\nchannel_tags: {channel_tags_yaml}\n---\n# Invalid")
    assert await SkillsLoader(tmp_path).list_skill_info() == []


@pytest.mark.parametrize("channel_tags, expected", [([], ["public"]), (["crm"], ["public", "crm", "shared"])])
@pytest.mark.parametrize("not_interactive", [False, True])
@pytest.mark.asyncio
async def test_mcp_only_registers_public_and_matching_servers(
    monkeypatch, tmp_path, channel_tags, expected, not_interactive
):
    config = ToolConfig.model_validate(
        {
            "mcpServers": {
                "public": {"command": "demo"},
                "crm": {"command": "demo", "channel_tags": ["crm"]},
                "erp": {"command": "demo", "channel_tags": ["erp"]},
                "shared": {"command": "demo", "channel_tags": ["crm", "erp"]},
            }
        }
    )
    requested = []
    registered = []

    async def get_registration(user_id, name, server_config, **kwargs):
        requested.append(name)
        return name

    def register_tool(tool_type="function"):
        if tool_type == "mcp":

            async def register(registration):
                registered.append(registration)

            return register
        return lambda *args, **kwargs: None

    monkeypatch.setattr(registry, "get_session_mcp_registration_from_config", get_registration)
    provider = SimpleNamespace(support_image=False, support_video=False, register_tool=register_tool)
    await registry.register_tools(
        user_id="tag-user",
        provider=provider,
        workspace=tmp_path,
        config=config,
        channel_tags=channel_tags,
        not_interactive=not_interactive,
    )
    assert requested == registered == expected


def test_mcp_preserves_channel_tags_in_registration_and_cache_key():
    public = mcp_client.create_mcp_registration_from_config("demo", MCPServerSettings(command="demo"))
    dedicated = mcp_client.create_mcp_registration_from_config(
        "demo",
        MCPServerSettings(command="demo", channel_tags=["crm", "crm"]),
    )
    assert dedicated.config.channel_tags == ["crm"]
    assert mcp_client._persistent_registration_key(public.config) != mcp_client._persistent_registration_key(
        dedicated.config
    )
