"""Custom JWT channels select public and matching Skill/MCP resources."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.prompt.system import build_system_prompt
from xpeech.agent.runner import AgentRunner
from xpeech.agent.server.api import app
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent.tools import mcp_client, registry
from xpeech.config.settings import JWTChannelConfig, JWTConfig, MCPServerSettings, ToolConfig, settings

CRM_KEY = "crm-channel-key-that-is-at-least-32-characters"
ERP_KEY = "erp-channel-key-that-is-at-least-32-characters"
UUID7_SESSION_ID = "018f0000-0000-7000-8000-000000000001"


@pytest.fixture
def channels(monkeypatch):
    channels = {
        **settings.jwt.channels,
        "crm": JWTChannelConfig(secret_key=CRM_KEY),
        "erp": JWTChannelConfig(secret_key=ERP_KEY),
    }
    monkeypatch.setattr(settings.jwt, "channels", channels)
    return channels


@pytest.mark.parametrize("name", ["internal", "", "bad name"])
def test_custom_channels_reject_reserved_or_invalid_names(name):
    config = settings.jwt.model_dump()
    config["channels"] = {name: {"secret_key": CRM_KEY}}

    with pytest.raises(ValueError, match="channel names"):
        JWTConfig.model_validate(config)


@pytest.mark.parametrize("key_source", ["internal", "web_client", "feishu"])
def test_custom_channel_cannot_reuse_builtin_key(key_source):
    config = settings.jwt.model_dump()
    secret_key = config["secret_key"] if key_source == "internal" else config["channels"][key_source]["secret_key"]
    config["channels"]["crm"] = {"secret_key": secret_key}

    with pytest.raises(ValueError, match="distinct values"):
        JWTConfig.model_validate(config)


def test_custom_channels_cannot_share_keys():
    config = settings.jwt.model_dump()
    config["channels"] = {name: {"secret_key": CRM_KEY} for name in ["crm", "erp"]}
    with pytest.raises(ValueError, match="distinct values"):
        JWTConfig.model_validate(config)


@pytest.mark.parametrize("session_id", ["default", UUID7_SESSION_ID])
def test_custom_channel_can_get_token_and_use_business_endpoints(channels, session_id):
    client = TestClient(app)
    response = client.post("/token", auth=("crm", CRM_KEY), data={"username": "tag-test"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    assert jwt.get_unverified_header(token)["kid"] == "crm"
    headers = {
        "Authorization": f"Bearer {token}",
        "x-session-id": session_id,
    }
    assert client.post("/answer_question", data={"answer": "ok"}, headers=headers).status_code == 200
    assert client.get("/statistics", headers=headers).status_code == 401
    assert client.post("/internal/scheduled-run", headers=headers).status_code == 401


def test_custom_channel_rejects_wrong_key(channels):
    token = jwt.encode(
        {"exp": datetime.now(UTC) + timedelta(seconds=60)},
        ERP_KEY,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "crm"},
    )
    response = TestClient(app).post(
        "/answer_question",
        data={"answer": "ok"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_skill_api_filters_using_channel(
    channels,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path)
    builtin_dirs = []
    for name, filter_expr in [
        ("public", None),
        ("crm-skill", 'channel("crm")'),
        ("erp-skill", 'channel("erp")'),
        ("shared", 'channel("crm", "erp")'),
        ("web-skill", 'channel("web_client")'),
    ]:
        skill_file = tmp_path / "builtin" / name / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        filter_line = f"filter: {filter_expr}\n" if filter_expr else ""
        skill_file.write_text(f"---\nname: {name}\n{filter_line}---\n# {name}\n")
        builtin_dirs.append(skill_file.parent)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: builtin_dirs)
    for name, filter_expr in [("user-public", None), ("user-crm", 'channel("crm")'), ("user-erp", 'channel("erp")')]:
        skill_file = tmp_path / "tag-user" / "skills" / name / "SKILL.md"
        skill_file.parent.mkdir(parents=True)
        filter_line = f"filter: {filter_expr}\n" if filter_expr else ""
        skill_file.write_text(f"---\nname: {name}\n{filter_line}---\n# {name}\n")

    client = TestClient(app)

    def visible_skills(kid, key, **extra_claims):
        token = jwt.encode(
            {"exp": datetime.now(UTC) + timedelta(seconds=60), "sub": "tag-user", **extra_claims},
            key,
            algorithm=settings.jwt.algorithm,
            headers={"kid": kid},
        )
        response = client.get(
            "/skills",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        return {skill["metadata"]["name"] for skill in response.json()}

    # 工作区技能由用户自己放置，不受过滤表达式限制；内置技能仍按渠道过滤。
    all_user_skills = {"user-public", "user-crm", "user-erp"}
    assert visible_skills(
        "crm", CRM_KEY, channel="erp"
    ) == all_user_skills | {"public", "crm-skill", "shared"}
    assert visible_skills("erp", ERP_KEY) == all_user_skills | {"public", "erp-skill", "shared"}
    assert visible_skills(
        "web_client", settings.jwt.channels["web_client"].secret_key, channel="crm"
    ) == all_user_skills | {"public", "web-skill"}


@pytest.mark.asyncio
async def test_generic_skill_tags_do_not_restrict_channel_access(monkeypatch, tmp_path):
    skill_dir = tmp_path / "builtin" / "classified"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: classified\ntags: [finance, 'Long Form']\n---\n# Instructions\n")
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: [skill_dir])

    [skill] = await SkillsLoader(tmp_path).list_skill_info()
    assert skill.metadata.filter is None
    assert skill.metadata.kwargs["tags"] == ["finance", "Long Form"]


@pytest.mark.parametrize("path, not_interactive", [("/chat", False), ("/chat_sync", True)])
@pytest.mark.parametrize("session_id", ["default", UUID7_SESSION_ID])
def test_chat_passes_verified_channel_to_registration(
    channels,
    monkeypatch,
    tmp_path,
    path,
    not_interactive,
    session_id,
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
        {
            "exp": datetime.now(UTC) + timedelta(seconds=60),
            "sub": "tag-user",
            "channel": "erp",

        },
        CRM_KEY,
        algorithm=settings.jwt.algorithm,
        headers={"kid": "crm"},
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "x-session-id": session_id,
        "x-sender-name": "test",
    }
    response = TestClient(app).post(
        path,
        headers=headers,
        data={
            "content": '[{"text":"hello"}]',
            "session_metadata": '{"channel":"erp","channel_name":"erp","channel_tag":"erp"}',
        },
    )
    assert response.status_code == 200
    assert created == [("tag-user", {"not_interactive": not_interactive, "channel": "crm"})]


@pytest.mark.parametrize("source", ["builtin", "workspace"])
@pytest.mark.asyncio
async def test_always_skill_filter_restricts_builtin_skills_only(monkeypatch, tmp_path, source):
    skill_root = tmp_path / "builtin" if source == "builtin" else tmp_path / "skills"
    skill_file = skill_root / "dedicated" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text('---\nname: dedicated\nfilter: channel("crm")\nalways: true\n---\n# CRM instructions')
    builtin_dirs = [skill_file.parent] if source == "builtin" else []
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: builtin_dirs)
    loader = SkillsLoader(tmp_path)
    prompt = await loader.build_skills_prompt()
    if source == "builtin":
        # 内置技能受过滤表达式限制。
        assert prompt == ""
        loader = SkillsLoader(tmp_path, channel="crm")
        prompt = await loader.build_skills_prompt()
    # 工作区技能由用户自己放置，不受访问过滤表达式限制。
    assert "# CRM instructions" in prompt
    skill = (await loader.list_skill_info())[0]
    assert skill.metadata.filter == 'channel("crm")'
    assert "filter" not in skill.metadata.kwargs
    system_prompt = await build_system_prompt(tmp_path, channel="crm")
    assert "# CRM instructions" in system_prompt["content"]


@pytest.mark.parametrize("filter_yaml", ["channel()", "[1]", "['crm']", "import os", "123"])
@pytest.mark.asyncio
async def test_invalid_skill_filters_are_skipped(monkeypatch, tmp_path, filter_yaml):
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", list)
    skill_file = tmp_path / "skills" / "invalid" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(f"---\nname: invalid\nfilter: {filter_yaml}\n---\n# Invalid")
    assert await SkillsLoader(tmp_path).list_skill_info() == []


@pytest.mark.parametrize(
    "channel, expected",
    [
        (None, ["public"]),
        ("crm", ["public", "crm", "shared"]),
        ("erp", ["public", "erp", "shared"]),
        ("sales", ["public"]),
        ("cr", ["public"]),
        ("CRM", ["public"]),
    ],
)
@pytest.mark.parametrize("not_interactive", [False, True])
@pytest.mark.parametrize("include_public", [False, True])
@pytest.mark.asyncio
async def test_mcp_only_registers_public_and_matching_servers(
    monkeypatch, tmp_path, channel, expected, not_interactive, include_public
):
    async def shell():
        return "unused"

    monkeypatch.setattr(registry, "build_shell_tools", lambda **_kwargs: shell)
    config = ToolConfig.model_validate(
        {
            "mcpServers": {
                "public": {"command": "demo"},
                "crm": {"command": "demo", "filter": 'channel("crm")'},
                "erp": {"command": "demo", "filter": 'channel("erp")'},
                "shared": {"command": "demo", "filter": 'channel("crm", "erp")'},
            }
        }
    )
    if not include_public:
        config.mcp_servers.pop("public")
        expected = [name for name in expected if name != "public"]
    requested = []
    registered = []
    function_tools = []
    for name, server_config in config.mcp_servers.items():
        server_config.description = f"purpose-{name}"

    async def get_registration(user_id, name, server_config, **kwargs):
        requested.append(name)
        async def tool():
            return name

        task = asyncio.get_running_loop().create_future()
        task.set_result([mcp_client.MCPToolBinding(schema={"function": {"name": name}}, func=tool)])
        return SimpleNamespace(
            config=server_config,
            start_tool_discovery=lambda: task,
            get_status=lambda: {"status": "ready"},
        )

    def register_mcp_tool(_tool, *, tool_json):
        registered.append(tool_json["function"]["name"])

    monkeypatch.setattr(registry, "get_session_mcp_registration_from_config", get_registration)
    provider = SimpleNamespace(
        support_image=False,
        support_video=False,
        register_function_tool=function_tools.append,
        register_mcp_tool=register_mcp_tool,
    )
    await registry.register_tools(
        user_id="tag-user",
        provider=provider,
        workspace=tmp_path,
        config=config,
        channel=channel,
        not_interactive=not_interactive,
    )
    assert requested == registered == expected
    status_tool = next((tool for tool in function_tools if tool.__name__ == "query_mcp_status"), None)
    if not expected:
        assert status_tool is None
        return
    assert status_tool is not None
    assert set(json.loads(await status_tool())["servers"]) == set(expected)
    for name in config.mcp_servers:
        assert (f"purpose-{name}" in status_tool.__doc__) is (name in expected)


def test_mcp_preserves_filter_in_registration_and_cache_key(tmp_path):
    public = mcp_client.create_mcp_registration_from_config(
        "demo", MCPServerSettings(command="demo"), workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    dedicated = mcp_client.create_mcp_registration_from_config(
        "demo",
        MCPServerSettings(command="demo", filter='channel("crm")'),
        workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    assert dedicated.config.filter == 'channel("crm")'
    assert mcp_client._persistent_registration_key(public.config) != mcp_client._persistent_registration_key(
        dedicated.config
    )
