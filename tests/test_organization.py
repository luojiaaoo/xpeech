"""验证技能和 MCP 服务统一按用户组织查询结果进行筛选。"""

import asyncio
import json
import shlex
import sys
from types import SimpleNamespace

import jwt
import pytest
from fastapi.testclient import TestClient

from xpeech.agent.prompt.system import build_system_prompt
from xpeech.agent.server.api import app
from xpeech.agent.skills import skill as skill_module
from xpeech.agent.skills.skill import SkillMetadata, SkillsLoader
from xpeech.agent.tools import mcp_client, registry
from xpeech.config.settings import MCPServerSettings, OrganizationConfig, ToolConfig, settings
from xpeech.utils import organization as organization_module
from xpeech.utils.organization import organization_allows_access, resolve_user_organization


@pytest.mark.parametrize("prefixes", [None, "company/", [1], [""], [" "], [" company/"]])
def test_resource_organization_prefix_validation(prefixes):
    with pytest.raises(ValueError):
        SkillMetadata(name="test", description="Test", organization_prefixes=prefixes)
    with pytest.raises(ValueError):
        MCPServerSettings(command="demo", organization_prefixes=prefixes)


@pytest.mark.parametrize(
    "prefixes, organization, allowed",
    [
        ([], None, True),
        (["公司/研发/"], "公司/研发/平台", True),
        (["公司/销售/", "公司/研发/"], "公司/研发/平台", True),
        (["company/eng"], "company/engineering", True),
        (["company/eng/"], "company/engineering", False),
        (["Company/"], "company/eng", False),
        (["company/*"], "company/eng", False),
        (["company/"], None, False),
        (["company/"], "", False),
    ],
)
def test_organization_prefix_matching(prefixes, organization, allowed):
    assert organization_allows_access(prefixes, organization) is allowed


@pytest.mark.asyncio
async def test_lookup_uses_literal_user_argument_and_strips_stdout():
    user_id = "a user; $(touch should-not-exist)"
    script = "import sys; print('  公司/研发/' + sys.argv[1] + '  ')"
    command = shlex.join([sys.executable, "-c", script, "${user_id}"])
    assert await resolve_user_organization(user_id, command_template=command) == "公司/研发/" + user_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script",
    [
        "print('')",
        "import sys; print('company/admin'); sys.exit(1)",
        "import sys; sys.stdout.buffer.write(b'\\xff')",
        "import time; time.sleep(10)",
    ],
)
async def test_lookup_failures_grant_no_organization(script):
    command = shlex.join([sys.executable, "-c", script, "$user_id"])
    assert await resolve_user_organization("alice", command_template=command, timeout=0.1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [None, "", "'", "/missing-command $user_id", "echo company/admin"])
async def test_disabled_or_invalid_lookup_grants_no_organization(command):
    assert await resolve_user_organization("alice", command_template=command) is None


@pytest.mark.parametrize("source", ["builtin", "workspace"])
@pytest.mark.asyncio
async def test_skills_and_always_prompt_filter_by_organization(monkeypatch, tmp_path, source):
    root = tmp_path / "builtin" if source == "builtin" else tmp_path / "skills"
    builtin_dirs = []
    for name, extra in [
        ("public", ""),
        ("engineering", "organization_prefixes: ['公司/研发/', '公司/研发/']\n"),
        ("sales", "organization_prefixes: ['公司/销售/']\n"),
        ("other-channel", "organization_prefixes: ['公司/研发/']\nchannel_tags: [other]\n"),
        ("invalid", "organization_prefixes: ['']\n"),
    ]:
        path = root / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {name}\nalways: true\n{extra}---\n# Instructions for {name}\n")
        if source == "builtin":
            builtin_dirs.append(path.parent)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: builtin_dirs)

    public = await SkillsLoader(tmp_path).list_skill_info()
    assert {skill.metadata.name for skill in public} == {"public"}
    loader = SkillsLoader(tmp_path, user_organization="公司/研发/平台")
    skills = {skill.metadata.name: skill for skill in await loader.list_skill_info()}
    expected = {"public", "engineering"} | ({"other-channel"} if source == "workspace" else set())
    assert set(skills) == expected
    assert skills["engineering"].metadata.organization_prefixes == ["公司/研发/"]
    assert "organization_prefixes" not in skills["engineering"].metadata.kwargs
    system = await build_system_prompt(tmp_path, user_organization="公司/研发/平台")
    assert "# Instructions for engineering" in system["content"]
    assert "# Instructions for sales" not in system["content"]
    assert "<name>sales</name>" not in system["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("user_organization", [None, "公司/研发/平台", "公司/销售/"])
async def test_mcp_filters_before_connections_status_and_discovery(monkeypatch, tmp_path, user_organization):
    config = ToolConfig.model_validate({"mcpServers": {
        "public": {"command": "demo"},
        "engineering": {"command": "demo", "organization_prefixes": ["公司/研发/"]},
        "sales": {"command": "demo", "organization_prefixes": ["公司/销售/"]},
        "other-channel": {"command": "demo", "channel_tags": ["other"]},
    }})
    requested = []
    discovered = []

    async def get_registration(user_id, name, server_config, **kwargs):
        requested.append((user_id, name))

        def discover():
            discovered.append(name)
            future = asyncio.get_running_loop().create_future()
            future.set_result([])
            return future

        return SimpleNamespace(config=server_config, start_tool_discovery=discover, get_status=lambda: {"status": "ready"})

    monkeypatch.setattr(registry, "get_session_mcp_registration_from_config", get_registration)
    tools = []
    provider = SimpleNamespace(support_image=False, support_video=False, register_function_tool=tools.append)
    await registry.register_tools(
        user_id="alice", provider=provider, workspace=tmp_path, config=config,
        user_organization=user_organization,
    )
    expected = ["public"]
    if user_organization:
        expected.append("engineering" if "研发" in user_organization else "sales")
    assert requested == [("alice", name) for name in expected]
    assert discovered == expected
    status = next(tool for tool in tools if tool.__name__ == "query_mcp_status")
    assert set(json.loads(await status())["servers"]) == set(expected)
    assert "other-channel" not in status.__doc__


def test_mcp_organization_settings_are_preserved_in_cache_key(tmp_path):
    public = mcp_client.create_mcp_registration_from_config(
        "demo", MCPServerSettings(command="demo"), workspace=tmp_path,
    )
    private = mcp_client.create_mcp_registration_from_config(
        "demo", MCPServerSettings(command="demo", organization_prefixes=["公司/", "公司/"]), workspace=tmp_path,
    )
    assert private.config.organization_prefixes == ["公司/"]
    assert mcp_client._persistent_registration_key(public.config) != mcp_client._persistent_registration_key(private.config)


def test_skill_api_looks_up_authenticated_user_and_ignores_claims(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path)
    monkeypatch.setattr(settings, "organization", OrganizationConfig(command_template="lookup $user_id"))
    directories = []
    for name in ("engineering", "sales"):
        path = tmp_path / "builtin" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {name}\norganization_prefixes: [company/{name}/]\n---\n")
        directories.append(path.parent)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", lambda: directories)
    calls = []

    async def execute(command, variables, **kwargs):
        calls.append(variables["user_id"])
        return "company/engineering/team" if variables["user_id"] == "alice" else "company/sales/team"

    monkeypatch.setattr(organization_module, "execute_command_template", execute)
    client = TestClient(app)
    for user_id, expected in [("alice", "engineering"), ("bob", "sales")]:
        token = jwt.encode(
            {"sub": user_id, "exp": 9999999999, "organization": "company/sales/team"},
            settings.jwt.channels["web_client"].secret_key,
            algorithm=settings.jwt.algorithm,
            headers={"kid": "web_client"},
        )
        response = client.get(
            "/skills", params={"user_id": "bob", "organization": "company/sales/team"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert {skill["metadata"]["name"] for skill in response.json()} == {expected}
    assert calls == ["alice", "bob"]
