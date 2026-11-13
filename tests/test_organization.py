"""验证技能和 MCP 服务统一按用户组织查询结果进行筛选。"""
from unittest.mock import AsyncMock

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
from xpeech.agent.skills.skill import SkillsLoader
from xpeech.agent.tools import mcp_client, registry
import xpeech.config.settings as settings_module
from xpeech.config.settings import MCPServerSettings, OrganizationConfig, ToolConfig, settings
from xpeech.utils.command_template import CommandTemplateError


@pytest.fixture
def organization_cache():
    cache = OrganizationConfig._query_user_organization
    cache.cache_clear()
    yield cache
    cache.cache_clear()


@pytest.mark.asyncio
async def test_organization_cache_reuses_results_and_merges_concurrent_queries(monkeypatch, organization_cache):
    async def execute(command, variables, **kwargs):
        await asyncio.sleep(0)
        return f"company/{variables['user_id']}"

    execute = AsyncMock(side_effect=execute)
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    config = OrganizationConfig(command_template="lookup $user_id")
    results = await asyncio.gather(*(config.resolve_user_organization("alice") for _ in range(10)))
    assert results == ["company/alice"] * 10
    assert await config.resolve_user_organization("alice") == "company/alice"
    assert execute.await_count == 1
    assert await config.resolve_user_organization("bob") == "company/bob"
    assert execute.await_count == 2
    changed = OrganizationConfig(command_template="other-lookup $user_id")
    assert await changed.resolve_user_organization("alice") == "company/alice"
    assert execute.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("first_organization", ["company/old", ""])
async def test_organization_cache_expires_after_ten_minutes(monkeypatch, organization_cache, first_organization):
    execute = AsyncMock(side_effect=[first_organization, "company/new"])
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    scheduled = []
    loop = asyncio.get_running_loop()
    call_later = loop.call_later

    def capture_expiry(delay, callback, *args, **kwargs):
        handle = call_later(delay, callback, *args, **kwargs)
        if delay == 600:
            scheduled.append((callback, args, handle))
        return handle

    monkeypatch.setattr(loop, "call_later", capture_expiry)
    config = OrganizationConfig(command_template="lookup $user_id")
    assert await config.resolve_user_organization("alice") == (first_organization or None)
    assert len(scheduled) == 1
    assert await config.resolve_user_organization("alice") == (first_organization or None)
    assert execute.await_count == 1
    callback, args, handle = scheduled.pop()
    handle.cancel()
    callback(*args)
    assert await config.resolve_user_organization("alice") == "company/new"
    assert execute.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "failed"])
async def test_unsuccessful_organization_queries_are_not_cached(monkeypatch, organization_cache, failure):
    first = CommandTemplateError(failure, operation="organization.command_template")
    execute = AsyncMock(side_effect=[first, "company/recovered"])
    monkeypatch.setattr(settings_module, "execute_command_template", execute)
    config = OrganizationConfig(command_template="lookup $user_id")
    assert await config.resolve_user_organization("alice") is None
    assert await config.resolve_user_organization("alice") == "company/recovered"
    assert execute.await_count == 2


@pytest.mark.asyncio
async def test_lookup_uses_literal_user_argument_and_strips_stdout():
    user_id = "a user; $(touch should-not-exist)"
    script = "import sys; print('  公司/研发/' + sys.argv[1] + '  ')"
    command = shlex.join([sys.executable, "-c", script, "${user_id}"])
    config = OrganizationConfig(command_template=command)
    assert await config.resolve_user_organization(user_id) == "公司/研发/" + user_id


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
    config = OrganizationConfig(command_template=command, timeout=0.1)
    assert await config.resolve_user_organization("alice") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [None, "'", "/missing-command $user_id", "echo company/admin"])
async def test_disabled_or_invalid_lookup_grants_no_organization(command):
    assert await OrganizationConfig(command_template=command).resolve_user_organization("alice") is None


@pytest.mark.parametrize("command", ["", "   "])
def test_empty_lookup_command_is_rejected(command):
    with pytest.raises(ValueError):
        OrganizationConfig(command_template=command)


@pytest.mark.parametrize("source", ["builtin", "workspace"])
@pytest.mark.asyncio
async def test_skills_and_always_prompt_filter_by_organization(monkeypatch, tmp_path, source):
    root = tmp_path / "builtin" if source == "builtin" else tmp_path / "skills"
    builtin_dirs = []
    for name, extra in [
        ("public", ""),
        ("engineering", 'filter: org("公司/研发/", "公司/研发/")\n'),
        ("sales", 'filter: org("公司/销售/")\n'),
        ("other-channel", 'filter: org("公司/研发/") and channel("other")\n'),
        ("invalid", 'filter: org("")\n'),
    ]:
        path = root / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"---\nname: {name}\nalways: true\n{extra}---\n# Instructions for {name}\n")
        if source == "builtin":
            builtin_dirs.append(path.parent)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", AsyncMock(return_value=builtin_dirs))

    # org("") 是非法过滤表达式，在元数据解析阶段即被跳过，与来源无关。
    unfiltered = {"public", "engineering", "sales", "other-channel"}
    expected_public = unfiltered if source == "workspace" else {"public"}
    expected_engineering = unfiltered if source == "workspace" else {"public", "engineering"}
    public = await SkillsLoader(tmp_path).list_skill_info()
    assert {skill.metadata.name for skill in public} == expected_public
    loader = SkillsLoader(tmp_path, user_organization="公司/研发/平台")
    skills = {skill.metadata.name: skill for skill in await loader.list_skill_info()}
    assert set(skills) == expected_engineering
    assert skills["engineering"].metadata.filter == 'org("公司/研发/", "公司/研发/")'
    assert "filter" not in skills["engineering"].metadata.kwargs
    system = await build_system_prompt(tmp_path, user_organization="公司/研发/平台")
    assert "# Instructions for engineering" in system["content"]
    if source == "workspace":
        # 工作区技能由用户自己放置，不受过滤表达式限制。
        assert "# Instructions for sales" in system["content"]
        assert "<name>sales</name>" in system["content"]
    else:
        assert "# Instructions for sales" not in system["content"]
        assert "<name>sales</name>" not in system["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("user_organization", [None, "公司/研发/平台", "公司/销售/"])
async def test_mcp_filters_before_connections_status_and_discovery(monkeypatch, tmp_path, user_organization):
    config = ToolConfig.model_validate({"mcpServers": {
        "public": {"command": "demo"},
        "engineering": {"command": "demo", "filter": 'org("公司/研发/")'},
        "sales": {"command": "demo", "filter": 'org("公司/销售/")'},
        "other-channel": {"command": "demo", "filter": 'channel("other")'},
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


def test_mcp_filter_is_preserved_in_cache_key(tmp_path):
    public = mcp_client.create_mcp_registration_from_config(
        "demo", MCPServerSettings(command="demo"), workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    private = mcp_client.create_mcp_registration_from_config(
        "demo", MCPServerSettings(command="demo", filter='org("公司/")'), workspace=tmp_path,
        skills_loader=SkillsLoader(tmp_path),
    )
    assert private.config.filter == 'org("公司/")'
    assert mcp_client._persistent_registration_key(public.config) != mcp_client._persistent_registration_key(private.config)


def test_skill_api_looks_up_authenticated_user_and_ignores_claims(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.path, "workspace_base_path", tmp_path)
    monkeypatch.setattr(settings, "organization", OrganizationConfig(command_template="lookup $user_id"))
    directories = []
    for name in ("engineering", "sales"):
        path = tmp_path / "builtin" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(f'---\nname: {name}\nfilter: org("company/{name}/")\n---\n')
        directories.append(path.parent)
    monkeypatch.setattr(skill_module, "iter_builtin_skill_dirs", AsyncMock(return_value=directories))
    calls = []

    async def execute(command, variables, **kwargs):
        calls.append(variables["user_id"])
        return "company/engineering/team" if variables["user_id"] == "alice" else "company/sales/team"

    monkeypatch.setattr(settings_module, "execute_command_template", execute)
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
