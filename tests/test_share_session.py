import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from xpeech.agent.skills.skill import BUILTIN_SKILLS_DIR, SkillsLoader
from xpeech.agent.tools.ask_user_question import QuestionArgs
from xpeech.agent.tools.browser_preview import BrowserPreviewArgs, build_browser_preview_tool


SCRIPT_PATH = BUILTIN_SKILLS_DIR / "share-session/scripts/render_share.py"
SPEC = spec_from_file_location("render_share", SCRIPT_PATH)
renderer = module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


@pytest.fixture
def conversation():
    return {
        "title": "排查问题",
        "scope_note": "仅当前可见内容",
        "summary": "问题已经解决，后续继续观察。",
        "messages": [
            {"role": "system", "content": "SYSTEM_PRIVATE"},
            {"role": "developer", "content": "DEVELOPER_PRIVATE"},
            {"role": "user", "content": "请排查这个问题"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "REASONING_PRIVATE",
                "tool_calls": [{"function": {"name": "check_status", "arguments": '{"target":"TOOL_INPUT"}'}}],
            },
            {"role": "tool", "name": "check_status", "content": "TOOL_OUTPUT"},
            {"role": "assistant", "content": "问题已修复", "reasoning_content": "REASONING_PRIVATE"},
        ],
    }


@pytest.mark.parametrize("level", ["detailed", "conversation", "summary"])
def test_share_levels_filter_actual_html(conversation, level):
    html = renderer.render_share({**conversation, "level": level})

    assert "排查问题" in html
    assert "仅当前可见内容" in html
    for hidden in ["SYSTEM_PRIVATE", "DEVELOPER_PRIVATE", "REASONING_PRIVATE"]:
        assert hidden not in html
    for tool_content in ["check_status", "TOOL_INPUT", "TOOL_OUTPUT"]:
        assert (tool_content in html) is (level == "detailed")
    for message in ["请排查这个问题", "问题已修复"]:
        assert (message in html) is (level != "summary")
    assert (conversation["summary"] in html) is (level == "summary")
    if level == "detailed":
        assert html.index("请排查这个问题") < html.index("TOOL_INPUT") < html.index("TOOL_OUTPUT") < html.index("问题已修复")


def test_render_escapes_untrusted_content_in_all_fields():
    payload = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    html = renderer.render_share({
        "level": "detailed",
        "title": payload,
        "scope_note": payload,
        "messages": [
            {"role": "user", "content": payload},
            {"role": "assistant", "tool_calls": [{"function": {"name": payload, "arguments": payload}}]},
            {"role": "tool", "name": payload, "content": payload},
        ],
    })

    assert "<script>" not in html
    assert "<img" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize("data", [
    {"level": "invalid"},
    {"level": "summary", "summary": ""},
    {"level": "summary", "summary": ["wrong type"]},
    {"level": "conversation", "messages": []},
    {"level": "conversation", "messages": [{"role": "tool", "content": "private"}]},
    {"level": "conversation", "messages": [{"role": "user", "content": {"html": "bad"}}]},
])
def test_invalid_or_empty_share_is_rejected(data):
    with pytest.raises(ValueError):
        renderer.render_share(data)


def test_summary_does_not_require_or_parse_transcript():
    html = renderer.render_share({"level": "summary", "summary": "只有结论", "messages": "ignored"})
    assert "只有结论" in html
    assert "ignored" not in html


def test_cli_and_browser_preview_publish_only_html(tmp_path: Path, conversation):
    workspace = tmp_path / "workspace"
    share_dir = workspace / "session-shares/share-1"
    share_dir.mkdir(parents=True)
    source = share_dir / "content.json"
    source.write_text(json.dumps({**conversation, "level": "conversation"}), encoding="utf-8")
    output = share_dir / "index.html"

    subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--input", str(source), "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    preview_root = tmp_path / "previews"
    preview = build_browser_preview_tool(workspace, preview_root, "https://preview.example.test")
    url = preview(BrowserPreviewArgs(path="session-shares/share-1/index.html"))

    assert url.startswith("https://preview.example.test/")
    assert url.endswith("/index.html")
    published = list(preview_root.rglob("*.*"))
    assert len(published) == 1
    assert published[0].name == "index.html"
    html = published[0].read_text(encoding="utf-8")
    assert html == output.read_text(encoding="utf-8")
    assert "TOOL_OUTPUT" not in html


@pytest.mark.asyncio
async def test_share_skill_is_discovered_and_form_uses_supported_schema(tmp_path: Path):
    skills = await SkillsLoader(tmp_path).list_builtin_skill_info()
    [skill] = [skill for skill in skills if skill.metadata.name == "share-session"]
    assert skill.metadata.kwargs["short_description"] == "当前会话分享"
    assert skill.metadata.internal is False
    assert skill.path == str(tmp_path / "skills/share-session/SKILL.md")
    form = json.loads(skill.body.split("```json\n", 1)[1].split("```", 1)[0])
    question = QuestionArgs(question=json.dumps(form))
    validated_form = json.loads(question.question)
    options = validated_form["fields"][0]["options"]
    assert {option["value"] for option in options} == set(renderer.LEVELS)
