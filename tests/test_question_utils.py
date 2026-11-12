import json

import pytest

from xpeech.agent.tools.ask_user_question import QuestionArgs
from xpeech.utils.question import (
    build_execution_question,
    validate_confirm_question_json,
    validate_question_form_json,
)


@pytest.mark.parametrize("serialized", [False, True])
def test_confirmation_question_has_no_form_fields(serialized):
    question = {"type": "confirm", "title": "执行确认", "description": "是否继续？\n请确认执行参数。"}
    assert validate_confirm_question_json(json.dumps(question) if serialized else question) == {
        **question, "tool_call_id": None, "tool_name": None, "arguments": None, "timeout_seconds": None,
    }


@pytest.mark.parametrize("extra", [{"fields": []}, {"submit_label": "执行"}, {"subtitle": "旧副标题"}, {"confirm_label": "运行"}, {"view_label": "输出"}])
def test_confirmation_question_rejects_form_options(extra):
    with pytest.raises(ValueError, match="确认问题 JSON 校验失败"):
        validate_confirm_question_json({"type": "confirm", **extra})


def test_missing_question_type_still_defaults_to_form():
    question = {"fields": []}
    assert validate_question_form_json(question)["type"] == "form"
    assert "type" not in question
    with pytest.raises(ValueError):
        validate_question_form_json({"type": "unknown", "fields": []})


def test_execution_confirmation_preserves_context_without_form_fields():
    arguments = {"command": "echo hello"}
    question = validate_confirm_question_json(build_execution_question("shell", arguments, "输出 hello"))
    assert question["type"] == "confirm"
    assert question["title"] == "执行确认"
    assert question["tool_name"] == "shell"
    assert question["description"] == "输出 hello"
    assert question["arguments"] == arguments
    assert question["timeout_seconds"] == 300
    assert "subtitle" not in question
    assert "fields" not in question
    assert arguments == {"command": "echo hello"}


def test_ask_user_question_still_builds_a_form_with_supplementary_input():
    question = json.loads(QuestionArgs(question='{"fields": []}').validate_and_format())
    assert question["type"] == "form"
    assert question["fields"][-1]["name"] == "user_customization_input"


@pytest.mark.parametrize("serialized", [False, True])
def test_validators_reject_the_other_question_type(serialized):
    form = {"type": "form", "fields": []}
    confirmation = {"type": "confirm"}
    with pytest.raises(ValueError, match="问题表单 JSON 校验失败"):
        validate_question_form_json(json.dumps(confirmation) if serialized else confirmation)
    with pytest.raises(ValueError, match="确认问题 JSON 校验失败"):
        validate_confirm_question_json(json.dumps(form) if serialized else form)


@pytest.mark.parametrize("validator", [validate_question_form_json, validate_confirm_question_json])
def test_validators_report_invalid_json(validator):
    with pytest.raises(ValueError, match="JSON 格式错误"):
        validator("{")


def test_ask_user_question_rejects_confirm_type_as_a_form_validation_error():
    with pytest.raises(ValueError, match="问题表单 JSON 校验失败"):
        QuestionArgs(question='{"type": "confirm"}').validate_and_format()


def test_execution_confirmation_uses_custom_copy():
    question = validate_confirm_question_json(build_execution_question(
        "shell", {"command": "echo hello"}, "输出 hello",
        title="运行测试命令",
    ))
    assert question["title"] == "运行测试命令"
    assert question["tool_name"] == "shell"
