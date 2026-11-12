from __future__ import annotations

import json
from typing import Any

from ....utils.question import validate_confirm_question_json, validate_question_form_json
from .formatting import ensure_markdown, plain_text

def build_feishu_background_task_card(
    content: str,
    schedule_type: str,
    schedule_value: str,
) -> dict[str, Any]:
    """Build a Feishu card for a background Agent task result."""
    return {
        "schema": "2.0",
        "header": {
            "title": plain_text("定时任务执行结果"),
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": ensure_markdown(
                        f"**调度类型**：{schedule_type}\n"
                        f"**调度表达式**：`{schedule_value}`"
                    ),
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                },
                {
                    "tag": "markdown",
                    "content": ensure_markdown(content),
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                },
            ]
        },
    }


def _feishu_label(content: str) -> dict[str, Any]:
    return {
        "tag": "div",
        "text": {
            "tag": "plain_text",
            "content": content,
            "text_size": "normal_v2",
            "text_align": "left",
            "text_color": "default",
        },
        "margin": "0px 0px 0px 0px",
    }


def _feishu_options(
    options: list[dict[str, str]],
    *,
    include_icon: bool = False,
) -> list[dict[str, Any]]:
    feishu_options = []
    for option in options:
        feishu_option: dict[str, Any] = {
            "text": plain_text(option["label"]),
            "value": option["value"],
        }
        if include_icon:
            feishu_option["icon"] = {
                "tag": "standard_icon",
                "token": "signature_outlined",
            }
        feishu_options.append(feishu_option)
    return feishu_options


def _feishu_field(field: dict[str, Any], *, disabled: bool = False, value: Any = None) -> list[dict[str, Any]]:
    placeholder = {
        "tag": "plain_text",
        "content": field.get("placeholder") or "请选择",
    }
    field_type = field["type"]
    elements: list[dict[str, Any]] = [_feishu_label(field["label"])]

    if field_type == "input":
        elements.append(
            {
                "tag": "input",
                "placeholder": {
                    "tag": "plain_text",
                    "content": field.get("placeholder") or "请输入",
                },
                "default_value": field.get("default_value") or "",
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "select":
        elements.append(
            {
                "tag": "select_static",
                "placeholder": placeholder,
                "options": _feishu_options(field["options"]),
                "type": "default",
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "multi_select":
        elements.append(
            {
                "tag": "multi_select_static",
                "placeholder": placeholder,
                "options": _feishu_options(field["options"], include_icon=True),
                "type": "default",
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "date":
        elements.append(
            {
                "tag": "date_picker",
                "placeholder": placeholder,
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    elif field_type == "datetime":
        elements.append(
            {
                "tag": "picker_datetime",
                "placeholder": placeholder,
                "width": "fill",
                "required": False,
                "name": field["name"],
                "margin": "0px 0px 0px 0px",
            }
        )
    else:
        raise ValueError(f"Unsupported question field type: {field_type}")

    component = elements[-1]
    component["disabled"] = disabled
    if value is not None:
        match field_type:
            case "input":
                component["default_value"] = value
            case "select":
                if value:
                    component["initial_option"] = value
            case "multi_select":
                component["selected_values"] = value
            case "date":
                if value:
                    component["initial_date"] = value[:10]
            case "datetime":
                if value:
                    component["initial_datetime"] = value[:16]
    return elements


def build_feishu_question_card(
    question_context: str, *, tool_result: dict[str, Any] | None = None, disabled: bool = False,
    answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        question = json.loads(question_context)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误: {exc}") from exc
    if not isinstance(question, dict):
        raise ValueError("问题 JSON 必须是对象")
    match question.get("type", "form"):
        case "confirm":
            return _build_feishu_confirm_card(validate_confirm_question_json(question), tool_result, disabled=disabled)
        case "form":
            return _build_feishu_form_card(validate_question_form_json(question), disabled=disabled, answers=answers)
        case question_type:
            raise ValueError(f"Unsupported question type: {question_type}")


def _build_feishu_confirm_card(
    question: dict[str, Any], tool_result: dict[str, Any] | None, *, disabled: bool = False,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    if question["description"]:
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": question["description"],
                    "text_size": "normal",
                    "text_align": "left",
                    "text_color": "default",
                },
                "margin": "0px 0px 0px 0px",
            }
        )

    failed = tool_result is not None and tool_result.get("success") is False
    if tool_result is not None:
        value = tool_result.get("result")
        result_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        elements.append({
            "tag": "markdown",
            "content": ensure_markdown("执行失败" if failed else "执行完成"),
        })
        elements.append({
            "tag": "collapsible_panel", "expanded": False,
            "header": {"title": plain_text(question["view_label"])},
            "elements": [{"tag": "markdown", "content": ensure_markdown(result_text, 6000)}],
        })
    buttons = [
        {
            "tag": "button",
            "text": plain_text(label),
            "type": "primary" if decision == "confirm" else "default",
            "width": "default",
            "disabled": disabled or tool_result is not None,
            "behaviors": [{"type": "callback", "value": {
                "type": question["type"],
                "decision": decision,
                "tool_call_id": question["tool_call_id"],
            }}],
        }
        for label, decision in ((question["confirm_label"], "confirm"), ("取消", "cancel"))
    ]
    elements.append(_feishu_question_actions(buttons))

    return _feishu_question_card(question["title"], "", elements, failed=failed)


def _build_feishu_form_card(
    form: dict[str, Any], *, disabled: bool = False, answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    for field in form["fields"]:
        elements.extend(_feishu_field(field, disabled=disabled, value=(answers or {}).get(field["name"])))
    elements.append(_feishu_question_actions([{
        "tag": "button",
        "text": plain_text(form.get("submit_label") or "提交"),
        "type": "primary",
        "width": "default",
        "disabled": disabled,
        "form_action_type": "submit",
        "name": "submit_question",
    }]))
    return _feishu_question_card(form["title"], form["subtitle"], [{
        "tag": "form",
        "elements": elements,
        "direction": "vertical",
        "padding": "4px 0px 4px 0px",
        "margin": "0px 0px 0px 0px",
        "name": "question_form",
    }])


def _feishu_question_actions(buttons: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tag": "column_set",
        "horizontal_align": "left",
        "columns": [
            {
                "tag": "column",
                "width": "auto",
                "elements": [button],
                "vertical_align": "top",
            }
            for button in buttons
        ],
    }


def _feishu_question_card(
    title: str, subtitle: str, elements: list[dict[str, Any]], *, failed: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "style": {
                "text_size": {
                    "normal_v2": {
                        "default": "normal",
                        "pc": "normal",
                        "mobile": "heading",
                    }
                }
            },
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": elements,
        },
        "header": {
            "title": plain_text(title),
            "subtitle": plain_text(subtitle),
            "template": "red" if failed else "blue",
            "padding": "12px 12px 12px 12px",
        },
    }
