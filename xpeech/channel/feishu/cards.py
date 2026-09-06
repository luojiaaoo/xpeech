from __future__ import annotations

from typing import Any

from ...agent.tools.question import validate_question_json

FINISH_CARD_CONTENT = {
    "schema": "2.0",
    "config": {
        "update_multi": True,
        "style": {"text_size": {"normal_v2": {"default": "normal", "pc": "normal", "mobile": "heading"}}},
    },
    "body": {
        "direction": "vertical",
        "horizontal_spacing": "8px",
        "vertical_spacing": "8px",
        "horizontal_align": "center",
        "vertical_align": "center",
        "padding": "12px 12px 12px 12px",
        "elements": [
            {
                "tag": "markdown",
                "content": ":OK:<font color='red'>表单填写完成</font>",
                "text_align": "left",
                "text_size": "normal_v2",
                "margin": "0px 0px 0px 0px",
            }
        ],
    },
}


def plain_text(content: str) -> dict[str, str]:
    return {"tag": "plain_text", "content": content}


def build_feishu_background_task_card(content: str) -> dict[str, Any]:
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
                    "content": content,
                    "text_align": "left",
                    "text_size": "normal",
                    "margin": "0px 0px 0px 0px",
                }
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


def _feishu_field(field: dict[str, Any]) -> list[dict[str, Any]]:
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

    return elements


def build_feishu_question_card(question_context: str) -> dict[str, Any]:
    form = validate_question_json(question_context)
    elements: list[dict[str, Any]] = []
    for field in form["fields"]:
        elements.extend(_feishu_field(field))

    elements.extend(
        [
            _feishu_label("自定义"),
            {
                "tag": "input",
                "placeholder": {
                    "tag": "plain_text",
                    "content": "请输入",
                },
                "default_value": "",
                "width": "fill",
                "required": False,
                "name": "user_customization",
                "margin": "0px 0px 0px 0px",
            },
        ]
    )

    elements.append(
        {
            "tag": "column_set",
            "horizontal_align": "left",
            "columns": [
                {
                    "tag": "column",
                    "width": "auto",
                    "elements": [
                        {
                            "tag": "button",
                            "text": plain_text(form.get("submit_label") or "提交"),
                            "type": "primary",
                            "width": "default",
                            "form_action_type": "submit",
                            "name": "submit_question",
                        }
                    ],
                    "vertical_align": "top",
                },
                {
                    "tag": "column",
                    "width": "auto",
                    "elements": [],
                    "vertical_align": "top",
                },
            ],
        }
    )

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
            "elements": [
                {
                    "tag": "form",
                    "elements": elements,
                    "direction": "vertical",
                    "padding": "4px 0px 4px 0px",
                    "margin": "0px 0px 0px 0px",
                    "name": "question_form",
                }
            ],
        },
        "header": {
            "title": plain_text(form["title"]),
            "subtitle": plain_text(form["subtitle"]),
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
    }
