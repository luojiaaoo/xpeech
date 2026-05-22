import json
from typing import Any, List, Union, Literal, Annotated
from pydantic import BaseModel, Field, ValidationError, ConfigDict, TypeAdapter

USER_TIMEOUT = 5 * 60


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlainText(StrictBaseModel):
    tag: Literal["plain_text"] = "plain_text"
    content: str


class PlaceholderText(StrictBaseModel):
    tag: Literal["plain_text"] = "plain_text"
    content: str = "请选择"


class InputPlaceholderText(StrictBaseModel):
    tag: Literal["plain_text"] = "plain_text"
    content: str = "请输入"


class StandardIcon(StrictBaseModel):
    tag: Literal["standard_icon"] = "standard_icon"
    token: Literal["signature_outlined"] = "signature_outlined"


class SelectOption(StrictBaseModel):
    text: PlainText
    value: str
    icon: StandardIcon = Field(default_factory=StandardIcon)


class EmptyLabel(StrictBaseModel):
    tag: Literal["plain_text"] = "plain_text"
    content: str = ""


class DivPlainText(StrictBaseModel):
    tag: Literal["plain_text"] = "plain_text"
    content: str
    text_size: Literal["normal_v2"] = "normal_v2"
    text_align: Literal["left"] = "left"
    text_color: Literal["default"] = "default"


Margin = Annotated[str, Field(pattern=r"^\d+px \d+px \d+px \d+px$")]


class Div(StrictBaseModel):
    tag: Literal["div"] = "div"
    text: DivPlainText
    margin: Margin = "0px 0px 0px 0px"


class SelectStatic(StrictBaseModel):
    tag: Literal["select_static"] = "select_static"
    placeholder: PlaceholderText = Field(default_factory=PlaceholderText)
    options: List[SelectOption]
    type: Literal["default"] = "default"
    width: Literal["fill"] = "fill"
    name: str
    margin: Margin = "0px 0px 0px 0px"


class MultiSelectStatic(StrictBaseModel):
    tag: Literal["multi_select_static"] = "multi_select_static"
    placeholder: PlaceholderText = Field(default_factory=PlaceholderText)
    options: List[SelectOption]
    type: Literal["default"] = "default"
    width: Literal["fill"] = "fill"
    name: str
    margin: Margin = "0px 0px 0px 0px"


class Input(StrictBaseModel):
    tag: Literal["input"] = "input"
    placeholder: InputPlaceholderText = Field(default_factory=InputPlaceholderText)
    default_value: str = ""
    width: Literal["fill"] = "fill"
    label: EmptyLabel = Field(default_factory=EmptyLabel)
    label_position: Literal["top"] = "top"
    name: str
    margin: Margin = "0px 0px 0px 0px"


class DatePicker(StrictBaseModel):
    tag: Literal["date_picker"] = "date_picker"
    placeholder: PlaceholderText = Field(default_factory=PlaceholderText)
    width: Literal["fill"] = "fill"
    name: str
    margin: Margin = "0px 0px 0px 0px"


class DateTimePicker(StrictBaseModel):
    tag: Literal["picker_datetime"] = "picker_datetime"
    placeholder: PlaceholderText = Field(default_factory=PlaceholderText)
    width: Literal["fill"] = "fill"
    name: str
    margin: Margin = "0px 0px 0px 0px"


FormComponent = Union[
    Div,
    SelectStatic,
    MultiSelectStatic,
    Input,
    DatePicker,
    DateTimePicker,
]
FormComponentListAdapter = TypeAdapter(List[FormComponent])


def validate_form_json(data: str) -> tuple[bool, Any]:
    try:
        data = json.loads(data)
        validated_data = FormComponentListAdapter.validate_python(data)
        return [item.model_dump() for item in validated_data]
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {str(e)}")
    except ValidationError as e:
        raise ValueError(f"表单 JSON 校验失败: {str(e)}")


def get_json(form: str, title: str, subtitle: str):
    return {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "style": {"text_size": {"normal_v2": {"default": "normal", "pc": "normal", "mobile": "heading"}}},
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": [
                {
                    "tag": "form",
                    "elements": [
                        *validate_form_json(form),
                        {
                            "tag": "div",
                            "text": {
                                "tag": "plain_text",
                                "content": "自定义",
                                "text_size": "normal_v2",
                                "text_align": "left",
                                "text_color": "default",
                            },
                            "margin": "0px 0px 0px 0px",
                        },
                        {
                            "tag": "input",
                            "placeholder": {"tag": "plain_text", "content": "请输入"},
                            "default_value": "",
                            "width": "fill",
                            "name": "user_customization",
                            "margin": "0px 0px 0px 0px",
                        },
                        {
                            "tag": "column_set",
                            "columns": [
                                {
                                    "tag": "column",
                                    "width": "auto",
                                    "elements": [
                                        {
                                            "tag": "button",
                                            "text": {"tag": "plain_text", "content": "提交"},
                                            "type": "primary",
                                            "width": "default",
                                            "form_action_type": "submit",
                                            "name": "Button_mpgy4lye",
                                        }
                                    ],
                                    "vertical_align": "top",
                                },
                                {"tag": "column", "width": "auto", "elements": [], "vertical_align": "top"},
                            ],
                        },
                    ],
                    "padding": "4px 0px 4px 0px",
                    "margin": "0px 0px 0px 0px",
                    "name": "Form_mpgy4lyd",
                }
            ],
        },
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "subtitle": {"tag": "plain_text", "content": subtitle},
            "template": "blue",
            "padding": "12px 12px 12px 12px",
        },
    }


def is_ok_question(question: str) -> bool:
    return question.startswith("OK ")


def extract_question(question: str) -> str:
    if is_ok_question(question):
        return question.split(" ", 1)[-1]
    else:
        raise ValueError("Invalid question format")


class QuestionArgs(StrictBaseModel):
    question: str = Field(description="飞书卡片 2.0 JSON 数组")
    title: str = Field(description="表单标题")

    @property
    def json(self):
        try:
            json_str: str = json.dumps(
                get_json(self.question, self.title, f"您有{USER_TIMEOUT}秒时间填写表单"), ensure_ascii=False, indent=4
            )
            return f"OK {json_str}"
        except ValidationError as error:
            return f"NOTOK 校验失败：{error}"


def joyride_request_human_input(args: QuestionArgs):
    """通过发送飞书卡片 2.0 表单 JSON 向用户追问关键信息"""
    return args.json
