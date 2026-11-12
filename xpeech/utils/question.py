"""共享问题表单模型、校验与执行确认表单构造。"""

import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

USER_TIMEOUT = 5 * 60


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuestionOption(StrictBaseModel):
    label: str
    value: str


Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class QuestionFieldBase(StrictBaseModel):
    name: Name
    label: str
    placeholder: str | None = None


class InputField(QuestionFieldBase):
    type: Literal["input"] = "input"
    default_value: str = ""


class SelectField(QuestionFieldBase):
    type: Literal["select"] = "select"
    options: list[QuestionOption]


class MultiSelectField(QuestionFieldBase):
    type: Literal["multi_select"] = "multi_select"
    options: list[QuestionOption]


class DateField(QuestionFieldBase):
    type: Literal["date"] = "date"


class DateTimeField(QuestionFieldBase):
    type: Literal["datetime"] = "datetime"


QuestionField = Annotated[
    Union[InputField, SelectField, MultiSelectField, DateField, DateTimeField],
    Field(discriminator="type"),
]


class QuestionForm(StrictBaseModel):
    type: Literal["form"] = "form"
    title: str = "需要补充信息"
    subtitle: str | None = None
    submit_label: str = "提交"
    fields: list[QuestionField]

    @model_validator(mode="after")
    def validate_unique_field_names(self):
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"field names must be unique: {', '.join(duplicates)}")
        return self


class ConfirmQuestion(StrictBaseModel):
    type: Literal["confirm"] = "confirm"
    title: str = "请确认"
    description: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    timeout_seconds: int | None = Field(default=None, gt=0)


def validate_question_form_json(data: str | dict[str, Any]) -> dict[str, Any]:
    """仅校验普通表单；省略 type 时默认为 form。"""
    try:
        raw_data = json.loads(data) if isinstance(data, str) else data
        return QuestionForm.model_validate(raw_data).model_dump()
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {str(e)}") from e
    except ValidationError as e:
        raise ValueError(f"问题表单 JSON 校验失败: {str(e)}") from e


def validate_confirm_question_json(data: str | dict[str, Any]) -> dict[str, Any]:
    """仅校验确认问题；省略 type 时默认为 confirm。"""
    try:
        raw_data = json.loads(data) if isinstance(data, str) else data
        return ConfirmQuestion.model_validate(raw_data).model_dump()
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {str(e)}") from e
    except ValidationError as e:
        raise ValueError(f"确认问题 JSON 校验失败: {str(e)}") from e


def build_execution_question(
    tool_name: str, arguments: dict[str, Any], description: str, *,
    tool_call_id: str | None = None, title: str | None = None,
) -> str:
    """将执行说明与工具详情分开传递，交由各端独立渲染。"""
    return ConfirmQuestion(
        title=title or "执行确认",
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        description=description,
        arguments=arguments,
        timeout_seconds=USER_TIMEOUT,
    ).model_dump_json()


