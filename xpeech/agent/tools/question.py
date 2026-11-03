import json
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

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


QuestionFormAdapter = TypeAdapter(QuestionForm)


def validate_question_json(data: str | dict[str, Any]) -> dict[str, Any]:
    try:
        raw_data = json.loads(data) if isinstance(data, str) else data
        validated_data = QuestionFormAdapter.validate_python(raw_data)
        return validated_data.model_dump()
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误: {str(e)}") from e
    except ValidationError as e:
        raise ValueError(f"问题表单 JSON 校验失败: {str(e)}") from e


class QuestionArgs(StrictBaseModel):
    question: str = Field(description="序列化为 JSON 字符串的完整表单对象，包含标题、副标题和问题字段列表")

    def validate_and_format(self) -> str:
        question = validate_question_json(self.question)
        question["fields"].append(
            InputField(
                name="user_customization_input",
                label="补充说明",
                placeholder="填写其他想法或补充信息（选填）",
            ).model_dump()
        )
        subtitle = question.get("subtitle")
        timeout_hint = f"您有{USER_TIMEOUT}秒时间填写表单"
        question["subtitle"] = f"{subtitle}\n{timeout_hint}" if subtitle else timeout_hint
        return json.dumps(question, ensure_ascii=False, indent=4)


def ask_user_question(args: QuestionArgs) -> str:
    """通过包含标题、副标题和问题列表的完整表单 JSON 向用户追问关键信息。

    返回包含 type、title、subtitle、submit_label、fields 的完整表单 JSON 字符串。
    """
    return args.validate_and_format()
