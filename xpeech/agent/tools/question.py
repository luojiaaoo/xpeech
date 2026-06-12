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
    include_customization: bool = True
    fields: list[QuestionField]

    @model_validator(mode="after")
    def validate_unique_field_names(self):
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"field names must be unique: {', '.join(duplicates)}")
        if self.include_customization and "user_customization" in names:
            raise ValueError("field name 'user_customization' is reserved when include_customization is true")
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


def is_ok_question(question: str) -> bool:
    return question.startswith("OK ")


def extract_question(question: str) -> str:
    if is_ok_question(question):
        return question.split(" ", 1)[-1]
    raise ValueError("Invalid question format")


class QuestionArgs(StrictBaseModel):
    question: str = Field(description="通用问题表单 JSON 对象")
    title: str | None = Field(default=None, description="表单标题。若 question.title 已提供，可省略。")

    @property
    def json(self):
        try:
            question = validate_question_json(self.question)
            if self.title and question.get("title") == QuestionForm.model_fields["title"].default:
                question["title"] = self.title
            question["subtitle"] = f"您有{USER_TIMEOUT}秒时间填写表单"
            json_str = json.dumps(question, ensure_ascii=False, indent=4)
            return f"OK {json_str}"
        except ValueError as error:
            return f"NOTOK 校验失败：{error}"


def ask_user_question(args: QuestionArgs):
    """通过通用问题表单 JSON 向用户追问关键信息。"""
    return args.json
