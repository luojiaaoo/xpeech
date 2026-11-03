import json

from pydantic import Field

from ...utils.question import USER_TIMEOUT, InputField, StrictBaseModel, validate_question_form_json


class QuestionArgs(StrictBaseModel):
    question: str = Field(description="序列化为 JSON 字符串的完整表单对象，包含标题、副标题和问题字段列表")

    def validate_and_format(self) -> str:
        question = validate_question_form_json(self.question)
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
        return json.dumps(validate_question_form_json(question), ensure_ascii=False, indent=4)


def ask_user_question(args: QuestionArgs) -> str:
    """通过包含标题、副标题和问题列表的完整表单 JSON 向用户追问关键信息。

    返回包含 type、title、subtitle、submit_label、fields 的完整表单 JSON 字符串。
    """
    return args.validate_and_format()
