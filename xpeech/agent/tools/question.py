from pydantic import BaseModel, Field
import json


SUPPORTED_FIELD_TAGS = {
    "select_static",
    "multi_select_static",
    "input",
    "date_picker",
    "picker_datetime",
}
REQUIRED_TOP_LEVEL_KEYS = {
    "schema",
    "config",
    "body",
    "header",
}


class ValidationError(Exception):
    pass


def load_json(json_str: str) -> dict:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as error:
        raise ValidationError(f"JSON 格式错误：第 {error.lineno} 行，第 {error.colno} 列，{error.msg}") from error
    if not isinstance(data, dict):
        raise ValidationError("JSON 根节点必须是对象")
    return data


def require_keys(obj: dict, keys: set[str], path: str) -> None:
    missing_keys = keys - set(obj.keys())
    if missing_keys:
        raise ValidationError(f"{path} 缺少字段：{', '.join(sorted(missing_keys))}")


def require_type(value, expected_type, path: str) -> None:
    if not isinstance(value, expected_type):
        expected_name = expected_type.__name__
        actual_name = type(value).__name__
        raise ValidationError(f"{path} 类型错误：期望 {expected_name}，实际 {actual_name}")


def validate_plain_text(obj: dict, path: str) -> None:
    require_type(obj, dict, path)
    if obj.get("tag") != "plain_text":
        raise ValidationError(f"{path}.tag 必须是 plain_text")
    if "content" not in obj:
        raise ValidationError(f"{path} 缺少 content")
    require_type(obj["content"], str, f"{path}.content")


def validate_header(card: dict) -> None:
    header = card["header"]
    require_type(header, dict, "header")
    require_keys(header, {"title", "subtitle"}, "header")
    validate_plain_text(header["title"], "header.title")
    validate_plain_text(header["subtitle"], "header.subtitle")
    if "template" in header:
        require_type(header["template"], str, "header.template")
    if "icon" in header:
        require_type(header["icon"], dict, "header.icon")
        require_keys(header["icon"], {"tag", "token"}, "header.icon")


def validate_config(card: dict) -> None:
    config = card["config"]
    require_type(config, dict, "config")
    if "update_multi" in config:
        require_type(config["update_multi"], bool, "config.update_multi")
    if "style" in config:
        require_type(config["style"], dict, "config.style")


def validate_title_div(element: dict, path: str) -> None:
    require_type(element, dict, path)
    if element.get("tag") != "div":
        raise ValidationError(f"{path}.tag 必须是 div")
    if "text" not in element:
        raise ValidationError(f"{path} 缺少 text")
    validate_plain_text(element["text"], f"{path}.text")


def validate_options(component: dict, path: str) -> None:
    if "options" not in component:
        raise ValidationError(f"{path} 缺少 options")
    require_type(component["options"], list, f"{path}.options")
    if not component["options"]:
        raise ValidationError(f"{path}.options 不能为空")
    seen_values = set()
    for index, option in enumerate(component["options"]):
        option_path = f"{path}.options[{index}]"
        require_type(option, dict, option_path)
        require_keys(option, {"text", "value"}, option_path)
        validate_plain_text(option["text"], f"{option_path}.text")
        require_type(option["value"], str, f"{option_path}.value")
        if option["value"] in seen_values:
            raise ValidationError(f"{option_path}.value 重复：{option['value']}")
        seen_values.add(option["value"])


def validate_field_component(component: dict, path: str) -> None:
    require_type(component, dict, path)
    tag = component.get("tag")
    if tag not in SUPPORTED_FIELD_TAGS:
        raise ValidationError(f"{path}.tag 不支持：{tag}，支持值：{', '.join(sorted(SUPPORTED_FIELD_TAGS))}")
    if "name" not in component:
        raise ValidationError(f"{path} 缺少 name")
    require_type(component["name"], str, f"{path}.name")
    if not component["name"].strip():
        raise ValidationError(f"{path}.name 不能为空")
    if tag in {"select_static", "multi_select_static"}:
        validate_options(component, path)
    if "placeholder" in component:
        validate_plain_text(component["placeholder"], f"{path}.placeholder")


def is_button_group(element: dict) -> bool:
    return isinstance(element, dict) and element.get("tag") == "column_set"


def validate_button_group(element: dict, path: str) -> None:
    require_type(element, dict, path)
    if element.get("tag") != "column_set":
        raise ValidationError(f"{path}.tag 必须是 column_set")
    require_keys(element, {"columns"}, path)
    require_type(element["columns"], list, f"{path}.columns")
    if len(element["columns"]) < 2:
        raise ValidationError(f"{path}.columns 至少需要提交和取消两个按钮")


def validate_form(form: dict, path: str) -> None:
    require_type(form, dict, path)
    if form.get("tag") != "form":
        raise ValidationError(f"{path}.tag 必须是 form")
    require_keys(form, {"elements", "name"}, path)
    require_type(form["elements"], list, f"{path}.elements")
    require_type(form["name"], str, f"{path}.name")
    elements = form["elements"]
    if not elements:
        raise ValidationError(f"{path}.elements 不能为空")
    if not is_button_group(elements[-1]):
        raise ValidationError(f"{path}.elements 最后一个元素必须是提交/取消按钮 column_set")
    validate_button_group(elements[-1], f"{path}.elements[-1]")
    field_elements = elements[:-1]
    if len(field_elements) % 2 != 0:
        raise ValidationError(f"{path}.elements 字段部分必须成对出现：标题 div + 字段组件")
    for index in range(0, len(field_elements), 2):
        title_path = f"{path}.elements[{index}]"
        component_path = f"{path}.elements[{index + 1}]"
        validate_title_div(field_elements[index], title_path)
        validate_field_component(field_elements[index + 1], component_path)


def validate_body(card: dict) -> None:
    body = card["body"]
    require_type(body, dict, "body")
    require_keys(body, {"elements"}, "body")
    require_type(body["elements"], list, "body.elements")
    if not body["elements"]:
        raise ValidationError("body.elements 不能为空")
    form = body["elements"][0]
    validate_form(form, "body.elements[0]")


def validate_card(card: dict) -> None:
    require_keys(card, REQUIRED_TOP_LEVEL_KEYS, "root")
    if card["schema"] != "2.0":
        raise ValidationError("schema 必须是 2.0")
    validate_config(card)
    validate_body(card)
    validate_header(card)


def is_ok_question(question: str) -> bool:
    return question.startswith("OK ")


def extract_question(question: str) -> str:
    if is_ok_question(question):
        return question.split(" ", 1)[-1]
    else:
        raise ValueError("Invalid question format")


class QuestionArgs(BaseModel):
    question: str = Field(description="飞书卡片 2.0 表单 JSON")
    timeout: int = Field(description="用户填写表单超时时间", default=20, ge=10, le=60)

    @property
    def json(self):
        try:
            card = load_json(self.question)
            validate_card(card)
            return f"OK {self.question}"
        except ValidationError as error:
            return f"NOTOK 校验失败：{error}"


def joyride_request_human_input(args: QuestionArgs):
    """通过发送飞书卡片 2.0 表单 JSON 向用户追问关键信息"""
    return args.json
