from re import A
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated
import json


# 校验
class Option(BaseModel):
    label: str = Field(description="The label of the option")
    description: str = Field(description="The description of the option")


class Options(BaseModel):
    options: list[Option] = Field(description="The list of options")


# 参数
class QuestionArgs(BaseModel):
    question: Annotated[str, Field(description="The question to ask")] = Field(description="The question to ask")


class OptionsArgs(BaseModel):
    options: Annotated[str, Field(description="The list of options to choose from.")] = Field(
        description="""
            The list of options to choose from. Json format for example:
            [
                {
                    "label": "npm",
                    "description": "Node package manager"
                },
                {
                    "label": "pnpm",
                    "description": "Fast, disk space efficient"
                },
                {
                    "label": "yarn",
                    "description": "Yarn classic or berry"
                }
            ]
        """
    )


class RadioQuestionArgs(QuestionArgs, OptionsArgs):
    def validated_options(self) -> Options:
        try:
            return Options.model_validate_json(self.options)
        except ValidationError as e:
            raise ValidationError(
                f"The question tool was called with invalid arguments: {e}. Please rewrite the input so it satisfies the expected schema.  "
            )

    @property
    def human_friendly_chinese(self) -> str:
        """Generate a Chinese human-readable representation of the radio question."""
        options_text = "\n".join([f"  ○ {opt.label}: {opt.description}" for opt in self.validated_options().options])
        return f"【单选】{self.question}\n\n选项：\n{options_text}"

    @property
    def human_friendly_english(self) -> str:
        """Generate an English human-readable representation of the radio question."""
        options_text = "\n".join([f"  ○ {opt.label}: {opt.description}" for opt in self.validated_options().options])
        return f"[Single Choice] {self.question}\n\nOptions:\n{options_text}"

    @property
    def json(self):
        return json.dumps(
            {
                "question": self.question,
                "options": self.validated_options().model_dump(),
                "human_friendly_chinese": self.human_friendly_chinese,
                "human_friendly_english": self.human_friendly_english,
            },
            ensure_ascii=False,
        )


class CheckboxQuestionArgs(QuestionArgs, OptionsArgs):
    def validated_options(self) -> Options:
        try:
            return Options.model_validate_json(self.options)
        except ValidationError as e:
            raise ValidationError(
                f"The question tool was called with invalid arguments: {e}. Please rewrite the input so it satisfies the expected schema.  "
            )

    @property
    def human_friendly_chinese(self) -> str:
        """Generate a Chinese human-readable representation of the checkbox question."""
        options_text = "\n".join([f"  ☐ {opt.label}: {opt.description}" for opt in self.validated_options().options])
        return f"【多选】{self.question}\n\n选项（可多选）：\n{options_text}"

    @property
    def human_friendly_english(self) -> str:
        """Generate an English human-readable representation of the checkbox question."""
        options_text = "\n".join([f"  ☐ {opt.label}: {opt.description}" for opt in self.validated_options().options])
        return f"[Multiple Choice] {self.question}\n\nSelect all that apply:\n{options_text}"

    @property
    def json(self):
        return json.dumps(
            {
                "question": self.question,
                "options": self.validated_options().model_dump(),
                "human_friendly_chinese": self.human_friendly_chinese,
                "human_friendly_english": self.human_friendly_english,
            },
            ensure_ascii=False,
        )


class InputQuestionArgs(QuestionArgs):
    @property
    def human_friendly_chinese(self) -> str:
        """Generate a Chinese human-readable representation of the input question."""
        return f"【填空】{self.question}\n\n请输入您的回答："

    @property
    def human_friendly_english(self) -> str:
        """Generate an English human-readable representation of the input question."""
        return f"[Text Input] {self.question}\n\nPlease enter your response:"

    @property
    def json(self):
        return json.dumps(
            {
                "question": self.question,
                "human_friendly_chinese": self.human_friendly_chinese,
                "human_friendly_english": self.human_friendly_english,
            },
            ensure_ascii=False,
        )


def question_radio(args: RadioQuestionArgs):
    """
    Ask the user a single-choice (radio) question and wait for their response.

    Use this tool when you need the user to select ONE option from multiple choices.
    The system will display the question with clickable options and return the user's selection.

    Examples:
    - Asking which package manager to use (npm/pnpm/yarn)
    - Confirming deployment environment (staging/production)
    - Selecting a theme (light/dark/auto)

    ⚠️ CRITICAL RULE: You can ONLY call question_radio, question_checkbox, or question_input tools together.
    DO NOT call any other tools (like read_file, write_file, shell, etc.) in the same turn as these question tools.
    Mixing question tools with other tools will cause all non-question tools to be IGNORED.
    Wait for the user's response before proceeding with other actions.
    
    IMPORTANT: Only use when you genuinely need user input. Do not use for rhetorical questions.
    """
    return args.json


def question_checkbox(args: CheckboxQuestionArgs):
    """
    Ask the user a multi-select (checkbox) question and wait for their response.

    Use this tool when you need the user to select ZERO, ONE, or MULTIPLE options from a list.
    The system will display the question with checkboxes and return all selected options.

    Examples:
    - Selecting features to enable (logging, monitoring, caching)
    - Choosing platforms to support (web, iOS, Android)
    - Picking notification preferences (email, SMS, push)

    ⚠️ CRITICAL RULE: You can ONLY call question_radio, question_checkbox, or question_input tools together.
    DO NOT call any other tools (like read_file, write_file, shell, etc.) in the same turn as these question tools.
    Mixing question tools with other tools will cause all non-question tools to be IGNORED.
    Wait for the user's response before proceeding with other actions.
    
    IMPORTANT: Only use when you need multiple selections. For single choice, use question_radio instead.
    """
    return args.json


def question_input(args: InputQuestionArgs):
    """
    Ask the user an open-ended text input question and wait for their response.

    Use this tool when you need free-form text input from the user that cannot be captured with predefined options.
    The system will display the question with a text input field and return the user's typed response.

    Examples:
    - Asking for a project name
    - Requesting an API key or token
    - Getting a custom configuration value
    - Collecting user feedback or comments

    ⚠️ CRITICAL RULE: You can ONLY call question_radio, question_checkbox, or question_input tools together.
    DO NOT call any other tools (like read_file, write_file, shell, etc.) in the same turn as these question tools.
    Mixing question tools with other tools will cause all non-question tools to be IGNORED.
    Wait for the user's response before proceeding with other actions.
    
    IMPORTANT: Only use when you need unstructured text. For structured choices, use question_radio or question_checkbox instead.
    """
    return args.json
