from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Annotated
from enum import StrEnum
import json


class QuestionType(StrEnum):
    radio = "radio"
    checkbox = "checkbox"
    input = "input"


class QuestionArgs(BaseModel):
    question_type: Annotated[Literal["radio", "checkbox", "input"], Field(description="The type of question")] = Field(
        description="The type of question"
    )
    question: Annotated[str, Field(description="The question to ask")] = Field(description="The question to ask")
    header: Annotated[str, Field(description="The header of the question")] = Field(
        description="The header of the question"
    )
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


class Option(BaseModel):
    label: str = Field(description="The label of the option")
    description: str = Field(description="The description of the option")


class Options(BaseModel):
    options: list[Option] = Field(description="The list of options")


def question(args: QuestionArgs):
    """ """
    # 校验格式
    try:
        options = Options.model_validate_json(args.options)
    except ValidationError as e:
        raise ValidationError(
            f"The question tool was called with invalid arguments: {e}. Please rewrite the input so it satisfies the expected schema.  "
        )
    return json.dumps(
        {
            "question_type": args.question_type,
            "question": args.question,
            "header": args.header,
            "options": options,
        }
    )
