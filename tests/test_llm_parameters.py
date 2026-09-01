import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from xpeech.config.settings import LLMConfig
from xpeech.provider.schema import LLMParameters


def base_llm_config(**overrides):
    return {
        "api_key": "test",
        "api_base": "https://example.test",
        "default_model": "test",
        "parameters": {},
        "tools_python_package": "custom_tools",
        "default_tools": [],
        **overrides,
    }


def test_llm_parameters_have_requested_defaults():
    parameters = LLMParameters()

    assert parameters.max_tokens == 32768
    assert parameters.max_context_tokens == 200000
    assert parameters.temperature is None
    assert parameters.top_p is None
    assert parameters.top_k is None
    assert parameters.min_p is None
    assert parameters.presence_penalty is None
    assert parameters.repetition_penalty is None
    assert parameters.reasoning_effort is None


def test_llm_parameters_copy_with_only_explicit_overrides():
    parameters = LLMParameters(
        max_tokens=4096,
        max_context_tokens=65536,
        temperature=0.8,
        top_p=0.9,
    )

    copied = parameters.copy_with(LLMParameters(max_tokens=1024, temperature=None))

    assert copied is not parameters
    assert copied.max_tokens == 1024
    assert copied.max_context_tokens == 65536
    assert copied.temperature is None
    assert copied.top_p == 0.9


def test_example_uses_nested_llm_parameters():
    with Path("conf.toml.example").open("rb") as example_file:
        config = LLMConfig.model_validate(tomllib.load(example_file)["llm"])

    assert config.parameters.model_dump() == {
        "max_tokens": 32768,
        "max_context_tokens": 262144,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": None,
    }
    assert config.summary_tokens == 8192
    assert config.max_iterations == 40


def test_llm_parameters_reject_aliases():
    with pytest.raises(ValidationError):
        LLMParameters.model_validate({"max_token": 4096})

    with pytest.raises(ValidationError):
        LLMParameters.model_validate({"default_top_p": 0.8})


def test_llm_config_requires_nested_parameters():
    config_data = base_llm_config()
    config_data.pop("parameters")

    with pytest.raises(ValidationError):
        LLMConfig.model_validate(config_data)


def test_llm_config_rejects_root_level_model_parameters():
    with pytest.raises(ValidationError):
        LLMConfig.model_validate(base_llm_config(default_top_p=0.8))


def test_llm_config_reads_nested_parameters():
    config = LLMConfig.model_validate(
        base_llm_config(
            parameters={
                "max_tokens": 4096,
                "max_context_tokens": 65536,
                "top_p": 0.8,
            },
        )
    )

    assert config.parameters.max_tokens == 4096
    assert config.parameters.max_context_tokens == 65536
    assert config.parameters.top_p == 0.8
