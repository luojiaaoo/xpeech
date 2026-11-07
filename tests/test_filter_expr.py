"""过滤表达式（filter）的语法、求值与注入防护。"""

import pytest

from xpeech.agent.skills.skill import SkillMetadata
from xpeech.config.settings import MCPServerSettings
from xpeech.utils.filter_expr import (
    FILTER_MAX_ARGS,
    FILTER_MAX_DEPTH,
    FILTER_MAX_LENGTH,
    FilterContext,
    FilterExpressionError,
    filter_allows_access,
    parse_filter,
)


@pytest.mark.parametrize(
    ("source", "context", "expected"),
    [
        ('channel("crm")', FilterContext(channel="crm"), True),
        ('channel("crm")', FilterContext(channel="erp"), False),
        ('channel("crm")', FilterContext(), False),
        ('channel("CRM")', FilterContext(channel="crm"), False),
        ('channel("crm", "erp")', FilterContext(channel="erp"), True),
        ('channel("crm", "erp")', FilterContext(channel="sales"), False),
        ('org("公司/研发/")', FilterContext(org="公司/研发/平台"), True),
        ('org("公司/研发/")', FilterContext(org="公司/销售/"), False),
        ('org("公司/研发/")', FilterContext(), False),
        ('org("公司/研发/")', FilterContext(org=""), False),
        ('org("company/eng")', FilterContext(org="company/engineering"), True),
        ('org("company/eng/")', FilterContext(org="company/engineering"), False),
        ('org("Company/")', FilterContext(org="company/eng"), False),
        ('org("公司/销售/", "公司/研发/")', FilterContext(org="公司/研发/平台"), True),
        ('not channel("crm")', FilterContext(), True),
        ('not channel("crm")', FilterContext(channel="crm"), False),
        ('channel("crm") or org("公司/")', FilterContext(org="公司/研发/"), True),
        ('channel("crm") or org("公司/")', FilterContext(channel="erp"), False),
        ('channel("crm") and org("公司/")', FilterContext(channel="crm", org="公司/研发/"), True),
        ('channel("crm") and org("公司/")', FilterContext(channel="crm"), False),
        ('channel("crm") and not org("公司/销售/")', FilterContext(channel="crm", org="公司/研发/"), True),
        ('channel("crm") and not org("公司/销售/")', FilterContext(channel="crm", org="公司/销售/"), False),
        ('(channel("crm") or channel("erp")) and org("p/")', FilterContext(channel="erp", org="p/q"), True),
        ('(channel("crm") or channel("erp")) and org("p/")', FilterContext(channel="erp", org="x"), False),
        ("True", FilterContext(), True),
        ("False", FilterContext(channel="crm"), False),
    ],
)
def test_filter_evaluation(source, context, expected):
    assert parse_filter(source).evaluate(context) is expected
    assert filter_allows_access(source, context) is expected


def test_missing_filter_is_public():
    assert filter_allows_access(None, FilterContext()) is True


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
        "channel()",
        "org()",
        "channel(1)",
        'channel("")',
        'channel(" a")',
        'org("a ")',
        "channel(name='a')",
        "channel(*['a'])",
        'channel("a") == "a"',
        "1",
        '"crm"',
        "None",
        "x",
        "tag",
        "1 + 1",
        "not",
        "True and",
        "(channel('a')",
        "channel('a') if True else False",
        "{'a': 1}",
        "f'{1}'",
        "lambda: True",
        "[x for x in ()]",
        "channel.__class__",
        "channel('a').upper",
        "().__class__",
        "[].__class__.__bases__",
        "import os",
        "channel('a'); import os",
        "__import__('os')",
        "__import__('os').system('id')",
        "eval('1')",
        "exec('x')",
        "open('x')",
        "compile('1', '1', 'eval')",
        "(lambda: __import__('os'))()",
        "getattr(channel, 'x')",
        "channel('x') or __import__('os')",
    ],
)
def test_invalid_or_malicious_filter_is_rejected(source):
    with pytest.raises(FilterExpressionError):
        parse_filter(source)


def test_filter_length_limit():
    with pytest.raises(FilterExpressionError, match="characters"):
        parse_filter(f"channel({'x' * FILTER_MAX_LENGTH})")
    parse_filter(f"channel('{'x' * (FILTER_MAX_LENGTH - 8)}')")


def test_filter_depth_limit():
    parse_filter("not " * FILTER_MAX_DEPTH + "True")
    with pytest.raises(FilterExpressionError, match="nesting"):
        parse_filter("not " * (FILTER_MAX_DEPTH + 1) + "True")


def test_filter_argument_count_limit():
    args = ", ".join(f"'{index}'" for index in range(FILTER_MAX_ARGS))
    parse_filter(f"channel({args})")
    args = ", ".join(f"'{index}'" for index in range(FILTER_MAX_ARGS + 1))
    with pytest.raises(FilterExpressionError, match="arguments"):
        parse_filter(f"channel({args})")


@pytest.mark.parametrize("source", [None, 'channel("crm") or org("公司/")', "channel()", "import os"])
def test_config_surfaces_validate_identically(source):
    if source is not None:
        try:
            parse_filter(source)
        except FilterExpressionError:
            with pytest.raises(ValueError):
                SkillMetadata(name="demo", description="Demo", filter=source)
            with pytest.raises(ValueError):
                MCPServerSettings(command="demo", filter=source)
            return
    assert SkillMetadata(name="demo", description="Demo", filter=source).filter == source
    assert MCPServerSettings(command="demo", filter=source).filter == source
