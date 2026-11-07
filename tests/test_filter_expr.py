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
        ('tag("crm")', FilterContext(tag="crm"), True),
        ('tag("crm")', FilterContext(tag="erp"), False),
        ('tag("crm")', FilterContext(), False),
        ('tag("CRM")', FilterContext(tag="crm"), False),
        ('tag("crm", "erp")', FilterContext(tag="erp"), True),
        ('tag("crm", "erp")', FilterContext(tag="sales"), False),
        ('org("公司/研发/")', FilterContext(org="公司/研发/平台"), True),
        ('org("公司/研发/")', FilterContext(org="公司/销售/"), False),
        ('org("公司/研发/")', FilterContext(), False),
        ('org("公司/研发/")', FilterContext(org=""), False),
        ('org("company/eng")', FilterContext(org="company/engineering"), True),
        ('org("company/eng/")', FilterContext(org="company/engineering"), False),
        ('org("Company/")', FilterContext(org="company/eng"), False),
        ('org("公司/销售/", "公司/研发/")', FilterContext(org="公司/研发/平台"), True),
        ('not tag("crm")', FilterContext(), True),
        ('not tag("crm")', FilterContext(tag="crm"), False),
        ('tag("crm") or org("公司/")', FilterContext(org="公司/研发/"), True),
        ('tag("crm") or org("公司/")', FilterContext(tag="erp"), False),
        ('tag("crm") and org("公司/")', FilterContext(tag="crm", org="公司/研发/"), True),
        ('tag("crm") and org("公司/")', FilterContext(tag="crm"), False),
        ('tag("crm") and not org("公司/销售/")', FilterContext(tag="crm", org="公司/研发/"), True),
        ('tag("crm") and not org("公司/销售/")', FilterContext(tag="crm", org="公司/销售/"), False),
        ('(tag("crm") or tag("erp")) and org("p/")', FilterContext(tag="erp", org="p/q"), True),
        ('(tag("crm") or tag("erp")) and org("p/")', FilterContext(tag="erp", org="x"), False),
        ("True", FilterContext(), True),
        ("False", FilterContext(tag="crm"), False),
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
        "tag()",
        "org()",
        "tag(1)",
        'tag("")',
        'tag(" a")',
        'org("a ")',
        "tag(name='a')",
        "tag(*['a'])",
        'tag("a") == "a"',
        "1",
        '"crm"',
        "None",
        "x",
        "tag",
        "1 + 1",
        "not",
        "True and",
        "(tag('a')",
        "tag('a') if True else False",
        "{'a': 1}",
        "f'{1}'",
        "lambda: True",
        "[x for x in ()]",
        "tag.__class__",
        "tag('a').upper",
        "().__class__",
        "[].__class__.__bases__",
        "import os",
        "tag('a'); import os",
        "__import__('os')",
        "__import__('os').system('id')",
        "eval('1')",
        "exec('x')",
        "open('x')",
        "compile('1', '1', 'eval')",
        "(lambda: __import__('os'))()",
        "getattr(tag, 'x')",
        "tag('x') or __import__('os')",
    ],
)
def test_invalid_or_malicious_filter_is_rejected(source):
    with pytest.raises(FilterExpressionError):
        parse_filter(source)


def test_filter_length_limit():
    with pytest.raises(FilterExpressionError, match="characters"):
        parse_filter(f"tag({'x' * FILTER_MAX_LENGTH})")
    parse_filter(f"tag('{'x' * (FILTER_MAX_LENGTH - 8)}')")


def test_filter_depth_limit():
    parse_filter("not " * FILTER_MAX_DEPTH + "True")
    with pytest.raises(FilterExpressionError, match="nesting"):
        parse_filter("not " * (FILTER_MAX_DEPTH + 1) + "True")


def test_filter_argument_count_limit():
    args = ", ".join(f"'{index}'" for index in range(FILTER_MAX_ARGS))
    parse_filter(f"tag({args})")
    args = ", ".join(f"'{index}'" for index in range(FILTER_MAX_ARGS + 1))
    with pytest.raises(FilterExpressionError, match="arguments"):
        parse_filter(f"tag({args})")


@pytest.mark.parametrize("source", [None, 'tag("crm") or org("公司/")', "tag()", "import os"])
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
