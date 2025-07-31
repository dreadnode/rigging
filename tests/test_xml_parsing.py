import typing as t
from contextlib import nullcontext as does_not_raise
from textwrap import dedent

import pytest

from rigging.model import (
    Answer,
    CommaDelimitedAnswer,
    DelimitedAnswer,
    Model,
    Question,
    QuestionAnswer,
    YesNoAnswer,
    attr,
    element,
)
from rigging.parsing import parse_many

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001, FBT001, FBT002

# Models to use during tests


class NameWithThings(Model):
    name: str = attr()
    things: list[str] = element("thing")


class ContentWithTypeAttr(Model, tag="inner"):
    type: str = attr()
    content: str


class Content(Model, tag="outer"):
    content: str


class ContentA(Model, tag="a"):
    content: str


class ContentB(Model, tag="b"):
    content: str


class ContentC(Model, tag="c"):
    content: str


class ContentTag(Model, tag="tag"):
    content: str


class Wrapped(Model):
    inners: list[ContentWithTypeAttr] = element()


class ContentAsElement(Model, tag="outer"):
    content: str = element()


class MultiFieldModel(Model):
    type_: str = attr(name="type")
    foo_field: str = element(tag="foo")
    bar_field: str = element(tag="bar")


class MixedModel(Model):
    content_field: str = element(tag="content")
    nested: ContentAsElement = element()


MULTI_LINE_TEXT = """\
Multiline content with indentation
some extra spaces

Some more text\
"""
MULTI_LINE_CONTENT_TAG = """\
<tag>
  Multiline content with indentation
  some extra spaces

  Some more text
</tag>
"""
MULTI_LINE_CONTENT_AS_ELEMENT = """\
<outer>
  <content>
    Multiline content with indentation
    some extra spaces

    Some more text
  </content>
</outer>
"""
MULTI_LINE_CONTENT_MULTI_FIELD = """\
<multi-field-model type="cmd">
  <foo>Process <pid> terminated</foo>
  <bar>
    Multiline content with indentation
    some extra spaces

    Some more text
  </bar>
</multi-field-model>
"""


@pytest.mark.parametrize(
    ("content", "models"),
    [
        pytest.param("<answer>hello</answer>", [Answer(content="hello")], id="single_answer"),
        pytest.param(
            "Random data <question>hello</question>",
            [Question(content="hello")],
            id="single_question",
        ),
        pytest.param(
            "<answer> <question>hello</question>",
            [Question(content="hello")],
            id="single_question_with_unrelated_tag",
        ),
        pytest.param(
            "<answer>hello</answer><question>world</question>",
            [Answer(content="hello"), Question(content="world")],
            id="answer_and_question",
        ),
        pytest.param(
            "<question>hello</question><answer>world</answer>",
            [Question(content="hello"), Answer(content="world")],
            id="question_and_answer",
        ),
        pytest.param(
            "Sure I'll answer between <answer> tags. <answer>hello</answer>",
            [Answer(content="hello")],
            id="answer_with_duplicate_start_tag",
        ),
        pytest.param(
            "Sure I'll answer between <answer></answer> tags. <answer>hello</answer>",
            [Answer(content=""), Answer(content="hello")],
            id="answer_with_duplicate_tags",
        ),
        pytest.param(
            "<question>Should I answer between <answer> tags?</question> <answer>hello</answer>",
            [Question(content="Should I answer between <answer> tags?"), Answer(content="hello")],
            id="question_with_answer_tag_1",
        ),
        pytest.param(
            "<question>Should I answer between <answer> tags?</question> <answer>hello</answer>",
            [Question(content="Should I answer between <answer> tags?"), Answer(content="hello")],
            id="question_with_answer_tag_2",
        ),
        pytest.param(
            dedent("""\
            <question-answer>
              <question>hello</question>
              <answer>world</answer>
            </question-answer>\
            """),
            [QuestionAnswer(question=Question(content="hello"), answer=Answer(content="world"))],
            id="question_answer",
        ),
        pytest.param(
            dedent("""\
            <delimited-answer>
              - hello
              - world
            </delimited-answer>\
            """),
            [DelimitedAnswer(content="- hello\n- world", _items=["hello", "world"])],
            id="newline_delimited_answer",
        ),
        pytest.param(
            "<delimited-answer>hello, world, foo | bar</delimited-answer>",
            [
                DelimitedAnswer(
                    content="hello, world, foo | bar",
                    _items=["hello", "world", "foo | bar"],
                ),
            ],
            id="comma_delimited_answer",
        ),
        pytest.param(
            "<delimited-answer>hello / world / foo / bar, test | value</delimited-answer>",
            [
                DelimitedAnswer(
                    content="hello / world / foo / bar, test | value",
                    _items=["hello", "world", "foo", "bar, test | value"],
                ),
            ],
            id="slash_delimited_answer",
        ),
        pytest.param(
            dedent("""\
            <name-with-things name="test">
              <thing>a</thing>
              <thing>b</thing>
            </name-with-things>\
            """),
            [NameWithThings(name="test", things=["a", "b"])],
            id="name_with_things",
        ),
        pytest.param(
            dedent("""\
            <wrapped>
              <inner type="cat">meow</inner>
              <inner type="dog">bark</inner>
            </wrapped>\
            """),
            [
                Wrapped(
                    inners=[
                        ContentWithTypeAttr(type="cat", content="meow"),
                        ContentWithTypeAttr(type="dog", content="bark"),
                    ],
                ),
            ],
            id="wrapped",
        ),
        pytest.param(
            "<outer>text before <inner>nested content</inner> text after</outer>",
            [Content(content="text before <inner>nested content</inner> text after")],
            id="outer_with_inner_tag",
        ),
        pytest.param(
            "<outer>Some text with <inner> incomplete tags</outer>",
            [Content(content="Some text with <inner> incomplete tags")],
            id="incomplete_inner_tag",
        ),
        pytest.param(
            "Commentary about the `<inner>` tag\n\n<outer>something is <inner> and </inner></outer>",
            [Content(content="something is <inner> and </inner>")],
            id="comment_before_tag",
        ),
        pytest.param(
            "<a>first <b>overlap <a>nested</a> continue</b> end</a>",
            [ContentA(content="nested")],
            id="overlapping_tags",
        ),
        pytest.param(
            "<outer>level 1 <middle>level 2 <inner>level 3</inner> still 2</middle> back to 1</outer>",
            [
                Content(
                    content="level 1 <middle>level 2 <inner>level 3</inner> still 2</middle> back to 1",
                ),
            ],
            id="multiple_nested_levels",
        ),
        pytest.param(
            "Here's how to use <tag>: <tag>actual content</tag>",
            [ContentTag(content="actual content")],
            id="tag_in_text_and_xml",
        ),
        pytest.param(
            MULTI_LINE_CONTENT_TAG,
            [ContentTag(content=MULTI_LINE_TEXT)],
            id="indented_multiline_content_tag",
        ),
        pytest.param(
            MULTI_LINE_CONTENT_AS_ELEMENT,
            [ContentAsElement(content=MULTI_LINE_TEXT)],
            id="indented_multiline_content_as_element",
        ),
        pytest.param(
            MULTI_LINE_CONTENT_MULTI_FIELD,
            [
                MultiFieldModel(
                    type_="cmd",
                    foo_field="Process <pid> terminated",
                    bar_field=MULTI_LINE_TEXT,
                ),
            ],
            id="indented_multiline_content_multi_field",
        ),
        pytest.param(
            "<answer>first</answer> text <answer>second</answer> more text <answer>third</answer>",
            [Answer(content="first"), Answer(content="second"), Answer(content="third")],
            id="multiple_same_tags_order",
        ),
        pytest.param(
            "<outer>Text with <tag> & XML entities</outer>",
            [Content(content="Text with <tag> & XML entities")],
            id="xml_entities_in_content",
        ),
        pytest.param(
            "<outer>Command output: <DIR> Program Files & <DIR> Users</outer>",
            [Content(content="Command output: <DIR> Program Files & <DIR> Users")],
            id="xml_breaking_chars_with_ampersand",
        ),
        pytest.param(
            dedent("""\
            <multi-field-model type="cmd">
              <foo>Process <pid> terminated</foo>
              <bar>Exit code <1></bar>
            </multi-field-model>\
            """),
            [
                MultiFieldModel(
                    type_="cmd",
                    foo_field="Process <pid> terminated",
                    bar_field="Exit code <1>",
                ),
            ],
            id="multi_field_with_xml_breaking_chars",
        ),
        pytest.param(
            dedent("""\
            <outer>
              Volume in drive C:
               Directory of C:\\
               01/02/2024 <DIR> Program Files
            </outer>\
            """),
            [
                Content(
                    content="Volume in drive C:\n Directory of C:\\\n 01/02/2024 <DIR> Program Files",
                ),
            ],
            id="shell_output_simulation",
        ),
        pytest.param(
            dedent("""\
            <mixed-model>
              <content>Error in <module> at line <42></content>
              <outer>
                <content>normal nested content</content>
              </outer>
            </mixed-model>\
            """),
            [
                MixedModel(
                    content_field="Error in <module> at line <42>",
                    nested=ContentAsElement(content="normal nested content"),
                ),
            ],
            id="mixed_model_with_nested_and_xml_breaking",
        ),
    ],
)
def test_xml_parsing(content: str, models: list[Model]) -> None:
    parsed = parse_many(content, *{type(m) for m in models})
    assert len(parsed) == len(models), "Failed to parse set"
    for (obj, slice_), expected in zip(parsed, models, strict=False):
        assert obj.model_dump() == expected.model_dump(), (
            f"Failed to parse model {expected.__class__.__name__} <- {obj!s} ({parsed})"
        )
        xml = obj.to_pretty_xml()
        assert xml == content[slice_], (
            f"Failed to serialize model {expected.__class__.__name__} back to XML: {xml!r} != {content!r}"
        )


@pytest.mark.parametrize(
    ("content", "models"),
    [
        # These cases parse correctly, but their XML representation doesn't yield
        # the original content as some escape sequences are not preserved.
        pytest.param(
            "<outer>Text with &lt;inner&gt; as escaped HTML entities</outer>",
            [Content(content="Text with <inner> as escaped HTML entities")],
            id="escaped_xml_in_content",
        ),
        pytest.param(
            "<outer><content><![CDATA[Already wrapped <content> & entities]]></content></outer>",
            [ContentAsElement(content="Already wrapped <content> & entities")],
            id="already_cdata_wrapped_content",
        ),
    ],
)
def test_xml_parsing_without_exact_roundtrip(content: str, models: list[Model]) -> None:
    parsed = parse_many(content, *{type(m) for m in models})
    assert len(parsed) == len(models), "Failed to parse set"
    for (obj, _), expected in zip(parsed, models, strict=False):
        assert obj.model_dump() == expected.model_dump(), (
            f"Failed to parse model {expected.__class__.__name__} <- {obj!s} ({parsed})"
        )


@pytest.mark.parametrize(
    ("content", "model", "expectation"),
    [
        pytest.param(
            "<yes-no-answer>yes</yes-no-answer>",
            YesNoAnswer,
            does_not_raise(),
            id="yes_no_answer_1",
        ),
        pytest.param(
            "<yes-no-answer>no</yes-no-answer>",
            YesNoAnswer,
            does_not_raise(),
            id="yes_no_answer_2",
        ),
        pytest.param(
            "<yes-no-answer>Yes</yes-no-answer>",
            YesNoAnswer,
            does_not_raise(),
            id="yes_no_answer_3",
        ),
        pytest.param(
            "<yes-no-answer>No, extra stuff</yes-no-answer>",
            YesNoAnswer,
            does_not_raise(),
            id="yes_no_answer_4",
        ),
        pytest.param(
            "<yes-no-answer>No, stuff <internal-tag></yes-no-answer>",
            YesNoAnswer,
            does_not_raise(),
            id="yes_no_answer_5",
        ),
        pytest.param(
            "<yes-no-answer>Invalid</yes-no-answer>",
            YesNoAnswer,
            pytest.raises(ValueError),
            id="yes_no_answer_invalid",
        ),
        pytest.param(
            "<delimited-answer>hello world</delimited-answer>",
            DelimitedAnswer,
            does_not_raise(),
            id="delimited_answer_1",
        ),
        pytest.param(
            "<comma-delimited-answer>hello,world</comma-delimited-answer>",
            CommaDelimitedAnswer,
            does_not_raise(),
            id="comma_delimited_answer_1",
        ),
        pytest.param(
            "<comma-delimited-answer>hello, world</comma-delimited-answer>",
            CommaDelimitedAnswer,
            does_not_raise(),
            id="comma_delimited_answer_2",
        ),
        pytest.param(
            "<comma-delimited-answer>hello, world, </comma-delimited-answer>",
            CommaDelimitedAnswer,
            does_not_raise(),
            id="comma_delimited_answer_3",
        ),
        pytest.param(
            "<comma-delimited-answer>hello</comma-delimited-answer>",
            CommaDelimitedAnswer,
            does_not_raise(),
            id="comma_delimited_answer_4",
        ),
        pytest.param(
            # This is a little strange, as you'd expect the wrong delimiter to fail,
            # however we now support the idea that a single item is a valid list
            # TODO: Decide if we want to keep this behavior
            "<comma-delimited-answer>hello;test;stuff</comma-delimited-answer>",
            CommaDelimitedAnswer,
            does_not_raise(),
            id="comma_delimited_answer_5",
        ),
    ],
)
def test_xml_parsing_with_validation(
    content: str,
    model: Model,
    expectation: t.ContextManager[t.Any],
) -> None:
    with expectation:
        model.from_text(content)


@pytest.mark.parametrize(
    ("content", "count", "model"),
    [
        pytest.param(
            "<yes-no-answer>yes</yes-no-answer><yes-no-answer>no</yes-no-answer>",
            2,
            YesNoAnswer,
            id="yes_no_many",
        ),
        pytest.param(
            "<delimited-answer><delimited-answer>1, 2, 3</delimited-answer>",
            1,
            DelimitedAnswer,
            id="delimited_single",
        ),
    ],
)
def test_xml_parsing_sets(content: str, count: int, model: Model) -> None:
    models = model.from_text(content)
    assert len(models) == count, "Failed to parse model set"


def test_tag_order_preservation() -> None:
    content = "<answer>first</answer> text <answer>second</answer> more text <answer>third</answer>"
    answers = [model for model, _ in Answer.from_text(content)]
    assert len(answers) == 3
    assert answers[0].content == "first"
    assert answers[1].content == "second"
    assert answers[2].content == "third"


def test_nested_tag_parsing() -> None:
    content = "<outer>text before <a>nested content</a> text after</outer>"

    # Test parsing the outer tag
    outer, _ = Content.one_from_text(content)
    assert outer.content == "text before <a>nested content</a> text after"

    # Test parsing the inner tag
    inner, _ = ContentA.one_from_text(content)
    assert inner.content == "nested content"


def test_nested_incomplete_tags() -> None:
    content = "<outer>Content with <inner> incomplete tag</outer>"

    outer, _ = Content.one_from_text(content)
    assert outer.content == "Content with <inner> incomplete tag"


def test_edge_case_xml_entities() -> None:
    content = "<outer>Content with &lt;tag&gt; as XML entities</outer>"
    outer, _ = Content.one_from_text(content)
    assert outer.content == "Content with <tag> as XML entities"


@pytest.mark.parametrize(
    "content_text",
    [
        "No XML breaking chars here",
        "Already &lt;escaped&gt; content",
        "<![CDATA[Already wrapped]]>",
        "",
        "   ",
    ],
)
def test_preprocess_for_cdata(content_text: str) -> None:
    input_xml = f"<outer><content>{content_text}</content></outer>"
    processed = ContentAsElement.preprocess_with_cdata(input_xml)

    # Should not add CDATA wrapper for safe content (unless already present)
    if "<![CDATA[" not in content_text:
        assert processed.count("<![CDATA[") == 0
