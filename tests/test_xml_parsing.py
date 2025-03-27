import typing as t
from contextlib import nullcontext as does_not_raise

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

# Models to use during tests


class NameWithThings(Model):
    name: str = attr()
    things: list[str] = element("thing")


class Inner(Model):
    type: str = attr()
    content: str


class Outer(Model):
    content: str


class A(Model):
    content: str


class B(Model):
    content: str


class C(Model):
    content: str


class Tag(Model):
    content: str


class Wrapped(Model):
    inners: list[Inner] = element()


# Tests


@pytest.mark.parametrize(
    "content, models",
    [
        pytest.param("<answer>hello</answer>", [Answer(content="hello")], id="single_answer"),
        pytest.param("Random data <question>hello</question>", [Question(content="hello")], id="single_question"),
        pytest.param(
            "<answer> <question>hello</question>", [Question(content="hello")], id="single_question_with_unrelated_tag"
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
            "<question> Should I answer between <answer> tags? </question> <answer>hello</answer>",
            [Question(content=" Should I answer between <answer> tags? "), Answer(content="hello")],
            id="question_with_answer_tag_1",
        ),
        pytest.param(
            "<question> Should I answer between <answer> tags? </question> <answer>hello</answer>",
            [Question(content=" Should I answer between <answer> tags? "), Answer(content="hello")],
            id="question_with_answer_tag_2",
        ),
        pytest.param(
            "<question-answer><question>hello</question><answer>world</answer></question-answer>",
            [QuestionAnswer(question=Question(content="hello"), answer=Answer(content="world"))],
            id="question_answer",
        ),
        pytest.param(
            "<delimited-answer>\n- hello\n - world</delimited-answer>",
            [DelimitedAnswer(content="\n- hello\n - world", _items=["hello", "world"])],
            id="newline_delimited_answer",
        ),
        pytest.param(
            "<delimited-answer>hello, world, foo | bar</delimited-answer>",
            [DelimitedAnswer(content="hello, world, foo | bar", _items=["hello", "world", "foo | bar"])],
            id="comma_delimited_answer",
        ),
        pytest.param(
            "<delimited-answer>hello / world / foo / bar, test | value</delimited-answer>",
            [
                DelimitedAnswer(
                    content="hello / world / foo / bar, test | value",
                    _items=["hello", "world", "foo", "bar, test | value"],
                )
            ],
            id="slash_delimited_answer",
        ),
        pytest.param(
            '<name-with-things name="test"><thing>a</thing><thing>b</thing></name-with-things>',
            [NameWithThings(name="test", things=["a", "b"])],
            id="name_with_things",
        ),
        pytest.param(
            '<wrapped><inner type="cat">meow</inner><inner type="dog">bark</inner></wrapped>',
            [Wrapped(inners=[Inner(type="cat", content="meow"), Inner(type="dog", content="bark")])],
            id="wrapped",
        ),
        pytest.param(
            "<outer>text before <inner>nested content</inner> text after</outer>",
            [Outer(content="text before <inner>nested content</inner> text after")],
            id="outer_with_inner_tag",
        ),
        pytest.param(
            "<outer>Some text with <inner> incomplete tags</outer>",
            [Outer(content="Some text with <inner> incomplete tags")],
            id="incomplete_inner_tag",
        ),
        pytest.param(
            "Commentary about the `<inner>` tag\n\n<outer>\nsomething is <inner> and </inner>\n</outer>",
            [Outer(content="\nsomething is <inner> and </inner>\n")],
            id="comment_before_tag",
        ),
        pytest.param(
            "<a>first <b>overlap <a>nested</a> continue</b> end</a>",
            [A(content="nested")],
            id="overlapping_tags",
        ),
        pytest.param(
            "<outer>level 1 <middle>level 2 <inner>level 3</inner> still 2</middle> back to 1</outer>",
            [Outer(content="level 1 <middle>level 2 <inner>level 3</inner> still 2</middle> back to 1")],
            id="multiple_nested_levels",
        ),
        pytest.param(
            "Here's how to use <tag>: <tag>actual content</tag>",
            [Tag(content="actual content")],
            id="tag_in_text_and_xml",
        ),
        pytest.param(
            "<outer>Text with &lt;inner&gt; as escaped HTML entities</outer>",
            [Outer(content="Text with <inner> as escaped HTML entities")],
            id="escaped_xml_in_content",
        ),
        pytest.param(
            "<answer>first</answer> text <answer>second</answer> more text <answer>third</answer>",
            [Answer(content="first"), Answer(content="second"), Answer(content="third")],
            id="multiple_same_tags_order",
        ),
    ],
)
def test_xml_parsing(content: str, models: list[Model]) -> None:
    parsed = parse_many(content, *{type(m) for m in models})
    print(models)
    print(parsed)

    assert len(parsed) == len(models), "Failed to parse set"
    for (obj, _), expected in zip(parsed, models):
        print(obj)
        assert (
            obj.model_dump() == expected.model_dump()
        ), f"Failed to parse model {expected.__class__.__name__} <- {str(obj)} ({parsed})"

    print("---")


@pytest.mark.parametrize(
    "content, model, expectation",
    [
        pytest.param("<yes-no-answer>yes</yes-no-answer>", YesNoAnswer, does_not_raise(), id="yes_no_answer_1"),
        pytest.param("<yes-no-answer>no</yes-no-answer>", YesNoAnswer, does_not_raise(), id="yes_no_answer_2"),
        pytest.param("<yes-no-answer>Yes</yes-no-answer>", YesNoAnswer, does_not_raise(), id="yes_no_answer_3"),
        pytest.param(
            "<yes-no-answer>No, extra stuff</yes-no-answer>", YesNoAnswer, does_not_raise(), id="yes_no_answer_4"
        ),
        pytest.param(
            "<yes-no-answer>No, stuff <internal-tag></yes-no-answer>",
            YesNoAnswer,
            does_not_raise(),
            id="yes_no_answer_5",
        ),
        pytest.param(
            "<yes-no-answer>Invalid</yes-no-answer>", YesNoAnswer, pytest.raises(ValueError), id="yes_no_answer_invalid"
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
def test_xml_parsing_with_validation(content: str, model: Model, expectation: t.ContextManager[t.Any]) -> None:
    with expectation:
        model.from_text(content)


@pytest.mark.parametrize(
    "content, count, model",
    [
        pytest.param(
            "<yes-no-answer>yes</yes-no-answer><yes-no-answer>no</yes-no-answer>", 2, YesNoAnswer, id="yes_no_many"
        ),
        pytest.param(
            "<delimited-answer><delimited-answer>1, 2, 3</delimited-answer>", 1, DelimitedAnswer, id="delimited_single"
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
    outer, _ = Outer.one_from_text(content)
    assert outer.content == "text before <a>nested content</a> text after"

    # Test parsing the inner tag
    inner, _ = A.one_from_text(content)
    assert inner.content == "nested content"


def test_same_tag_mentioned_in_text() -> None:
    """Test that mentions of tags in text don't confuse the parser."""
    content = "Commentary about the `<outer>` tag\n\n<outer>\nsomething is <inner> and </inner>\n</outer>"

    outer, _ = Outer.one_from_text(content)
    assert outer.content == "\nsomething is <inner> and </inner>\n"


def test_nested_incomplete_tags() -> None:
    """Test handling of content with incomplete nested tags."""
    content = "<outer>Content with <inner> incomplete tag</outer>"

    outer, _ = Outer.one_from_text(content)
    assert outer.content == "Content with <inner> incomplete tag"


def test_edge_case_xml_entities() -> None:
    content = "<outer>Content with &lt;tag&gt; as XML entities</outer>"
    outer, _ = Outer.one_from_text(content)
    assert outer.content == "Content with <tag> as XML entities"
