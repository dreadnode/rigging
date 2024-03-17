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


class NameWithThings(Model):
    name: str = attr()
    things: list[str] = element("thing")


class Inner(Model):
    type: str = attr()
    content: str


class Wrapped(Model):
    inners: list[Inner] = element()


@pytest.mark.parametrize(
    "content, expectations",
    [
        pytest.param("<answer>hello</answer>", [(Answer, "hello")], id="single_answer"),
        pytest.param("Random data <question>hello</question>", [(Question, "hello")], id="single_question"),
        pytest.param(
            "<answer> <question>hello</question>", [(Question, "hello")], id="single_question_with_unrelated_tag"
        ),
        pytest.param(
            "<answer>hello</answer><question>world</question>",
            [(Answer, "hello"), (Question, "world")],
            id="answer_and_question",
        ),
        pytest.param(
            "<question>hello</question><answer>world</answer>",
            [(Question, "hello"), (Answer, "world")],
            id="question_and_answer",
        ),
        pytest.param(
            "Sure I'll answer between <answer> tags. <answer>hello</answer>",
            [(Answer, "hello")],
            id="answer_with_duplicate_start_tag",
        ),
        pytest.param(
            "Sure I'll answer between <answer></answer> tags. <answer>hello</answer>",
            [(Answer, "hello")],
            id="answer_with_duplicate_tags",
        ),
        pytest.param(
            "<question> Should I answer between <answer> tags? </question> <answer>hello</answer>",
            [(Question, " Should I answer between <answer> tags? "), (Answer, "hello")],
            id="question_with_answer_tag",
        ),
        pytest.param(
            "<question> Should I answer between <answer> tags? </question> <answer>hello</answer>",
            [(Question, " Should I answer between <answer> tags? "), (Answer, "hello")],
            id="question_with_answer_tag",
        ),
        pytest.param(
            "<question-answer><question>hello</question><answer>world</answer></question-answer>",
            [(QuestionAnswer, QuestionAnswer(question=Question(content="hello"), answer=Answer(content="world")))],
            id="question_answer",
        ),
        pytest.param(
            "<delimited-answer>\n- hello\n - world</delimited-answer>",
            [(DelimitedAnswer, ["hello", "world"])],
        ),
        pytest.param(
            "<delimited-answer>hello, world, foo | bar</delimited-answer>",
            [(DelimitedAnswer, ["hello", "world", "foo | bar"])],
        ),
        pytest.param(
            "<delimited-answer>hello / world / foo / bar, test | value</delimited-answer>",
            [(DelimitedAnswer, ["hello", "world", "foo", "bar, test | value"])],
        ),
        pytest.param(
            '<name-with-things name="test"><thing>a</thing><thing>b</thing></name-with-things>',
            [(NameWithThings, NameWithThings(name="test", things=["a", "b"]))],
        ),
        pytest.param(
            '<wrapped><inner type="cat">meow</inner><inner type="dog">bark</inner></wrapped>',
            [(Wrapped, Wrapped(inners=[Inner(type="cat", content="meow"), Inner(type="dog", content="bark")]))],
        ),
    ],
)
def test_xml_parsing(content: str, expectations: list[tuple[Model, str]]) -> None:
    for model, expected in expectations:
        obj, _ = model.extract_xml(content)  # type: ignore [var-annotated]
        assert obj is not None, "Failed to parse model"
        if isinstance(expected, str):
            assert getattr(obj, "content", None) == expected, "Failed to parse content"
        if isinstance(expected, list):
            assert getattr(obj, "items", None) == expected, "Failed to parse items"
        elif isinstance(expected, Model):
            assert obj.model_dump() == expected.model_dump(), "Failed to parse model"


@pytest.mark.parametrize(
    "content, model, expectation",
    [
        ("<yes-no-answer>yes</yes-no-answer>", YesNoAnswer, does_not_raise()),
        ("<yes-no-answer>no</yes-no-answer>", YesNoAnswer, does_not_raise()),
        ("<yes-no-answer>Yes</yes-no-answer>", YesNoAnswer, does_not_raise()),
        ("<yes-no-answer>No, extra stuff</yes-no-answer>", YesNoAnswer, does_not_raise()),
        ("<yes-no-answer>No, stuff <internal-tag></yes-no-answer>", YesNoAnswer, does_not_raise()),
        ("<yes-no-answer>Invalid</yes-no-answer>", YesNoAnswer, pytest.raises(ValueError)),
        ("<delimited-answer>hello world</delimited-answer>", DelimitedAnswer, pytest.raises(ValueError)),
        ("<comma-delimited-answer>hello,world</comma-delimited-answer>", CommaDelimitedAnswer, does_not_raise()),
        ("<comma-delimited-answer>hello, world</comma-delimited-answer>", CommaDelimitedAnswer, does_not_raise()),
        ("<comma-delimited-answer>hello, world, </comma-delimited-answer>", CommaDelimitedAnswer, does_not_raise()),
        ("<comma-delimited-answer>hello</comma-delimited-answer>", CommaDelimitedAnswer, pytest.raises(ValueError)),
        (
            "<comma-delimited-answer>hello;test;stuff</comma-delimited-answer>",
            CommaDelimitedAnswer,
            pytest.raises(ValueError),
        ),
    ],
)
def test_xml_parsing_with_validation(content: str, model: Model, expectation: t.ContextManager[t.Any]) -> None:
    with expectation:
        obj, _ = model.extract_xml(content)  # type: ignore [var-annotated]
