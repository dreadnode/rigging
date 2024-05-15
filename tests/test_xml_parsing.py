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
            id="newline_delimited_answer",
        ),
        pytest.param(
            "<delimited-answer>hello, world, foo | bar</delimited-answer>",
            [(DelimitedAnswer, ["hello", "world", "foo | bar"])],
            id="comma_delimited_answer",
        ),
        pytest.param(
            "<delimited-answer>hello / world / foo / bar, test | value</delimited-answer>",
            [(DelimitedAnswer, ["hello", "world", "foo", "bar, test | value"])],
            id="slash_delimited_answer",
        ),
        pytest.param(
            '<name-with-things name="test"><thing>a</thing><thing>b</thing></name-with-things>',
            [(NameWithThings, NameWithThings(name="test", things=["a", "b"]))],
            id="name_with_things",
        ),
        pytest.param(
            '<wrapped><inner type="cat">meow</inner><inner type="dog">bark</inner></wrapped>',
            [(Wrapped, Wrapped(inners=[Inner(type="cat", content="meow"), Inner(type="dog", content="bark")]))],
            id="wrapped",
        ),
    ],
)
def test_xml_parsing(content: str, expectations: list[tuple[Model, str]]) -> None:
    # TODO: Sort out the types here a break this into focused functions
    for model, expected in expectations:
        obj, _ = t.cast(tuple[Model, slice], model.one_from_text(content))
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
        model.from_text(content)  # type: ignore [var-annotated]


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
    models = model.from_text(content)  # type: ignore [var-annotated]
    assert len(models) == count, "Failed to parse model set"
