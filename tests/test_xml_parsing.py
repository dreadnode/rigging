import typing as t
from contextlib import nullcontext as does_not_raise

import pytest
from pydantic import field_validator

from rigging.model import CoreModel


class Answer(CoreModel):
    content: str


class Question(CoreModel):
    content: str


class QuestionAnswer(CoreModel):
    question: Question
    answer: Answer


class YesNoAnswer(CoreModel):
    boolean: bool

    @field_validator("boolean", mode="before")
    def parse_str_to_bool(cls, v: t.Any) -> t.Any:
        if isinstance(v, str):
            if v.strip().lower().startswith("yes"):
                return True
            elif v.strip().lower().startswith("no"):
                return False
        return v


class CommaDelimitedAnswer(CoreModel, tag="delimited-answer"):
    content: str

    @property
    def items(self) -> list[str]:
        return [i.strip() for i in self.content.split(",")]

    @field_validator("content", mode="before")
    def parse_str_to_list(cls, v: t.Any) -> t.Any:
        if not isinstance(v, str) or "," not in v:
            raise ValueError(f"Cannot parse content as a comma delimited list: {v}")
        return v


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
    ],
)
def test_xml_parsing(content: str, expectations: list[tuple[CoreModel, str]]) -> None:
    for model, expected in expectations:
        obj, _ = model.extract_xml(content)  # type: ignore [var-annotated]
        assert obj is not None, "Failed to parse model"
        if isinstance(expected, str):
            assert getattr(obj, "content", None) == expected, "Failed to parse content"
        elif isinstance(expected, CoreModel):
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
        ("<delimited-answer>hello,world</delimited-answer>", CommaDelimitedAnswer, does_not_raise()),
        ("<delimited-answer>hello, world</delimited-answer>", CommaDelimitedAnswer, does_not_raise()),
        ("<delimited-answer>hello, world, </delimited-answer>", CommaDelimitedAnswer, does_not_raise()),
        ("<delimited-answer>hello</delimited-answer>", CommaDelimitedAnswer, pytest.raises(ValueError)),
        ("<delimited-answer>hello;test;stuff</delimited-answer>", CommaDelimitedAnswer, pytest.raises(ValueError)),
    ],
)
def test_xml_parsing_with_validation(content: str, model: CoreModel, expectation: t.ContextManager[t.Any]) -> None:
    with expectation:
        obj, _ = model.extract_xml(content)  # type: ignore [var-annotated]
