import pytest

from rigging.completion import Completion, CompletionPipeline
from rigging.generator import GenerateParams, get_generator

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001


def test_completion_generator_id() -> None:
    generator = get_generator("gpt-3.5")
    completion = Completion("foo", "bar", generator)
    assert completion.generator_id == "litellm!gpt-3.5"

    completion.generator = None
    assert completion.generator_id is None


def test_completion_properties() -> None:
    generator = get_generator("gpt-3.5")
    completion = Completion("foo", "bar", generator)
    assert completion.text == "foo"
    assert completion.generated == "bar"
    assert completion.generator == generator
    assert len(completion) == len("foo") + len("bar")
    assert completion.all == "foobar"


def test_completion_restart() -> None:
    generator = get_generator("gpt-3.5")
    completion = Completion("foo", "bar", generator)
    assert len(completion.restart()) == 3
    assert len(completion.restart(include_all=True)) == 6

    assert len(completion.fork("baz")) == 6
    assert len(completion.continue_("baz")) == 9

    completion.generator = None
    with pytest.raises(ValueError):
        completion.restart()


def test_completion_clone() -> None:
    generator = get_generator("gpt-3.5")
    original = Completion("foo", "bar", generator).meta(key="value")
    clone = original.clone()
    assert clone.text == original.text
    assert clone.generated == original.generated
    assert clone.metadata == original.metadata

    clone_2 = original.clone(only_messages=True)
    assert clone.metadata != clone_2.metadata


def test_completion_pipeline_with() -> None:
    pipeline = CompletionPipeline(get_generator("gpt-3.5"), "foo")
    with_pipeline = pipeline.with_(GenerateParams(max_tokens=123))
    assert with_pipeline == pipeline
    assert with_pipeline.params is not None
    assert with_pipeline.params.max_tokens == 123

    with_pipeline_2 = with_pipeline.with_(top_p=0.5)
    assert with_pipeline_2 != with_pipeline
    assert with_pipeline_2.params is not None
    assert with_pipeline_2.params.max_tokens == 123
    assert with_pipeline_2.params.top_p == 0.5


def test_completion_pipeline_fork() -> None:
    pipeline = CompletionPipeline(get_generator("gpt-3.5"), "foo")
    forked_1 = pipeline.fork("bar")
    forked_2 = pipeline.fork("baz")

    assert pipeline != forked_1 != forked_2
    assert pipeline.text == "foo"
    assert forked_1.text == "foobar"
    assert forked_2.text == "foobaz"


def test_completion_pipeline_meta() -> None:
    pipeline = CompletionPipeline(get_generator("gpt-3.5"), "foo")
    with_meta = pipeline.meta(key="value")
    assert with_meta == pipeline
    assert with_meta.metadata == {"key": "value"}


def test_completion_pipeline_apply() -> None:
    pipeline = CompletionPipeline(get_generator("gpt-3.5"), "Hello $name")
    applied = pipeline.apply(name="World", noexist="123")
    assert pipeline != applied
    assert pipeline.text == "Hello $name"
    assert applied.text == "Hello World"
