from __future__ import annotations

import pytest

from rigging.completion import Completion, PendingCompletion
from rigging.generator import GenerateParams, get_generator


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


def test_pending_completion_with() -> None:
    pending = PendingCompletion(get_generator("gpt-3.5"), "foo")
    with_pending = pending.with_(GenerateParams(max_tokens=123))
    assert with_pending == pending
    assert with_pending.params is not None
    assert with_pending.params.max_tokens == 123

    with_pending_2 = with_pending.with_(top_p=0.5)
    assert with_pending_2 != with_pending
    assert with_pending_2.params is not None
    assert with_pending_2.params.max_tokens == 123
    assert with_pending_2.params.top_p == 0.5


def test_pending_completion_fork() -> None:
    pending = PendingCompletion(get_generator("gpt-3.5"), "foo")
    forked_1 = pending.fork("bar")
    forked_2 = pending.fork("baz")

    assert pending != forked_1 != forked_2
    assert pending.text == "foo"
    assert forked_1.text == "foobar"
    assert forked_2.text == "foobaz"


def test_pending_completion_meta() -> None:
    pending = PendingCompletion(get_generator("gpt-3.5"), "foo")
    with_meta = pending.meta(key="value")
    assert with_meta == pending
    assert with_meta.metadata == {"key": "value"}


def test_pending_completion_apply() -> None:
    pending = PendingCompletion(get_generator("gpt-3.5"), "Hello $name")
    applied = pending.apply(name="World", noexist="123")
    assert pending != applied
    assert pending.text == "Hello $name"
    assert applied.text == "Hello World"
