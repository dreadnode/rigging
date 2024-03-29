import typing as t

import pytest

from rigging import Message
from rigging.generator import GenerateParams, Generator
from rigging.model import YesNoAnswer


class EchoGenerator(Generator):
    def complete(self, messages: t.Sequence[Message], overloads: GenerateParams) -> Message:
        return Message(role="assistant", content=messages[-1].content)


class CallbackGenerator(Generator):
    callback: t.Callable[["CallbackGenerator", t.Sequence[Message]], str] | None = None

    def complete(self, messages: t.Sequence[Message], overloads: GenerateParams) -> Message:
        assert self.callback is not None, "Callback must be defined for CallbackGenerator"
        return Message(role="assistant", content=self.callback(self, messages))


def test_until_parsed_as_with_reset() -> None:
    generator = CallbackGenerator(model="callback", params=GenerateParams())

    def valid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 1
        assert messages[0].content == "original"
        return "<yes-no-answer>yes</yes-no-answer>"

    def invalid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        self.callback = valid_cb
        return "dropped"

    generator.callback = invalid_cb
    chat = generator.chat([{"role": "user", "content": "original"}]).until_parsed_as(YesNoAnswer).run()
    assert len(chat) == 2
    assert chat.last.try_parse(YesNoAnswer) is not None


@pytest.mark.parametrize("drop_dialog", [True, False])
def test_until_parsed_as_with_recovery(drop_dialog: bool) -> None:
    generator = CallbackGenerator(model="callback", params=GenerateParams())

    def valid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 5
        assert messages[0].content == "original"
        assert messages[1].content == "invalid1"
        assert messages[3].content == "invalid2"
        return "<yes-no-answer>yes</yes-no-answer>"

    def invalid_cb_2(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 3, messages
        assert messages[0].content == "original"
        assert messages[1].content == "invalid1"
        assert "<system-error-model>" in messages[2].content
        self.callback = valid_cb
        return "invalid2"

    def invalid_cb_1(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 1
        assert messages[0].content == "original"
        self.callback = invalid_cb_2
        return "invalid1"

    generator.callback = invalid_cb_1
    chat = (
        generator.chat([{"role": "user", "content": "original"}])
        .until_parsed_as(YesNoAnswer, attempt_recovery=True, drop_dialog=drop_dialog)
        .run()
    )

    assert len(chat) == (2 if drop_dialog else 6)
    assert chat.last.try_parse(YesNoAnswer) is not None
