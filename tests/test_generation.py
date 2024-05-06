import typing as t

import pytest

from rigging import Message
from rigging.generator import GenerateParams, Generator
from rigging.model import YesNoAnswer


class EchoGenerator(Generator):
    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
        *,
        prefix: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        if prefix is not None:
            messages = [list(m) + list(prefix) for m in messages]

        assert len(messages) == 1
        return [Message(role="assistant", content=messages[-1][-1].content) for m in messages]


class CallbackGenerator(Generator):
    callback: t.Callable[["CallbackGenerator", t.Sequence[Message]], str] | None = None

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
        *,
        prefix: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        if prefix is not None:
            messages = [list(prefix) + list(m) for m in messages]

        assert len(messages) == 1
        assert self.callback is not None
        return [Message(role="assistant", content=self.callback(self, m)) for m in messages]


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
