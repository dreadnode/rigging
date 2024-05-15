import typing as t

import pytest

from rigging import Message
from rigging.generator import GenerateParams, Generator
from rigging.model import YesNoAnswer
from rigging.parsing import try_parse


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

    def generate_texts(
        self, texts: t.Sequence[str], params: t.Sequence[GenerateParams], *, prefix: str | None = None
    ) -> t.Sequence[str]:
        if prefix is not None:
            texts = [t + prefix for t in texts]

        assert len(texts) == 1
        return [texts[-1]]


class CallbackGenerator(Generator):
    message_callback: t.Callable[["CallbackGenerator", t.Sequence[Message]], str] | None = None
    text_callback: t.Callable[["CallbackGenerator", str], str] | None = None

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
        assert self.message_callback is not None
        return [Message(role="assistant", content=self.message_callback(self, m)) for m in messages]

    def generate_texts(
        self, texts: t.Sequence[str], params: t.Sequence[GenerateParams], *, prefix: str | None = None
    ) -> t.Sequence[str]:
        if prefix is not None:
            texts = [prefix + t for t in texts]

        assert len(texts) == 1
        assert self.text_callback is not None
        return [self.text_callback(self, text) for text in texts]


def test_chat_until_parsed_as_with_reset() -> None:
    generator = CallbackGenerator(model="callback", params=GenerateParams())

    def valid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 1
        assert messages[0].content == "original"
        return "<yes-no-answer>yes</yes-no-answer>"

    def invalid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        self.message_callback = valid_cb
        return "dropped"

    generator.message_callback = invalid_cb
    chat = generator.chat([{"role": "user", "content": "original"}]).until_parsed_as(YesNoAnswer).run()
    assert len(chat) == 2
    assert chat.last.try_parse(YesNoAnswer) is not None


@pytest.mark.parametrize("drop_dialog", [True, False])
def test_chat_until_parsed_as_with_recovery(drop_dialog: bool) -> None:
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
        self.message_callback = valid_cb
        return "invalid2"

    def invalid_cb_1(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 1
        assert messages[0].content == "original"
        self.message_callback = invalid_cb_2
        return "invalid1"

    generator.message_callback = invalid_cb_1
    chat = (
        generator.chat([{"role": "user", "content": "original"}])
        .until_parsed_as(YesNoAnswer, attempt_recovery=True, drop_dialog=drop_dialog)
        .run()
    )

    assert len(chat) == (2 if drop_dialog else 6)
    assert chat.last.try_parse(YesNoAnswer) is not None


def test_completion_until_parsed_as_with_reset() -> None:
    generator = CallbackGenerator(model="callback", params=GenerateParams())

    def valid_cb(self: CallbackGenerator, text: str) -> str:
        assert text == "original"
        return "<yes-no-answer>yes</yes-no-answer>"

    def invalid_cb(self: CallbackGenerator, text: str) -> str:
        self.text_callback = valid_cb
        return "dropped"

    generator.text_callback = invalid_cb
    completion = generator.complete("original").until_parsed_as(YesNoAnswer).run()
    assert try_parse(completion.generated, YesNoAnswer) is not None
