from __future__ import annotations

import typing as t

import pytest

from rigging import Message
from rigging.error import ExhaustedMaxRoundsError
from rigging.generator import GenerateParams, Generator
from rigging.generator.base import GeneratedMessage, GeneratedText
from rigging.model import YesNoAnswer
from rigging.parsing import try_parse


class FixedGenerator(Generator):
    text: str

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        return [GeneratedMessage.from_text(self.text) for _ in messages]

    async def generate_texts(
        self, texts: t.Sequence[str], params: t.Sequence[GenerateParams]
    ) -> t.Sequence[GeneratedText]:
        return [GeneratedText.from_text(self.text) for _ in texts]


class EchoGenerator(Generator):
    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        return [GeneratedMessage.from_text(m[-1].content) for m in messages]

    async def generate_texts(
        self, texts: t.Sequence[str], params: t.Sequence[GenerateParams]
    ) -> t.Sequence[GeneratedText]:
        return [GeneratedText.from_text(t) for t in texts]


class CallbackGenerator(Generator):
    message_callback: t.Callable[[CallbackGenerator, t.Sequence[Message]], str] | None = None
    text_callback: t.Callable[[CallbackGenerator, str], str] | None = None

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        assert self.message_callback is not None
        return [GeneratedMessage.from_text(self.message_callback(self, m)) for m in messages]

    async def generate_texts(
        self, texts: t.Sequence[str], params: t.Sequence[GenerateParams]
    ) -> t.Sequence[GeneratedText]:
        assert len(texts) == 1
        assert self.text_callback is not None
        return [GeneratedText.from_text(self.text_callback(self, text)) for text in texts]


@pytest.mark.asyncio
async def test_chat_until_parsed_as_with_reset() -> None:
    generator = CallbackGenerator(model="callback", params=GenerateParams())

    def valid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 1
        assert messages[0].content == "original"
        return "<yes-no-answer>yes</yes-no-answer>"

    def invalid_cb(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        self.message_callback = valid_cb
        return "dropped"

    generator.message_callback = invalid_cb
    chat = await generator.chat([{"role": "user", "content": "original"}]).until_parsed_as(YesNoAnswer).run()
    assert len(chat) == 2
    assert chat.last.try_parse(YesNoAnswer) is not None


@pytest.mark.parametrize("drop_dialog", [True, False])
@pytest.mark.asyncio
async def test_chat_until_parsed_as_with_recovery(drop_dialog: bool) -> None:
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
        assert "<system-error>" in messages[2].content
        self.message_callback = valid_cb
        return "invalid2"

    def invalid_cb_1(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        assert len(messages) == 1
        assert messages[0].content == "original"
        self.message_callback = invalid_cb_2
        return "invalid1"

    generator.message_callback = invalid_cb_1
    chat = (
        await generator.chat([{"role": "user", "content": "original"}])
        .until_parsed_as(YesNoAnswer, attempt_recovery=True, drop_dialog=drop_dialog)
        .run()
    )

    assert len(chat) == (2 if drop_dialog else 6)
    assert chat.last.try_parse(YesNoAnswer) is not None


@pytest.mark.asyncio
async def test_completion_until_parsed_as_with_reset() -> None:
    generator = CallbackGenerator(model="callback", params=GenerateParams())

    def valid_cb(self: CallbackGenerator, text: str) -> str:
        assert text == "original"
        return "<yes-no-answer>yes</yes-no-answer>"

    def invalid_cb(self: CallbackGenerator, text: str) -> str:
        self.text_callback = valid_cb
        return "dropped"

    generator.text_callback = invalid_cb
    completion = await generator.complete("original").until_parsed_as(YesNoAnswer).run()
    assert try_parse(completion.generated, YesNoAnswer) is not None


@pytest.mark.parametrize("attempt_recovery", [True, False])
@pytest.mark.asyncio
async def test_chat_run_allowed_failed(attempt_recovery: bool) -> None:
    generator = EchoGenerator(model="callback", params=GenerateParams())
    max_rounds = 3

    chat = (
        await generator.chat([{"role": "user", "content": "test"}])
        .until_parsed_as(YesNoAnswer, attempt_recovery=attempt_recovery, max_rounds=max_rounds)
        .run(allow_failed=True)
    )

    assert chat.failed is True
    assert isinstance(chat.error, ExhaustedMaxRoundsError)
    assert len(chat) == ((max_rounds * 2) + 2 if attempt_recovery else 2)
    assert chat.last.role == "assistant"


@pytest.mark.parametrize("text", ["test", "<yes-no-answer>yes</yes-no-answer>"])
@pytest.mark.asyncio
async def test_chat_run_many_include_failed(text: str) -> None:
    generator = FixedGenerator(model="callback", params=GenerateParams(), text=text)

    chats = (
        await generator.chat([{"role": "user", "content": "test"}])
        .until_parsed_as(YesNoAnswer)
        .run_many(3, on_failed="include")
    )

    assert len(chats) == 3
    for chat in chats:
        assert chat.failed is (True if "<yes-no-answer>" not in text else False)
        assert len(chat) == 2
        assert chat.last.content == text


@pytest.mark.asyncio
async def test_chat_run_batch_include_failed() -> None:
    generator = EchoGenerator(model="callback", params=GenerateParams())

    chats = (
        await generator.chat()
        .until_parsed_as(YesNoAnswer)
        .run_batch([[Message(role="user", content=f"test-{i}")] for i in range(3)], on_failed="include")
    )

    assert len(chats) == 3
    for i, chat in enumerate(chats):
        assert chat.failed is True
        assert len(chat) == 2
        assert chat.last.content == f"test-{i}"
