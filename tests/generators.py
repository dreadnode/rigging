import typing as t

from rigging import Message
from rigging.generator import GenerateParams, Generator
from rigging.generator.base import GeneratedMessage, GeneratedText

# ruff: noqa: S101, ARG002


class FixedGenerator(Generator):
    text: str

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        return [GeneratedMessage.from_text(self.text, stop_reason="stop") for _ in messages]

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        return [GeneratedText.from_text(self.text, stop_reason="stop") for _ in texts]


class EchoGenerator(Generator):
    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        return [GeneratedMessage.from_text(m[-1].content, stop_reason="stop") for m in messages]

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        return [GeneratedText.from_text(t, stop_reason="stop") for t in texts]


class CallbackGenerator(Generator):
    message_callback: t.Callable[["CallbackGenerator", t.Sequence[Message]], str] | None = None
    text_callback: t.Callable[["CallbackGenerator", str], str] | None = None

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        assert self.message_callback is not None
        return [
            GeneratedMessage.from_text(self.message_callback(self, m), stop_reason="stop")
            for m in messages
        ]

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        assert len(texts) == 1
        assert self.text_callback is not None
        return [
            GeneratedText.from_text(self.text_callback(self, text), stop_reason="stop")
            for text in texts
        ]


class FailingGenerator(Generator):
    _exception: Exception = RuntimeError("Intentional failure")

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        raise self._exception

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        raise self._exception
