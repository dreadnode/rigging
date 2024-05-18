import asyncio
import typing as t

import litellm  # type: ignore

from rigging.generator.base import GenerateParams, Generator, register_generator, trace_messages, trace_str
from rigging.message import (
    Message,
)

# We should probably let people configure
# this independently, but for now we'll
# fix it to prevent confusion
litellm.drop_params = True


class LiteLLMGenerator(Generator):
    """
    Generator backed by the LiteLLM library.

    Note:
        Find more information about supported models and formats [in their docs.](https://docs.litellm.ai/docs/providers).

    Note:
        Batching support is not performant and simply a loop over inputs.
    """

    def _generate_message(self, messages: t.Sequence[Message], params: GenerateParams) -> Message:
        result = litellm.completion(
            self.model,
            [message.model_dump(include={"role", "content"}) for message in messages],
            api_key=self.api_key,
            **self.params.merge_with(params).to_dict(),
        )
        response = result.choices[-1].message.content.strip()
        return Message(role="assistant", content=response)

    async def _agenerate_message(self, messages: t.Sequence[Message], params: GenerateParams) -> Message:
        result = await litellm.acompletion(
            self.model,
            [message.model_dump(include={"role", "content"}) for message in messages],
            api_key=self.api_key,
            **self.params.merge_with(params).to_dict(),
        )
        response = result.choices[-1].message.content.strip()
        return Message(role="assistant", content=response)

    def _generate_text(self, text: str, params: GenerateParams) -> str:
        result = litellm.text_completion(
            text, self.model, api_key=self.api_key, **self.params.merge_with(params).to_dict()
        )
        return t.cast(str, result.choices[-1]["text"])

    async def _agenerate_text(self, text: str, params: GenerateParams) -> str:
        result = await litellm.atext_completion(
            text, self.model, api_key=self.api_key, **self.params.merge_with(params).to_dict()
        )
        return t.cast(str, result.choices[-1]["text"])

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[Message]:
        generated: list[Message] = []
        for i, (_messages, _params) in enumerate(zip(messages, params, strict=True)):
            trace_messages(_messages, f"Messages {i+1}/{len(messages)}")
            next_message = self._generate_message(_messages, _params)
            generated.append(next_message)
            trace_messages([next_message], f"Response {i+1}/{len(messages)}")

        return generated

    async def agenerate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[Message]:
        generated: list[Message] = await asyncio.gather(
            *[self._agenerate_message(_messages, _params) for _messages, _params in zip(messages, params, strict=True)]
        )

        for i, (_messages, _generated) in enumerate(zip(messages, generated, strict=True)):
            trace_messages(_messages, f"Messages {i+1}/{len(messages)}")
            trace_messages([_generated], f"Response {i+1}/{len(messages)}")

        return generated

    def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[str]:
        generated: list[str] = []
        for i, (text, _params) in enumerate(zip(texts, params, strict=True)):
            trace_str(text, f"Text {i+1}/{len(texts)}")
            response = self._generate_text(text, _params)
            generated.append(response)
            trace_str(response, f"Generated {i+1}/{len(texts)}")

        return generated

    async def agenerate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[str]:
        generated: list[str] = await asyncio.gather(
            *[self._agenerate_text(text, _params) for text, _params in zip(texts, params, strict=True)]
        )

        for i, (text, response) in enumerate(zip(texts, generated, strict=True)):
            trace_str(text, f"Text {i+1}/{len(texts)}")
            trace_str(response, f"Generated {i+1}/{len(texts)}")

        return generated


register_generator("litellm", LiteLLMGenerator)
