import asyncio
import typing as t

import litellm  # type: ignore

from rigging.generator.base import GenerateParams, Generator, register_generator, trace_messages, trace_str
from rigging.message import Message

# We should probably let people configure
# this independently, but for now we'll
# fix it to prevent confusion
litellm.drop_params = True


class LiteLLMGenerator(Generator):
    """
    Generator backed by the LiteLLM library.

    Find more information about supported models and formats [in their docs.](https://docs.litellm.ai/docs/providers).

    Note:
        Batching support is not performant and simply a loop over inputs.

    Warning:
        While some providers support passing `n` to produce a batch
        of completions per request, we don't currently use this in the
        implementation due to it's brittle requirements.

    Tip:
        Consider setting [`max_requests`][rigging.generator.litellm_.LiteLLMGenerator.max_requests]
        to a lower value if you run into API limits. You can pass this directly in the generator id:

        ```py
        get_generator("litellm!openai/gpt-4o,max_requests=5")
        ```
    """

    max_requests: int | None = None
    """
    When using async variants, how many simultaneous requests to pool at one time.
    This is useful to set when you run into API limits at a provider.
    Defaults to `None` for no limit.
    """

    # TODO: Some model providers support using `n` as a batch
    # parameter to generate multiple completions at once. Which
    # could help us optimize run_many calls.
    #
    # If we wanted this, we'd need to check the model provider
    # and see if it was supported, and all our messages/texts
    # were equal before overriding that parameter to the call.
    #
    # This seems like a brittle feature at the moment, so we'll
    # leave it out for now.

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
        generated: list[Message] = []
        max_requests = self.max_requests or len(messages)
        for i in range(0, len(messages), max_requests):
            chunk_messages = messages[i : i + max_requests]
            chunk_params = params[i : i + max_requests]
            chunk_generated = await asyncio.gather(
                *[
                    self._agenerate_message(_messages, _params)
                    for _messages, _params in zip(chunk_messages, chunk_params, strict=True)
                ]
            )
            generated.extend(chunk_generated)

            for j, (_messages, _generated) in enumerate(zip(chunk_messages, chunk_generated, strict=True)):
                trace_messages(_messages, f"Messages {i+j+1}/{len(messages)}")
                trace_messages([_generated], f"Response {i+j+1}/{len(messages)}")

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
        generated: list[str] = []
        max_requests = self.max_requests or len(texts)
        for i in range(0, len(texts), max_requests or len(texts)):
            chunk_texts = texts[i : i + max_requests]
            chunk_params = params[i : i + max_requests]
            chunk_generated = await asyncio.gather(
                *[self._agenerate_text(text, _params) for text, _params in zip(chunk_texts, chunk_params, strict=True)]
            )
            generated.extend(chunk_generated)

            for i, (text, response) in enumerate(zip(chunk_texts, chunk_generated, strict=True)):
                trace_str(text, f"Text {i+1}/{len(texts)}")
                trace_str(response, f"Generated {i+1}/{len(texts)}")

        return generated


register_generator("litellm", LiteLLMGenerator)
