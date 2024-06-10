from __future__ import annotations

import asyncio
import typing as t

import litellm

from rigging.generator.base import (
    GeneratedMessage,
    GeneratedText,
    GenerateParams,
    Generator,
    trace_messages,
    trace_str,
)
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
        Consider setting [`max_connections`][rigging.generator.litellm_.LiteLLMGenerator.max_connections]
        to a lower value if you run into API limits. You can pass this directly in the generator id:

        ```py
        get_generator("litellm!openai/gpt-4o,max_connections=5")
        ```
    """

    max_connections: int = 4
    """
    How many simultaneous requests to pool at one time.
    This is useful to set when you run into API limits at a provider.

    Set to 0 to remove the limit.
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

    def _parse_model_response(self, response: litellm.utils.ModelResponse) -> GeneratedMessage:
        choice = response.choices[-1]
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = response.usage.model_dump()  # type: ignore
            usage["input_tokens"] = usage.pop("prompt_tokens")
            usage["output_tokens"] = usage.pop("completion_tokens")
        return GeneratedMessage(
            message=Message(role="assistant", content=choice.message.content),  # type: ignore
            stop_reason=choice.finish_reason,
            usage=usage,
            extra={"response_id": response.id},
        )

    def _parse_text_completion_response(self, response: litellm.utils.TextCompletionResponse) -> GeneratedText:
        choice = response.choices[-1]
        usage = None
        if response.usage is not None:
            usage = response.usage.model_dump()
            usage["input_tokens"] = usage.pop("prompt_tokens")
            usage["output_tokens"] = usage.pop("completion_tokens")
        return GeneratedText(
            text=choice["text"],
            stop_reason=choice.finish_reason,
            usage=usage,
            extra={"response_id": response.id},
        )

    async def _generate_message(self, messages: t.Sequence[Message], params: GenerateParams) -> GeneratedMessage:
        return self._parse_model_response(
            await litellm.acompletion(
                self.model,
                [message.model_dump(include={"role", "content"}) for message in messages],
                api_key=self.api_key,
                **self.params.merge_with(params).to_dict(),
            )
        )

    async def _generate_text(self, text: str, params: GenerateParams) -> GeneratedText:
        return self._parse_text_completion_response(
            await litellm.atext_completion(
                text, self.model, api_key=self.api_key, **self.params.merge_with(params).to_dict()
            )
        )

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        generated: list[GeneratedMessage] = []
        max_connections = self.max_connections if self.max_connections > 0 else len(messages)
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=max_connections)

        async def worker(i: int, _messages: t.Sequence[Message], _params: GenerateParams) -> None:
            _generated = await self._generate_message(_messages, _params)
            generated.append(_generated)
            trace_messages(_messages, f"Messages {i+1}/{len(messages)}")
            trace_messages([_generated], f"Response {i+1}/{len(messages)}")
            queue.get_nowait()
            queue.task_done()

        tasks: list[asyncio.Task[None]] = []
        for i, (_messages, _params) in enumerate(zip(messages, params)):
            await queue.put(None)
            task = asyncio.create_task(worker(i, _messages, _params))
            tasks.append(task)

        await queue.join()
        await asyncio.gather(*tasks)

        return generated

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        generated: list[GeneratedText] = []
        max_connections = self.max_connections if self.max_connections > 0 else len(texts)
        for i in range(0, len(texts), max_connections):
            chunk_texts = texts[i : i + max_connections]
            chunk_params = params[i : i + max_connections]
            chunk_generated = await asyncio.gather(
                *[self._generate_text(text, _params) for text, _params in zip(chunk_texts, chunk_params)]
            )
            generated.extend(chunk_generated)

            for i, (text, response) in enumerate(zip(chunk_texts, chunk_generated)):
                trace_str(text, f"Text {i+1}/{len(texts)}")
                trace_str(response, f"Generated {i+1}/{len(texts)}")

        return generated
