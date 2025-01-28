from __future__ import annotations

import asyncio
import datetime
import typing as t

import litellm
import litellm.types.utils
from loguru import logger

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
        or [`min_delay_between_requests`][rigging.generator.litellm_.LiteLLMGenerator.min_delay_between_requests
        if you run into API limits. You can pass this directly in the generator id:

        ```py
        get_generator("litellm!openai/gpt-4o,max_connections=2,min_delay_between_requests=1000")
        ```
    """

    max_connections: int = 10
    """
    How many simultaneous requests to pool at one time.
    This is useful to set when you run into API limits at a provider.

    Set to 0 to remove the limit.
    """

    min_delay_between_requests: float = 0.0
    """
    Minimum time (ms) between each request.
    This is useful to set when you run into API limits at a provider.
    """

    _semaphore: asyncio.Semaphore | None = None
    _last_request_time: datetime.datetime | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            # TODO: This is hacky
            max_connections = self.max_connections if self.max_connections > 0 else 10_000
            self._semaphore = asyncio.Semaphore(max_connections)
        return self._semaphore

    async def _ensure_delay_between_requests(self) -> None:
        if self._last_request_time is None:
            return

        delta = datetime.datetime.now() - self._last_request_time
        delta_ms = delta.total_seconds() * 1000

        if delta_ms < self.min_delay_between_requests:
            wait_seconds = (self.min_delay_between_requests - delta_ms) / 1000
            logger.trace(f"Waiting {wait_seconds} seconds")
            await asyncio.sleep(wait_seconds)

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

    def _parse_model_response(self, response: litellm.types.utils.ModelResponse) -> GeneratedMessage:
        choice = response.choices[-1]
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = response.usage.model_dump()  # type: ignore
            usage["input_tokens"] = usage.pop("prompt_tokens")
            usage["output_tokens"] = usage.pop("completion_tokens")

        if isinstance(choice, litellm.types.utils.StreamingChoices):
            raise ValueError("Streaming choices are not supported")

        tool_calls: list[dict[str, t.Any]] | None = None
        if (
            isinstance(choice.message, litellm.types.utils.Message)
            and choice.message.tool_calls is not None
            and all(
                isinstance(call, litellm.types.utils.ChatCompletionMessageToolCall)
                for call in choice.message.tool_calls
            )
        ):
            tool_calls = [call.model_dump() for call in choice.message.tool_calls]

        extra = {"response_id": response.id}
        if hasattr(response, "provider"):
            extra["provider"] = response.provider
        if choice.message.provider_specific_fields is not None:
            extra.update(choice.message.provider_specific_fields)

        return GeneratedMessage(
            message=Message(role="assistant", content=choice.message.content, tool_calls=tool_calls),
            stop_reason=choice.finish_reason,
            usage=usage,
            extra=extra,
        )

    def _parse_text_completion_response(self, response: litellm.types.utils.TextCompletionResponse) -> GeneratedText:
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
        async with self.semaphore:
            # if params.max_tokens is None:
            #     params.max_tokens = get_max_tokens_for_model(self.model)
            await self._ensure_delay_between_requests()

            acompletion = litellm.acompletion
            if self._wrap is not None:
                acompletion = self._wrap(acompletion)

            response = await acompletion(
                model=self.model,
                messages=[message.to_openai_spec() for message in messages],
                api_key=self.api_key,
                **self.params.merge_with(params).to_dict(),
            )

            self._last_request_time = datetime.datetime.now()
            return self._parse_model_response(response)

    async def _generate_text(self, text: str, params: GenerateParams) -> GeneratedText:
        async with self.semaphore:
            # if params.max_tokens is None:
            #     params.max_tokens = get_max_tokens_for_model(self.model)
            await self._ensure_delay_between_requests()

            atext_completion = litellm.atext_completion
            if self._wrap is not None:
                atext_completion = self._wrap(atext_completion)

            response = await atext_completion(
                prompt=text, model=self.model, api_key=self.api_key, **self.params.merge_with(params).to_dict()
            )

            self._last_request_time = datetime.datetime.now()
            return self._parse_text_completion_response(response)

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        coros = [self._generate_message(_messages, _params) for _messages, _params in zip(messages, params)]
        generated = await asyncio.gather(*coros)

        for i, (_messages, response) in enumerate(zip(messages, generated)):
            trace_messages(_messages, f"Messages {i+1}/{len(messages)}")
            trace_messages([response], f"Response {i+1}/{len(messages)}")

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


def get_max_tokens_for_model(model: str) -> int | None:
    """
    Try to get the maximum number of tokens for a model from litellm mappings.

    Args:
        model: The model name.

    Returns:
        The maximum number of tokens.
    """
    while model not in litellm.model_cost:
        if "/" not in model:
            return None
        model = "/".join(model.split("/")[1:])

    return litellm.model_cost[model].get("max_tokens")  # type: ignore
