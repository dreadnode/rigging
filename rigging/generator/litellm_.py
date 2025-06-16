import asyncio
import base64
import datetime
import re
import typing as t

import litellm
import litellm.types.utils
from loguru import logger

from rigging.generator.base import (
    Fixup,
    GeneratedMessage,
    GeneratedText,
    GenerateParams,
    Generator,
    trace_messages,
    trace_str,
    with_fixups,
)
from rigging.message import ContentAudioInput, ContentImageUrl, ContentText, Message
from rigging.tools.base import FunctionDefinition, ToolDefinition
from rigging.tracing import tracer

# We should probably let people configure
# this independently, but for now we'll
# fix it to prevent confusion
litellm.drop_params = True

# Prevent the small debug statements
# from being printed to the console
litellm.suppress_debug_info = True


class OpenAIToolsWithImageURLsFixup(Fixup):
    # As of writing, openai doesn't support multi-part messages
    # associated with the `tool` role. This is complicated by
    # the fact that we need to resolve the tool call(s) in the
    # following messages. To get around this, we'll resolve the tool
    # call with empty content, and duplicate the multi-part data
    # into a user message immediately following it. We also need
    # to take care of multiple tool calls next to each other and ensure
    # we don't add the user message in between them.

    def can_fix(self, exception: Exception) -> bool:
        return (
            "Image URLs are only allowed for messages with role 'user', but this message with role 'tool' contains an image URL."
            in str(exception)
        )

    def fix(self, items: t.Sequence[Message]) -> t.Sequence[Message]:
        updated_messages: list[Message] = []
        append_queue: list[Message] = []
        for message in items:
            if message.role == "tool" and isinstance(message.content_parts, list):
                updated_messages.append(
                    message.model_copy(
                        deep=True,
                        update={"content_parts": [ContentText(text="See next message")]},
                    ),
                )
                append_queue.append(message.model_copy(deep=True, update={"role": "user"}))
            else:
                updated_messages.extend(append_queue)
                append_queue = []
                updated_messages.append(message)

        updated_messages.extend(append_queue)
        return updated_messages


class CacheTooSmallFixup(Fixup):
    # Attempt to enable caching on chat messages which
    # are below a certain threshold can result in a 400
    # error from APIs (Vertex/Gemini).

    def can_fix(self, exception: Exception) -> bool | t.Literal["once"]:
        return "once" if "Cached content is too small." in str(exception) else False

    def fix(self, messages: t.Sequence[Message]) -> t.Sequence[Message]:
        return [message.cache(False) for message in messages]


class GroqAssistantContentFixup(Fixup):
    # Groq can complain if we try to send fully
    # structured content parts when working with
    # the assistant role.
    #
    # Compatibility flags are a poor workaround for the
    # fact that we don't have direct control over the
    # conversion to the OpenAI spec.

    def can_fix(self, exception: Exception) -> bool:
        return "Groq" in str(exception) and "content' : value must be a string" in str(exception)

    def fix(self, messages: t.Sequence[Message]) -> t.Sequence[Message]:
        updated_messages: list[Message] = []
        for message in messages:
            if message.role == "assistant":
                message = message.clone()  # noqa: PLW2901
                message.compatibility_flags.add("content_as_str")
            updated_messages.append(message)
        return updated_messages


g_fixups = [
    OpenAIToolsWithImageURLsFixup(),
    CacheTooSmallFixup(),
    GroqAssistantContentFixup(),
]


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

        ```
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
    _supports_function_calling: bool | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            # TODO: This is hacky
            max_connections = self.max_connections if self.max_connections > 0 else 10_000
            self._semaphore = asyncio.Semaphore(max_connections)
        return self._semaphore

    async def supports_function_calling(self) -> bool | None:
        if self._supports_function_calling is not None:
            return self._supports_function_calling

        self._supports_function_calling = litellm.utils.supports_function_calling(self.model)
        if self._supports_function_calling:
            return self._supports_function_calling

        self._supports_function_calling = False

        # Otherwise we'll run a small check to see if we can

        with tracer.span(f"Checking '{self.model}' for function calling support") as span:
            try:
                generated = await self.generate_messages(
                    [[Message(role="user", content="Call the test function")]],
                    [
                        GenerateParams(
                            tools=[
                                ToolDefinition(
                                    function=FunctionDefinition(
                                        name="test_function",
                                        description="Test function",
                                    ),
                                ),
                            ],
                        ),
                    ],
                )

                if generated:
                    if isinstance(generated[0], BaseException):
                        raise generated[0]  # noqa: TRY301

                    if (
                        isinstance(generated[0], GeneratedMessage)
                        and generated[0].message.tool_calls
                    ):
                        self._supports_function_calling = True
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to check for function calling support: {e}")
                span.set_attribute("error", str(e))

            span.set_attribute("supports_function_calling", self._supports_function_calling)

        return self._supports_function_calling

    async def _ensure_delay_between_requests(self) -> None:
        if self._last_request_time is None:
            return

        delta = datetime.datetime.now(tz=datetime.timezone.utc) - self._last_request_time
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

    def _parse_model_response(
        self,
        response: litellm.types.utils.ModelResponse,
    ) -> GeneratedMessage:
        choice = response.choices[-1]
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = response.usage.model_dump()  # type: ignore [attr-defined]
            usage["input_tokens"] = usage.pop("prompt_tokens")
            usage["output_tokens"] = usage.pop("completion_tokens")

        if isinstance(choice, litellm.types.utils.StreamingChoices):
            raise TypeError("Streaming choices are not supported")

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

        extra: dict[str, t.Any] = {"response_id": response.id}
        if hasattr(response, "provider"):
            extra["provider"] = response.provider
        if (
            hasattr(choice.message, "provider_specific_fields")
            and choice.message.provider_specific_fields is not None
        ):
            extra.update(choice.message.provider_specific_fields)
        if (
            hasattr(choice.message, "reasoning_content")
            and choice.message.reasoning_content is not None
        ):
            extra["reasoning_content"] = choice.message.reasoning_content
        if (
            hasattr(choice.message, "thinking_blocks")
            and choice.message.thinking_blocks is not None
        ):
            extra["thinking_blocks"] = choice.message.thinking_blocks

        message = Message(
            role="assistant",
            content=[],
            tool_calls=tool_calls,
        )

        if choice.message.content is not None:
            # Check for lazy litellm handling
            # https://github.com/BerriAI/litellm/blob/0f9ebc23a5c1e386195267dfc8d91ba7169c4508/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L578C1-L599C48
            if match := re.match(r"(data:[\w/]+?;base64,[A-Za-z0-9+/=]+)", choice.message.content):
                encoded_data = match.group(1)
                choice.message.content = choice.message.content.replace(encoded_data, "").strip()
                message.content_parts.append(ContentImageUrl.from_url(encoded_data))

            message.content_parts.append(
                ContentText(
                    text=choice.message.content,
                ),
            )

        if hasattr(choice.message, "audio") and choice.message.audio is not None:
            message.content_parts.append(
                ContentAudioInput.from_bytes(
                    base64.b64decode(choice.message.audio.data),
                    transcript=choice.message.audio.transcript,
                ),
            )

        return GeneratedMessage(
            message=message,
            stop_reason=choice.finish_reason,
            usage=usage,
            extra=extra,
        )

    def _parse_text_completion_response(
        self,
        response: litellm.types.utils.TextCompletionResponse,
    ) -> GeneratedText:
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

    @with_fixups(*g_fixups)
    async def _generate_message(
        self,
        messages: t.Sequence[Message],
        params: GenerateParams,
    ) -> GeneratedMessage:
        async with self.semaphore:
            # if params.max_tokens is None:
            #     params.max_tokens = get_max_tokens_for_model(self.model)
            await self._ensure_delay_between_requests()

            acompletion = litellm.acompletion
            if self._wrap is not None:
                acompletion = self._wrap(acompletion)

            response = await acompletion(
                model=self.model,
                messages=[message.to_openai() for message in messages],
                api_key=self.api_key,
                **self.params.merge_with(params).to_dict(),
            )

            self._last_request_time = datetime.datetime.now(tz=datetime.timezone.utc)
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
                prompt=text,
                model=self.model,
                api_key=self.api_key,
                **self.params.merge_with(params).to_dict(),
            )

            self._last_request_time = datetime.datetime.now(tz=datetime.timezone.utc)
            return self._parse_text_completion_response(response)

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage | BaseException]:
        coros = [
            self._generate_message(_messages, _params)
            for _messages, _params in zip(messages, params, strict=True)
        ]
        generated = await asyncio.gather(*coros, return_exceptions=True)

        for i, (_messages, response) in enumerate(zip(messages, generated, strict=True)):
            trace_messages(_messages, f"Messages {i + 1}/{len(messages)}")
            if isinstance(response, BaseException):
                trace_str(str(response), f"Response {i + 1}/{len(messages)}")
            else:
                trace_messages([response], f"Response {i + 1}/{len(messages)}")

        return generated

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText | BaseException]:
        coros = [
            self._generate_text(text, _params) for text, _params in zip(texts, params, strict=True)
        ]
        generated = await asyncio.gather(*coros, return_exceptions=True)

        for i, (text, response) in enumerate(zip(texts, generated, strict=True)):
            trace_str(text, f"Text {i + 1}/{len(texts)}")
            trace_str(response, f"Response {i + 1}/{len(texts)}")

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

    return litellm.model_cost[model].get("max_tokens")  # type: ignore [no-any-return]
