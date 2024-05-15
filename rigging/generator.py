"""
Generators produce completions for a given set of messages or text.
"""

import asyncio
import typing as t

import litellm  # type: ignore
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rigging.error import InvalidModelSpecifiedError
from rigging.message import (
    Message,
    MessageDict,
)

if t.TYPE_CHECKING:
    from rigging.chat import PendingChat
    from rigging.completion import PendingCompletion

# We should probably let people configure
# this independently, but for now we'll
# fix it to prevent confusion
litellm.drop_params = True

# Global provider map
g_providers: dict[str, type["Generator"]] = {}


# TODO: Ideally we flex this to support arbitrary
# generator params, but we'll limit things
# for now until we understand the use cases
#
# TODO: We also would like to support N-style
# parallel generation eventually -> need to
# update our interfaces to support that
class GenerateParams(BaseModel):
    """
    Parameters for generating text using a language model.

    These are designed to generally overlap with underlying
    APIs like litellm, but will be extended as needed.

    Note:
        Use the `extra` field to pass additional parameters to the API.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    """The sampling temperature."""

    max_tokens: int | None = None
    """The maximum number of tokens to generate."""

    top_p: float | None = None
    """The nucleus sampling probability."""

    stop: list[str] | None = None
    """A list of stop sequences to stop generation at."""

    presence_penalty: float | None = None
    """The presence penalty."""

    frequency_penalty: float | None = None
    """The frequency penalty."""

    api_base: str | None = None
    """The base URL for the API."""

    timeout: int | None = None
    """The timeout for the API request."""

    seed: int | None = None
    """The random seed."""

    extra: dict[str, t.Any] = Field(default_factory=dict)
    """Extra parameters to be passed to the API."""

    @field_validator("stop", mode="before")
    def validate_stop(cls, value: t.Any) -> t.Any:
        if isinstance(value, str):
            return value.split(";")
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        raise ValueError("Stop sequences must be a list or a string separated by ';'")

    def merge_with(self, *others: t.Optional["GenerateParams"]) -> "GenerateParams":
        """
        Apply a series of parameter overrides to the current instance and return a copy.

        Args:
            *others: The parameters to be merged with the current instance's parameters.
                Can be multiple and overrides will be applied in order.

        Returns:
            The merged parameters instance.
        """
        if len(others) == 0 or all(p is None for p in others):
            return self

        updates: dict[str, t.Any] = {}
        for other in [o for o in others if o is not None]:
            other_dict = other.model_dump(exclude_unset=True)
            for name, value in other_dict.items():
                if value is not None:
                    updates[name] = value

        return self.model_copy(update=updates)

    def to_dict(self) -> dict[str, t.Any]:
        """
        Convert the parameters to a dictionary.

        Returns:
            The parameters as a dictionary.
        """
        params = self.model_dump(exclude_unset=True)
        if "extra" in params:
            params.update(params.pop("extra"))
        return params


class Generator(BaseModel):
    """
    Base class for all rigging generators.

    This class provides common functionality and methods for generating completion messages.

    A subclass of this can implement any of the following:

    - `generate_messages`: Process a batch of messages.
    - `generate_texts`: Process a batch of texts.

    (In addition to async variants of these functions)
    """

    model: str
    """The model name to be used by the generator."""
    api_key: str | None = Field(None, exclude=True)
    """The API key used for authentication."""
    params: GenerateParams
    """The parameters used for generating completion messages."""

    def to_identifier(self, params: GenerateParams | None = None) -> str:
        """
        Converts the generator instance back into a rigging identifier string.

        This calls [rigging.generator.get_identifier][] with the current instance.

        Args:
            params: The generation parameters.

        Returns:
            The identifier string.
        """
        return get_identifier(self, params)

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
        *,
        prefix: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        """
        Generate a batch of messages using the specified parameters.

        Note:
            The length of `params` must be the same as the length of `many`.

        Args:
            messages: A sequence of sequences of messages.
            params: A sequence of GenerateParams objects.
            prefix: A sequence of fixed messages to be prefixed before every item of `many`.

        Returns:
            A sequence of generated messages.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("`generate_messages` is not supported by this generator.")

    async def agenerate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
        *,
        prefix: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        """async version of [rigging.generator.Generator.generate_messages][]"""
        raise NotImplementedError("`agenerate_messages` is not supported by this generator.")

    def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
        *,
        prefix: str | None = None,
    ) -> t.Sequence[str]:
        """
        Generate a batch of text completions using the generator.

        Note:
            This method falls back to looping over the inputs and calling `generate_text` for each item.

        Note:
            If supplied, the length of `params` must be the same as the length of `many`.

        Args:
            texts: The input texts for generating the batch.
            params: Additional parameters for generating each text in the batch.
            prefix: A fixed input text to be used as a prefix for all of `many`.

        Returns:
            The generated texts.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("`generate_texts` is not supported by this generator.")

    async def agenerate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
        *,
        prefix: str | None = None,
    ) -> t.Sequence[str]:
        """async version of [rigging.generator.Generator.generate_texts][]"""
        raise NotImplementedError("`agenerate_texts` is not supported by this generator.")

    # Helper alternative to chat(generator) -> generator.chat(...)
    #
    # params seem odd, but mypy doesn't like the TypedDict in a list otherwise

    @t.overload
    def chat(
        self,
        messages: t.Sequence[MessageDict],
        params: GenerateParams | None = None,
    ) -> "PendingChat":
        ...

    @t.overload
    def chat(
        self,
        messages: t.Sequence[Message] | MessageDict | Message | str | None = None,
        params: GenerateParams | None = None,
    ) -> "PendingChat":
        ...

    def chat(
        self,
        messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str | None = None,
        params: GenerateParams | None = None,
    ) -> "PendingChat":
        """
        Build a pending chat with the given messages and optional params overloads.

        Args:
            messages: The messages to be sent in the chat.
            params: Optional parameters for generating responses.

        Returns:
            Pending chat to run.
        """
        from rigging.chat import PendingChat

        return PendingChat(self, Message.fit_as_list(messages) if messages else [], params)

    # Helper alternative to complete(generator) -> generator.complete(...)

    def complete(self, text: str, params: GenerateParams | None = None) -> "PendingCompletion":
        """
        Build a pending string completion of the given text with optional param overloads.

        Args:
            text: The input text to be completed.
            params: The parameters to be used for completion.

        Returns:
            The completed text.
        """
        from rigging.completion import PendingCompletion

        return PendingCompletion(self, text, params)


@t.overload
def chat(
    generator: "Generator",
    messages: t.Sequence[MessageDict],
    params: GenerateParams | None = None,
) -> "PendingChat":
    ...


@t.overload
def chat(
    generator: "Generator",
    messages: t.Sequence[Message] | MessageDict | Message | str | None = None,
    params: GenerateParams | None = None,
) -> "PendingChat":
    ...


def chat(
    generator: "Generator",
    messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str | None = None,
    params: GenerateParams | None = None,
) -> "PendingChat":
    """
    Creates a pending chat using the given generator, messages, and params.

    Args:
        generator: The generator to use for creating the chat.
        messages:
            The messages to include in the chat. Can be a single message or a sequence of messages.
        params: Additional parameters for generating the chat.

    Returns:
        Pending chat to run.
    """
    return generator.chat(messages, params)


def complete(
    generator: Generator,
    text: str,
    params: GenerateParams | None = None,
) -> "PendingCompletion":
    return generator.complete(text, params)


def get_identifier(generator: Generator, params: GenerateParams | None = None) -> str:
    """
    Converts the generator instance back into a rigging identifier string.

    Warning:
        The `extra` parameter field is not currently supported in identifiers.

    Args:
        generator: The generator object.
        params: The generation parameters.

    Returns:
        The identifier string for the generator.
    """

    provider = next(name for name, klass in g_providers.items() if isinstance(generator, klass))
    identifier = f"{provider}!{generator.model}"

    merged_params = generator.params.merge_with(params)
    if merged_params.extra:
        logger.warning("Extra parameters are not supported in identifiers.")
        merged_params.extra = {}

    params_dict = merged_params.to_dict()
    if params_dict:
        if "stop" in params_dict:
            params_dict["stop"] = ";".join(params_dict["stop"])
        identifier += f",{','.join([f'{k}={v}' for k, v in params_dict.items()])}"

    return identifier


def get_generator(identifier: str) -> Generator:
    """
    Get a generator by an identifier string. Uses LiteLLM by default.

    Identifier strings are formatted like `<provider>!<model>,<**kwargs>`

    (provider is optional andif not specified)

    Examples:

    - "gpt-3.5-turbo" -> `LiteLLMGenerator(model="gpt-3.5-turbo")`
    - "litellm!claude-2.1" -> `LiteLLMGenerator(model="claude-2.1")`
    - "mistral/mistral-tiny" -> `LiteLLMGenerator(model="mistral/mistral-tiny")`

    You can also specify arguments to the generator by comma-separating them:

    - "mistral/mistral-medium,max_tokens=1024"
    - "gpt-4-0613,temperature=0.9,max_tokens=512"
    - "claude-2.1,stop_sequences=Human:;test,max_tokens=100"

    (These get parsed as [rigging.generator.GenerateParams][])

    Args:
        identifier: The identifier string to use to get a generator.

    Returns:
        The generator object.

    Raises:
        InvalidModelSpecified: If the identifier is invalid.
    """

    provider: str = list(g_providers.keys())[0]
    model: str = identifier
    api_key: str | None = None
    params: GenerateParams = GenerateParams()

    # Split provider, model, and kwargs

    try:
        if "!" in identifier:
            provider, model = identifier.split("!")

        if "," in model:
            model, kwargs_str = model.split(",", 1)
            kwargs = dict(arg.split("=") for arg in kwargs_str.split(","))
            api_key = kwargs.pop("api_key", None)
            params = GenerateParams(**kwargs)
    except Exception as e:
        raise InvalidModelSpecifiedError(identifier) from e

    if provider not in g_providers:
        raise InvalidModelSpecifiedError(identifier)

    generator_cls = g_providers[provider]
    return generator_cls(model=model, api_key=api_key, params=params)


def register_generator(provider: str, generator_cls: type[Generator]) -> None:
    """
    Register a generator class for a provider id.

    This let's you use [rigging.generator.get_generator][] with a custom generator class.

    Args:
        provider: The name of the provider.
        generator_cls: The generator class to register.
    """
    global g_providers
    g_providers[provider] = generator_cls


def trace_messages(messages: t.Sequence[Message], title: str) -> None:
    """
    Helper function to trace log a sequence of Message objects.

    Args:
        messages: A sequence of Message objects to be logged.
        title: The title to be displayed in the log.

    Returns:
        None
    """
    logger.trace(f"--- {title} ---")
    logger.trace("\n".join([str(msg) for msg in messages]))
    logger.trace("---")


def trace_str(content: str, title: str) -> None:
    """
    Helper function to trace log a string.

    Parameters:
        content: The string content to be logged.
        title: The title of the log entry.

    Returns:
        None
    """
    logger.trace(f"--- {title} ---")
    logger.trace(content)
    logger.trace("---")


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
        *,
        prefix: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        if prefix is not None:
            messages = [list(prefix) + list(messages) for messages in messages]

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
        *,
        prefix: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        if prefix is not None:
            messages = [list(prefix) + list(messages) for messages in messages]

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
        *,
        prefix: str | None = None,
    ) -> t.Sequence[str]:
        if prefix is not None:
            texts = [prefix + text for text in texts]

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
        *,
        prefix: str | None = None,
    ) -> t.Sequence[str]:
        if prefix is not None:
            texts = [prefix + text for text in texts]

        generated: list[str] = await asyncio.gather(
            *[self._agenerate_text(text, _params) for text, _params in zip(texts, params, strict=True)]
        )

        for i, (text, response) in enumerate(zip(texts, generated, strict=True)):
            trace_str(text, f"Text {i+1}/{len(texts)}")
            trace_str(response, f"Generated {i+1}/{len(texts)}")

        return generated


g_providers["litellm"] = LiteLLMGenerator
