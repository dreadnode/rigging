"""
Generators produce completions for a given set of messages or text.
"""

import asyncio
import typing as t

import litellm  # type: ignore
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rigging.chat import PendingChat
from rigging.completion import PendingCompletion
from rigging.error import InvalidModelSpecifiedError
from rigging.message import (
    Message,
    MessageDict,
)

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

    Attributes:
        temperature (float | None): The sampling temperature.
        max_tokens (int | None): The maximum number of tokens to generate.
        top_p (float | None): The nucleus sampling probability.
        stop (list[str] | None): A list of stop sequences to stop generation at.
        presence_penalty (float | None): The presence penalty.
        frequency_penalty (float | None): The frequency penalty.
        api_base (str | None): The base URL for the API.
        timeout (int | None): The timeout for the API request.
        seed (int | None): The seed.
        extra (dict[str, t.Any]): Extra parameters.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    api_base: str | None = None
    timeout: int | None = None
    seed: int | None = None
    extra: dict[str, t.Any] = Field(default_factory=dict)

    @field_validator("stop", mode="before")
    def validate_stop(cls, value: t.Any) -> t.Any:
        if isinstance(value, str):
            return value.split(";")
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        raise ValueError("Stop sequences must be a list or a string separated by ';'")


class Generator(BaseModel):
    """
    Base class for all rigging generators.

    This class provides common functionality and methods for generating completion messages.

    A subclass of this can implement any of the following:

    - `generate_message`: Generate the next message for a given set of messages.
    - `generate_text`: Generate a string completion of the given text.
    - `batch_messages`: Process a batch of messages.
    - `batch_texts`: Process a batch of texts.

    (In addition to async variants of these functions)

    Attributes:
        model (str): The model used by the generator.
        api_key (str | None): The API key used for authentication. Defaults to None.
        params (GenerateParams): The parameters used for generating completion messages.
    """

    model: str
    api_key: str | None = Field(None, exclude=True)
    params: GenerateParams

    def to_identifier(self, overloads: GenerateParams | None = None) -> str:
        """
        Converts the generator instance back into a rigging identifier string.

        Note:
            Extra parameters are not supported in identifiers.

        Args:
            overloads (GenerateParams | None, optional): The parameters to be used for generating the identifier.

        Returns:
            str: The identifier string.
        """
        provider = next(name for name, klass in g_providers.items() if isinstance(self, klass))
        params_dict = self._merge_params(overloads)
        if not params_dict:
            return f"{provider}!{self.model}"

        if "extra" in params_dict:
            logger.warning("Extra parameters are not supported in identifiers.")
            params_dict.pop("extra")

        if "stop" in params_dict:
            params_dict["stop"] = ";".join(params_dict["stop"])

        params = ",".join([f"{k}={v}" for k, v in params_dict.items()])

        return f"{provider}!{self.model},{params}"

    def _merge_params(self, overloads: GenerateParams | None = None) -> dict[str, t.Any]:
        """
        Helper to merge the parameters of the current instance with the provided `overloads` parameters.

        Typically used to prepare a dictionary of API parameters for a request.

        Args:
            overloads (GenerateParams): The parameters to be merged with the current instance's parameters.

        Returns:
            dict[str, t.Any]: The merged parameters.
        """
        params: dict[str, t.Any] = self.params.model_dump(exclude_unset=True) if self.params else {}
        if overloads is None:
            return params

        overloads_dict = overloads.model_dump(exclude_unset=True)
        if "extra" in overloads_dict:
            params.update(overloads_dict.pop("extra"))

        for name, value in overloads_dict.items():
            if value is not None:
                params[name] = value

        return params

    # Message generation

    def generate_message(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> Message:
        """
        Generates the next message for a given set of messages.

        Args:
            messages (Sequence[Message]): The list of messages to generate completion for.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            Message: The generated completion message.

        Raises:
            NotImplementedError: This generator does not support this method.
        """
        raise NotImplementedError("generate_message is not supported by this generator.")

    async def agenerate_message(
        self, messages: t.Sequence[Message], overloads: GenerateParams | None = None
    ) -> Message:
        """
        Asynchronously generates the next message for a given set of messages.

        Args:
            messages (Sequence[Message]): A sequence of messages.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            Coroutine[None, None, Message]: A coroutine that yields completion messages.
        """
        raise NotImplementedError("agenerate_message is not supported by this generator.")

    # Text generation

    def generate_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        """
        Generates a string completion of the given text.

        Args:
            text (str): The input text to be completed.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            str: The completed text.

        Raises:
            NotImplementedError: This generator does not support this method.
        """
        raise NotImplementedError("generate_text is not supported by this generator.")

    async def agenerate_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        """
        Asynchronously generates a string completion of the given text.

        Args:
            text (str): The input text to be completed.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            Coroutine[None, None, str]: A coroutine that yields the completed text.

        Raises:
            NotImplementedError: This generator does not support this method.
        """
        raise NotImplementedError("agenerate_text is not supported by this generator.")

    # Batching messages

    def batch_messages(
        self,
        many: t.Sequence[t.Sequence[Message]],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        """
        Generate a batch of messages using the specified parameters.

        Note:
            If supplied, the length of `overloads` must be the same as the length of `many`.

        Args:
            many (Sequence[Sequence[Message]]): A sequence of sequences of messages.
            overloads (Sequence[GenerateParams | None], optional): A sequence of GenerateParams objects or None. Defaults to None.
            fixed (Sequence[Message], optional): A sequence of fixed messages to be prefixed before every item of `many`. Defaults to None.

        Returns:
            Sequence[Message]: A sequence of generated messages.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("batch_messages is not supported by this generator.")

    async def abatch_messages(
        self,
        many: t.Sequence[t.Sequence[Message]],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: t.Sequence[Message],
    ) -> t.Sequence[Message]:
        """
        Asynchronously Generate a batch of messages based on the given parameters.

        Note:
            If supplied, the length of `overloads` must be the same as the length of `many`.

        Args:
            many (Sequence[Sequence[Message]]): A sequence of sequences of messages.
            overloads (Sequence[GenerateParams | None], optional): A sequence of GenerateParams or None. Defaults to None.
            fixed (Sequence[Message]): A sequence of fixed messages to be prefixed before every item of `many`. Defaults to None.

        Returns:
            Sequence[Message]: A sequence of generated messages.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("abatch_messages is not supported by this generator.")

    # Batching texts

    def batch_texts(
        self,
        many: t.Sequence[str],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: str | None = None,
    ) -> t.Sequence[str]:
        """
        Generate a batch of texts using the generator.

        Note:
            If supplied, the length of `overloads` must be the same as the length of `many`.

        Args:
            many (Sequence[str]): The input texts for generating the batch.
            overloads (Sequence[GenerateParams | None] | None, optional): Additional parameters for generating each text in the batch. Defaults to None.
            fixed (str | None, optional): A fixed input text to be used as a prefix for all of `many`. Defaults to None.

        Returns:
            Sequence[str]: The generated texts in the batch.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("batch_texts is not supported by this generator.")

    async def abatch_texts(
        self,
        many: t.Sequence[str],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: str | None = None,
    ) -> t.Sequence[str]:
        """
        Asynchronously Generate multiple texts in batch.

        Args:
            many (Sequence[str]): A sequence of texts to generate.
            overloads (Sequence[GenerateParams | None] | None, optional): A sequence of optional parameters for each text. Defaults to None.
            fixed (str | None, optional): A fixed parameter for all texts. Defaults to None.

        Returns:
            Sequence[str]: A sequence of generated texts.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("abatch_texts is not supported by this generator.")

    # Helper alternative to chat(generator) -> generator.chat(...)
    #
    # Overloads seem odd, but mypy doesn't like the TypedDict in a list otherwise

    @t.overload
    def chat(
        self,
        messages: t.Sequence[MessageDict],
        overloads: GenerateParams | None = None,
    ) -> PendingChat:
        ...

    @t.overload
    def chat(
        self, messages: t.Sequence[Message] | MessageDict | Message | str, overloads: GenerateParams | None = None
    ) -> PendingChat:
        ...

    def chat(
        self,
        messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str,
        overloads: GenerateParams | None = None,
    ) -> PendingChat:
        """
        Builds a pending chat with the given messages and optional overloads.

        Args:
            messages (Sequence[MessageDict] | Sequence[Message] | str): The messages to be sent in the chat.
            overloads (GenerateParams | None, optional): Optional parameters for generating responses. Defaults to None.

        Returns:
            PendingChat: Pending chat to run.
        """
        return PendingChat(self, Message.fit_as_list(messages), overloads)

    # Helper alternative to complete(generator) -> generator.complete(...)

    def complete(self, text: str, overloads: GenerateParams | None = None) -> PendingCompletion:
        """
        Generates a pending string completion of the given text.

        Args:
            text (str): The input text to be completed.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            str: The completed text.
        """
        return PendingCompletion(self, text, overloads)


@t.overload
def chat(
    generator: "Generator",
    messages: t.Sequence[MessageDict],
    overloads: GenerateParams | None = None,
) -> PendingChat:
    ...


@t.overload
def chat(
    generator: "Generator",
    messages: t.Sequence[Message] | MessageDict | Message | str,
    overloads: GenerateParams | None = None,
) -> PendingChat:
    ...


def chat(
    generator: "Generator",
    messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str,
    overloads: GenerateParams | None = None,
) -> PendingChat:
    """
    Creates a pending chat using the given generator, messages, and overloads.

    Args:
        generator (Generator): The generator to use for creating the chat.
        messages (Sequence[MessageDict] | Sequence[Message] | MessageDict | Message | str):
            The messages to include in the chat. Can be a single message or a sequence of messages.
        overloads (GenerateParams | None, optional): Additional parameters for generating the chat.
            Defaults to None.

    Returns:
        PendingChat: Pending chat to run.
    """
    return generator.chat(messages, overloads)


def complete(
    generator: Generator,
    text: str,
    overloads: GenerateParams | None = None,
) -> PendingCompletion:
    return generator.complete(text, overloads)


def get_identifier(generator: Generator, overloads: GenerateParams | None = None) -> str:
    """
    Returns the identifier for the given generator.

    Delegates to [rigging.generator.Generator.to_identifier][]

    Args:
        generator (Generator): The generator object.
        overloads (GenerateParams | None, optional): The generate parameters. Defaults to None.

    Returns:
        str: The identifier for the generator.
    """
    return generator.to_identifier(overloads)


def get_generator(identifier: str) -> Generator:
    """
    Get a generator by an identifier string. Uses LiteLLM by default.

    Identifier strings are formatted like `<provider>!<model>,<**kwargs>`

    (provider is optional and defaults to "litellm" if not specified)

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
        identifier (str): The identifier string to use to get a generator.

    Returns:
        Generator: The generator object.

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
        provider (str): The name of the provider.
        generator_cls (type[Generator]): The generator class to register.

    Returns:
        None
    """
    global g_providers
    g_providers[provider] = generator_cls


def trace_messages(messages: t.Sequence[Message], title: str) -> None:
    """
    Helper function to trace log a sequence of Message objects.

    Args:
        messages (Sequence[Message]): A sequence of Message objects to be logged.
        title (str): The title to be displayed in the log.

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
        content (str): The string content to be logged.
        title (str): The title of the log entry.

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
        While this generator implements the batch methods, they are not performant and simply loop over the inputs.
    """

    def generate_message(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> Message:
        trace_messages(messages, "Conversations")

        messages_as_dicts = [message.model_dump(include={"role", "content"}) for message in messages]
        params = self._merge_params(overloads)
        result = litellm.completion(self.model, messages_as_dicts, api_key=self.api_key, **params)
        response = result.choices[-1].message.content.strip()
        next_message = Message(role="assistant", content=response)

        trace_messages([next_message], "Response")

        return next_message

    async def agenerate_message(
        self, messages: t.Sequence[Message], overloads: GenerateParams | None = None
    ) -> Message:
        trace_messages(messages, "Conversations")

        messages_as_dicts = [message.model_dump(include={"role", "content"}) for message in messages]
        params = self._merge_params(overloads)
        result = await litellm.acompletion(self.model, messages_as_dicts, api_key=self.api_key, **params)
        response = result.choices[-1].message.content.strip()
        next_message = Message(role="assistant", content=response)

        trace_messages([next_message], "Response")

        return next_message

    def generate_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        trace_str(text, "Text")

        params = self._merge_params(overloads)
        result = litellm.text_completion(text, self.model, api_key=self.api_key, **params)
        completion: str = result.choices[-1]["text"]

        trace_str(completion, "Completion")

        return completion

    async def agenerate_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        trace_str(text, "Text")

        params = self._merge_params(overloads)
        result = await litellm.atext_completion(text, self.model, api_key=self.api_key, **params)
        completion: str = result.choices[-1]["text"]

        trace_str(completion, "Completion")

        return completion

    def batch_messages(
        self,
        many: t.Sequence[t.Sequence[Message]],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: t.Sequence[Message] | None = None,
    ) -> t.Sequence[Message]:
        if overloads is not None and len(overloads) != len(many):
            raise ValueError("Length of overloads must match the length of many.")

        overloads = [None] * len(many) if overloads is None else overloads
        if fixed is not None:
            many = [list(fixed) + list(messages) for messages in many]

        return [self.generate_message(messages, overload) for messages, overload in zip(many, overloads, strict=True)]

    async def abatch_messages(
        self,
        many: t.Sequence[t.Sequence[Message]],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: t.Sequence[Message],
    ) -> t.Sequence[Message]:
        if overloads is not None and len(overloads) != len(many):
            raise ValueError("Length of overloads must match the length of many.")

        overloads = [None] * len(many) if overloads is None else overloads
        if fixed is not None:
            many = [list(fixed) + list(messages) for messages in many]

        return await asyncio.gather(
            *[self.agenerate_message(messages, overload) for messages, overload in zip(many, overloads, strict=True)]
        )

    def batch_texts(
        self,
        many: t.Sequence[str],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: str | None = None,
    ) -> t.Sequence[str]:
        if overloads is not None and len(overloads) != len(many):
            raise ValueError("Length of overloads must match the length of many.")

        overloads = [None] * len(many) if overloads is None else overloads
        if fixed is not None:
            many = [fixed + message for message in many]

        return [self.generate_text(message, overload) for message, overload in zip(many, overloads, strict=True)]

    async def abatch_texts(
        self,
        many: t.Sequence[str],
        overloads: t.Sequence[GenerateParams | None] | None = None,
        *,
        fixed: str | None = None,
    ) -> t.Sequence[str]:
        if overloads is not None and len(overloads) != len(many):
            raise ValueError("Length of overloads must match the length of many.")

        overloads = [None] * len(many) if overloads is None else overloads
        if fixed is not None:
            many = [fixed + message for message in many]

        return await asyncio.gather(
            *[self.agenerate_text(message, overload) for message, overload in zip(many, overloads, strict=True)]
        )


g_providers["litellm"] = LiteLLMGenerator
