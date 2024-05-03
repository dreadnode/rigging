"""
Generators produce completions for a given set of messages or text.
"""

import abc
import typing as t

import litellm  # type: ignore
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rigging.chat import PendingChat
from rigging.error import InvalidModelSpecifiedError
from rigging.message import (
    Message,
    MessageDict,
)

# We should probably let people configure
# this independently, but for now we'll
# fix it to prevent confusion
litellm.drop_params = True


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


class Generator(BaseModel, abc.ABC):
    """
    Base class for all rigging generators.

    This class provides common functionality and methods for generating completion messages.

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

    def complete_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        """
        Generates a string completion of the given text.

        Args:
            text (str): The input text to be completed.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            str: The completed text.

        Raises:
            NotImplementedError: This generator does not support the `complete_text` method.
        """
        raise NotImplementedError("complete_text is not supported by this generator.")

    async def acomplete_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        """
        Asynchronously generates a string completion of the given text.

        Args:
            text (str): The input text to be completed.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            Coroutine[None, None, str]: A coroutine that yields the completed text.

        Raises:
            NotImplementedError: This generator does not support the `acomplete_text` method.
        """
        raise NotImplementedError("acomplete_text is not supported by this generator.")

    @abc.abstractmethod
    def complete(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> Message:
        """
        Generates the next message for a given set of messages.

        Args:
            messages (Sequence[Message]): The list of messages to generate completion for.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            Message: The generated completion message.
        """
        ...

    @abc.abstractmethod
    async def acomplete(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> Message:
        """
        Asynchronously generates the next message for a given set of messages.

        Args:
            messages (Sequence[Message]): A sequence of messages.
            overloads (GenerateParams | None, optional): The parameters to be used for completion.

        Returns:
            Coroutine[None, None, Message]: A coroutine that yields completion messages.
        """
        ...

    @t.overload
    def chat(self, messages: t.Sequence[MessageDict]) -> PendingChat:
        ...

    @t.overload
    def chat(self, messages: t.Sequence[Message] | str) -> PendingChat:
        ...

    def chat(
        self, messages: t.Sequence[MessageDict] | t.Sequence[Message] | str, overloads: GenerateParams | None = None
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


# Helper function external to a generator


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
    messages: t.Sequence[Message] | str,
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
    return PendingChat(generator, Message.fit_as_list(messages), overloads)


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

    !!! note
        Find more information about supported models and formats [in their docs.](https://docs.litellm.ai/docs/providers).
    """

    def complete(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> Message:
        trace_messages(messages, "Conversations")

        messages_as_dicts = [message.model_dump() for message in messages]
        params = self._merge_params(overloads)
        result = litellm.completion(self.model, messages_as_dicts, api_key=self.api_key, **params)
        response = result.choices[-1].message.content.strip()
        next_message = Message(role="assistant", content=response)

        trace_messages([next_message], "Response")

        return next_message

    async def acomplete(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> Message:
        trace_messages(messages, "Conversations")

        messages_as_dicts = [message.model_dump() for message in messages]
        params = self._merge_params(overloads)
        result = await litellm.acompletion(self.model, messages_as_dicts, api_key=self.api_key, **params)
        response = result.choices[-1].message.content.strip()
        next_message = Message(role="assistant", content=response)

        trace_messages([next_message], "Response")

        return next_message

    def complete_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        trace_str(text, "Text")
        params = self._merge_params(overloads)
        result = litellm.text_completion(self.model, text, api_key=self.api_key, **params)
        completion: str = result.choices[-1]["text"]
        trace_str(completion, "Completion")
        return completion

    async def acomplete_text(self, text: str, overloads: GenerateParams | None = None) -> str:
        trace_str(text, "Text")
        params = self._merge_params(overloads)
        result = await litellm.atext_completion(self.model, text, api_key=self.api_key, **params)
        completion: str = result.choices[-1]["text"]
        trace_str(completion, "Completion")
        return completion


g_providers: dict[str, type["Generator"]] = {
    "litellm": LiteLLMGenerator,
}


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
