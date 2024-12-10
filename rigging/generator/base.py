from __future__ import annotations

import inspect
import typing as t

from loguru import logger
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator
from typing_extensions import Self

from rigging.error import InvalidModelSpecifiedError
from rigging.message import Message, MessageDict
from rigging.tool.api import ToolChoice, ToolDefinition

if t.TYPE_CHECKING:
    from rigging.chat import ChatPipeline, WatchChatCallback
    from rigging.completion import CompletionPipeline, WatchCompletionCallback
    from rigging.prompt import P, Prompt, R

    WatchCallbacks = t.Union[WatchChatCallback, WatchCompletionCallback]


CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])

# Global provider map


@t.runtime_checkable
class LazyGenerator(t.Protocol):
    def __call__(self) -> type[Generator]:
        ...


g_providers: dict[str, type[Generator] | LazyGenerator] = {}


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

    top_k: int | None = None
    """The top-k sampling parameter."""

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

    tools: list[ToolDefinition] | None = None
    """The tools to be used in the generation."""

    tool_choice: ToolChoice | None = None
    """The tool choice to be used in the generation."""

    parallel_tool_calls: bool | None = None
    """Whether to run allow tool calls in parallel."""

    extra: dict[str, t.Any] = Field(default_factory=dict)
    """Extra parameters to be passed to the API."""

    @field_validator("tools", mode="before")
    def validate_tools(cls, value: t.Any) -> t.Any:
        if isinstance(value, list) and all(isinstance(v, dict) for v in value):
            return [ToolDefinition.model_validate(v) for v in value]
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            return [ToolDefinition.model_validate_json(v) for v in value]
        return value

    @field_validator("stop", mode="before")
    def validate_stop(cls, value: t.Any) -> t.Any:
        if value is None:
            return None
        if isinstance(value, str):
            return value.split(";")
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        raise ValueError("Stop sequences must be a list or a string separated by ';'")

    def merge_with(self, *others: t.Optional[GenerateParams]) -> GenerateParams:
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
            other_dict = other.model_dump(exclude_unset=True, exclude_none=True)
            for name in other_dict.keys():
                updates[name] = getattr(other, name)

        return self.model_copy(update=updates)

    def to_dict(self) -> dict[str, t.Any]:
        """
        Convert the parameters to a dictionary.

        Returns:
            The parameters as a dictionary.
        """
        params = self.model_dump(exclude_none=True)
        if "extra" in params:
            params.update(params.pop("extra"))
        return params


StopReason = t.Literal["stop", "length", "content_filter", "tool_calls", "unknown"]
"""Reporting reason for generation completing."""


def convert_stop_reason(reason: t.Optional[str]) -> StopReason:
    if reason in ["stop", "eos"]:
        return "stop"
    elif reason in ["model_length"]:
        return "length"
    elif reason in ["length"]:
        return "length"
    elif reason in ["content_filter"]:
        return "content_filter"
    elif reason and "tool" in reason:
        return "tool_calls"
    return "unknown"


class Usage(BaseModel):
    input_tokens: int
    """The number of input tokens."""
    output_tokens: int
    """The number of output tokens."""
    total_tokens: int
    """The total number of tokens processed."""


GeneratedT = t.TypeVar("GeneratedT", Message, str)


class GeneratedMessage(BaseModel):
    """A generated message with additional generation information."""

    message: Message
    """The generated message."""

    stop_reason: t.Annotated[StopReason, BeforeValidator(convert_stop_reason)] = "unknown"
    """The reason for stopping generation."""

    usage: t.Optional[Usage] = None
    """The usage statistics for the generation if available."""

    extra: dict[str, t.Any] = Field(default_factory=dict)
    """Any additional information from the generation."""

    def __str__(self) -> str:
        return str(self.message)

    @classmethod
    def from_text(cls, text: str, stop_reason: StopReason = "unknown") -> GeneratedMessage:
        return cls(message=Message(role="assistant", content=text), stop_reason=stop_reason)


class GeneratedText(BaseModel):
    """A generated text with additional generation information."""

    text: str
    """The generated text."""

    stop_reason: t.Annotated[StopReason, BeforeValidator(convert_stop_reason)] = "unknown"
    """The reason for stopping generation."""

    usage: t.Optional[Usage] = None
    """The usage statistics for the generation if available."""

    extra: dict[str, t.Any] = Field(default_factory=dict)
    """Any additional information from the generation."""

    def __str__(self) -> str:
        return self.text

    @classmethod
    def from_text(cls, text: str, stop_reason: StopReason = "unknown") -> GeneratedText:
        return cls(text=text, stop_reason=stop_reason)

    def to_generated_message(self) -> GeneratedMessage:
        return GeneratedMessage(
            message=Message(role="assistant", content=self.text),
            stop_reason=self.stop_reason,
            usage=self.usage,
            extra=self.extra,
        )


class Generator(BaseModel):
    """
    Base class for all rigging generators.

    This class provides common functionality and methods for generating completion messages.

    A subclass of this can implement both or one of the following:

    - `generate_messages`: Process a batch of messages.
    - `generate_texts`: Process a batch of texts.
    """

    model: str
    """The model name to be used by the generator."""
    api_key: str | None = Field(None, exclude=True)
    """The API key used for authentication."""
    params: GenerateParams
    """The parameters used for generating completion messages."""

    _watch_callbacks: list[WatchCallbacks] = []
    _wrap: t.Callable[[CallableT], CallableT] | None = None

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

    def watch(self, *callbacks: WatchCallbacks, allow_duplicates: bool = False) -> Generator:
        """
        Registers watch callbacks to be passed to any created
        [rigging.chat.ChatPipeline][] or [rigging.completion.CompletionPipeline][].

        Args:
            *callbacks: The callback functions to be executed.
            allow_duplicates: Whether to allow (seemingly) duplicate callbacks to be added.

        ```
        async def log(chats: list[Chat]) -> None:
            ...

        await pipeline.watch(log).run()
        ```

        Returns:
            The current instance of the chat.
        """
        for callback in callbacks:
            if allow_duplicates or callback not in self._watch_callbacks:
                self._watch_callbacks.append(callback)
        return self

    def load(self) -> Self:
        """
        If supported, trigger underlying loading and preparation of the model.

        Returns:
            The generator.
        """
        return self

    def unload(self) -> Self:
        """
        If supported, clean up resources used by the underlying model.

        Returns:
            The generator.
        """
        return self

    def wrap(self, func: t.Callable[[CallableT], CallableT] | None) -> Self:
        """
        If supported, wrap any underlying interior framework calls with this function.

        This is useful for adding things like backoff or rate limiting.

        Args:
            func: The function to wrap the calls with.

        Returns:
            The generator.
        """
        # TODO: Not sure why mypy is complaining here
        self._wrap = func  # type: ignore [assignment]
        return self

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        """
        Generate a batch of messages using the specified parameters.

        Note:
            The length of `params` must be the same as the length of `many`.

        Args:
            messages: A sequence of sequences of messages.
            params: A sequence of GenerateParams objects.

        Returns:
            A sequence of generated messages.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("`generate_messages` is not supported by this generator.")

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        """
        Generate a batch of text completions using the generator.

        Note:
            This method falls back to looping over the inputs and calling `generate_text` for each item.

        Note:
            If supplied, the length of `params` must be the same as the length of `many`.

        Args:
            texts: The input texts for generating the batch.
            params: Additional parameters for generating each text in the batch.

        Returns:
            The generated texts.

        Raises:
            NotImplementedError: This method is not supported by this generator.
        """
        raise NotImplementedError("`generate_texts` is not supported by this generator.")

    # Helper alternative to chat(generator) -> generator.chat(...)
    #
    # params seem odd, but mypy doesn't like the TypedDict in a list otherwise

    @t.overload
    def chat(
        self,
        messages: t.Sequence[MessageDict],
        params: GenerateParams | None = None,
    ) -> ChatPipeline:
        ...

    @t.overload
    def chat(
        self,
        messages: t.Sequence[Message] | MessageDict | Message | str | None = None,
        params: GenerateParams | None = None,
    ) -> ChatPipeline:
        ...

    def chat(
        self,
        messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str | None = None,
        params: GenerateParams | None = None,
    ) -> ChatPipeline:
        """
        Build a chat pipeline with the given messages and optional params overloads.

        Args:
            messages: The messages to be sent in the chat.
            params: Optional parameters for generating responses.

        Returns:
            chat pipeline to run.
        """
        from rigging.chat import ChatPipeline, WatchChatCallback

        chat_watch_callbacks = [cb for cb in self._watch_callbacks if isinstance(cb, (WatchChatCallback))]

        return ChatPipeline(
            self,
            Message.fit_as_list(messages) if messages else [],
            params=params,
            watch_callbacks=chat_watch_callbacks,
        )

    # Helper alternative to complete(generator) -> generator.complete(...)

    def complete(self, text: str, params: GenerateParams | None = None) -> CompletionPipeline:
        """
        Build a completion pipeline of the given text with optional param overloads.

        Args:
            text: The input text to be completed.
            params: The parameters to be used for completion.

        Returns:
            The completed text.
        """
        from rigging.completion import CompletionPipeline, WatchCompletionCallback

        completion_watch_callbacks = [cb for cb in self._watch_callbacks if isinstance(cb, (WatchCompletionCallback))]

        return CompletionPipeline(self, text, params=params, watch_callbacks=completion_watch_callbacks)

    def prompt(self, func: t.Callable[P, t.Coroutine[None, None, R]]) -> Prompt[P, R]:
        """
        Decorator to convert a function into a prompt bound to this generator.

        See [rigging.prompt.prompt][] for more information.

        Args:
            func: The function to be converted into a prompt.

        Returns:
            The prompt.
        """
        from rigging.prompt import prompt

        return prompt(func, generator=self)


@t.overload
def chat(
    generator: Generator,
    messages: t.Sequence[MessageDict],
    params: GenerateParams | None = None,
) -> ChatPipeline:
    ...


@t.overload
def chat(
    generator: Generator,
    messages: t.Sequence[Message] | MessageDict | Message | str | None = None,
    params: GenerateParams | None = None,
) -> ChatPipeline:
    ...


def chat(
    generator: Generator,
    messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str | None = None,
    params: GenerateParams | None = None,
) -> ChatPipeline:
    """
    Creates a chat pipeline using the given generator, messages, and params.

    Args:
        generator: The generator to use for creating the chat.
        messages:
            The messages to include in the chat. Can be a single message or a sequence of messages.
        params: Additional parameters for generating the chat.

    Returns:
        chat pipeline to run.
    """
    return generator.chat(messages, params)


def complete(
    generator: Generator,
    text: str,
    params: GenerateParams | None = None,
) -> CompletionPipeline:
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

    provider = next(
        name for name, klass in g_providers.items() if isinstance(klass, type) and isinstance(generator, klass)
    )
    identifier = f"{provider}!{generator.model}"

    extra_cls_args = generator.model_dump(exclude_unset=True, exclude={"model", "api_key", "params"})
    if extra_cls_args:
        identifier += f",{','.join([f'{k}={v}' for k, v in extra_cls_args.items()])}"

    merged_params = generator.params.merge_with(params)
    if merged_params.extra:
        logger.debug("Extra parameters are not supported in identifiers.")
        merged_params.extra = {}

    params_dict = merged_params.to_dict()
    if params_dict:
        if "stop" in params_dict:
            params_dict["stop"] = ";".join(params_dict["stop"])
        identifier += f",{','.join([f'{k}={v}' for k, v in params_dict.items()])}"

    return identifier


def get_generator(identifier: str, *, params: GenerateParams | None = None) -> Generator:
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
        params: The generation parameters to use for the generator.
            These will override any parameters specified in the identifier string.

    Returns:
        The generator object.

    Raises:
        InvalidModelSpecified: If the identifier is invalid.
    """

    provider: str = list(g_providers.keys())[0]
    model: str = identifier

    # Split provider, model, and kwargs

    if "!" in identifier:
        try:
            provider, model = identifier.split("!")
        except Exception as e:
            raise InvalidModelSpecifiedError(identifier) from e

    if provider not in g_providers:
        raise InvalidModelSpecifiedError(identifier)

    if not isinstance(g_providers[provider], type):
        lazy_generator = t.cast(LazyGenerator, g_providers[provider])
        g_providers[provider] = lazy_generator()

    generator_cls = t.cast(type[Generator], g_providers[provider])

    kwargs = {}
    if "," in model:
        try:
            model, kwargs_str = model.split(",", 1)
            kwargs = dict(arg.split("=") for arg in kwargs_str.split(","))
        except Exception as e:
            raise InvalidModelSpecifiedError(identifier) from e

    # See if any of the kwargs would apply to the cls constructor directly
    init_signature = inspect.signature(generator_cls)
    init_kwargs: dict[str, t.Any] = {k: kwargs.pop(k) for k in list(kwargs.keys())[:] if k in init_signature.parameters}

    # Do some subtle type conversion
    for k, v in init_kwargs.items():
        try:
            init_kwargs[k] = float(v)
            continue
        except ValueError:
            pass

        try:
            init_kwargs[k] = int(v)
            continue
        except ValueError:
            pass

        if isinstance(v, str) and v.lower() in ["true", "false"]:
            init_kwargs[k] = v.lower() == "true"

    try:
        merged_params = GenerateParams(**kwargs).merge_with(params)
    except Exception as e:
        raise InvalidModelSpecifiedError(identifier) from e

    return generator_cls(model=model, params=merged_params, **init_kwargs)


def register_generator(provider: str, generator_cls: type[Generator] | LazyGenerator) -> None:
    """
    Register a generator class for a provider id.

    This let's you use [rigging.generator.get_generator][] with a custom generator class.

    Args:
        provider: The name of the provider.
        generator_cls: The generator class to register.
    """
    global g_providers
    g_providers[provider] = generator_cls


def trace_messages(messages: t.Sequence[Message] | t.Sequence[GeneratedMessage], title: str) -> None:
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


def trace_str(content: str | GeneratedText, title: str) -> None:
    """
    Helper function to trace log a string.

    Parameters:
        content: The string content to be logged.
        title: The title of the log entry.

    Returns:
        None
    """
    logger.trace(f"--- {title} ---")
    logger.trace(str(content))
    logger.trace("---")
