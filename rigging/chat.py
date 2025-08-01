"""
Chats are used pre and post generation to hold messages.

They are the primary way to interact with the generator.
"""

import asyncio
import contextlib
import inspect
import types
import typing as t
import warnings
from contextlib import aclosing, asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import runtime_checkable
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    WithJsonSchema,
    computed_field,
)

from rigging.caching import CacheMode, apply_cache_mode_to_messages
from rigging.error import MaxDepthError, PipelineWarning
from rigging.generator import GenerateParams, Generator, get_generator
from rigging.generator.base import StopReason, Usage
from rigging.message import (
    Content,
    Message,
    MessageDict,
    Messages,
    MessageSlice,
    SliceType,
)
from rigging.message import (
    inject_system_content as inject_system_content_into_messages,
)
from rigging.model import Model, ModelT, SystemErrorModel, ValidationErrorModel
from rigging.tokenizer import TokenizedChat, Tokenizer, get_tokenizer
from rigging.tools import Tool, ToolCall, ToolChoice, ToolMode
from rigging.transform import (
    PostTransform,
    Transform,
    get_transform,
    make_tools_to_xml_transform,
    tools_to_json_in_xml_transform,
    tools_to_json_transform,
    tools_to_json_with_tag_transform,
)
from rigging.util import flatten_list, get_callable_name

if t.TYPE_CHECKING:
    from dreadnode.metric import Scorer, ScorerCallable
    from dreadnode.scorers.rigging import ChatFilterFunction, ChatFilterMode
    from elasticsearch import AsyncElasticsearch  # type: ignore [import-not-found, unused-ignore]

    from rigging.data import ElasticOpType
    from rigging.prompt import Prompt

P = t.ParamSpec("P")
R = t.TypeVar("R")

CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])

DEFAULT_MAX_ROUNDS = 5
"""Maximum number of internal callback rounds to attempt during generation before giving up."""

DEFAULT_MAX_DEPTH = 20
"""Maximum depth of nested pipeline generations to attempt before giving up."""

FailMode = t.Literal["raise", "skip", "include"]
"""
How to handle failures in pipelines.

- raise: Raise an exception when a failure is encountered.
- skip: Ignore the error and do not include the failed chat in the final output.
- include: Mark the message as failed and include it in the final output.
"""


class Chat(BaseModel):
    """
    A completed chat interaction.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, json_schema_extra={"rigging.type": "chat"}
    )

    uuid: UUID = Field(default_factory=uuid4)
    """The unique identifier for the chat."""
    timestamp: datetime = Field(default_factory=datetime.now, repr=False)
    """The timestamp when the chat was created."""
    messages: list[Message]
    """The list of messages prior to generation."""
    generated: list[Message] = Field(default_factory=list)
    """The list of messages resulting from the generation."""
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    """Additional metadata for the chat."""

    stop_reason: StopReason = Field(default="unknown")
    """The reason the generation stopped."""
    usage: Usage | None = Field(None, repr=False)
    """The usage statistics for the generation if available."""
    extra: dict[str, t.Any] = Field(default_factory=dict, repr=False)
    """Any additional information from the generation."""

    generator: Generator | None = Field(None, exclude=True, repr=False)
    """The generator associated with the chat."""
    params: GenerateParams | None = Field(None, repr=False)
    """Any additional generation params used for this chat."""

    error: (
        t.Annotated[
            BaseException,
            PlainSerializer(
                lambda x: str(x),
                return_type=str,
                when_used="json-unless-none",
            ),
            WithJsonSchema({"type": "string", "description": "Error message"}),
        ]
        | None
    ) = Field(None, repr=False)
    """Holds any exception that was caught during the generation pipeline."""
    failed: bool = Field(False, exclude=False, repr=True)
    """
    Indicates whether conditions during generation were not met.
    This is typically used for graceful error handling when parsing.
    """

    _pipeline: "ChatPipeline | None"

    @computed_field(repr=False)  # type: ignore [prop-decorator]
    @property
    def generator_id(self) -> str | None:
        """The identifier of the generator used to create the chat"""
        if self.generator is not None:
            return self.generator.to_identifier()
        return None

    def __init__(
        self,
        messages: Messages,
        generated: Messages | None = None,
        generator: Generator | None = None,
        pipeline: "ChatPipeline | None" = None,
        params: GenerateParams | None = None,
        **kwargs: t.Any,
    ):
        """
        Initialize a Chat object.

        Args:
            messages: The messages for the chat.
            generated: The next messages for the chat.
            generator: The generator associated with this chat.
            **kwargs: Additional keyword arguments (typically used for deserialization)
        """

        if "generator_id" in kwargs and generator is None:
            # TODO: Should we move params to self.params?
            generator_id = kwargs.pop("generator_id")
            if generator_id:
                generator = get_generator(generator_id)

        # We can't deserialize an error
        if isinstance(kwargs.get("error"), str):
            kwargs.pop("error")

        super().__init__(
            messages=Message.fit_as_list(messages),
            generated=Message.fit_as_list(generated) if generated is not None else [],
            generator=generator,
            params=params,
            **kwargs,
        )

        self._pipeline = pipeline

    def __len__(self) -> int:
        return len(self.all)

    @property
    def all(self) -> list[Message]:
        """Returns all messages in the chat, including the next messages."""
        return self.messages + self.generated

    @property
    def prev(self) -> list[Message]:
        """Alias for the .messages property"""
        return self.messages

    @property
    def next(self) -> list[Message]:
        """Alias for the .generated property"""
        return self.generated

    @property
    def last(self) -> Message:
        """Alias for .all[-1]"""
        return self.all[-1]

    @property
    def conversation(self) -> str:
        """Returns a string representation of the chat."""
        conversation = "\n\n".join([str(m) for m in self.all])
        if self.error:
            conversation += f"\n\n[error]: {self.error}"
        return conversation

    def __str__(self) -> str:
        formatted = f"--- Chat {self.uuid}"
        formatted += f"\n |- timestamp:   {self.timestamp.isoformat()}"
        if self.usage:
            formatted += f"\n |- usage:       {self.usage}"
        if self.generator:
            formatted += f"\n |- generator:   {self.generator.to_identifier(short=True)}"
        if self.stop_reason:
            formatted += f"\n |- stop_reason: {self.stop_reason}"
        if self.metadata:
            formatted += f"\n |- metadata:    {self.metadata}"
        formatted += f"\n\n{self.conversation}\n"
        return formatted

    @property
    def message_dicts(self) -> list[MessageDict]:
        """Returns the chat as a minimal message dictionaries."""
        return [t.cast("MessageDict", m.to_openai()) for m in self.all]

    @property
    def message_metadata(self) -> dict[str, t.Any]:
        """Returns a merged dictionary of metadata from all messages in the chat."""
        metadata: dict[str, t.Any] = {}
        for message in self.all:
            if message.metadata:
                metadata.update(message.metadata)
        return metadata

    def message_slices(
        self,
        slice_type: SliceType | None = None,
        filter_fn: t.Callable[[MessageSlice], bool] | None = None,
        *,
        reverse: bool = False,
    ) -> list[MessageSlice]:
        """
        Get all slices across all messages with optional filtering.

        See Message.find_slices() for more information.

        Args:
            slice_type: Filter by slice type
            filter_fn: A function to filter slices. If provided, only slices for which
                `filter_fn(slice)` returns True will be included.
            reverse: If True, the slices will be returned in reverse order.

        Returns:
            List of all matching slices across all messages
        """
        all_slices = []
        for message in self.messages:
            all_slices.extend(
                message.find_slices(slice_type=slice_type, filter_fn=filter_fn, reverse=reverse),
            )
        return all_slices

    def meta(self, **kwargs: t.Any) -> "Chat":
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated chat.
        """
        self.metadata.update(kwargs)
        return self

    def restart(
        self,
        *,
        generator: Generator | None = None,
        include_all: bool = False,
    ) -> "ChatPipeline":
        """
        Attempt to convert back to a ChatPipeline for further generation.

        Args:
            generator: The generator to use for the restarted chat. Otherwise
                the generator from the original ChatPipeline will be used.
            include_all: Whether to include the next messages in the restarted chat.

        Returns:
            The restarted chat.

        Raises:
            ValueError: If the chat was not created with a ChatPipeline and no generator is provided.
        """
        messages = self.all if include_all else self.messages
        if generator is None:
            if self._pipeline is not None:
                return self._pipeline.clone(chat=Chat(messages))
            generator = self.generator
        if generator is None:
            raise ValueError("Cannot restart a chat without an associated generator")
        return generator.chat(messages, self.params)

    def fork(
        self,
        messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | MessageDict | str,
        *,
        include_all: bool = False,
    ) -> "ChatPipeline":
        """
        Forks the chat by creating calling [rigging.chat.Chat.restart][] and appending the specified messages.

        Args:
            messages:
                The messages to be added to the new `ChatPipeline` instance.
            include_all: Whether to include the next messages in the restarted chat.

        Returns:
            A new pipeline with specified messages added.
        """
        return self.restart(include_all=include_all).add(messages)

    def continue_(
        self,
        messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | str,
    ) -> "ChatPipeline":
        """Alias for the [rigging.chat.Chat.fork][] with `include_all=True`."""
        return self.fork(messages, include_all=True)

    def clone(self, *, only_messages: bool = False) -> "Chat":
        """
        Creates a deep copy of the chat.

        Args:
            only_messages: If True, only the messages will be cloned.
                If False (default), the entire chat object will be cloned.

        Returns:
            A cloned chat.
        """
        new = Chat(
            [m.model_copy() for m in self.messages],
            [m.model_copy() for m in self.generated],
            self.generator,
        )
        if not only_messages:
            new.metadata = deepcopy(self.metadata)
            new.params = self.params.model_copy() if self.params is not None else None
            new.stop_reason = self.stop_reason
            new.usage = self.usage.model_copy() if self.usage is not None else None
            new.extra = deepcopy(self.extra)
            new.failed = self.failed
            new.error = self.error
        return new

    def apply(self, **kwargs: str) -> "Chat":
        """
        Calls [rigging.message.Message.apply][] on the last message in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The updated chat.
        """
        if self.generated:
            self.generated[-1] = self.generated[-1].apply(**kwargs)
        else:
            self.messages[-1] = self.messages[-1].apply(**kwargs)
        return self

    def apply_to_all(self, **kwargs: str) -> "Chat":
        """
        Calls [rigging.message.Message.apply][] on all messages in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The updated chat.
        """
        self.messages = Message.apply_to_list(self.messages, **kwargs)
        self.generated = Message.apply_to_list(self.generated, **kwargs)
        return self

    def inject_system_content(self, content: str) -> "Chat":
        """
        Injects content into the chat as a system message.

        Note:
            If the chat is empty or the first message is not a system message,
            a new system message with the given content is inserted at the beginning of the chat.
            If the first message is a system message, the content is appended to it.

        Args:
            content: The content to be injected.

        Returns:
            The updated chat.
        """
        self.messages = inject_system_content_into_messages(self.messages, content)
        return self

    def to_df(self) -> t.Any:
        """
        Converts the chat to a Pandas DataFrame.

        See [rigging.data.chats_to_df][] for more information.

        Returns:
            The chat as a DataFrame.
        """
        # Late import for circular
        from rigging.data import chats_to_df

        return chats_to_df(self)

    async def to_elastic(
        self,
        index: str,
        client: "AsyncElasticsearch",
        *,
        op_type: "ElasticOpType" = "index",
        create_index: bool = True,
        **kwargs: t.Any,
    ) -> int:
        """
        Converts the chat data to Elasticsearch format and indexes it.

        See [rigging.data.chats_to_elastic][] for more information.

        Returns:
            The number of chats indexed.
        """
        from rigging.data import chats_to_elastic

        return await chats_to_elastic(
            self,
            index,
            client,
            op_type=op_type,
            create_index=create_index,
            **kwargs,
        )

    def to_openai(self) -> list[dict[str, t.Any]]:
        """
        Converts the chat messages to the OpenAI-compatible JSON format.

        See Message.to_openai() for more information.

        Returns:
            The serialized chat.
        """
        return [m.to_openai() for m in self.all]

    async def to_tokens(
        self,
        tokenizer: str | Tokenizer,
        transform: str | Transform | None = None,
    ) -> TokenizedChat:
        """
        Converts the chat messages to a list of tokenized messages.

        Args:
            tokenizer: The tokenizer to use for tokenization. Can be a string identifier or a Tokenizer instance.
            transform: An optional transform to apply to the chat before tokenization. Can be a well-known transform
                identifier or a Transform instance.

        Returns:
            The serialized chat as a list of token lists.
        """

        if isinstance(tokenizer, str):
            tokenizer = get_tokenizer(tokenizer)

        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(
                f"Expected a Tokenizer instance, got {type(tokenizer).__name__}",
            )

        if isinstance(transform, str):
            transform = get_transform(transform)

        if transform and not isinstance(transform, Transform):
            raise TypeError(
                f"Expected a Transform instance, got {type(transform).__name__}",
            )

        chat = await self.transform(transform) if transform else self
        return await tokenizer.tokenize_chat(chat)

    async def transform(self, transform: Transform | str) -> "Chat":
        """
        Applies a transform to the chat.

        Args:
            transform: The transform to apply.

        Returns:
            A new chat with the transform applied to its messages and parameters.
        """
        if isinstance(transform, str):
            transform = get_transform(transform)
        messages = [m.clone() for m in self.messages]
        params = self.params.clone() if self.params else GenerateParams()
        messages, params, _ = await transform(self.messages, params)
        new = self.clone()
        new.messages = messages
        new.params = params
        return new


# List Helper Type


class ChatList(list[Chat]):
    """
    Represents a list of chat objects.

    Inherits from the built-in `list` class and is specialized for storing `Chat` objects.
    """

    def to_df(self) -> t.Any:
        """
        Converts the chat list to a Pandas DataFrame.

        See [rigging.data.chats_to_df][] for more information.

        Returns:
            The chat list as a DataFrame.
        """
        # Late import for circular
        from rigging.data import chats_to_df

        return chats_to_df(self)

    async def to_elastic(
        self,
        index: str,
        client: "AsyncElasticsearch",
        *,
        op_type: "ElasticOpType" = "index",
        create_index: bool = True,
        **kwargs: t.Any,
    ) -> int:
        """
        Converts the chat list to Elasticsearch format and indexes it.

        See [rigging.data.chats_to_elastic][] for more information.

        Returns:
            The number of chats indexed.
        """
        from rigging.data import chats_to_elastic

        return await chats_to_elastic(
            self,
            index,
            client,
            op_type=op_type,
            create_index=create_index,
            **kwargs,
        )

    def to_json(self) -> list[dict[str, t.Any]]:
        """
        Helper to convert the chat list to a list of dictionaries.
        """
        return [chat.model_dump() for chat in self]

    def to_openai(self) -> list[list[dict[str, t.Any]]]:
        """
        Converts the chat list to a list of OpenAI-compatible JSON format.

        See Message.to_openai() for more information.

        Returns:
            The serialized chat list.
        """
        return [chat.to_openai() for chat in self]

    async def to_tokens(
        self,
        tokenizer: str | Tokenizer,
        transform: str | Transform | None = None,
    ) -> list[TokenizedChat]:
        """
        Converts the chat list to a list of tokenized chats.

        Args:
            tokenizer: The tokenizer to use for tokenization. Can be a string identifier or a Tokenizer instance.
            transform: An optional transform to apply to each chat before tokenization. Can be a well-known transform
                identifier or a Transform instance.

        Returns:
            A list of tokenized chats.
        """
        # Resolve the tokenizer first so we don't duplicate effort
        if isinstance(tokenizer, str):
            tokenizer = get_tokenizer(tokenizer)

        return await asyncio.gather(
            *(chat.to_tokens(tokenizer, transform) for chat in self),
        )


# Callbacks


@runtime_checkable
class _ThenChatCallback(t.Protocol):
    def __call__(
        self,
        chat: Chat,
        /,
    ) -> t.Awaitable[Chat | None] | Chat | None: ...


@runtime_checkable
class _ThenChatStepCallback(t.Protocol):
    def __call__(
        self,
        chat: Chat,
        /,
    ) -> "PipelineStepGenerator | PipelineStepContextManager | t.Awaitable[PipelineStepGenerator | PipelineStepContextManager | None]": ...


ThenChatCallback = _ThenChatCallback | _ThenChatStepCallback
"""
Passed a finalized chat to process and can return a new chat to replace it.
"""


@runtime_checkable
class _MapChatCallback(t.Protocol):
    def __call__(
        self,
        chats: list[Chat],
        /,
    ) -> t.Awaitable[list[Chat]] | list[Chat]: ...


@runtime_checkable
class _MapChatStepCallback(t.Protocol):
    def __call__(
        self,
        chats: list[Chat],
        /,
    ) -> "PipelineStepGenerator | PipelineStepContextManager | t.Awaitable[PipelineStepGenerator | PipelineStepContextManager]": ...


MapChatCallback = _MapChatCallback | _MapChatStepCallback
"""
Passed a finalized chats to process. Can replace chats in the pipeline
by returning any number of new or existing chats.
"""


@runtime_checkable
class WatchChatCallback(t.Protocol):
    def __call__(self, chats: list[Chat], /) -> t.Awaitable[None]:
        """
        Passed any created chat objects for monitoring/logging.
        """
        ...


# Pipeline Step

g_pipeline_step_ctx: "ContextVar[ChatList | None]" = ContextVar(
    "g_pipeline_step_ctx",
    default=None,
)

PipelineState = t.Literal["generated", "callback", "final"]


@dataclass
class PipelineStep:
    """
    An intermediate step during pipeline generation.
    """

    state: PipelineState
    """The current state of the generation."""
    chats: ChatList
    """The chats associated with this step."""
    pipeline: "ChatPipeline"
    """The pipeline associated with this step."""
    parent: "PipelineStep | None" = None
    """The parent step of pipelines which are running above this step."""
    callback: "ThenChatCallback | MapChatCallback | None" = None
    """The associated callback function if state is 'callback'."""

    def copy(self) -> "PipelineStep":
        """
        Clone the current step.
        """
        return PipelineStep(
            state=self.state,
            chats=self.chats,
            pipeline=self.pipeline,
            parent=self.parent.copy() if self.parent else None,
            callback=self.callback,
        )

    def with_parent(self, parent: "PipelineStep") -> "PipelineStep":
        """
        Clone the current step and append a parent to it's hierarchy.
        """
        if self is parent:
            raise RuntimeError("Cannot set parent to self")

        copy = self.copy()

        if copy.parent is None:
            copy.parent = parent
            return copy

        next_parent = copy.parent
        while next_parent is not None:
            if next_parent is next_parent.parent:
                raise RuntimeError("Parent is self-referential")

            if next_parent.parent is None:
                next_parent.parent = parent
                return copy

            next_parent = next_parent.parent

        raise RuntimeError("Unable to set parent step")

    def __str__(self) -> str:
        callback_name = get_callable_name(self.callback) if self.callback else "None"
        self_str = f"PipelineStep(pipeline={id(self.pipeline)}, state={self.state}, chats={len(self.chats)}, callback={callback_name})"
        if self.parent is not None:
            self_str += f" <- {self.parent!s}"
        return self_str

    @property
    def depth(self) -> int:
        """
        Returns the depth of this pipeline step in the pipeline tree.

        This is useful for setting constraints on recursion depth.
        """
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth


PipelineStepGenerator = t.AsyncGenerator[PipelineStep, None]
PipelineStepContextManager = t.AsyncContextManager[PipelineStepGenerator]

# Tracing wrappers


def _wrap_watch_callback(callback: WatchChatCallback) -> WatchChatCallback:
    import dreadnode as dn

    callback_name = get_callable_name(callback)

    async def wrapped_callback(chats: list[Chat]) -> None:
        with dn.span(
            name=callback_name,
            attributes={"rigging.type": "chat_pipeline.watch_callback"},
        ):
            await callback(chats)

    return wrapped_callback


# Pipeline


class ChatPipeline:
    """
    Pipeline to manipulate and produce chats.
    """

    def __init__(
        self,
        generator: Generator,
        messages: t.Sequence[Message],
        *,
        params: GenerateParams | None = None,
        watch_callbacks: list[WatchChatCallback] | None = None,
    ):
        self.generator: Generator = generator
        """The generator object responsible for generating the chat."""
        self.chat: Chat = Chat(messages)
        """The chat object representing the conversation."""
        self.params = params
        """The parameters for generating messages."""
        self.metadata: dict[str, t.Any] = {}
        """Additional metadata associated with the chat."""
        self.errors_to_catch: set[type[Exception]] = {MaxDepthError, ValidationError}
        """The list of exceptions to catch during generation if you are including or skipping failures."""
        self.errors_to_exclude: set[type[Exception]] = set()
        """The list of exceptions to exclude from the catch list."""
        self.on_failed: FailMode = "raise"
        """How to handle failures in the pipeline unless overridden in calls."""
        self.caching: CacheMode | None = None
        """How to handle cache_control entries on messages."""
        self.task_name: str = f"Chat with {generator.to_identifier(short=True)}"
        """The name of the pipeline task, used for logging and debugging."""
        self.scorers: list[Scorer[Chat]] = []
        """List of dreadnode scorers to evaluate the generated chat upon completion."""

        self.until_parsed_as_types: list[type[Model]] = []
        self.until_parsed_as_catch: bool = True
        self.tools: list[Tool[..., t.Any]] = []
        self.tool_mode: ToolMode = "auto"
        self.inject_tool_prompt = True
        self.add_tool_stop_token = True
        self.then_callbacks: list[
            # callback, max_depth, as_task
            tuple[ThenChatCallback, int, bool]
        ] = []
        self.map_callbacks: list[
            # callback, max_depth, as_task
            tuple[MapChatCallback, int, bool]
        ] = []
        self.watch_callbacks: list[WatchChatCallback] = watch_callbacks or []
        self.transforms: list[Transform] = []

    def __len__(self) -> int:
        return len(self.chat)

    def with_(
        self,
        params: GenerateParams | None = None,
        **kwargs: t.Any,
    ) -> "ChatPipeline":
        """
        Assign specific generation parameter overloads for this chat.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            params: The parameters to set for the chat.
            **kwargs: An alternative way to pass parameters as keyword arguments.

        Returns:
            The updated pipeline.
        """
        if params is None:
            params = GenerateParams(**kwargs)

        if self.params is not None:
            new = self.clone()
            new.params = self.params.merge_with(params)
            return new

        self.params = params
        return self

    def catch(
        self,
        *errors: type[Exception],
        on_failed: FailMode | None = None,
        exclude: list[type[Exception]] | None = None,
    ) -> "ChatPipeline":
        """
        Adds exceptions to catch during generation when including or skipping failures.

        Args:
            *errors: The exception types to catch.
            on_failed: How to handle failures in the pipeline unless overridden in calls.

        Returns:
            The updated pipeline.
        """
        self.errors_to_catch.update(errors)
        self.errors_to_exclude.update(exclude or [])
        self.on_failed = on_failed or self.on_failed
        return self

    def watch(
        self,
        *callbacks: WatchChatCallback,
        allow_duplicates: bool = False,
    ) -> "ChatPipeline":
        """
        Registers a callback to monitor any chats produced.

        Args:
            *callbacks: The callback functions to be executed.
            allow_duplicates: Whether to allow (seemingly) duplicate callbacks to be added.

        Returns:
            The updated pipeline.

        Example:
            ```
            async def log(chats: list[Chat]) -> None:
                ...

            await pipeline.watch(log).run()
            ```
        """
        for callback in callbacks:
            if not allow_duplicates and callback in self.watch_callbacks:
                raise ValueError(
                    f"Callback '{get_callable_name(callback)}' is already registered.",
                )

        self.watch_callbacks.extend(callbacks)
        return self

    def add(
        self,
        messages: t.Sequence[MessageDict]
        | t.Sequence[Message]
        | MessageDict
        | Message
        | Content
        | str,
        *,
        merge_strategy: t.Literal["only-user-role", "all", "none"] = "none",
    ) -> "ChatPipeline":
        """
        Appends new message(s) to the internal chat before generation.

        Note:
            `merge_strategy` configures behavior when the last message in the chat
            is the same role as the first incoming message. This is useful for appending content
            automatically to avoid duplicate messages of the same role - which may cause issues with
            some inference models. In version >=3.0, the default has been set to `none` to avoid
            unexpected behavior.

        Args:
            messages: The messages to be added to the chat. It can be a single message or a sequence of messages.
            merge_strategy: The strategy to use when merging message content when the roles match.
                - "only-user-role": Only merge content of the last existing message and the first incoming message if the last message role is "user".
                - "all": Merge content of the last existing message and the first incoming message if their roles match.
                - "none": Keep messages independent and do not merge any content.

        Returns:
            The updated pipeline.
        """
        message_list = Message.fit_as_list(messages)

        if (
            merge_strategy != "none"
            and self.chat.all
            and self.chat.all[-1].role == message_list[0].role
            and (
                merge_strategy == "all"
                or (merge_strategy == "only-user-role" and self.chat.all[-1].role == "user")
            )
        ):
            self.chat.all[-1].content_parts += message_list[0].content_parts
            message_list = message_list[1:]

        self.chat.generated += message_list
        return self

    def fork(
        self,
        messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str,
    ) -> "ChatPipeline":
        """
        Creates a new instance of `ChatPipeline` by forking the current chat and adding the specified messages.

        This is a convenience method for calling `clone().add(messages)`.

        Args:
            messages: A sequence of messages or a single message to be added to the new chat.

        Returns:
            The cloned pipeline with messages added.
        """
        return self.clone().add(messages)

    def clone(
        self,
        *,
        only_messages: bool = False,
        chat: Chat | None = None,
        callbacks: bool | t.Sequence[MapChatCallback | ThenChatCallback | Transform] = True,
    ) -> "ChatPipeline":
        """
        Creates a clone of the current `ChatPipeline` instance.

        Args:
            only_messages: If True, only the messages will be cloned.
                If False (default), the entire `ChatPipeline` instance will be cloned
                including until callbacks, types, tools, metadata, etc.
            chat: An optional chat object clone for use in the new pipeline, otherwise the current
                internal chat object will be cloned.
            callbacks: If True (default), all callbacks will be cloned. If False, no callbacks will be cloned.
                Otherwise provide a sequence of callbacks which should be maintained in the new pipeline.

        Returns:
            The cloned ChatPipeline.
        """
        new = ChatPipeline(
            self.generator,
            [],
            params=self.params.model_copy() if self.params is not None else None,
            watch_callbacks=self.watch_callbacks,
        )
        new.chat = (chat or self.chat).clone()
        if not only_messages:
            new.until_parsed_as_types = self.until_parsed_as_types.copy()
            new.tools = self.tools.copy()
            new.tool_mode = self.tool_mode
            new.metadata = deepcopy(self.metadata)
            new.on_failed = self.on_failed
            new.errors_to_catch = self.errors_to_catch.copy()
            new.errors_to_exclude = self.errors_to_exclude.copy()
            new.caching = self.caching
            new.task_name = self.task_name
            new.scorers = self.scorers.copy()
            new.transforms = self.transforms.copy()

            new.watch_callbacks = self.watch_callbacks.copy()

            # Check if any of our callbacks are bound methods to a Chatpipeline.
            # If so, we should rebind them to `self` to ensure they work correctly
            # and aren't operating with old state.

            if callbacks is False:
                return new

            new.then_callbacks = [
                (callback, max_depth, as_task)
                if not hasattr(callback, "__self__")
                or not isinstance(callback.__self__, ChatPipeline)
                else (types.MethodType(callback.__func__, new), max_depth, as_task)  # type: ignore [union-attr]
                for callback, max_depth, as_task in self.then_callbacks.copy()
            ]
            new.map_callbacks = [
                (callback, max_depth, as_task)
                if not hasattr(callback, "__self__")
                or not isinstance(callback.__self__, ChatPipeline)
                else (types.MethodType(callback.__func__, new), max_depth, as_task)  # type: ignore [union-attr]
                for callback, max_depth, as_task in self.map_callbacks.copy()
            ]
            new.transforms = [
                callback
                if not hasattr(callback, "__self__")
                or not isinstance(callback.__self__, ChatPipeline)
                else types.MethodType(callback.__func__, new)  # type: ignore [attr-defined]
                for callback in self.transforms
            ]

            if not isinstance(callbacks, bool):
                new.then_callbacks = [
                    (callback, max_depth, as_task)
                    for callback, max_depth, as_task in self.then_callbacks
                    if callback in callbacks
                ]
                new.map_callbacks = [
                    (callback, max_depth, as_task)
                    for callback, max_depth, as_task in self.map_callbacks
                    if callback in callbacks
                ]
                new.transforms = [callback for callback in self.transforms if callback in callbacks]

        return new

    def meta(self, **kwargs: t.Any) -> "ChatPipeline":
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated pipeline.
        """
        self.metadata.update(kwargs)
        return self

    def name(self, name: str) -> "ChatPipeline":
        """
        Sets the name of the pipeline.

        Args:
            name: The name to set for the pipeline.

        Returns:
            The updated pipeline.
        """
        self.task_name = name
        return self

    def then(
        self,
        *callbacks: ThenChatCallback,
        max_depth: int = DEFAULT_MAX_DEPTH,
        allow_duplicates: bool = False,
        as_task: bool = True,
    ) -> "ChatPipeline":
        """
        Registers one or many callbacks to be executed after the generation process completes.

        Note:
            Returning a Chat object from the callback will replace that chat
            for the remainder of the callbacks and the final return value
            from the pipeline.

        Args:
            callbacks: The callback functions to be added.
            max_depth: The maximum depth to allow recursive pipeline calls during this callback.
            allow_duplicates: Whether to allow (seemingly) duplicate callbacks to be added.
            as_task: Whether to create a task for this callback.

        Returns:
            The updated pipeline.

        Example:
            ```
            async def process(chat: Chat) -> Chat | None:
                ...

            await pipeline.then(process).run()
            ```
        """
        for callback in callbacks:
            if allow_duplicates:
                continue

            if callback in [c[0] for c in self.then_callbacks]:
                raise ValueError(
                    f"Callback '{get_callable_name(callback)}' is already registered.",
                )

        self.then_callbacks.extend([(callback, max_depth, as_task) for callback in callbacks])
        return self

    def map(
        self,
        *callbacks: MapChatCallback,
        max_depth: int = DEFAULT_MAX_DEPTH,
        allow_duplicates: bool = False,
        as_task: bool = True,
    ) -> "ChatPipeline":
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            You must return a list of Chat objects from the callback which will
            represent the state of chats for the remainder of the callbacks and
            the final return value from the pipeline.

        Args:
            callbacks: The callback function to be executed.
            max_depth: The maximum depth to allow recursive pipeline calls during this callback.
            allow_duplicates: Whether to allow (seemingly) duplicate callbacks to be added.
            as_task: Whether to create a task for this callback.

        Returns:
            The updated pipeline.

        Example:
            ```
            async def process(chats: list[Chat]) -> list[Chat]:
                ...

            await pipeline.map(process).run()
            ```
        """
        for callback in callbacks:
            if allow_duplicates:
                continue

            if callback in [c[0] for c in self.map_callbacks]:
                raise ValueError(
                    f"Callback '{get_callable_name(callback)}' is already registered.",
                )

        self.map_callbacks.extend([(callback, max_depth, as_task) for callback in callbacks])
        return self

    def transform(
        self,
        *callbacks: Transform,
        allow_duplicates: bool = False,
    ) -> "ChatPipeline":
        """
        Registers a callback to be executed just before generation, and optionally return
        a callback to executed just after generation.

        Transform callbacks are low-level callbacks used to modify messages and parameters based
        on pipeline state and conditions. They are not emitted as pipeline steps and all other
        callbacks (watch, then, map) occur after all transform callbacks have been executed.

        Args:
            callbacks: The callback function to be executed.
            allow_duplicates: Whether to allow (seemingly) duplicate callbacks to be added.

        Returns:
            The updated pipeline.

        Example:
            ```
            async def transform(
                messages: list[Message],
                params: GenerateParams
            ) -> tuple[list[Message], GenerateParams, PostTransformChatCallback | None]:

                async def post_transform(chat: Chat) -> Chat | None:
                    ...

                return messages, params, post_transform

            await pipeline.transform(transform).run()
            ```
        """
        for callback in callbacks:
            if not allow_duplicates and callback in self.transforms:
                raise ValueError(
                    f"Callback '{get_callable_name(callback)}' is already registered.",
                )

        self.transforms.extend(callbacks)
        return self

    def apply(self, **kwargs: str) -> "ChatPipeline":
        """
        Clones this chat pipeline and calls [rigging.chat.Chat.apply][] with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be applied to the chat.

        Returns:
            A new pipeline with updated chat.
        """
        new = self.clone()
        new.chat.apply(**kwargs)
        return new

    def apply_to_all(self, **kwargs: str) -> "ChatPipeline":
        """
        Clones this chat pipeline and calls [rigging.chat.Chat.apply_to_all][] with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be applied to the chat.

        Returns:
            A new pipeline with updated chat.
        """
        new = self.clone()
        new.chat.apply_to_all(**kwargs)
        return new

    def cache(
        self,
        mode: CacheMode | None | t.Literal[False] = "latest",
    ) -> "ChatPipeline":
        """
        Sets the caching mode for the pipeline.

        Args:
            mode: The caching mode to use. Defaults to "latest".

        Returns:
            The updated pipeline.
        """
        if mode is False:
            mode = None
        self.caching = mode
        return self

    def wrap(self, func: t.Callable[[CallableT], CallableT]) -> "ChatPipeline":
        """
        Helper for [rigging.generator.base.Generator.wrap][].

        Args:
            func: The function to wrap the calls with.

        Returns:
            The current pipeline.
        """
        self.generator = self.generator.wrap(func)
        return self

    def using(
        self,
        *tools: Tool[..., t.Any]
        | t.Callable[..., t.Any]
        | t.Sequence[Tool[..., t.Any] | t.Callable[..., t.Any]],
        mode: ToolMode | None = None,
        choice: ToolChoice | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        add_stop_token: bool | None = None,
        catch: bool | t.Iterable[type[Exception]] | None = None,
    ) -> "ChatPipeline":
        """
        Adds a tool or a sequence of tools to participate in the generation process.

        Note:
            By default, the tool mode is set to "auto" which will attempt to use
            api function calling if available, otherwise it will fallback to json arguments
            wrapped in xml tags.

        Args:
            *tools: The tools to be added to the pipeline, these can be either:
                - A Tool instance (e.g., Tool.from_callable() or @tool decorator).
                - A callable function that can be converted to a Tool.
                - An instance of a class with @tool_method decorated methods.
                - A sequence of any of the above.
            mode: The tool calling mode to use (e.g., "xml", "json-with-tag", "json-in-xml", "api") - default is "auto".
            choice: The API tool choice to use. This is only relevant when using the "api" tool mode.
            max_depth: The maximum depth for recursive tool calls (this is shared between all tools).
            add_stop_token: When using "xml" tool transforms, use stop tokens to
                immediately process a tool call when observed.
            catch: Override the catch setting for all incoming tools, or leave `None` to use the tool's default.
                By default, catches `json.JSONDecodeError` and `ValidationError`. Set to `{}` to let the pipeline
                handle all tool exceptions.

        Returns:
            The updated pipeline.

        Example:
            ```
            async def get_weather(city: Annotated[str, "The city name to get weather for"]) -> str:
                "Get the weather"
                city = city.replace(" ", "+")
                return requests.get(f"http://wttr.in/{city}?format=2").text.strip()

            chat = (
                await rg.get_generator("openai/gpt-4o-mini")
                .chat("What's the weather in london?")
                .using(get_weather)
                .run()
            )
            ```
        """
        if len(tools) == 0:
            return self

        _tools: list[Tool[..., t.Any]] = []
        for tool in flatten_list(list(tools)):
            interior_tools = [
                val
                for _, val in inspect.getmembers(
                    tool,
                    predicate=lambda x: isinstance(x, Tool),
                )
            ]
            if interior_tools:
                _tools.extend(interior_tools)
            elif not isinstance(tool, Tool):
                _tools.append(Tool.from_callable(tool))
            else:
                _tools.append(tool)

        if catch is not None:
            _tools = [tool.with_(catch=catch) for tool in _tools]

        existing_names = {tool.name for tool in self.tools}
        new_names = {tool.name for tool in _tools}
        for name in existing_names & new_names:
            warnings.warn(
                f"Overwriting existing tool '{name}'.",
                PipelineWarning,
                stacklevel=2,
            )

        self.tools = [tool for tool in self.tools if tool.name not in new_names] + _tools

        self.then_callbacks = [
            (callback, max_depth, as_task)
            for callback, max_depth, as_task in self.then_callbacks
            if callback != self._then_tools  # Always remove to update max_depth
        ]
        self.then_callbacks.insert(
            0,  # make sure this is first
            (self._then_tools, max_depth, False),
        )

        if mode is not None:
            self.tool_mode = mode

        if add_stop_token is not None:
            self.add_tool_stop_token = add_stop_token

        # We would install the transform here for native tool calls,
        # but we want to do it lazily because it's a closure that requires
        # the current state of the pipeline. Having to re-construct it during
        # cloning would be a pain.

        self.params = self.params or GenerateParams()
        self.params.tools = [tool.api_definition for tool in self.tools]
        if choice is not None:
            self.params.tool_choice = choice

        return self

    def score(
        self,
        *scorers: "Scorer[Chat] | ScorerCallable[Chat]",
        filter: "ChatFilterMode | ChatFilterFunction" = "last",
    ) -> "ChatPipeline":
        """
        Adds one or more scorers to the pipeline to evaluate the generated chat upon completion.

        Args:
            *scorers: The scorer or scorers to be added. These can be either:
                - A dreadnode.Scorer instance.
                - A callable function that can be converted to a dreadnode.Scorer.
            filter: The strategy for filtering which messages to include:
                - "all": Use all messages in the chat.
                - "last": Use only the last message.
                - "first": Use only the first message.
                - "user": Use only user messages.
                - "assistant": Use only assistant messages.
                - "last_user": Use only the last user message.
                - "last_assistant": Use only the last assistant message.
                - A callable that takes a list of `Message` objects and returns a filtered list.

        Returns:
            The updated pipeline.
        """
        import dreadnode as dn

        self.scorers.extend(
            [
                dn.scorers.wrap_chat(
                    scorer if isinstance(scorer, dn.Scorer) else dn.Scorer.from_callable(scorer),
                    filter=filter,
                )
                for scorer in scorers
            ]
        )
        return self

    # Internal callbacks for handling tools and parsing

    def until_parsed_as(
        self,
        *types: type[ModelT],
        max_depth: int = DEFAULT_MAX_DEPTH,
        catch: bool | None = None,
        # deprecated
        attempt_recovery: bool | None = None,
        drop_dialog: bool | None = None,
        max_rounds: int | None = None,
    ) -> "ChatPipeline":
        """
        Adds the specified types to the list of types which should successfully parse
        before the generation process completes.

        Args:
            *types: The type or types of models to wait for.
            max_depth: The maximum depth to re-attempt parsing using recursive pipelines (this is shared between all types).
            catch: Whether to catch exceptions and return them as messages automatically, otherwise raise them to the pipeline.
            attempt_recovery: deprecated, recovery is always attempted.
            drop_dialog: deprecated, the full dialog is always returned.
            max_rounds: deprecated, use `max_depth` instead.

        Returns:
            The updated pipeline.
        """
        if attempt_recovery is not None:
            warnings.warn(
                "The 'attempt_recovery' argument is deprecated and has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
        if drop_dialog is not None:
            warnings.warn(
                "The 'drop_dialog' argument is deprecated and has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_rounds is not None:
            warnings.warn(
                "The 'max_rounds' argument is deprecated, use 'max_depth'.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.until_parsed_as_types = list(types)
        self.until_parsed_as_catch = catch or self.until_parsed_as_catch

        max_depth = max_rounds or max_depth
        self.then_callbacks = [
            (callback, max_depth, as_task)
            for callback, max_depth, as_task in self.then_callbacks
            if callback != self._then_parse
        ]
        self.then_callbacks.append((self._then_parse, max_depth, False))

        return self

    async def _then_tools(self, chat: Chat) -> PipelineStepContextManager | None:
        if not chat.last.tool_calls:
            return None

        next_pipeline = self.clone(chat=chat, callbacks=[self._then_tools])

        async def _process_tool_call(tool_call: ToolCall) -> bool:
            if (tool := next((t for t in self.tools if t.name == tool_call.name), None)) is None:
                next_pipeline.add(
                    Message.from_model(
                        SystemErrorModel(
                            content=f"Tool '{tool_call.name}' not found.",
                        ),
                    ),
                )
                return False

            message, stop = await tool.handle_tool_call(tool_call)
            next_pipeline.add(message)
            return stop

        # Process all tool calls in parallel

        stop = max(
            await asyncio.gather(
                *[_process_tool_call(tool_call) for tool_call in chat.last.tool_calls],
            ),
        )

        # Need to prevent infinite loops and treat tool_choice like
        # an ephemeral setting which resets after the first tool call.

        if self.tool_mode == "api" and next_pipeline.params:
            next_pipeline.params.tool_choice = None

        if stop:
            # TODO(nick): Type hints here stop us from mixing step generators and basic chat returns.
            return next_pipeline.chat  # type: ignore [return-value]

        return next_pipeline.step()

    async def _then_parse(self, chat: Chat) -> PipelineStepContextManager | None:
        import dreadnode as dn

        if chat.error:  # If we have an error, we should not attempt to parse.
            return None

        next_pipeline = self.clone(chat=chat)

        type_names = " | ".join(
            sorted(until_type.__name__ for until_type in self.until_parsed_as_types)
        )
        task_name = f"parse - {type_names}"

        try:
            with dn.task_span(task_name, attributes={"rigging.type": "chat_pipeline.parse"}):
                dn.log_input("message", chat.last)
                parsed = chat.last.parse_many(*self.until_parsed_as_types)
                dn.log_output("parsed", parsed)
        except Exception as e:
            if not self.until_parsed_as_catch:
                raise

            error_model_cls = (
                ValidationErrorModel if isinstance(e, ValidationError) else SystemErrorModel
            )
            next_pipeline.add(
                Message.from_model(
                    error_model_cls(content=str(e)),
                    suffix="Rewrite your entire message with all of the required xml elements.",
                ),
            )
        else:  # parsed successfully
            return None

        return next_pipeline.step()

    # Run helper methods

    def _fit_params(
        self,
        count: int,
        params: t.Sequence[GenerateParams | None] | None = None,
    ) -> list[GenerateParams]:
        params = [None] * count if params is None else list(params)
        if len(params) != count:
            raise ValueError(f"The number of params must be {count}")
        if self.params is not None:
            params = [self.params.merge_with(p) for p in params]
        return [(p or GenerateParams()) for p in params]

    @dataclass
    class CallbackState:
        chat: Chat
        ready_event: asyncio.Event
        continue_event: asyncio.Event
        step: PipelineStep | None = None
        completed: bool = False

    async def _process_then_callback(
        self,
        callback: ThenChatCallback,
        state: CallbackState,
    ) -> None:
        async def complete() -> None:
            state.completed = True
            state.ready_event.set()

        async with contextlib.AsyncExitStack() as exit_stack:
            exit_stack.push_async_callback(complete)

            result = callback(state.chat)
            if inspect.isawaitable(result):
                result = await result

            if result is None or isinstance(result, Chat):
                state.chat = result or state.chat
                return

            if isinstance(result, contextlib.AbstractAsyncContextManager):
                result = await exit_stack.enter_async_context(result)

            if not inspect.isasyncgen(result):
                raise TypeError(
                    f"Callback '{get_callable_name(callback)}' must return a Chat, PipelineStepGenerator, or None",
                )

            generator = t.cast(
                "PipelineStepGenerator",
                await exit_stack.enter_async_context(aclosing(result)),
            )
            async for step in generator:
                state.step = step

                state.ready_event.set()
                await state.continue_event.wait()

                state.ready_event.clear()
                state.continue_event.clear()
                state.step = None

                state.chat = step.chats[-1] if step.chats else state.chat

    async def _score_chats(self, chats: list[Chat]) -> None:
        import dreadnode as dn

        if not self.scorers:
            return

        for scorer in self.scorers:
            for metric in await asyncio.gather(
                *[scorer(chat) for chat in chats],
            ):
                dn.log_metric(scorer.name, metric)

    def _raise_if_failed(
        self,
        chats: list[Chat | BaseException] | ChatList,
        on_failed: FailMode | None = None,
    ) -> None:
        for chat in chats:
            error = chat.error if isinstance(chat, Chat) else chat
            if error is not None and (
                on_failed == "raise"
                or not any(isinstance(error, t) for t in self.errors_to_catch)
                or any(isinstance(error, t) for t in self.errors_to_exclude)
            ):
                raise error

    # Run methods

    async def _step(  # noqa: PLR0915, PLR0912
        self,
        messages: list[list[Message]],
        params: list[GenerateParams],
        on_failed: FailMode,
    ) -> PipelineStepGenerator:
        import dreadnode as dn

        chats: ChatList = ChatList([])

        # Some pre-run work

        if self.tool_mode == "auto" and self.tools:
            self.tool_mode = (
                "api" if await self.generator.supports_function_calling() else "json-in-xml"
            )

        # Transform callbacks (pre)

        transforms = self.transforms

        # If we are using tool parsing, add the transform here
        # as we need our latest tool states for XML
        # TODO: We don't need everything this early

        if self.tools:
            match self.tool_mode:
                case "xml":
                    transforms.append(
                        make_tools_to_xml_transform(
                            self.tools,
                            add_tool_stop_token=self.add_tool_stop_token,
                        ),
                    )
                case "json-in-xml":
                    transforms.append(tools_to_json_in_xml_transform)
                case "json-with-tag":
                    transforms.append(tools_to_json_with_tag_transform)
                case "json":
                    transforms.append(tools_to_json_transform)

        post_transforms: list[list[PostTransform | None]] = []
        for i, (_messages, _params) in enumerate(zip(messages, params, strict=True)):
            _post_transforms: list[PostTransform | None] = []
            for transform_callback in transforms:
                _messages, _params, post_transform = await transform_callback(_messages, _params)
                _post_transforms.append(post_transform)

            messages[i] = _messages
            params[i] = _params
            post_transforms.append(_post_transforms)

        # Pass the messages to the generator

        try:
            messages = apply_cache_mode_to_messages(self.caching, messages)

            with dn.span(
                f"generate - {self.generator.to_identifier(short=True)}",
                attributes={"rigging.type": "chat_pipeline.generate"},
            ):
                # dn.log_input("messages", messages)
                # dn.log_input("params", params)
                generated = await self.generator.generate_messages(messages, params)
                # dn.log_output("generated", generated)

        # If we got a total failure here for generation as a whole,
        # we can't distinguish between incoming messages in terms
        # of what caused the error, so so we need to set the error
        # on all of them.

        except Exception as error:  # noqa: BLE001
            chats = ChatList(
                [
                    Chat(
                        messages_,
                        [],
                        generator=self.generator,
                        pipeline=self,
                        metadata=self.metadata,
                        params=params_,
                        failed=True,
                        error=error,
                    )
                    for messages_, params_ in zip(messages, params, strict=True)
                ],
            )

        # Otherwise we can construct individual chats with
        # error states per-generation.

        else:
            chats = ChatList(
                [
                    Chat(
                        messages_,
                        [],
                        generator=self.generator,
                        pipeline=self,
                        metadata=self.metadata,
                        params=params_,
                        failed=True,
                        error=generated_,
                    )
                    if isinstance(generated_, BaseException)
                    else Chat(
                        messages_,
                        [generated_.message],
                        generator=self.generator,
                        pipeline=self,
                        metadata=self.metadata,
                        params=params_,
                        stop_reason=generated_.stop_reason,
                        usage=generated_.usage,
                        extra=generated_.extra,
                    )
                    for messages_, params_, generated_ in zip(
                        messages,
                        params,
                        generated,
                        strict=True,
                    )
                ],
            )

        # Transform callbacks (post)

        for i, (_chat, _post_transforms) in enumerate(zip(chats, post_transforms, strict=True)):
            for post_transform in [transform for transform in _post_transforms if transform]:
                _chat = await post_transform(_chat) or _chat
            chats[i] = _chat

        # Watch callbacks

        await asyncio.gather(
            *[_wrap_watch_callback(callback)(chats) for callback in self.watch_callbacks],
        )

        # Yield what we generated

        current_step = PipelineStep(
            state="generated",
            chats=chats,
            pipeline=self,
        )
        yield current_step

        # Check if we should immediately raise

        self._raise_if_failed(chats, on_failed)

        # Chat cleanup

        if on_failed == "skip":
            chats = ChatList([chat for chat in chats if not chat.failed])

        if len(chats) == 0 or all(chat.failed for chat in chats):
            yield PipelineStep(
                state="final",
                chats=chats,
                pipeline=self,
            )
            return

        # Then callbacks

        for then_callback, max_depth, as_task in self.then_callbacks:
            callback_name = get_callable_name(then_callback, short=True)

            states = [
                self.CallbackState(
                    chat=chat,
                    ready_event=asyncio.Event(),
                    continue_event=asyncio.Event(),
                )
                for chat in chats
            ]

            callback_task = (
                dn.task(
                    name=callback_name,
                    attributes={"rigging.type": "chat_pipeline.then_callback"},
                    log_inputs=True,
                    log_output=True,
                )(then_callback)
                if as_task
                else then_callback
            )

            tasks = [
                asyncio.create_task(self._process_then_callback(callback_task, state))  # type: ignore [arg-type]
                for state in states
            ]

            try:
                while not all(state.completed for state in states):
                    await asyncio.gather(
                        *[state.ready_event.wait() for state in states if not state.completed],
                    )

                    for state, task in zip(states, tasks, strict=True):
                        if task.done() and (exception := task.exception()):
                            state.chat.error = exception
                            state.chat.failed = True

                        elif state.ready_event.is_set() and state.step:
                            step = state.step.with_parent(current_step)

                            if step.depth > max_depth:
                                max_depth_error = MaxDepthError(
                                    max_depth,
                                    step,
                                    callback_name,
                                )
                                if on_failed == "raise":
                                    raise max_depth_error

                                state.chat.error = max_depth_error
                                state.chat.failed = True
                                state.completed = True
                                continue

                            yield step
                            state.continue_event.set()
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            chats = ChatList([state.chat for state in states if state.chat])

            self._raise_if_failed(chats, on_failed)

            current_step = PipelineStep(
                state="callback",
                chats=chats,
                pipeline=self,
                callback=then_callback,
            )
            yield current_step

        # Chat cleanup

        if on_failed == "skip":
            chats = ChatList([chat for chat in chats if not chat.failed])

        if len(chats) == 0 or all(chat.failed for chat in chats):
            yield PipelineStep(
                state="final",
                chats=chats,
                pipeline=self,
            )
            return

        # Map callbacks

        for map_callback, max_depth, as_task in self.map_callbacks:
            callback_name = get_callable_name(map_callback, short=True)

            map_task = (
                dn.task(
                    name=callback_name,
                    attributes={"rigging.type": "chat_pipeline.map_callback"},
                    log_inputs=True,
                    log_output=True,
                )(map_callback)
                if as_task
                else map_callback
            )

            async with contextlib.AsyncExitStack() as exit_stack:
                try:
                    result = map_task(chats)
                    if inspect.isawaitable(result):
                        result = await result
                except Exception as e:  # noqa: BLE001
                    # If the map raised an exception, assign it to all the chats
                    for chat in chats:
                        chat.error = e
                        chat.failed = True

                    self._raise_if_failed(chats, on_failed)
                    continue

                if isinstance(result, contextlib.AbstractAsyncContextManager):
                    result = await exit_stack.enter_async_context(result)

                if inspect.isasyncgen(result):
                    generator = t.cast(
                        "PipelineStepGenerator",
                        await exit_stack.enter_async_context(
                            aclosing(result),
                        ),
                    )
                    async for step in generator:
                        _step = step.with_parent(current_step)
                        if _step.depth > max_depth:
                            max_depth_error = MaxDepthError(
                                max_depth,
                                _step,
                                callback_name,
                            )
                            if on_failed == "raise":
                                raise max_depth_error

                            chats = ChatList(chats)
                            for chat in chats:
                                chat.error = max_depth_error
                                chat.failed = True
                        else:
                            yield _step
                            chats = step.chats

                    chats = step.chats

                elif isinstance(result, list) and all(isinstance(c, Chat) for c in result):
                    chats = ChatList(result)

                current_step = PipelineStep(
                    state="callback",
                    chats=chats,
                    pipeline=self,
                    callback=map_callback,
                )
                yield current_step

        if on_failed == "skip":
            chats = ChatList([chat for chat in chats if not chat.failed])

        yield PipelineStep(
            state="final",
            chats=chats,
            pipeline=self,
        )

    # Single messages

    @asynccontextmanager
    async def step(
        self,
        *,
        on_failed: FailMode | None = None,
    ) -> t.AsyncIterator[PipelineStepGenerator]:
        """
        Step through the generation process for a single message.

        Args:
            on_failed: The behavior when a message fails to generate.
                (this is used as an alternative to allow_failed)

        Yields:
            Pipeline steps.
        """

        if on_failed == "skip":
            raise ValueError(
                "Cannot use 'skip' mode with single message generation (pass allow_failed=True or on_failed='include'/'raise')",
            )

        on_failed = on_failed or self.on_failed

        messages = [self.chat.all]
        params = self._fit_params(1, [self.params])

        async with aclosing(
            self._step(messages, params, on_failed),
        ) as generator:
            yield generator

    async def run(
        self,
        *,
        name: str | None = None,
        on_failed: FailMode | None = None,
        allow_failed: bool = False,
    ) -> Chat:
        """
        Execute the generation process for a single message.

        Args:
            name: The name of the task for logging purposes.
            on_failed: The behavior when a message fails to generate.
            allow_failed: Deprecated, use `on_failed="include"`.

        Returns:
            The generated Chat.
        """
        import dreadnode as dn

        if allow_failed:
            warnings.warn(
                "The 'allow_failed' argument is deprecated, use 'on_failed=\"include\"'.",
                DeprecationWarning,
                stacklevel=2,
            )

        if on_failed is None:
            on_failed = "include" if allow_failed else self.on_failed

        if on_failed == "skip":
            raise ValueError(
                "Cannot use 'skip' mode with single message generation (pass allow_failed=True or on_failed='include'/'raise')",
            )

        messages = [self.chat.all]
        params = self._fit_params(1, [self.params])

        last: PipelineStep | None = None
        with dn.task_span(
            name or self.task_name,
            label=name or f"pipeline_{self.task_name}",
            tags=["rigging/pipeline"],
            attributes={"rigging.type": "chat_pipeline.run"},
        ) as task:
            dn.log_inputs(
                messages=messages[0],
                params=params[0],
                generator_id=self.generator.to_identifier(),
            )

            try:
                async with aclosing(
                    self._step(messages, params, on_failed),
                ) as steps:
                    async for step in steps:
                        last = step
            finally:
                if last is not None and last.chats:
                    dn.log_output("chat", last.chats[-1])
                    await self._score_chats(last.chats)
                    # TODO: Remove once Strikes UI is ported
                    task.set_attribute("chats", last.chats)

            if last is None or last.state != "final":
                raise RuntimeError("The pipeline did not complete successfully")

            if not last.chats:
                raise RuntimeError("The pipeline process did not produce any chats")

        return last.chats[-1]

    __call__ = run

    # Many messages

    @asynccontextmanager
    async def step_many(
        self,
        count: int,
        *,
        params: t.Sequence[GenerateParams | None] | None = None,
        on_failed: FailMode | None = None,
    ) -> t.AsyncIterator[PipelineStepGenerator]:
        """
        Step through the generation process in parallel over the same input.

        Args:
            count: The number of parallel generations.
            params: A sequence of parameters to be used for each execution.
            on_failed: The behavior when a message fails to generate.

        Yields:
            Pipeline steps.
        """
        on_failed = on_failed or self.on_failed

        messages = [self.chat.all] * count
        params = self._fit_params(count, params)

        async with aclosing(
            self._step(messages, params, on_failed),
        ) as generator:
            yield generator

    async def run_many(
        self,
        count: int,
        *,
        params: t.Sequence[GenerateParams | None] | None = None,
        name: str | None = None,
        mode: t.Literal["merged", "parallel"] = "parallel",
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process in parallel over the same input.

        Args:
            count: The number of times to execute the generation process.
            params: A sequence of parameters to be used for each execution.
            name: The name of the task for logging purposes.
            mode: The mode of execution, either "merged" or "parallel".
                - In "merged" mode, a single pipeline manages all generation simultaneously
                - In "parallel" mode, independent pipelines are created for each generation
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generated Chats.
        """
        import dreadnode as dn

        if count < 1:
            raise ValueError("Count must be greater than 0")

        on_failed = on_failed or self.on_failed

        messages = [self.chat.all] * count
        params = self._fit_params(count, params)

        last: PipelineStep | None = None
        with dn.task_span(
            name or f"{self.task_name} (x{count})",
            label=name or f"pipeline_many_{self.task_name}",
            tags=["rigging/pipeline"],
            attributes={"rigging.type": "chat_pipeline.run_many"},
        ) as task:
            dn.log_inputs(
                count=count,
                messages=messages[0],
                params=params[0],
                generator_id=self.generator.to_identifier(),
            )

            if mode == "merged":
                try:
                    async with aclosing(
                        self._step(messages, params, on_failed),
                    ) as steps:
                        async for step in steps:
                            last = step
                finally:
                    if last is not None:
                        dn.log_output("chats", last.chats)
                        await self._score_chats(last.chats)
                        # TODO: Remove once Strikes UI is ported
                        task.set_attribute("chats", last.chats)

                if last is None or last.state != "final":
                    raise RuntimeError("The pipeline did not complete successfully")

                return last.chats

            if mode == "parallel":
                tasks = [asyncio.create_task(self.run(on_failed="include")) for _ in range(count)]
                chats_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

                self._raise_if_failed(chats_or_errors, on_failed)

                chats = [
                    chat
                    for chat in chats_or_errors
                    if isinstance(chat, Chat) and (on_failed != "skip" or not chat.failed)
                ]

                dn.log_output("chats", chats)
                # TODO: Remove once Strikes UI is ported
                task.set_attribute("chats", chats)

                return ChatList(chats)

        raise ValueError(
            f"Invalid mode '{mode}', expected 'merged' or 'parallel'",
        )

    # Batch messages

    def _fit_batch_args(
        self,
        many: t.Sequence[t.Sequence[Message]]
        | t.Sequence[Message]
        | t.Sequence[MessageDict]
        | t.Sequence[str]
        | MessageDict
        | str,
        params: t.Sequence[GenerateParams | None] | None = None,
    ) -> tuple[int, list[list[Message]], list[GenerateParams]]:
        # Get the maximum of either incoming messages or params

        count = max(len(many), len(params) if params is not None else 0)

        # If we have less messages than params, we need to either:
        # 1. Error because we have >1 messages that we can't reasonably
        #    zip with our parameters of a different length
        # 2. Duplicate a single message we have len(params) times as the
        #    user is just batching only over parameters

        messages = [[*self.chat.all, *Message.fit_as_list(m)] for m in many]
        if len(messages) < count:
            if len(messages) != 1:
                raise ValueError(
                    f"Can't fit {len(messages)} messages to {count} params",
                )
            messages = messages * count

        params = self._fit_params(count, params)

        return count, messages, params

    @asynccontextmanager
    async def step_batch(
        self,
        many: t.Sequence[t.Sequence[Message]]
        | t.Sequence[Message]
        | t.Sequence[MessageDict]
        | t.Sequence[str]
        | MessageDict
        | str,
        params: t.Sequence[GenerateParams | None] | None = None,
        *,
        on_failed: FailMode | None = None,
    ) -> t.AsyncIterator[PipelineStepGenerator]:
        """
        Step through the generation process over multiple inputs.

        Note:
            Anything already in this chat pipeline will be prepended to the input messages.

        Args:
            many: A sequence of sequences of messages to be generated.
            params: A sequence of parameters to be used for each set of messages.
            on_failed: The behavior when a message fails to generate.

        Yields:
            Pipeline steps.
        """
        on_failed = on_failed or self.on_failed
        _, messages, params = self._fit_batch_args(many, params)

        async with aclosing(
            self._step(messages, params, on_failed),
        ) as generator:
            yield generator

    async def run_batch(
        self,
        many: t.Sequence[t.Sequence[Message]]
        | t.Sequence[Message]
        | t.Sequence[MessageDict]
        | t.Sequence[str]
        | MessageDict
        | str,
        params: t.Sequence[GenerateParams | None] | None = None,
        *,
        name: str | None = None,
        mode: t.Literal["merged", "parallel"] = "parallel",
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process over multiple input messages.

        Note:
            Anything already in this chat pipeline will be prepended to the input messages.

        Args:
            many: A sequence of sequences of messages to be generated.
            params: A sequence of parameters to be used for each set of messages.
            name: The name of the task for logging purposes.
            mode: The mode of execution, either "merged" or "parallel".
                - In "merged" mode, a single pipeline manages all generation simultaneously
                - In "parallel" mode, independent pipelines are created for each generation
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """
        import dreadnode as dn

        on_failed = on_failed or self.on_failed
        count, messages, params = self._fit_batch_args(many, params)

        last: PipelineStep | None = None
        with dn.task_span(
            name or f"{self.task_name} (batch x{count})",
            label=name or f"pipeline_batch_{self.task_name}",
            tags=["rigging/pipeline"],
            attributes={"rigging.type": "chat_pipeline.run_batch"},
        ) as task:
            dn.log_inputs(
                count=count,
                messages=messages,
                params=params,
                generator_id=self.generator.to_identifier(),
            )

            if mode == "merged":
                try:
                    async with aclosing(
                        self._step(messages, params, on_failed),
                    ) as steps:
                        async for step in steps:
                            last = step
                finally:
                    if last is not None:
                        dn.log_output("chats", last.chats)
                        await self._score_chats(last.chats)
                        # TODO: Remove once Strikes UI is ported
                        task.set_attribute("chats", last.chats)

                if last is None or last.state != "final":
                    raise RuntimeError("The pipeline did not complete successfully")

                return last.chats

            if mode == "parallel":
                tasks = [
                    asyncio.create_task(
                        self.clone().add(_messages).with_(_params).run(on_failed="include")
                    )
                    for _messages, _params in zip(messages, params, strict=True)
                ]
                chats_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

                self._raise_if_failed(chats_or_errors, on_failed)

                chats = [
                    chat
                    for chat in chats_or_errors
                    if isinstance(chat, Chat) and (on_failed != "skip" or not chat.failed)
                ]

                dn.log_output("chats", chats)
                # TODO: Remove once Strikes UI is ported
                task.set_attribute("chats", chats)

                return ChatList(chats)

        raise ValueError(
            f"Invalid mode '{mode}', expected 'merged' or 'separate'",
        )

    # Generator iteration

    async def run_over(
        self,
        *generators: Generator | str,
        include_original: bool = True,
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process across multiple generators.

        For each generator, this pipeline is cloned and the generator is replaced
        before the run call. All callbacks and parameters are preserved.

        Args:
            *generators: A sequence of generators to be used for the generation process.
            include_original: Whether to include the original generator in the list of runs.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """
        import dreadnode as dn

        on_failed = on_failed or self.on_failed

        _generators: list[Generator] = [
            g if isinstance(g, Generator) else get_generator(g) for g in generators
        ]
        if include_original:
            _generators.append(self.generator)

        coros: list[t.Coroutine[t.Any, t.Any, Chat]] = []
        for generator in _generators:
            sub = self.clone()
            sub.generator = generator
            coros.append(sub.run(allow_failed=(on_failed != "raise")))

        short_generators = [g.to_identifier(short=True) for g in _generators]
        task_name = "iterate - " + ", ".join(short_generators)

        with dn.task_span(
            task_name,
            label="iterate_over",
            attributes={"rigging.type": "chat_pipeline.run_over"},
        ):
            dn.log_input("generators", [g.to_identifier() for g in _generators])
            return ChatList(await asyncio.gather(*coros))

    # Prompt binding

    def prompt(self, func: t.Callable[P, t.Coroutine[None, None, R]]) -> "Prompt[P, R]":
        """
        Decorator to convert a function into a prompt bound to this pipeline.

        See [rigging.prompt.prompt][] for more information.

        Args:
            func: The function to be converted into a prompt.

        Returns:
            The prompt.
        """
        from rigging.prompt import prompt

        return prompt(func, pipeline=self)
