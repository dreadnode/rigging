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

from elasticsearch import AsyncElasticsearch
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    WithJsonSchema,
    computed_field,
)

from rigging.error import MaxDepthError, UnknownToolError
from rigging.generator import GenerateParams, Generator, get_generator
from rigging.generator.base import StopReason, Usage
from rigging.message import Content, Message, MessageDict, Messages
from rigging.model import Model, ModelT, SystemErrorModel, ValidationErrorModel
from rigging.tool.api import ApiToolCall, ApiToolChoice
from rigging.tool.base import Tool, ToolMode
from rigging.tool.native import (
    TOOL_CALLS_TAG,
    JsonInXmlToolCall,
    JsonInXmlToolDefinition,
    XmlToolCall,
    XmlToolDefinition,
    tool_description_prompt_part,
)
from rigging.tracing import Span, tracer
from rigging.util import flatten_list, get_qualified_name

if t.TYPE_CHECKING:
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

CacheMode = t.Literal["latest"]
"""
How to handle cache_control entries on messages.

- latest: Assign cache_control to the latest 2 non-assistant messages in the pipeline before inference.
"""


class Chat(BaseModel):
    """
    A completed chat interaction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    params: GenerateParams | None = Field(None, exclude=True, repr=False)
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
            return self.generator.to_identifier(self.params)
        return None

    def __init__(
        self,
        messages: Messages,
        generated: Messages | None = None,
        generator: Generator | None = None,
        pipeline: "ChatPipeline | None" = None,
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

    @property
    def message_dicts(self) -> list[MessageDict]:
        """
        Returns the chat as a minimal message dictionaries.

        Returns:
            The MessageDict list
        """
        return [
            t.cast(
                MessageDict,
                m.model_dump(include={"role", "content_parts"}, exclude_none=True),
            )
            for m in self.all
        ]

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

    def strip(self, model_type: type[Model], fail_on_missing: bool = False) -> "Chat":  # noqa: FBT001, FBT002 (historical)
        """
        Strips all parsed parts of a particular type from the message content.

        Args:
            model_type: The type of model to keep in the chat.
            fail_on_missing: Whether to raise an exception if a message of the specified model type is not found.

        Returns:
            A new chat with only the messages of the specified model type.
        """
        new = self.clone()
        for message in new.all:
            message.strip(model_type, fail_on_missing=fail_on_missing)
        return new

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
        if len(self.messages) == 0 or self.messages[0].role != "system":
            self.messages.insert(0, Message(role="system", content=content))
        elif self.messages[0].role == "system" and content not in self.messages[0].content:
            self.messages[0].content += "\n\n" + content
        return self

    def inject_tool_prompt(
        self,
        tools: t.Sequence[Tool[..., t.Any]],
        mode: ToolMode,
    ) -> "Chat":
        """
        Injects a default tool use prompt into the system prompt.

        Args:
            tools: A sequence of Tool objects.

        Returns:
            The updated chat.
        """
        if mode not in ["xml", "json-in-xml"]:
            return self

        definitions: list[XmlToolDefinition] | list[JsonInXmlToolDefinition]
        if mode == "xml":
            definitions = [tool.xml_definition for tool in tools]
        else:
            definitions = [tool.json_definition for tool in tools]

        tool_system_prompt = tool_description_prompt_part(
            definitions,
            t.cast(t.Literal["xml", "json-in-xml"], mode),
        )
        return self.inject_system_content(tool_system_prompt)

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
        client: AsyncElasticsearch,
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
        client: AsyncElasticsearch,
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


# Callbacks


@runtime_checkable
class _ThenChatCallback(t.Protocol):
    def __call__(
        self,
        chat: Chat,
        /,
    ) -> t.Awaitable[Chat | None]:
        ...


@runtime_checkable
class _ThenChatStepCallback(t.Protocol):
    def __call__(
        self,
        chat: Chat,
        /,
    ) -> "PipelineStepGenerator | PipelineStepContextManager | t.Awaitable[PipelineStepGenerator | PipelineStepContextManager | None]":
        ...


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
    ) -> t.Awaitable[list[Chat]]:
        ...


@runtime_checkable
class _MapChatStepCallback(t.Protocol):
    def __call__(
        self,
        chats: list[Chat],
        /,
    ) -> "PipelineStepGenerator | PipelineStepContextManager | t.Awaitable[PipelineStepGenerator | PipelineStepContextManager]":
        ...


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
        Clone the current step and append a parent to it's heirarchy.
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
        callback_name = get_qualified_name(self.callback) if self.callback else "None"
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
        while self.parent is not None:
            depth += 1
            self = self.parent
        return depth


PipelineStepGenerator = t.AsyncGenerator[PipelineStep, None]
PipelineStepContextManager = t.AsyncContextManager[PipelineStepGenerator]

# Tracing wrappers


def _wrap_watch_callback(callback: WatchChatCallback) -> WatchChatCallback:
    callback_name = get_qualified_name(callback)

    async def traced_watch_callback(chats: list[Chat]) -> None:
        with tracer.span(
            f"Watch with {callback_name}()",
            callback=callback_name,
            chat_count=len(chats),
            chat_ids=[str(c.uuid) for c in chats],
        ):
            await callback(chats)

    return traced_watch_callback


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
        """How to handle failures in the pipeline unless overriden in calls."""
        self.caching: CacheMode | None = None
        """How to handle cache_control entries on messages."""

        self.until_types: list[type[Model]] = []
        self.tools: list[Tool[..., t.Any]] = []
        self.tool_mode: ToolMode = "auto"
        self.api_tool_choice: ApiToolChoice | None = None
        self.inject_tool_prompt = True
        self.stop_on_tool_calls = True
        self.then_callbacks: list[tuple[ThenChatCallback, int]] = []
        self.map_callbacks: list[tuple[MapChatCallback, int]] = []
        self.watch_callbacks: list[WatchChatCallback] = watch_callbacks or []

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
            on_failed: How to handle failures in the pipeline unless overriden in calls.

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
                    f"Callback '{get_qualified_name(callback)}' is already registered.",
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
                or merge_strategy == "only-user-role"
                and self.chat.all[-1].role == "user"
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
    ) -> "ChatPipeline":
        """
        Creates a clone of the current `ChatPipeline` instance.

        Args:
            only_messages: If True, only the messages will be cloned.
                If False (default), the entire `ChatPipeline` instance will be cloned
                including until callbacks, types, tools, metadata, etc.
            chat: An optional chat object clone for use in the new pipeline, otherwise the current
                internal chat object will be cloned.

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
            new.until_types = self.until_types.copy()
            new.tools = self.tools.copy()
            new.tool_mode = self.tool_mode
            new.metadata = deepcopy(self.metadata)
            new.map_callbacks = self.map_callbacks.copy()
            new.on_failed = self.on_failed
            new.errors_to_catch = self.errors_to_catch.copy()
            new.errors_to_exclude = self.errors_to_exclude.copy()
            new.caching = self.caching

            # Check if any of our callbacks are bound methods to a ChatPipline.
            # If so, we should rebind them to `self` to ensure they work correctly
            # and aren't operating with old state.

            new.then_callbacks = [
                (callback, max_depth)
                if not hasattr(callback, "__self__")
                or not isinstance(callback.__self__, ChatPipeline)
                else (types.MethodType(callback.__func__, new), max_depth)  # type: ignore [union-attr]
                for callback, max_depth in self.then_callbacks.copy()
            ]
            new.map_callbacks = [
                (callback, max_depth)
                if not hasattr(callback, "__self__")
                or not isinstance(callback.__self__, ChatPipeline)
                else (types.MethodType(callback.__func__, new), max_depth)  # type: ignore [union-attr]
                for callback, max_depth in self.map_callbacks.copy()
            ]

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

    def then(
        self,
        *callbacks: ThenChatCallback,
        max_depth: int = DEFAULT_MAX_DEPTH,
        allow_duplicates: bool = False,
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
            if not asyncio.iscoroutinefunction(callback):
                raise TypeError(
                    f"Callback '{get_qualified_name(callback)}' must be an async function",
                )

            if allow_duplicates:
                continue

            if callback in [c[0] for c in self.then_callbacks]:
                raise ValueError(
                    f"Callback '{get_qualified_name(callback)}' is already registered.",
                )

        self.then_callbacks.extend([(callback, max_depth) for callback in callbacks])
        return self

    def map(
        self,
        *callbacks: MapChatCallback,
        max_depth: int = DEFAULT_MAX_DEPTH,
        allow_duplicates: bool = False,
    ) -> "ChatPipeline":
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            You must return a list of Chat objects from the callback which will
            represent the state of chats for the remainder of the callbacks and
            the final return value from the pipeline.

        Args:
            callback: The callback function to be executed.
            max_depth: The maximum depth to allow recursive pipeline calls during this callback.

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
            if not asyncio.iscoroutinefunction(callback):
                raise TypeError(
                    f"Callback '{get_qualified_name(callback)}' must be an async function",
                )

            if allow_duplicates:
                continue

            if callback in [c[0] for c in self.map_callbacks]:
                raise ValueError(
                    f"Callback '{get_qualified_name(callback)}' is already registered.",
                )

        self.map_callbacks.extend([(callback, max_depth) for callback in callbacks])
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
        choice: ApiToolChoice | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        stop_on_tool_calls: bool | None = None,
    ) -> "ChatPipeline":
        """
        Adds a tool or a sequence of tools to participate in the generation process.

        Note:
            By default, the tool mode is set to "auto" which will attempt to use
            api function calling if available, otherwise it will fallback to `xml`.

        Args:
            *tools: The tools to be added to the pipeline.
            mode: The tool calling mode to use (e.g., "xml", "json-in-xml", "api").
            choice: The API tool choice to use. This is only relevant when using the "api" tool mode.
            max_depth: The maximum depth for recursive tool calls (this is shared between all tools).
            stop_on_tool_calls: When using natively parsed tools, whether to stop generation when a tool call block is observed.

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

        new_tools = [
            tool if isinstance(tool, Tool) else Tool.from_callable(tool)
            for tool in flatten_list(list(tools))  # in case the user gave us lists
        ]

        existing_names = {tool.name for tool in self.tools}
        for tool in new_tools:
            if tool.name in existing_names:
                raise ValueError(
                    f"Tool with name '{tool.name}' already exists in the pipeline.",
                )

        self.tools += new_tools

        self.then_callbacks = [
            (callback, max_depth)
            for callback, max_depth in self.then_callbacks
            if callback != self._then_tools
        ]
        self.then_callbacks.insert(
            0,
            (self._then_tools, max_depth),
        )  # make sure this is first

        if mode is not None:
            self.tool_mode = mode

        if choice is not None:
            self.api_tool_choice = choice

        if stop_on_tool_calls is not None:
            self.stop_on_tool_calls = stop_on_tool_calls

        return self

    def until_parsed_as(
        self,
        *types: type[ModelT],
        max_depth: int = DEFAULT_MAX_DEPTH,
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
            max_rounds: The maximum number of rounds to try to parse successfully.
            append: Whether to append the types to the existing list or replace it.
            max_depth: The maximum depth to re-attempt parsing using recursive pipelines  (this is shared between all types).
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

        self.until_types = list(types)

        max_depth = max_rounds or max_depth
        self.then_callbacks = [
            (callback, max_depth)
            for callback, max_depth in self.then_callbacks
            if callback != self._then_parse
        ]
        self.then_callbacks.append((self._then_parse, max_depth))

        return self

    # Internal callbacks for handling tools and parsing

    async def _then_tools(self, chat: Chat) -> PipelineStepContextManager | None:
        if (
            self.stop_on_tool_calls
            and self.tool_mode in ["xml", "json-in-xml"]
            and chat.stop_reason == "stop"
        ):
            # If we:
            # 1. Are using native tools
            # 2. Set a stop token for the tool calls
            # 3. Hit that stop token
            #
            # Then we should re-inject the closing tag for completeness.

            for part in chat.last.content_parts:
                if (
                    part.type == "text"
                    and f"<{TOOL_CALLS_TAG}>" in part.text
                    and f"</{TOOL_CALLS_TAG}>" not in part.text
                ):
                    part.text += f"</{TOOL_CALLS_TAG}>"
                    break

        # Parse the actual tool calls

        tool_calls: (list[ApiToolCall] | list[XmlToolCall] | list[JsonInXmlToolCall] | None) = None
        if self.tool_mode == "api":
            tool_calls = chat.last.tool_calls
        if self.tool_mode == "xml":
            tool_calls = chat.last.try_parse_set(XmlToolCall)
        elif self.tool_mode == "json-in-xml":
            tool_calls = chat.last.try_parse_set(JsonInXmlToolCall)

        if not tool_calls:
            return None

        next_pipeline = self.clone(chat=chat)

        should_continue = True

        for tool_call in tool_calls:
            tool = next((t for t in self.tools if t.name == tool_call.name), None)
            if tool is None:
                raise UnknownToolError(tool_call.name)

            message, _should_continue = await tool.handle_tool_call(tool_call)
            next_pipeline.add(message)

            # If the tool returns none, we should resolve tool calls, but
            # not continue the pipeline.

            if not _should_continue:
                should_continue = _should_continue

        # Need to prevent infinite loops and treat tool_choice like
        # an ephemeral setting which resets after the first tool call.

        if self.tool_mode == "api" and next_pipeline.params:
            next_pipeline.params.tool_choice = None

        if not should_continue:
            # TODO(nick): Type hints here stop us from mixing step generators
            # and basic chat returns.
            return next_pipeline.chat  # type: ignore [return-value]

        return next_pipeline.step()

    async def _then_parse(self, chat: Chat) -> PipelineStepContextManager | None:
        next_pipeline = self.clone(chat=chat)

        try:
            chat.last.parse_many(*self.until_types)
        except ValidationError as e:
            next_pipeline.add(
                Message.from_model(
                    ValidationErrorModel(content=str(e)),
                    suffix="Rewrite your entire message with all the required xml structure.",
                ),
            )
        except Exception as e:  # noqa: BLE001
            next_pipeline.add(
                Message.from_model(
                    SystemErrorModel(content=str(e)),
                    suffix="Rewrite your entire message with all the required xml structure.",
                ),
            )
        else:  # parsed successfully
            return None

        return next_pipeline.step()

    # Run helper methods

    async def _pre_run(self) -> None:
        if self.tool_mode == "auto" and self.tools:
            self.tool_mode = "api" if await self.generator.supports_function_calling() else "xml"

        if self.tools and self.tool_mode in ["xml", "json-in-xml"]:
            if self.inject_tool_prompt:
                self.chat.inject_tool_prompt(self.tools, self.tool_mode)
                self.inject_native_tool_prompt = False

            if self.stop_on_tool_calls:
                self.params = self.params = GenerateParams()
                self.params.stop = self.params.stop or []
                self.params.stop.append(f"</{TOOL_CALLS_TAG}>")

        if self.tools and self.tool_mode == "api":
            if self.params is None:
                self.params = GenerateParams()
            self.params.tools = [tool.api_definition for tool in self.tools]
            self.params.tool_choice = self.api_tool_choice

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

    def _apply_cache_mode_to_messages(
        self,
        messages: list[list[Message]],
    ) -> list[list[Message]]:
        if self.caching is None:
            return messages

        if self.caching != "latest":
            logger.warning(
                f"Unknown caching mode '{self.caching}', defaulting to 'latest'",
            )

        # first remove existing cache settings
        updated: list[list[Message]] = []
        for _messages in messages:
            updated = [
                *updated,
                [m.clone().cache(cache_control=False) for m in _messages],
            ]

        # then apply the latest cache settings
        for _messages in updated:
            for message in [m for m in _messages if m.role != "assistant"][-2:]:
                message.cache(cache_control=True)

        return updated

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
        callback_name = get_qualified_name(callback)

        async def complete() -> None:
            state.completed = True
            state.ready_event.set()

        with tracer.span(
            f"Then with {callback_name}()",
            callback=callback_name,
            chat_id=str(state.chat.uuid),
        ):
            async with contextlib.AsyncExitStack() as exit_stack:
                exit_stack.push_async_callback(complete)

                result = callback(state.chat)

                if inspect.isawaitable(result):
                    result = await result  # type: ignore [assignment]

                if result is None or isinstance(result, Chat):
                    state.chat = result or state.chat
                    return

                if isinstance(result, contextlib.AbstractAsyncContextManager):
                    result = await exit_stack.enter_async_context(result)

                if not inspect.isasyncgen(result):
                    raise TypeError(
                        f"Callback '{callback_name}' must return a Chat, PipelineStepGenerator, or None",
                    )

                generator = t.cast(
                    PipelineStepGenerator,
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

    # Run methods

    async def _step(  # noqa: PLR0915, PLR0912
        self,
        span: Span,
        messages: list[list[Message]],
        params: list[GenerateParams],
        on_failed: FailMode,
    ) -> PipelineStepGenerator:
        chats: ChatList = ChatList([])

        # Pass the messages to the generator

        try:
            messages = self._apply_cache_mode_to_messages(messages)
            generated = await self.generator.generate_messages(messages, params)

        # If we got a total failure here for generation as a whole,
        # we can't distinguish between incoming messages in terms
        # of what caused the error, so so we need to set the error
        # on all of them.

        except Exception as error:  # noqa: BLE001
            span.set_attribute("failed", True)
            span.set_attribute("error", error)

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

        # Watch callbacks

        await asyncio.gather(
            *[_wrap_watch_callback(callback)(chats) for callback in self.watch_callbacks],
        )

        # Yield what we generated

        span.set_attribute("chats", chats)
        current_step = PipelineStep(
            state="generated",
            chats=chats,
            pipeline=self,
        )
        yield current_step

        # Check if we should immediately raise

        for chat in chats:
            if chat.error is not None and (
                on_failed == "raise"
                or not any(isinstance(chat.error, t) for t in self.errors_to_catch)
                or any(isinstance(chat.error, t) for t in self.errors_to_exclude)
            ):
                span.set_attribute("error", chat.error)
                span.set_attribute("failed", True)
                raise chat.error

        # Chat cleanup

        if on_failed == "skip":
            chats = ChatList([chat for chat in chats if not chat.failed])

        span.set_attribute("chats", chats)

        if len(chats) == 0 or all(chat.failed for chat in chats):
            yield PipelineStep(
                state="final",
                chats=chats,
                pipeline=self,
            )
            return

        # Then callbacks

        for then_callback, max_depth in self.then_callbacks:
            callback_name = get_qualified_name(then_callback)

            states = [
                self.CallbackState(
                    chat=chat,
                    ready_event=asyncio.Event(),
                    continue_event=asyncio.Event(),
                )
                for chat in chats
            ]

            tasks = [
                asyncio.create_task(self._process_then_callback(then_callback, state))
                for state in states
            ]

            try:
                while not all(state.completed for state in states):
                    await asyncio.gather(
                        *[state.ready_event.wait() for state in states if not state.completed],
                    )

                    # TODO(nick): Are we good to throw exceptions here?
                    for task in tasks:
                        if task.done() and (exception := task.exception()):
                            raise exception

                    for state in states:
                        if state.ready_event.is_set() and state.step:
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
                    await asyncio.gather(*tasks)  # TODO(nick): return_exceptions=True ?

            chats = ChatList([state.chat for state in states if state.chat])

            span.set_attribute("chats", chats)
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

        span.set_attribute("chats", chats)

        if len(chats) == 0 or all(chat.failed for chat in chats):
            yield PipelineStep(
                state="final",
                chats=chats,
                pipeline=self,
            )
            return

        # Map callbacks

        for map_callback, max_depth in self.map_callbacks:
            callback_name = get_qualified_name(map_callback)

            with tracer.span(
                f"Map with {callback_name}()",
                callback=callback_name,
                chat_count=len(chats),
                chat_ids=[str(c.uuid) for c in chats],
            ):
                async with contextlib.AsyncExitStack() as exit_stack:
                    result = map_callback(chats)
                    chats_or_generator = await result if inspect.isawaitable(result) else result

                    if isinstance(result, contextlib.AbstractAsyncContextManager):
                        result = await exit_stack.enter_async_context(result)

                    if inspect.isasyncgen(chats_or_generator):
                        generator = t.cast(
                            PipelineStepGenerator,
                            await exit_stack.enter_async_context(
                                aclosing(chats_or_generator),
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

                    elif isinstance(chats_or_generator, list) and all(
                        isinstance(c, Chat) for c in chats_or_generator
                    ):
                        chats = ChatList(chats_or_generator)

                span.set_attribute("chats", chats)
                current_step = PipelineStep(
                    state="callback",
                    chats=chats,
                    pipeline=self,
                    callback=map_callback,
                )
                yield current_step

        if on_failed == "skip":
            chats = ChatList([chat for chat in chats if not chat.failed])

        span.set_attribute("chats", chats)
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

        await self._pre_run()

        on_failed = on_failed or self.on_failed

        messages = [self.chat.all]
        params = self._fit_params(1, [self.params])

        with tracer.span(
            f"Chat with {self.generator.to_identifier()}",
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            async with aclosing(
                self._step(span, messages, params, on_failed),
            ) as generator:
                yield generator

    async def run(
        self,
        *,
        on_failed: FailMode | None = None,
        allow_failed: bool = False,
    ) -> Chat:
        """
        Execute the generation process for a single message.

        Args:
            on_failed: The behavior when a message fails to generate.
            allow_failed: Deprecated, use `on_failed="include"`.

        Returns:
            The generated Chat.
        """
        if allow_failed:
            warnings.warn(
                "The 'allow_failed' argument is deprecated, use 'on_failed=\"include\"'.",
                DeprecationWarning,
                stacklevel=2,
            )

        if on_failed is None:
            on_failed = "include" if allow_failed else self.on_failed

        last: PipelineStep | None = None
        async with self.step(on_failed=on_failed) as steps:
            async for step in steps:
                last = step

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
        await self._pre_run()

        on_failed = on_failed or self.on_failed

        messages = [self.chat.all] * count
        params = self._fit_params(count, params)

        with tracer.span(
            f"Chat with {self.generator.to_identifier()} (x{count})",
            count=count,
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            async with aclosing(
                self._step(span, messages, params, on_failed),
            ) as generator:
                yield generator

    async def run_many(
        self,
        count: int,
        *,
        params: t.Sequence[GenerateParams | None] | None = None,
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process in parallel over the same input.

        Args:
            count: The number of times to execute the generation process.
            params: A sequence of parameters to be used for each execution.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """

        last: PipelineStep | None = None
        async with self.step_many(count, params=params, on_failed=on_failed) as steps:
            async for step in steps:
                last = step

        if last is None or last.state != "final":
            raise ValueError("The generation process did not complete successfully")

        return last.chats

    # Batch messages

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
        await self._pre_run()

        on_failed = on_failed or self.on_failed

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

        with tracer.span(
            f"Chat batch with {self.generator.to_identifier()} ({count})",
            count=count,
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            async with aclosing(
                self._step(span, messages, params, on_failed),
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
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process over multiple input messages.

        Note:
            Anything already in this chat pipeline will be prepended to the input messages.

        Args:
            many: A sequence of sequences of messages to be generated.
            params: A sequence of parameters to be used for each set of messages.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """

        last: PipelineStep | None = None
        async with self.step_batch(many, params=params, on_failed=on_failed) as steps:
            async for step in steps:
                last = step

        if last is None or last.state != "final":
            raise ValueError("The generation process did not complete successfully")

        return last.chats

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
        await self._pre_run()

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

        with tracer.span(f"Chat over {len(coros)} generators", count=len(coros)):
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
