"""
Chats are used pre and post generation to hold messages.

They are the primary way to interact with the generator.
"""

from __future__ import annotations

import asyncio
import typing as t
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import runtime_checkable
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, ValidationError, computed_field

from rigging.error import MessagesExhaustedMaxRoundsError, UnknownToolError
from rigging.generator import GenerateParams, Generator, get_generator
from rigging.generator.base import GeneratedMessage, StopReason, Usage  # noqa: TCH001
from rigging.message import Message, MessageDict, Messages
from rigging.model import Model, ModelT, SystemErrorModel, ValidationErrorModel
from rigging.tool.api import ApiTool, ToolChoice
from rigging.tool.native import Tool, ToolCalls, ToolDescriptionList, ToolResult, ToolResults, system_tool_extension
from rigging.tracing import Span, tracer
from rigging.util import get_qualified_name

if t.TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

    from rigging.data import ElasticOpType
    from rigging.prompt import P, Prompt, R

CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])

DEFAULT_MAX_ROUNDS = 5
"""Maximum number of internal callback rounds to attempt during generation before giving up."""

FailMode = t.Literal["raise", "skip", "include"]
"""
How to handle failures in pipelines.

- raise: Raise an exception when a failure is encountered.
- skip: Ignore the error and do not include the failed chat in the final output.
- include: Mark the message as failed and include it in the final output.
"""


class Chat(BaseModel):
    """
    Represents a completed chat conversation.
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
    usage: t.Optional[Usage] = Field(None, repr=False)
    """The usage statistics for the generation if available."""
    extra: dict[str, t.Any] = Field(default_factory=dict, repr=False)
    """Any additional information from the generation."""

    generator: t.Optional[Generator] = Field(None, exclude=True, repr=False)
    """The generator associated with the chat."""
    params: t.Optional[GenerateParams] = Field(None, exclude=True, repr=False)
    """Any additional generation params used for this chat."""

    error: t.Optional[
        t.Annotated[Exception, PlainSerializer(lambda x: str(x), return_type=str, when_used="json-unless-none")]
    ] = Field(None, repr=False)
    """Holds any exception that was caught during the generation pipeline."""
    failed: bool = Field(False, exclude=False, repr=True)
    """
    Indicates whether conditions during generation were not met.
    This is typically used for graceful error handling when parsing.
    """

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
        generator: t.Optional[Generator] = None,
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
            generator = get_generator(kwargs.pop("generator_id"))

        # We can't deserialize an error
        if isinstance(kwargs.get("error"), str):
            kwargs.pop("error")

        super().__init__(
            messages=Message.fit_as_list(messages),
            generated=Message.fit_as_list(generated) if generated is not None else [],
            generator=generator,
            **kwargs,
        )

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
        return "\n\n".join([str(m) for m in self.all])

    @property
    def message_dicts(self) -> list[MessageDict]:
        """
        Returns the chat as a minimal message dictionaries.

        Returns:
            The MessageDict list
        """
        return [t.cast(MessageDict, m.model_dump(include={"role", "all_content"})) for m in self.all]

    def meta(self, **kwargs: t.Any) -> Chat:
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated chat object.
        """
        self.metadata.update(kwargs)
        return self

    def restart(self, *, generator: t.Optional[Generator] = None, include_all: bool = False) -> ChatPipeline:
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
            generator = self.generator
        if generator is None:
            raise ValueError("Cannot restart a chat without an associated generator")
        return generator.chat(messages, self.params)

    def fork(
        self,
        messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | MessageDict | str,
        *,
        include_all: bool = False,
    ) -> ChatPipeline:
        """
        Forks the chat by creating calling [rigging.chat.Chat.restart][] and appending the specified messages.

        Args:
            messages:
                The messages to be added to the new `ChatPipeline` instance.
            include_all: Whether to include the next messages in the restarted chat.

        Returns:
            A new instance of `ChatPipeline` with the specified messages added.

        """
        return self.restart(include_all=include_all).add(messages)

    def continue_(self, messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | str) -> ChatPipeline:
        """Alias for the [rigging.chat.Chat.fork][] with `include_all=True`."""
        return self.fork(messages, include_all=True)

    def clone(self, *, only_messages: bool = False) -> Chat:
        """
        Creates a deep copy of the chat.

        Args:
            only_messages: If True, only the messages will be cloned.
                If False (default), the entire chat object will be cloned.

        Returns:
            A new instance of Chat.
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

    def apply(self, **kwargs: str) -> Chat:
        """
        Calls [rigging.message.Message.apply][] on the last message in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The modified Chat object.
        """
        if self.generated:
            self.generated[-1] = self.generated[-1].apply(**kwargs)
        else:
            self.messages[-1] = self.messages[-1].apply(**kwargs)
        return self

    def apply_to_all(self, **kwargs: str) -> Chat:
        """
        Calls [rigging.message.Message.apply][] on all messages in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The modified chat object.
        """
        self.messages = Message.apply_to_list(self.messages, **kwargs)
        self.generated = Message.apply_to_list(self.generated, **kwargs)
        return self

    def strip(self, model_type: type[Model], fail_on_missing: bool = False) -> Chat:
        """
        Strips all parsed parts of a particular type from the message content.

        Args:
            model_type: The type of model to keep in the chat.
            fail_on_missing: Whether to raise an exception if a message of the specified model type is not found.

        Returns:
            A new Chat object with only the messages of the specified model type.
        """
        new = self.clone()
        for message in new.all:
            message.strip(model_type, fail_on_missing=fail_on_missing)
        return new

    def inject_system_content(self, content: str) -> Message:
        """
        Injects content into the chat as a system message.

        Note:
            If the chat is empty or the first message is not a system message,
            a new system message with the given content is inserted at the beginning of the chat.
            If the first message is a system message, the content is appended to it.

        Args:
            content: The content to be injected.

        Returns:
            The updated system message.
        """
        if len(self.messages) == 0 or self.messages[0].role != "system":
            self.messages.insert(0, Message(role="system", content=content))
        elif self.messages[0].role == "system":
            self.messages[0].content += "\n\n" + content
        return self.messages[0]

    def inject_tool_prompt(self, tools: t.Sequence[Tool]) -> None:
        """
        Injects a default tool use prompt into the system prompt.

        Args:
            tools: A sequence of Tool objects.
        """
        call_format = ToolCalls.xml_example()
        tool_description_list = ToolDescriptionList(tools=[t.get_description() for t in tools])
        tool_system_prompt = system_tool_extension(call_format, tool_description_list.to_pretty_xml())
        self.inject_system_content(tool_system_prompt)

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
        op_type: ElasticOpType = "index",
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

        return await chats_to_elastic(self, index, client, op_type=op_type, create_index=create_index, **kwargs)


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
        op_type: ElasticOpType = "index",
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

        return await chats_to_elastic(self, index, client, op_type=op_type, create_index=create_index, **kwargs)

    def to_json(self) -> list[dict[str, t.Any]]:
        """
        Helper to convert the chat list to a list of dictionaries.
        """
        return [chat.model_dump() for chat in self]


# Callbacks


@runtime_checkable
class UntilMessageCallback(t.Protocol):
    def __call__(self, message: Message, /) -> tuple[bool, list[Message]]:
        """
        Passed the next message, returns whether or not to continue and an
        optional list of messages to append before continuing.
        """
        ...


@runtime_checkable
class ThenChatCallback(t.Protocol):
    def __call__(self, chat: Chat, /) -> t.Awaitable[Chat | None]:
        """
        Passed a finalized chat to process and can return a new chat to replace it.
        """
        ...


@runtime_checkable
class MapChatCallback(t.Protocol):
    def __call__(self, chats: list[Chat], /) -> t.Awaitable[list[Chat]]:
        """
        Passed a finalized chats to process. Can replace chats in the pipeline by returning
        a new chat object.
        """
        ...


@runtime_checkable
class WatchChatCallback(t.Protocol):
    def __call__(self, chats: list[Chat], /) -> t.Awaitable[None]:
        """
        Passed any created chat objects for monitoring/logging.
        """
        ...


# Helper classes to manage complexity inside the run functions


@dataclass
class RunState:
    inputs: list[Message]
    messages: list[Message]
    params: GenerateParams
    processor: t.Generator[list[Message], Message, list[Message]]
    chat: Chat | None = None
    watched: bool = False


class ChatPipeline:
    """
    Pipeline to manipulate and produce chats.
    """

    def __init__(
        self,
        generator: Generator,
        messages: t.Sequence[Message],
        *,
        params: t.Optional[GenerateParams] = None,
        watch_callbacks: t.Optional[list[WatchChatCallback]] = None,
    ):
        self.generator: Generator = generator
        """The generator object responsible for generating the chat."""
        self.chat: Chat = Chat(messages)
        """The chat object representing the conversation."""
        self.params = params
        """The parameters for generating messages."""
        self.metadata: dict[str, t.Any] = {}
        """Additional metadata associated with the chat."""
        self.errors_to_fail_on: set[type[Exception]] = set()
        """
        The list of exceptions to catch during generation if you are including or skipping failures.

        ExhuastedMaxRounds is implicitly included.
        """
        self.errors_to_exclude: set[type[Exception]] = set()
        """The list of exceptions to exclude from the catch list."""
        self.on_failed: FailMode = "raise"
        """How to handle failures in the pipeline unless overriden in calls."""

        # (callback, attempt_recovery, drop_dialog, max_rounds)
        self.until_callbacks: list[tuple[UntilMessageCallback, bool, bool, int]] = []
        self.until_types: list[type[Model]] = []
        self.api_tools: list[ApiTool] = []
        self.native_tools: list[Tool] = []
        self.inject_native_tool_prompt: bool = True
        self.force_native_tool: bool = False
        self.then_callbacks: list[ThenChatCallback] = []
        self.map_callbacks: list[MapChatCallback] = []
        self.watch_callbacks: list[WatchChatCallback] = watch_callbacks or []

    def __len__(self) -> int:
        return len(self.chat)

    def with_(self, params: t.Optional[GenerateParams] = None, **kwargs: t.Any) -> ChatPipeline:
        """
        Assign specific generation parameter overloads for this chat.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            params: The parameters to set for the chat.
            **kwargs: An alternative way to pass parameters as keyword arguments.

        Returns:
            A new instance of ChatPipeline with the updated parameters.
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
        self, *errors: type[Exception], on_failed: FailMode | None = None, exclude: list[type[Exception]] | None = None
    ) -> ChatPipeline:
        """
        Adds exceptions to catch during generation when including or skipping failures.

        Args:
            *errors: The exception types to catch.
            on_failed: How to handle failures in the pipeline unless overriden in calls.

        Returns:
            The updated ChatPipeline object.
        """
        self.errors_to_fail_on.update(errors)
        self.errors_to_exclude.update(exclude or [])
        self.on_failed = on_failed or self.on_failed
        return self

    def watch(self, *callbacks: WatchChatCallback, allow_duplicates: bool = False) -> ChatPipeline:
        """
        Registers a callback to monitor any chats produced.

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
            if allow_duplicates or callback not in self.watch_callbacks:
                self.watch_callbacks.append(callback)
        return self

    def add(
        self, messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str
    ) -> ChatPipeline:
        """
        Appends new message(s) to the internal chat before generation.

        Note:
            If the last message in the chat is the same role as the first new message,
            the content will be appended. instead of a new message being created.

        Args:
            messages: The messages to be added to the chat. It can be a single message or a sequence of messages.

        Returns:
            The updated ChatPipeline object.
        """
        message_list = Message.fit_as_list(messages)
        # If the last message is the same role as the first new message, append to it
        if self.chat.all and self.chat.all[-1].role == message_list[0].role:
            self.chat.all[-1].content += "\n" + message_list[0].content
            message_list = message_list[1:]
        else:
            self.chat.generated += message_list
        return self

    def fork(
        self, messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str
    ) -> ChatPipeline:
        """
        Creates a new instance of `ChatPipeline` by forking the current chat and adding the specified messages.

        This is a convenience method for calling `clone().add(messages)`.

        Args:
            messages: A sequence of messages or a single message to be added to the new chat.

        Returns:
            A new instance the pipeline with the specified messages added.
        """
        return self.clone().add(messages)

    def clone(self, *, only_messages: bool = False) -> ChatPipeline:
        """
        Creates a clone of the current `ChatPipeline` instance.

        Args:
            only_messages: If True, only the messages will be cloned.
                If False (default), the entire `ChatPipeline` instance will be cloned
                including until callbacks, types, tools, metadata, etc.

        Returns:
            A new instance of `ChatPipeline` that is a clone of the current instance.
        """
        new = ChatPipeline(
            self.generator,
            [],
            params=self.params.model_copy() if self.params is not None else None,
            watch_callbacks=self.watch_callbacks,
        )
        new.chat = self.chat.clone()
        if not only_messages:
            new.until_callbacks = self.until_callbacks.copy()
            new.until_types = self.until_types.copy()
            new.native_tools = self.native_tools.copy()
            new.api_tools = self.api_tools.copy()
            new.inject_native_tool_prompt = self.inject_native_tool_prompt
            new.force_native_tool = self.force_native_tool
            new.metadata = deepcopy(self.metadata)
            new.then_callbacks = self.then_callbacks.copy()
            new.map_callbacks = self.map_callbacks.copy()
            new.on_failed = self.on_failed
            new.errors_to_fail_on = self.errors_to_fail_on.copy()
            new.errors_to_exclude = self.errors_to_exclude.copy()
        return new

    def meta(self, **kwargs: t.Any) -> ChatPipeline:
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated chat object.
        """
        self.metadata.update(kwargs)
        return self

    def then(self, callback: ThenChatCallback) -> ChatPipeline:
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            Returning a Chat object from the callback will replace the current chat.
            for the remainder of the callbacks + return value of `run()`. This is
            optional.

        ```
        async def process(chat: Chat) -> Chat | None:
            ...

        await pipeline.then(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the chat.
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError(f"Callback '{get_qualified_name(callback)}' must be an async function")

        self.then_callbacks.append(callback)
        return self

    def map(self, callback: MapChatCallback) -> ChatPipeline:
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            You must return a list of Chat objects from the callback which will
            represent the state of chats for the remainder of the callbacks and
            the final return of control.

        ```
        async def process(chats: list[Chat]) -> list[Chat]:
            ...

        await pipeline.map(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the chat.
        """
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError(f"Callback '{get_qualified_name(callback)}' must be an async function")

        self.map_callbacks.append(callback)
        return self

    def apply(self, **kwargs: str) -> ChatPipeline:
        """
        Clones this chat pipeline and calls [rigging.chat.Chat.apply][] with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be applied to the chat.

        Returns:
            A new instance of ChatPipeline with the applied arguments.
        """
        new = self.clone()
        new.chat.apply(**kwargs)
        return new

    def apply_to_all(self, **kwargs: str) -> ChatPipeline:
        """
        Clones this chat pipeline and calls [rigging.chat.Chat.apply_to_all][] with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be applied to the chat.

        Returns:
            A new instance of ChatPipeline with the applied arguments.
        """
        new = self.clone()
        new.chat.apply_to_all(**kwargs)
        return new

    def until(
        self,
        callback: UntilMessageCallback,
        *,
        attempt_recovery: bool = True,
        drop_dialog: bool = True,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> ChatPipeline:
        """
        Registers a callback to participate in validating the generation process.

        ```py
        # Takes the next message being generated, and returns whether or not to continue
        # generating new messages in addition to a list of messages to append before continuing

        def callback(message: Message) -> tuple[bool, list[Message]]:
            if is_valid(message):
                return (False, [message])
            else:
                return (True, [message, ...])

        await pipeline.until(callback).run()
        ```

        Note:
            Users might prefer the `.then` or `.map` callbacks as they are easier to work with.

        Note:
            In general, your callback function should always include the message that was passed to it.

            Whether these messages get used or discarded in the next round depends on `attempt_recovery`.

        Args:
            callback: The callback function to be executed.
            attempt_recovery: Whether to attempt recovery by continuing to append prior messages
                before the next round of generation.
            drop_dialog: Whether to drop the intermediate dialog of recovery before returning
                the final chat back to the caller.
            max_rounds: The maximum number of rounds to attempt generation + callbacks
                before giving uop.

        Returns:
            The current instance of the chat.
        """
        self.until_callbacks.append((callback, attempt_recovery, drop_dialog, max_rounds))
        return self

    def wrap(self, func: t.Callable[[CallableT], CallableT]) -> ChatPipeline:
        """
        Helper for [rigging.generator.base.Generator.wrap][].

        Args:
            func: The function to wrap the calls with.

        Returns:
            The current instance of the pipeline.
        """
        self.generator = self.generator.wrap(func)
        return self

    @t.overload
    def using(self, *tools: t.Callable[..., t.Any], choice: ToolChoice | None = None) -> ChatPipeline:
        ...

    @t.overload
    def using(
        self,
        *tools: Tool,
        force: bool = False,
        attempt_recovery: bool = True,
        drop_dialog: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        inject_prompt: bool | None = None,
    ) -> ChatPipeline:
        ...

    def using(
        self,
        *tools: Tool | ApiTool | t.Callable[..., t.Any],
        choice: ToolChoice | None = None,
        force: bool = False,
        attempt_recovery: bool = True,
        drop_dialog: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        inject_prompt: bool | None = None,
    ) -> ChatPipeline:
        """
        Adds a tool or a sequence of tools to participate in the generation process.

        These can be either:
        - Native tools (rigging.tool.native.Tool) which use manual parsing and schema insertion
        - API tools (rigging.tool.api.ApiTool or any callable) which uses api-provided tool integrations

        Args:
            tools: The tool or sequence of tools to be added.
            force: Whether to force the use of the tool(s) at least once.
            attempt_recovery: Whether to attempt recovery if the tool(s) fail by providing
                validation feedback to the model before the next round.
            drop_dialog: Whether to drop the intermediate dialog of recovery efforts
                before returning the final chat to the caller.
            max_rounds: The maximum number of rounds to attempt recovery.
            inject_prompt: Whether to inject the tool guidance prompt into a
                system message.and will override self.inject_tool_prompt if provided.

        Returns:
            The updated ChatPipeline object.
        """
        native_tools = [tool for tool in tools if isinstance(tool, Tool)]
        if native_tools and len(native_tools) != len(tools):
            raise ValueError("All tools must be of the same type (api or native)")

        if native_tools:
            return self.using_native_tools(
                *t.cast(list[Tool], tools),
                force=force,
                attempt_recovery=attempt_recovery,
                drop_dialog=drop_dialog,
                max_rounds=max_rounds,
                inject_prompt=inject_prompt,
            )
        else:
            return self.using_api_tools(*t.cast(list[ApiTool], tools), choice=choice)

    def using_native_tools(
        self,
        *tools: Tool,
        force: bool = False,
        attempt_recovery: bool = True,
        drop_dialog: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        inject_prompt: bool | None = None,
    ) -> ChatPipeline:
        """
        Adds a tool or a sequence of tools to participate in the generation process.

        Args:
            tools: The tool or sequence of tools to be added.
            force: Whether to force the use of the tool(s) at least once.
            attempt_recovery: Whether to attempt recovery if the tool(s) fail by providing
                validation feedback to the model before the next round.
            drop_dialog: Whether to drop the intermediate dialog of recovery efforts
                before returning the final chat to the caller.
            max_rounds: The maximum number of rounds to attempt recovery.
            inject_prompt: Whether to inject the tool guidance prompt into a
                system message.and will override self.inject_tool_prompt if provided.

        Returns:
            The updated ChatPipeline object.
        """
        if len(tools) == 0:
            return self

        if self.api_tools:
            raise ValueError("Cannot mix native and API tools in the same pipeline")

        self.native_tools += tools
        self.inject_native_tool_prompt = inject_prompt or self.inject_native_tool_prompt
        self.force_native_tool = force
        if next((c for c in self.until_callbacks if c[0] == self._until_native_tools_callback), None) is None:
            self.until_callbacks.append(
                (
                    self._until_native_tools_callback,
                    attempt_recovery,
                    drop_dialog,
                    max_rounds,
                )
            )
        return self

    def using_api_tools(
        self, *tools: ApiTool | t.Callable[..., t.Any], choice: ToolChoice | None = None
    ) -> ChatPipeline:
        """
        Adds an API tool or a sequence of API tools to participate in the generation process.

        Args:
            tools: The API tool or sequence of API tools to be added.

        Returns:
            The updated ChatPipeline object.
        """
        if len(tools) == 0:
            return self

        if self.native_tools:
            raise ValueError("Cannot mix native and API tools in the same pipeline")

        self.api_tools += [tool if isinstance(tool, ApiTool) else ApiTool(tool) for tool in tools]

        if self.params is None:
            self.params = GenerateParams()
        self.params.tools = [tool.definition for tool in self.api_tools]

        if choice is not None:
            self.params.tool_choice = choice

        if next((c for c in self.then_callbacks if c == self._then_api_tools), None) is None:
            self.then_callbacks.append(self._then_api_tools)

        return self

    def until_parsed_as(
        self,
        *types: type[ModelT],
        attempt_recovery: bool = False,
        drop_dialog: bool = True,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> ChatPipeline:
        """
        Adds the specified types to the list of types which should successfully parse
        before the generation process completes.

        Args:
            *types: The type or types of models to wait for.
            attempt_recovery: Whether to attempt recovery if parsing fails by providing
                validation feedback to the model before the next round.
            drop_dialog: Whether to drop the intermediate dialog of recovery efforts
                before returning the final chat to the caller.
            max_rounds: The maximum number of rounds to try to parse successfully.

        Returns:
            The updated ChatPipeline object.
        """
        self.until_types += types
        if next((c for c in self.until_callbacks if c[0] == self._until_parse_callback), None) is None:
            self.until_callbacks.append((self._until_parse_callback, attempt_recovery, drop_dialog, max_rounds))

        return self

    def _until_native_tools_callback(self, message: Message) -> tuple[bool, list[Message]]:
        generated: list[Message] = [message]

        try:
            tool_calls = message.try_parse(ToolCalls)
        except ValidationError as e:
            generated.append(Message.from_model(ValidationErrorModel(content=str(e))))
            return (True, generated)

        if tool_calls is None:
            if self.force_native_tool:
                logger.debug("No tool calls or types, returning error")
                generated.append(Message.from_model(SystemErrorModel(content="You must use a tool")))
            else:
                logger.debug("No tool calls or types, returning message")
            return (self.force_native_tool, generated)

        self.force_native_tool = False

        tool_results: list[ToolResult] = []
        errors: list[SystemErrorModel] = []
        for call in tool_calls.calls:
            if call.tool not in [tool.name for tool in self.native_tools]:
                errors.append(SystemErrorModel(content=f"Tool '{call.tool}' does not exist"))
                continue

            tool = next(t for t in self.native_tools if t.name == call.tool)
            tool_description = tool.get_description()

            if call.function not in [f.name for f in tool_description.functions]:
                errors.append(SystemErrorModel(content=f"Function '{call.function}' does not exist on '{tool.name}'"))
                continue

            tool_results.append(tool(call))

        if errors:
            generated.append(Message.from_model(errors, suffix="Rewrite your message with all the required tags."))
        else:
            generated.append(Message.from_model(ToolResults(results=tool_results)))

        return (True, generated)

    async def _then_api_tools(self, chat: Chat) -> Chat | None:
        # If there are no tool calls, we can continue
        if not chat.last.tool_calls:
            return None

        # Slightly strange cloning behavior here, but we are abusing
        # the .then() mechanic slightly and want our pipeline to maintain
        # all existing state

        next_pipeline = self.clone()
        next_pipeline.chat = chat.clone()

        for tool_call in chat.last.tool_calls:
            tool = next((t for t in self.api_tools if t.name == tool_call.function.name), None)
            if tool is None:
                raise UnknownToolError(tool_call.function.name)
            next_pipeline.add(await tool.execute(tool_call))

        # Need to prevent infinite loops and treat tool_choice like
        # an ephemeral setting which resets after each tool call.
        #
        # TODO: Seems like this is surfacing a larger architectural issue we should look at

        if next_pipeline.params:
            next_pipeline.params.tool_choice = None

        return await next_pipeline.run()

    def _until_parse_callback(self, message: Message) -> tuple[bool, list[Message]]:
        should_continue: bool = False
        generated: list[Message] = [message]

        try:
            message.parse_many(*self.until_types)
        except ValidationError as e:
            should_continue = True
            generated.append(
                Message.from_model(
                    ValidationErrorModel(content=str(e)),
                    suffix="Rewrite your entire message with all the required elements.",
                )
            )
        except Exception as e:
            should_continue = True
            generated.append(
                Message.from_model(
                    SystemErrorModel(content=str(e)),
                    suffix="Rewrite your entire message with all the required elements.",
                )
            )

        return (should_continue, generated)

    def _until(
        self,
        message: Message,
        callback: UntilMessageCallback,
        attempt_recovery: bool,
        drop_dialog: bool,
        max_rounds: int,
    ) -> t.Generator[list[Message], Message, list[Message]]:
        should_continue, step_messages = callback(message)
        if not should_continue:
            return step_messages

        running_messages = step_messages if attempt_recovery else []
        next_message: Message

        for _round in range(1, max_rounds + 1):
            callback_name = get_qualified_name(callback)
            with tracer.span(
                f"Until with {callback_name}() ({_round}/{max_rounds})",
                callback=callback_name,
                round=_round,
                max_rounds=max_rounds,
                attempt_recovery=attempt_recovery,
                drop_dialog=drop_dialog,
            ):
                logger.trace(
                    f"_until({callback_name}) round {_round}/{max_rounds} (attempt_recovery={attempt_recovery})"
                )
                next_message = yield running_messages
                should_continue, step_messages = callback(next_message)
                logger.trace(f" |- returned {should_continue} with {len(step_messages)} new messages)")

                if attempt_recovery:
                    running_messages += step_messages

                if not should_continue:
                    return step_messages if drop_dialog else running_messages

        # !attempt_recovery -> Return just the latest generation
        # attempt_recovery & drop_dialog -> Return just the latest generation
        # attempt_recovery & !drop_dialog -> Return intermediate and the latest

        logger.warning(f"Exhausted max rounds ({max_rounds})")
        raise MessagesExhaustedMaxRoundsError(
            max_rounds, [next_message] if not attempt_recovery and next_message else running_messages[:-1]
        )

    # TODO: Much like the CompletionPipeline code, it's opaque
    # exactly how multiple callbacks should be blended together
    # when generating. I think we should look at limiting it to
    # one callback in total, but I'll leave the behavior as is
    # for now with the knowledge that behavior might be a bit
    # unpredictable.
    def _process(self) -> t.Generator[list[Message], Message, list[Message]]:
        self._pre_run()
        first_response = yield []
        new_messages = [first_response]

        # If we need to process tool calls, we should do that first
        # before proceeding with our until callbacks
        if first_response.tool_calls and self.api_tools:
            return new_messages

        for callback, reset_between, drop_internal, max_rounds in self.until_callbacks:
            generated = yield from self._until(new_messages[-1], callback, reset_between, drop_internal, max_rounds)
            new_messages = new_messages[:-1] + generated
        return new_messages

    async def _watch_callback(self, chats: list[Chat]) -> None:
        # Given that these watch callbacks don't return a value,
        # we should be safe to run them internally.

        def wrap_watch_callback(callback: WatchChatCallback) -> WatchChatCallback:
            async def traced_watch_callback(chats: list[Chat]) -> None:
                callback_name = get_qualified_name(callback)
                with tracer.span(
                    f"Watch with {callback_name}()",
                    callback=callback_name,
                    chat_count=len(chats),
                    chat_ids=[str(c.uuid) for c in chats],
                ):
                    await callback(chats)

            return traced_watch_callback

        coros = [wrap_watch_callback(callback)(chats) for callback in self.watch_callbacks]
        await asyncio.gather(*coros)

    # Run helper methods

    async def _post_run(self, chats: list[Chat], on_failed: FailMode) -> ChatList:
        if on_failed == "skip":
            chats = [c for c in chats if not c.failed]

        # These have to be sequenced to support the concept of
        # a pipeline where future then/map calls can depend on
        # previous calls being ran.

        for map_callback in self.map_callbacks:
            callback_name = get_qualified_name(map_callback)
            with tracer.span(
                f"Map with {callback_name}()",
                callback=callback_name,
                chat_count=len(chats),
                chat_ids=[str(c.uuid) for c in chats],
            ):
                chats = await map_callback(chats)
                if not all(isinstance(c, Chat) for c in chats):
                    raise ValueError(f".map() callback must return a Chat object or None ({callback_name})")

        def wrap_then_callback(callback: ThenChatCallback) -> ThenChatCallback:
            async def traced_then_callback(chat: Chat) -> Chat | None:
                callback_name = get_qualified_name(callback)
                with tracer.span(f"Then with {callback_name}()", callback=callback_name, chat_id=str(chat.uuid)):
                    return await callback(chat)

            return traced_then_callback

        for then_callback in self.then_callbacks:
            coros = [wrap_then_callback(then_callback)(chat) for chat in chats]
            new_chats = await asyncio.gather(*coros)
            if not all(isinstance(c, Chat) or c is None for c in new_chats):
                raise ValueError(
                    f".then() callback must return a Chat object or None ({get_qualified_name(then_callback)})"
                )

            chats = [new or chat for new, chat in zip(new_chats, chats)]

        return ChatList(chats)

    def _pre_run(self) -> None:
        if self.native_tools:
            if self.inject_native_tool_prompt:
                self.chat.inject_tool_prompt(self.native_tools)
                self.inject_native_tool_prompt = False

            # TODO: This can cause issues when certain APIs do not return
            # the stop sequence as part of the response. This behavior
            # seems like a larger issue than the model continuining after
            # requesting a tool call, so we'll remove it for now.
            #
            # self.params.stop = [ToolCalls.xml_end_tag()]

    def _fit_params(
        self, count: int, params: t.Sequence[t.Optional[GenerateParams] | None] | None = None
    ) -> list[GenerateParams]:
        params = [None] * count if params is None else list(params)
        if len(params) != count:
            raise ValueError(f"The number of params must be {count}")
        if self.params is not None:
            params = [self.params.merge_with(p) for p in params]
        return [(p or GenerateParams()) for p in params]

    # Run methods

    def _initialize_states(
        self, count: int, params: t.Sequence[t.Optional[GenerateParams]] | None = None
    ) -> list[RunState]:
        states = [RunState([], [], p, self._process()) for p in self._fit_params(count, params)]
        for state in states:
            next(state.processor)
        return states

    def _initialize_batch_states(
        self,
        many: t.Sequence[t.Sequence[Message]]
        | t.Sequence[Message]
        | t.Sequence[MessageDict]
        | t.Sequence[str]
        | MessageDict
        | str,
        params: t.Sequence[t.Optional[GenerateParams]] | None = None,
    ) -> list[RunState]:
        if isinstance(many, dict) or isinstance(many, str):  # Some strange typechecking here
            many = t.cast(t.Union[t.Sequence[str], t.Sequence[MessageDict]], [many])

        count = max(len(many), len(params) if params is not None else 0)

        many = [Message.fit_as_list(m) for m in many]
        if len(many) < count:
            if len(many) != 1:
                raise ValueError(f"Can't fit many of length {len(many)} to {count}")
            many = many * count

        params = self._fit_params(count, params)

        states: list[RunState] = [RunState(self.chat.all + m, [], p, self._process()) for m, p in zip(many, params)]
        for state in states:
            next(state.processor)

        return states

    def _create_chat(
        self,
        state: RunState,
        outputs: list[Message],
        inbound: GeneratedMessage,
        batch_mode: bool,
        failed: bool = False,
        error: Exception | None = None,
    ) -> Chat:
        return Chat(
            state.inputs if batch_mode else self.chat.all,
            outputs,
            generator=self.generator,
            metadata=self.metadata,
            params=state.params,
            stop_reason=inbound.stop_reason,
            usage=inbound.usage,
            extra=inbound.extra,
            failed=failed,
            error=error,
        )

    def _create_failed_chat(self, state: RunState, error: Exception, batch_mode: bool) -> Chat:
        return Chat(
            state.inputs if batch_mode else self.chat.all,
            [],
            generator=self.generator,
            metadata=self.metadata,
            params=state.params,
            failed=True,
            error=error,
        )

    async def _run(self, span: Span, states: list[RunState], on_failed: FailMode, batch_mode: bool = False) -> ChatList:
        pending_states = states
        while pending_states:
            try:
                inbounds = await self.generator.generate_messages(
                    [(s.inputs if batch_mode else self.chat.all) + s.messages for s in pending_states],
                    [s.params for s in pending_states],
                )

            except Exception as error:
                # Handle core generator errors
                if (
                    on_failed == "raise"
                    or not any(isinstance(error, t) for t in self.errors_to_fail_on)
                    or any(isinstance(error, t) for t in self.errors_to_exclude)
                ):
                    raise

                # We will apply the error to all chats in the batch as we can't
                # tell which chat caused the error right now.

                span.set_attribute("failed", True)
                span.set_attribute("error", error)

                for state in states:
                    state.chat = self._create_failed_chat(state, error, batch_mode)

            else:
                # Process each inbound message and individual errors
                for inbound, state in zip(inbounds, pending_states):
                    try:
                        # Process for parsing callbacks, etc.
                        state.messages = state.processor.send(inbound.message)
                    except StopIteration as stop:
                        # StopIteration implies we are done and the chat is good to go
                        outputs = t.cast(list[Message], stop.value)
                        state.chat = self._create_chat(state, outputs, inbound, batch_mode)
                    except MessagesExhaustedMaxRoundsError as exhausted:
                        if on_failed == "raise":
                            raise
                        # exhausted.messages holds the current messages when the error occured,
                        # so we'll pass them into the chat as the last generated messages.
                        span.set_attribute("failed", True)
                        span.set_attribute("error", exhausted)
                        state.chat = self._create_chat(
                            state, exhausted.messages, inbound, batch_mode, failed=True, error=exhausted
                        )
                    except Exception as error:
                        # Check to see if we should be handling any specific errors
                        # and gracefully marking the chat as failed instead of raising (.catch)
                        if (
                            on_failed == "raise"
                            or not any(isinstance(error, t) for t in self.errors_to_fail_on)
                            or any(isinstance(error, t) for t in self.errors_to_exclude)
                        ):
                            raise
                        span.set_attribute("failed", True)
                        span.set_attribute("error", error)
                        state.chat = self._create_chat(state, [], inbound, batch_mode, failed=True, error=error)

            pending_states = [s for s in pending_states if s.chat is None]
            completed_states = [s for s in states if s.chat is not None and not s.watched]

            if not completed_states:
                continue

            # We want to deliver chats to the watch callback as soon as possible, so we'll
            # track whether we've already done so and only deliver new chats.

            await self._watch_callback([s.chat for s in completed_states if s.chat is not None])
            for state in completed_states:
                state.watched = True

        chats = await self._post_run([s.chat for s in states if s.chat is not None], on_failed)
        span.set_attribute("chats", chats)
        return chats

    # Single messages

    async def run(self, *, allow_failed: bool = False, on_failed: FailMode | None = None) -> Chat:
        """
        Execute the generation process to produce the final chat.

        Parameters:
            allow_failed: Ignore any errors and potentially
                return the chat in a failed state.
            on_failed: The behavior when a message fails to generate.
                (this is used as an alternative to allow_failed)

        Returns:
            The generated Chat.
        """
        if on_failed is None:
            on_failed = "include" if allow_failed else self.on_failed

        if on_failed == "skip":
            raise ValueError(
                "Cannot use 'skip' mode with single message generation (pass allow_failed=True or on_failed='include'/'raise')"
            )

        on_failed = on_failed or self.on_failed
        states = self._initialize_states(1)

        with tracer.span(
            f"Chat with {self.generator.to_identifier()}",
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            return (await self._run(span, states, on_failed))[0]

    __call__ = run

    # Many messages

    async def run_many(
        self,
        count: int,
        *,
        params: t.Sequence[t.Optional[GenerateParams]] | None = None,
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process multiple times with the same inputs.

        Parameters:
            count: The number of times to execute the generation process.
            params: A sequence of parameters to be used for each execution.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """
        on_failed = on_failed or self.on_failed
        states = self._initialize_states(count, params)

        with tracer.span(
            f"Chat with {self.generator.to_identifier()} (x{count})",
            count=count,
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            return await self._run(span, states, on_failed)

    # Batch messages

    async def run_batch(
        self,
        many: t.Sequence[t.Sequence[Message]]
        | t.Sequence[Message]
        | t.Sequence[MessageDict]
        | t.Sequence[str]
        | MessageDict
        | str,
        params: t.Sequence[t.Optional[GenerateParams]] | None = None,
        *,
        on_failed: FailMode | None = None,
    ) -> ChatList:
        """
        Executes the generation process accross multiple input messages.

        Note:
            Anything already in this chat pipeline will be prepended to the input messages.

        Parameters:
            many: A sequence of sequences of messages to be generated.
            params: A sequence of parameters to be used for each set of messages.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """
        on_failed = on_failed or self.on_failed
        states = self._initialize_batch_states(many, params)

        with tracer.span(
            f"Chat batch with {self.generator.to_identifier()} ({len(states)})",
            count=len(states),
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            return await self._run(span, states, on_failed, batch_mode=True)

    # Generator iteration

    async def run_over(
        self, *generators: Generator | str, include_original: bool = True, on_failed: FailMode | None = None
    ) -> ChatList:
        """
        Executes the generation process across multiple generators.

        For each generator, this pipeline is cloned and the generator is replaced
        before the run call. All callbacks and parameters are preserved.

        Parameters:
            *generators: A sequence of generators to be used for the generation process.
            include_original: Whether to include the original generator in the list of runs.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Chats.
        """
        on_failed = on_failed or self.on_failed

        _generators: list[Generator] = [g if isinstance(g, Generator) else get_generator(g) for g in generators]
        if include_original:
            _generators.append(self.generator)

        coros: list[t.Coroutine[t.Any, t.Any, Chat]] = []
        for generator in _generators:
            sub = self.clone()
            sub.generator = generator
            coros.append(sub.run(allow_failed=(on_failed != "raise")))

        with tracer.span(f"Chat over {len(coros)} generators", count=len(coros)):
            chats = await asyncio.gather(*coros)
            return await self._post_run(chats, on_failed)

    # Prompt functions

    def prompt(self, func: t.Callable[P, t.Coroutine[None, None, R]]) -> Prompt[P, R]:
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

    async def run_prompt(self, prompt: Prompt[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Calls [rigging.prompt.Prompt.run][] with this pipeline.

        Warning:
            This method is deprecated and will be removed in a future release.
            Use [Prompt.bind(pipeline)][rigging.prompt.Prompt.bind] instead.
        """
        warnings.warn("run_prompt is deprecated, use Prompt.bind(pipeline) instead", DeprecationWarning, stacklevel=2)
        return await prompt.bind(self)(*args, **kwargs)

    async def run_prompt_many(self, prompt: Prompt[P, R], count: int, /, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """
        Calls [rigging.prompt.Prompt.run_many][] with this pipeline.

        Warning:
            This method is deprecated and will be removed in a future release.
            Use [Prompt.bind_many(pipeline)][rigging.prompt.Prompt.bind_many] instead.
        """
        warnings.warn(
            "run_prompt_many is deprecated, use Prompt.bind_many(pipeline) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return await prompt.bind_many(self)(count, *args, **kwargs)

    async def run_prompt_over(
        self, prompt: Prompt[P, R], generators: t.Sequence[Generator | str], /, *args: P.args, **kwargs: P.kwargs
    ) -> list[R]:
        """
        Calls [rigging.prompt.Prompt.run_over][] with this pipeline.

        Warning:
            This method is deprecated and will be removed in a future release.
            Use [Prompt.bind_over(pipeline)][rigging.prompt.Prompt.bind_over] instead.
        """
        warnings.warn(
            "run_prompt_over is deprecated, use Prompt.bind_over(pipeline) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return await prompt.bind_over(self)(generators, *args, **kwargs)
