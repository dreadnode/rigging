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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from rigging.error import MessagesExhaustedMaxRoundsError
from rigging.generator import GenerateParams, Generator, get_generator
from rigging.generator.base import StopReason, Usage  # noqa: TCH001
from rigging.message import Message, MessageDict, Messages
from rigging.model import Model, ModelT, SystemErrorModel, ValidationErrorModel
from rigging.tool import Tool, ToolCalls, ToolDescriptionList, ToolResult, ToolResults, system_tool_extension

if t.TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

    from rigging.data import ElasticOpType
    from rigging.prompt import P, Prompt, R

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

    error: t.Optional[Exception] = Field(None, exclude=True, repr=False)
    """Holds any exception that was caught during the generation pipeline."""
    failed: bool = Field(False, exclude=False, repr=False)
    """
    Indicates whether conditions during generation were not met.
    This is typically used for graceful error handling when parsing.
    """

    @computed_field(repr=False)  # type: ignore[misc]
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
        return [t.cast(MessageDict, m.model_dump(include={"role", "content"})) for m in self.all]

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
    def __call__(self, message: Message) -> tuple[bool, list[Message]]:
        """
        Passed the next message, returns whether or not to continue and an
        optional list of messages to append before continuing.
        """
        ...


@runtime_checkable
class ThenChatCallback(t.Protocol):
    async def __call__(self, chat: Chat) -> Chat | None:
        """
        Passed a finalized chat to process and can return a new chat to replace it.
        """
        ...


@runtime_checkable
class MapChatCallback(t.Protocol):
    async def __call__(self, chats: list[Chat]) -> list[Chat]:
        """
        Passed a finalized chats to process. Can replace chats in the pipeline by returning
        a new chat object.
        """
        ...


@runtime_checkable
class WatchChatCallback(t.Protocol):
    async def __call__(self, chats: list[Chat]) -> None:
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
        self.on_failed: FailMode = "raise"
        """How to handle failures in the pipeline unless overriden in calls."""

        # (callback, attempt_recovery, drop_dialog, max_rounds)
        self.until_callbacks: list[tuple[UntilMessageCallback, bool, bool, int]] = []
        self.until_types: list[type[Model]] = []
        self.until_tools: list[Tool] = []
        self.inject_tool_prompt: bool = True
        self.force_tool: bool = False
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

    def catch(self, *errors: type[Exception], on_failed: FailMode | None = None) -> ChatPipeline:
        """
        Adds exceptions to catch during generation when including or skipping failures.

        Args:
            *errors: The exception types to catch.
            on_failed: How to handle failures in the pipeline unless overriden in calls.

        Returns:
            The updated ChatPipeline object.
        """
        self.errors_to_fail_on.update(errors)
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
            new.until_tools = self.until_tools.copy()
            new.inject_tool_prompt = self.inject_tool_prompt
            new.force_tool = self.force_tool
            new.metadata = deepcopy(self.metadata)
            new.then_callbacks = self.then_callbacks.copy()
            new.map_callbacks = self.map_callbacks.copy()
            new.on_failed = self.on_failed
            new.errors_to_fail_on = self.errors_to_fail_on.copy()
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
            raise TypeError(f"Callback '{callback.__name__}' must be an async function")  # type: ignore

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
            raise TypeError(f"Callback '{callback.__name__}' must be an async function")  # type: ignore

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

    def using(
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
        self.until_tools += tools
        self.inject_tool_prompt = inject_prompt or self.inject_tool_prompt
        self.force_tool = force
        if next((c for c in self.until_callbacks if c[0] == self._until_tools_callback), None) is None:
            self.until_callbacks.append(
                (
                    self._until_tools_callback,
                    attempt_recovery,
                    drop_dialog,
                    max_rounds,
                )
            )
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

    def _until_tools_callback(self, message: Message) -> tuple[bool, list[Message]]:
        generated: list[Message] = [message]

        try:
            tool_calls = message.try_parse(ToolCalls)
        except ValidationError as e:
            generated.append(Message.from_model(ValidationErrorModel(content=str(e))))
            return (True, generated)

        if tool_calls is None:
            if self.force_tool:
                logger.debug("No tool calls or types, returning error")
                generated.append(Message.from_model(SystemErrorModel(content="You must use a tool")))
            else:
                logger.debug("No tool calls or types, returning message")
            return (self.force_tool, generated)

        self.force_tool = False

        tool_results: list[ToolResult] = []
        errors: list[SystemErrorModel] = []
        for call in tool_calls.calls:
            if call.tool not in [tool.name for tool in self.until_tools]:
                errors.append(SystemErrorModel(content=f"Tool '{call.tool}' does not exist"))
                continue

            tool = next(t for t in self.until_tools if t.name == call.tool)
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

        for _ in range(max_rounds):
            logger.trace(
                f"_until({callback.__call__.__name__}) round {_ + 1}/{max_rounds} (attempt_recovery={attempt_recovery})"
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
        for callback, reset_between, drop_internal, max_rounds in self.until_callbacks:
            generated = yield from self._until(new_messages[-1], callback, reset_between, drop_internal, max_rounds)
            new_messages = new_messages[:-1] + generated
        return new_messages

    async def _watch_callback(self, chats: list[Chat]) -> None:
        # Given that these watch callbacks don't return a value,
        # we should be safe to run them internally.

        coros = [callback(chats) for callback in self.watch_callbacks]
        await asyncio.gather(*coros)

    async def _post_run(self, chats: list[Chat]) -> ChatList:
        # These have to be sequenced to support the concept of
        # a pipeline where future then/map calls can depend on
        # previous calls being ran.

        for map_callback in self.map_callbacks:
            chats = await map_callback(chats)

        for then_callback in self.then_callbacks:
            coros = [then_callback(chat) for chat in chats]
            new_chats = await asyncio.gather(*coros)
            chats = [new or chat for new, chat in zip(new_chats, chats)]

        return ChatList(chats)

    def _pre_run(self) -> None:
        if self.until_tools:
            if self.inject_tool_prompt:
                self.chat.inject_tool_prompt(self.until_tools)
                self.inject_tool_prompt = False

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

    def _fit_many(
        self,
        count: int,
        many: t.Sequence[t.Sequence[Message]] | t.Sequence[Message] | t.Sequence[MessageDict] | t.Sequence[str],
    ) -> list[list[Message]]:
        many = [Message.fit_as_list(m) for m in many]
        if len(many) < count:
            if len(many) != 1:
                raise ValueError(f"Can't fit many of length {len(many)} to {count}")
            many = many * count
        return many

    # TODO: There is an embarrassing amount of code duplication here
    # between the async and non-async methods, batch and many, etc.

    # Single messages

    async def run(self, *, allow_failed: bool = False) -> Chat:
        """
        Execute the generation process to produce the final chat.

        Parameters:
            allow_failed: Ignore any errors and potentially
                return the chat in a failed state.

        Returns:
            The generated Chat.
        """
        chats = await self.run_many(1, on_failed="include" if allow_failed else "raise")
        return chats[0]

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
        states: list[RunState] = [RunState([], [], p, self._process()) for p in self._fit_params(count, params)]
        _ = [next(state.processor) for state in states]

        pending_states = states
        while pending_states:
            inbounds = await self.generator.generate_messages(
                [self.chat.all + s.messages for s in pending_states], [s.params for s in pending_states]
            )

            for inbound, state in zip(inbounds, pending_states):
                outputs: list[Message] = []
                failed: bool = False
                error: Exception | None = None

                try:
                    state.messages = state.processor.send(inbound.message)
                    continue
                except StopIteration as stop:
                    outputs = t.cast(list[Message], stop.value)
                except MessagesExhaustedMaxRoundsError as exhausted:
                    if on_failed == "raise":
                        raise
                    failed = True
                    outputs = exhausted.messages
                    error = exhausted
                except Exception as e:
                    if on_failed == "raise" or not any(isinstance(e, t) for t in self.errors_to_fail_on):
                        raise
                    failed = True
                    error = e

                state.chat = Chat(
                    self.chat.all,
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

            pending_states = [s for s in pending_states if s.chat is None]
            to_watch_states = [s for s in states if s.chat is not None and not s.watched]

            await self._watch_callback([s.chat for s in to_watch_states if s.chat is not None])

            for state in to_watch_states:
                state.watched = True

        if on_failed == "skip":
            chats = [s.chat for s in states if s.chat is not None and not s.chat.failed]
        else:
            chats = [s.chat for s in states if s.chat is not None]

        return await self._post_run(chats)

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

        if isinstance(many, dict) or isinstance(many, str):  # Some strange typechecking here
            many = t.cast(t.Union[t.Sequence[str], t.Sequence[MessageDict]], [many])

        count = max(len(many), len(params) if params is not None else 0)
        many = self._fit_many(count, many)
        params = self._fit_params(count, params)

        states: list[RunState] = [RunState(self.chat.all + m, [], p, self._process()) for m, p in zip(many, params)]
        _ = [next(state.processor) for state in states]

        pending_states = states
        while pending_states:
            inbounds = await self.generator.generate_messages(
                [s.inputs + s.messages for s in pending_states],
                [s.params for s in pending_states],
            )

            for inbound, state in zip(inbounds, pending_states):
                outputs: list[Message] = []
                failed: bool = False
                error: Exception | None = None

                try:
                    state.messages = state.processor.send(inbound.message)
                    continue
                except StopIteration as stop:
                    outputs = t.cast(list[Message], stop.value)
                except MessagesExhaustedMaxRoundsError as exhausted:
                    if on_failed == "raise":
                        raise
                    failed = True
                    outputs = exhausted.messages
                    error = exhausted
                except Exception as e:
                    if on_failed == "raise" or not any(isinstance(e, t) for t in self.errors_to_fail_on):
                        raise
                    failed = True
                    error = e

                state.chat = Chat(
                    state.inputs,
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

            pending_states = [s for s in pending_states if s.chat is None]
            to_watch_states = [s for s in states if s.chat is not None and not s.watched]

            await self._watch_callback([s.chat for s in to_watch_states if s.chat is not None])

            for state in to_watch_states:
                state.watched = True

        if on_failed == "skip":
            chats = [s.chat for s in states if s.chat is not None and not s.chat.failed]
        else:
            chats = [s.chat for s in states if s.chat is not None]

        return await self._post_run(chats)

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

        chats = await asyncio.gather(*coros)

        if on_failed == "skip":
            chats = [c for c in chats if not c.failed]

        return ChatList(chats)

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
