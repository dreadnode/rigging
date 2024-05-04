"""
Chats are used pre and post generation to hold messages.

They are the primary way to interact with the generator.
"""

import asyncio
import typing as t
from copy import deepcopy
from datetime import datetime
from uuid import UUID, uuid4

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
)

from rigging.error import ExhaustedMaxRoundsError
from rigging.message import Message, MessageDict, Messages
from rigging.model import (
    Model,
    ModelT,
    SystemErrorModel,
    ValidationErrorModel,
)
from rigging.prompt import system_tool_extension
from rigging.tool import Tool, ToolCalls, ToolDescriptionList, ToolResult, ToolResults

if t.TYPE_CHECKING:
    from rigging.generator import GenerateParams, Generator

DEFAULT_MAX_ROUNDS = 5


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

    pending: t.Optional["PendingChat"] = Field(None, exclude=True, repr=False)
    """The pending chat associated with the chat."""

    @computed_field(repr=False)
    def generator_id(self) -> str | None:
        """The identifier of the generator used to create the chat"""
        if self.pending is not None:
            return self.pending.generator.to_identifier(self.pending.params)
        return None

    def __init__(
        self,
        messages: Messages,
        generated: Messages | None = None,
        pending: t.Optional["PendingChat"] = None,
        **kwargs: t.Any,
    ):
        """
        Initialize a Chat object.

        Args:
            messages: The messages for the chat.
            generated: The next messages for the chat.
            pending: The pending chat.
            **kwargs: Additional keyword arguments (typically used for deserialization)
        """
        from rigging.generator import get_generator

        if "generator_id" in kwargs and pending is None:
            generator = get_generator(kwargs.pop("generator_id"))
            pending = generator.chat(messages)

        super().__init__(
            messages=Message.fit_as_list(messages),
            generated=Message.fit_as_list(generated) if generated is not None else [],
            pending=pending,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.messages) + len(self.generated)

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
        """Alias for .generated[-1]"""
        return self.generated[-1]

    @property
    def conversation(self) -> str:
        """Returns a string representation of the chat."""
        return "\n\n".join([str(m) for m in self.all])

    def meta(self, **kwargs: t.Any) -> "Chat":
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated chat object.
        """
        self.metadata.update(kwargs)
        return self

    def restart(self, *, generator: t.Optional["Generator"] = None, include_all: bool = False) -> "PendingChat":
        """
        Attempt to convert back to a PendingChat for further generation.

        Args:
            generator: The generator to use for the restarted chat. Otherwise
                the generator from the original PendingChat will be used.
            include_all: Whether to include the next messages in the restarted chat.

        Returns:
            The restarted chat.

        Raises:
            ValueError: If the chat was not created with a PendingChat and no generator is provided.
        """

        messages = self.all if include_all else self.messages
        if generator is not None:
            return generator.chat(messages)
        elif self.pending is None:
            raise ValueError("Cannot restart chat that was not created with a PendingChat")
        return PendingChat(self.pending.generator, messages, self.pending.params)

    def fork(
        self,
        messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | MessageDict | str,
        *,
        include_all: bool = False,
    ) -> "PendingChat":
        """
        Forks the chat by creating calling [rigging.chat.Chat.restart][] and appending the specified messages.

        Args:
            messages:
                The messages to be added to the new `PendingChat` instance.
            include_all: Whether to include the next messages in the restarted chat.

        Returns:
            A new instance of `PendingChat` with the specified messages added.

        """
        return self.restart(include_all=include_all).add(messages)

    def continue_(self, messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | str) -> "PendingChat":
        """Alias for the [rigging.chat.Chat.fork][] with `include_all=True`."""
        return self.fork(messages, include_all=True)

    def clone(self, *, only_messages: bool = False) -> "Chat":
        """Creates a deep copy of the chat."""
        new = Chat(
            [m.model_copy() for m in self.messages],
            [m.model_copy() for m in self.generated],
            self.pending,
        )
        if not only_messages:
            new.metadata = deepcopy(self.metadata)
        return new

    def apply(self, **kwargs: str) -> "Chat":
        """
        Calls [rigging.message.Message.apply][] on the last message in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The modified Chat object.
        """
        self.last.apply(**kwargs)
        return self

    def apply_to_all(self, **kwargs: str) -> "Chat":
        """
        Calls [rigging.message.Message.apply][] on all messages in the chat with the given keyword arguments.

        Args:
            **kwargs: The string mapping of replacements.

        Returns:
            The modified chat object.
        """
        for message in self.all:
            message.apply(**kwargs)
        return self

    def strip(self, model_type: type[Model], fail_on_missing: bool = False) -> "Chat":
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


# Passed the next message, returns whether or not to continue
# and an optional list of messages to append before continuing
UntilMessageCallback = t.Callable[[Message], tuple[bool, list[Message]]]

ThenChatCallback = t.Callable[[Chat], Chat | None]


class PendingChat:
    """
    Represents a pending chat that can be modified and executed.
    """

    def __init__(
        self, generator: "Generator", messages: t.Sequence[Message], params: t.Optional["GenerateParams"] = None
    ):
        self.generator: "Generator" = generator
        """The generator object responsible for generating the chat."""
        self.chat: Chat = Chat(messages, pending=self)
        """The chat object representing the conversation."""
        self.params = params
        """The parameters for generating messages."""
        self.metadata: dict[str, t.Any] = {}
        """Additional metadata associated with the chat."""

        # (callback, attempt_recovery, drop_dialog, max_rounds)
        self.until_callbacks: list[tuple[UntilMessageCallback, bool, bool, int]] = []
        self.until_types: list[type[Model]] = []
        self.until_tools: list[Tool] = []
        self.inject_tool_prompt: bool = True
        self.force_tool: bool = False
        self.then_callbacks: list[ThenChatCallback] = []

    def overload(self, **kwargs: t.Any) -> "PendingChat":
        """
        Overloads the current chat with the given parameters.

        This is a convenience method for calling `with_params(GenerateParams(**kwargs))`.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            **kwargs: Keyword arguments representing the parameters to be overloaded.

        Returns:
            A new instance of PendingChat with the overloaded parameters.
        """
        from rigging.generator import GenerateParams

        return self.with_params(GenerateParams(**kwargs))

    def with_params(self, params: "GenerateParams") -> "PendingChat":
        """
        Sets the generation parameter overloads for the chat.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            params: The parameters to set for the chat.

        Returns:
            A new instance of PendingChat with the updated parameters.
        """
        if self.params is not None:
            new = self.clone()
            new.params = params
            return new

        self.params = params
        return self

    def add(
        self, messages: t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | str
    ) -> "PendingChat":
        """
        Appends new message(s) to the internal chat before generation.

        Note:
            If the last message in the chat is the same role as the first new message,
            the content will be appended. instead of a new message being created.

        Args:
            messages: The messages to be added to the chat. It can be a single message or a sequence of messages.

        Returns:
            The updated PendingChat object.
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
    ) -> "PendingChat":
        """
        Creates a new instance of `PendingChat` by forking the current chat and adding the specified messages.

        This is a convenience method for calling `clone().add(messages)`.

        Args:
            messages: A sequence of messages or a single message to be added to the new chat.

        Returns:
            A new instance the pending chat with the specified messages added.
        """
        return self.clone().add(messages)

    def clone(self, *, only_messages: bool = False) -> "PendingChat":
        """
        Creates a clone of the current `PendingChat` instance.

        Args:
            only_messages: If True, only the messages will be cloned.
                If False (default), the entire `PendingChat` instance will be cloned
                including until callbacks, types, and tools.

        Returns:
            A new instance of `PendingChat` that is a clone of the current instance.
        """
        new = PendingChat(self.generator, [], self.params)
        new.chat = self.chat.clone()
        if not only_messages:
            new.until_callbacks = self.until_callbacks.copy()
            new.until_types = self.until_types.copy()
            new.until_tools = self.until_tools.copy()
            new.inject_tool_prompt = self.inject_tool_prompt
            new.force_tool = self.force_tool
            new.metadata = deepcopy(self.metadata)
        return new

    def meta(self, **kwargs: t.Any) -> "PendingChat":
        """
        Updates the metadata of the chat with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated chat object.
        """
        self.metadata.update(kwargs)
        return self

    def then(self, callback: ThenChatCallback) -> "PendingChat":
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            Returning a Chat object from the callback will replace the current chat.
            for the remainder of the callbacks + return value of `run()`.

        ```
        def process(chat: Chat) -> Chat | None:
            ...

        pending.then(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the chat.
        """
        self.then_callbacks.append(callback)
        return self

    def apply(self, **kwargs: str) -> "PendingChat":
        """
        Clones this pending chat and calls [rigging.chat.Chat.apply][] with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be applied to the chat.

        Returns:
            A new instance of PendingChat with the applied arguments.
        """
        new = self.clone()
        new.chat.apply(**kwargs)
        return new

    def apply_to_all(self, **kwargs: str) -> "PendingChat":
        """
        Clones this pending chat and calls [rigging.chat.Chat.apply_to_all][] with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be applied to the chat.

        Returns:
            A new instance of PendingChat with the applied arguments.
        """
        new = self.clone()
        new.chat.apply_to_all(**kwargs)
        return new

    def until(
        self,
        callback: UntilMessageCallback,
        *,
        attempt_recovery: bool = False,
        drop_dialog: bool = True,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> "PendingChat":
        """
        Registers a callback to participate in validating the generation process.

        ```python
        # Takes the next message being generated, and returns whether or not to continue
        # generating new messages in addition to a list of messages to append before continuing

        def callback(message: Message) -> tuple[bool, list[Message]]:
            if is_valid(message):
                return (False, [message])
            else:
                return (True, [message, ...])

        pending.until(callback).run()
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
        tool: Tool | t.Sequence[Tool],
        *,
        force: bool = False,
        attempt_recovery: bool = True,
        drop_dialog: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        inject_prompt: bool | None = None,
    ) -> "PendingChat":
        """
        Adds a tool or a sequence of tools to participate in the generation process.

        Args:
            tool: The tool or sequence of tools to be added.
            force: Whether to force the use of the tool(s) at least once.
            attempt_recovery: Whether to attempt recovery if the tool(s) fail by providing
                validation feedback to the model before the next round.
            drop_dialog: Whether to drop the intermediate dialog of recovery efforts
                before returning the final chat to the caller.
            max_rounds: The maximum number of rounds to attempt recovery.
            inject_prompt: Whether to inject the tool guidance prompt into a
                system message.and will override self.inject_tool_prompt if provided.

        Returns:
            The updated PendingChat object.

        """
        self.until_tools += tool if isinstance(tool, t.Sequence) else [tool]
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
    ) -> "PendingChat":
        """
        Adds the specified types to the list of types which should successfully parse
        before the generation process completes.

        Args:
            *types: The type or types of models to wait for.
            attempt_recovery: Whether to attempt recovery if parsing fails by providing
                validation feedback to the model before the next round.
            drop_dialog: Whether to drop the intermediate dialog of recovery efforts
                before returning the final chat to the caller.
            max_rounds: The maximum number of rounds to try to parse
                successfully.

        Returns:
            The updated PendingChat object.
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
            generated.append(Message.from_model(ValidationErrorModel(content=e)))
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
                    ValidationErrorModel(content=e),
                    suffix="Rewrite your entire message with all the required elements.",
                )
            )
        except Exception as e:
            should_continue = True
            generated.append(
                Message.from_model(
                    SystemErrorModel(content=e), suffix="Rewrite your entire message with all the required elements."
                )
            )

        return (should_continue, generated)

    def _until(
        self,
        messages: list[Message],
        callback: UntilMessageCallback,
        attempt_recovery: bool,
        drop_dialog: bool,
        max_rounds: int,
    ) -> t.Generator[list[Message], Message, list[Message]]:
        should_continue, step_messages = callback(messages[-1])
        if not should_continue:
            return step_messages

        running_messages = step_messages if attempt_recovery else []

        for _ in range(max_rounds):
            logger.trace(
                f"_until({callback.__name__}) round {_ + 1}/{max_rounds} (attempt_recovery={attempt_recovery})"
            )
            next_message = yield messages[:-1] + running_messages
            should_continue, step_messages = callback(next_message)
            logger.trace(f" |- returned {should_continue} with {len(step_messages)} new messages)")

            if not should_continue:
                return step_messages if drop_dialog else running_messages + step_messages

            if attempt_recovery:
                running_messages += step_messages

        logger.warning(f"Exhausted max rounds ({max_rounds})")
        raise ExhaustedMaxRoundsError(max_rounds)

    def _then(self, chat: Chat) -> Chat:
        # TODO: Adding async support here would be nice
        for callback in self.then_callbacks:
            chat = callback(chat) or chat
        return chat

    def _execute(self) -> t.Generator[list[Message], Message, list[Message]]:
        # TODO: Much like the PendingCompletion code, it's opaque
        # exactly how multiple callbacks should be blended together
        # when generating. I think we should look at limiting it to
        # one callback in total, but I'll leave the behavior as is
        # for now with the knowledge that behavior might be a bit
        # unpredictable.

        if self.until_tools:
            # TODO: This can cause issues when certain APIs do not return
            # the stop sequence as part of the response. This behavior
            # seems like a larger issue than the model continuining after
            # requesting a tool call, so we'll remove it for now.
            #
            # self.params.stop = [ToolCalls.xml_end_tag()]

            if self.inject_tool_prompt:
                self.chat.inject_tool_prompt(self.until_tools)
                self.inject_tool_prompt = False

        first_message = yield self.chat.all

        new_messages = [first_message]
        for callback, reset_between, drop_internal, max_rounds in self.until_callbacks:
            generated = yield from self._until(
                self.chat.all + new_messages, callback, reset_between, drop_internal, max_rounds
            )
            new_messages = new_messages[:-1] + generated

        return new_messages

    @t.overload
    def run(self, count: t.Literal[None] = None) -> Chat:
        ...

    @t.overload
    def run(self, count: int) -> list[Chat]:
        ...

    def run(self, count: int | None = None) -> Chat | list[Chat]:
        """
        Execute the generation process to produce the final chat.

        If `count` is provided, `run_many` will be called instead.

        Args:
            count: The number of times to generate using the same inputs.

        Returns:
            Chat | list[Chat]: The chat object or a list of chat objects, depending on the value of `count`.
        """

        if count is not None:
            return self.run_many(count)

        executor = self._execute()
        outbound = next(executor)

        try:
            while True:
                inbound = self.generator.generate_message(outbound, self.params)
                outbound = executor.send(inbound)
        except StopIteration as stop:
            outbound = t.cast(list[Message], stop.value)

        return self._then(Chat(self.chat.all, outbound, pending=self, metadata=self.metadata))

    def run_many(self, count: int) -> list[Chat]:
        """
        Executes the generation process multiple times with the same inputs.

        Parameters:
            count: The number of times to execute the generation process.

        Returns:
            list[ChatA list of Chat objects representing the results of each execution.
        """
        return [self.run() for _ in range(count)]

    __call__ = run

    @t.overload
    async def arun(self, count: t.Literal[None] = None) -> Chat:
        ...

    @t.overload
    async def arun(self, count: int) -> list[Chat]:
        ...

    async def arun(self, count: int | None = None) -> Chat | list[Chat]:
        """async variant of the [rigging.chat.PendingChat.run][] method."""
        if count is not None:
            return await self.arun_many(count)

        executor = self._execute()
        outbound = next(executor)

        try:
            while True:
                inbound = await self.generator.agenerate_message(outbound, self.params)
                outbound = executor.send(inbound)
        except StopIteration as stop:
            outbound = t.cast(list[Message], stop.value)

        return self._then(Chat(self.chat.all, outbound, pending=self, metadata=self.metadata))

    async def arun_many(self, count: int) -> list[Chat]:
        """async variant of the [rigging.chat.PendingChat.run_many][] method."""
        chats = await asyncio.gather(*[self.arun() for _ in range(count)])
        return [self._then(chat) for chat in chats]
