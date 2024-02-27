import typing as t

from loguru import logger
from pydantic import ValidationError

from rigging.error import ExhaustedMaxRoundsError
from rigging.message import Message, MessageDict, Messages
from rigging.model import (
    Model,
    ModelGeneric,
    SystemErrorModel,
    ValidationErrorModel,
)
from rigging.prompt import system_tool_extension
from rigging.tool import Tool, ToolCalls, ToolDescriptionList, ToolResult, ToolResults

if t.TYPE_CHECKING:
    from rigging.generator import GenerateParams, Generator

DEFAULT_MAX_ROUNDS = 5


class Chat:
    def __init__(
        self,
        messages: Messages,
        next_messages: Messages | None = None,
        pending: t.Optional["PendingChat"] = None,
    ):
        self.messages: list[Message] = Message.fit_list(messages)
        self.next_messages: list[Message] = []
        if next_messages is not None:
            self.next_messages = Message.fit_list(next_messages)
        self.pending_chat = pending

    def __len__(self) -> int:
        return len(self.messages) + len(self.next_messages)

    @property
    def all(self) -> list[Message]:
        return self.messages + self.next_messages

    @property
    def prev(self) -> list[Message]:
        return self.messages

    @property
    def next(self) -> list[Message]:
        return self.next_messages

    @property
    def last(self) -> Message:
        return self.next_messages[-1]

    @property
    def json(self) -> list[MessageDict]:
        return [t.cast(MessageDict, message.model_dump()) for message in self.all]

    def restart(self) -> "PendingChat":
        if self.pending_chat is None:
            raise ValueError("Cannot restart chat that was not created with a PendingChat")
        return PendingChat(self.pending_chat.generator, self.messages, self.pending_chat.params)

    @t.overload
    def continue_(self, messages: t.Sequence[MessageDict]) -> "PendingChat":
        ...

    @t.overload
    def continue_(self, messages: MessageDict) -> "PendingChat":
        ...

    @t.overload
    def continue_(self, messages: t.Sequence[Message]) -> "PendingChat":
        ...

    @t.overload
    def continue_(self, messages: Message) -> "PendingChat":
        ...

    def continue_(
        self, messages: t.Sequence[Message] | t.Sequence[MessageDict] | Message | MessageDict
    ) -> "PendingChat":
        if self.pending_chat is None:
            raise ValueError("Cannot continue chat that was not created with a PendingChat")

        messages_list: list[Message] = (
            Message.fit_list(messages) if isinstance(messages, t.Sequence) else [Message.fit(messages)]
        )
        return PendingChat(self.pending_chat.generator, self.all + messages_list, self.pending_chat.params)

    def clone(self) -> "Chat":
        return Chat(
            [m.model_copy() for m in self.messages], [m.model_copy() for m in self.next_messages], self.pending_chat
        )

    def apply(self, **kwargs: str) -> "Chat":
        self.messages[-1].apply(**kwargs)
        return self

    def apply_to_all(self, **kwargs: str) -> "Chat":
        for message in self.messages:
            message.apply(**kwargs)
        return self

    def strip(self, model_type: type[Model], fail_on_missing: bool = False) -> "Chat":
        new = self.clone()
        for message in new.all:
            message.strip(model_type, fail_on_missing)
        return new

    def inject_system_content(self, content: str) -> Message:
        if len(self.messages) == 0 or self.messages[0].role != "system":
            self.messages.insert(0, Message(role="system", content=content))
        elif self.messages[0].role == "system":
            self.messages[0].content += "\n\n" + content
        return self.messages[0]

    def inject_tool_prompt(self, tools: t.Sequence[Tool]) -> None:
        call_format = ToolCalls.xml_example()
        tool_description_list = ToolDescriptionList(tools=[t.get_description() for t in tools])
        tool_system_prompt = system_tool_extension(call_format, tool_description_list.to_pretty_xml())
        self.inject_system_content(tool_system_prompt)


# Passed the next message, returns whether or not to continue
# and an optional list of messages to append before continuing
UntilCallback = t.Callable[[Message], tuple[bool, list[Message]]]


class PendingChat:
    def __init__(self, generator: "Generator", messages: t.Sequence[Message], params: "GenerateParams"):
        self.generator: "Generator" = generator
        self.chat: Chat = Chat(messages, pending=self)

        # (callback, attempt_recovery, drop_dialog, max_rounds)
        self.until_callbacks: list[tuple[UntilCallback, bool, bool, int]] = []
        self.until_types: list[type[Model]] = []
        self.until_tools: list[Tool] = []
        self.inject_tool_prompt: bool = True

        self.params = params

    def overload(self, **kwargs: t.Any) -> "PendingChat":
        from rigging.generator import GenerateParams

        return self.with_params(GenerateParams(**kwargs))

    def with_params(self, params: "GenerateParams") -> "PendingChat":
        if params is not None:
            self.params = params
        return self

    def clone(self) -> "PendingChat":
        new = PendingChat(self.generator, [], self.params)
        new.chat = self.chat.clone()
        return new

    def apply(self, **kwargs: str) -> "PendingChat":
        new = self.clone()
        new.chat.apply(**kwargs)
        return new

    def apply_to_all(self, **kwargs: str) -> "PendingChat":
        new = self.clone()
        new.chat.apply_to_all(**kwargs)
        return new

    def until(
        self,
        callback: UntilCallback,
        *,
        attempt_recovery: bool = False,
        drop_dialog: bool = True,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> "PendingChat":
        self.until_callbacks.append((callback, attempt_recovery, drop_dialog, max_rounds))
        return self

    def using(
        self,
        tool: Tool | t.Sequence[Tool],
        *,
        attempt_recovery: bool = True,
        drop_dialog: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        inject_prompt: bool | None = None,
    ) -> "PendingChat":
        self.until_tools += tool if isinstance(tool, t.Sequence) else [tool]
        self.inject_tool_prompt = inject_prompt or self.inject_tool_prompt
        if next((c for c in self.until_callbacks if c[0] == self._until_tools_callback), None) is None:
            self.until_callbacks.append((self._until_tools_callback, attempt_recovery, drop_dialog, max_rounds))
        return self

    def until_parsed_as(
        self,
        types: type[ModelGeneric] | t.Sequence[type[ModelGeneric]],
        *,
        attempt_recovery: bool = False,
        drop_dialog: bool = True,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> "PendingChat":
        self.until_types += types if isinstance(types, t.Sequence) else [types]
        if next((c for c in self.until_callbacks if c[0] == self._until_parse_callback), None) is None:
            self.until_callbacks.append((self._until_parse_callback, attempt_recovery, drop_dialog, max_rounds))

        return self

    def _until_tools_callback(self, message: Message) -> tuple[bool, list[Message]]:
        next_messages: list[Message] = [message]

        try:
            tool_calls = message.try_parse(ToolCalls)
        except ValidationError as e:
            next_messages.append(Message.from_model(ValidationErrorModel(content=e)))
            return (True, next_messages)

        if tool_calls is None:
            logger.debug("No tool calls or types, returning message")
            return (False, next_messages)

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
            next_messages.append(Message.from_model(errors, suffix="Rewrite your message with all the required tags."))
        else:
            next_messages.append(Message.from_model(ToolResults(results=tool_results)))

        return (True, next_messages)

    def _until_parse_callback(self, message: Message) -> tuple[bool, list[Message]]:
        should_continue: bool = False
        next_messages: list[Message] = [message]

        try:
            message.parse_many(self.until_types)
        except ValidationError as e:
            should_continue = True
            next_messages.append(
                Message.from_model(
                    ValidationErrorModel(content=e), suffix="Rewrite your message with all the required tags."
                )
            )
        except Exception as e:
            should_continue = True
            next_messages.append(
                Message.from_model(
                    SystemErrorModel(content=e), suffix="Rewrite your message with all the required tags."
                )
            )

        return (should_continue, next_messages)

    def _until(
        self,
        messages: list[Message],
        callback: UntilCallback,
        attempt_recovery: bool,
        drop_dialog: bool,
        max_rounds: int,
    ) -> list[Message]:
        should_continue, step_messages = callback(messages[-1])
        if not should_continue:
            return step_messages

        running_messages = step_messages if attempt_recovery else []

        for _ in range(max_rounds):
            logger.trace(
                f"_until({callback.__name__}) round {_ + 1}/{max_rounds} (attempt_recovery={attempt_recovery})"
            )
            next_message = self.generator.complete(messages[:-1] + running_messages, self.params)
            should_continue, step_messages = callback(next_message)
            logger.trace(f" |- returned {should_continue} with {len(step_messages)} new messages)")

            if not should_continue:
                return step_messages if drop_dialog else running_messages + step_messages

            if attempt_recovery:
                running_messages += step_messages

        logger.warning(f"Exhausted max rounds ({max_rounds})")
        raise ExhaustedMaxRoundsError(max_rounds)

    def _execute(self) -> list[Message]:
        if self.until_tools:
            self.params.stop = [ToolCalls.xml_end_tag()]
            if self.inject_tool_prompt:
                self.chat.inject_tool_prompt(self.until_tools)

        new_messages: list[Message] = [self.generator.complete(self.chat.all, self.params)]

        for callback, reset_between, drop_internal, max_rounds in self.until_callbacks:
            next_messages = self._until(
                self.chat.all + new_messages, callback, reset_between, drop_internal, max_rounds
            )
            new_messages = new_messages[:-1] + next_messages

        return new_messages

    @t.overload
    def run(self, count: t.Literal[None] = None) -> Chat:
        ...

    @t.overload
    def run(self, count: int) -> list[Chat]:
        ...

    def run(self, count: int | None = None) -> Chat | list[Chat]:
        if count is not None:
            return self.run_many(count)
        else:
            return Chat(self.chat.all, self._execute(), pending=self)

    def run_many(self, count: int) -> list[Chat]:
        return [Chat(self.chat.all, self._execute(), pending=self) for _ in range(count)]

    __call__ = run
