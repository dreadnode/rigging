import json
import typing as t
from textwrap import dedent

from rigging.generator.base import GenerateParams
from rigging.message import Message
from rigging.tokenize.base import FormattedChat, Slice, SliceMessage, SliceOriginal
from rigging.tools.native import NativeToolCall, NativeToolResponse

if t.TYPE_CHECKING:
    from rigging.chat import Chat

# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template


def format_llama(chat: "Chat") -> FormattedChat:
    text = ""
    slices = []
    current_pos = 0

    def add_text(content: str) -> None:
        nonlocal current_pos, text
        text += content
        current_pos += len(content)

    def add_slice(
        content: str,
        slice_type: SliceMessage,
        original: SliceOriginal | None = None,
        metadata: dict[str, t.Any] | None = None,
    ) -> Slice:
        nonlocal current_pos
        start = current_pos
        add_text(content)
        slice_ = Slice(start, current_pos, slice_type, original, metadata)
        slices.append(slice_)
        return slice_

    def add_message(message: Message) -> Slice:
        slice_ = add_slice(
            message.content,
            "message",
            message,
            {"role": message.role},
        )

        # Handle parsed parts within the message
        for part in message.parts:
            part_start = slice_.start + part.slice_.start
            part_end = slice_.start + part.slice_.stop

            if isinstance(part.model, NativeToolCall | NativeToolResponse):
                slices.append(
                    Slice(
                        start=part_start,
                        end=part_end,
                        type="tool_call"
                        if isinstance(part.model, NativeToolCall)
                        else "tool_response",
                        original=part.model,
                        metadata={"tool_call_id": part.model.id},
                    ),
                )
            else:
                slices.append(
                    Slice(
                        start=part_start,
                        end=part_end,
                        type="parsed_part",
                        original=part,
                    ),
                )

        return slice_

    messages = chat.messages
    params = chat.params or GenerateParams()

    system_message: Message | None = None
    if messages and messages[0].role == "system":
        system_message, messages = messages[0], messages[1:]

    # System + Tools

    add_text("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")

    if params.tools:
        add_text(
            dedent(
                """\
            You have access to the following functions. To call a function, please \
            respond with JSON for a function call. Respond in the format \
            {"name": function name, "parameters": dictionary of argument name and its value}. \
            Do not use variables.\n
            """,
            ),
        )
        for tool in params.tools or []:
            add_text(tool.model_dump_json(indent=4) + "\n\n")

    if system_message:
        add_message(system_message)

    add_text("<|eot_id|>")

    # Process remaining messages

    for message in messages:
        # Tool calls
        if message.tool_calls:
            add_text("<|start_header_id|>assistant<|end_header_id|>\n\n")

            # Technically this should be limited to a single tool call
            for tool_call in message.tool_calls:
                tool_json = {
                    "name": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments),
                }
                add_slice(
                    json.dumps(tool_json),
                    "tool_call",
                    tool_call,
                    {"tool_call_id": tool_call.id},
                )
            add_text("<|eot_id|>")

        # Tool response
        elif message.role == "tool":
            add_text("<|start_header_id|>ipython<|end_header_id|>\n\n")
            add_slice(
                message.content,
                "tool_response",
                message,
                {"tool_call_id": message.tool_call_id},
            )
            add_text("<|eot_id|>")

        # User or assistant messages
        elif message.role in ["user", "assistant"]:
            add_text(f"<|start_header_id|>{message.role}<|end_header_id|>\n\n")
            add_message(message)
            add_text("<|eot_id|>")

    # Generation prompt

    add_text("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return FormattedChat(text, slices, chat, chat.metadata)
