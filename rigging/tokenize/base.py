import typing as t
from dataclasses import dataclass

from rigging.message import (
    Message,
    ParsedMessagePart,
)
from rigging.tools.base import ToolCall
from rigging.tools.native import NativeToolCall, NativeToolResponse

if t.TYPE_CHECKING:
    from rigging.chat import Chat


# Common

SliceMessage = t.Literal["message", "tool_call", "tool_response", "parsed_part"]
SliceOriginal = Message | ParsedMessagePart | ToolCall | NativeToolCall | NativeToolResponse


@dataclass
class Slice:
    start: int
    end: int
    type: SliceMessage
    original: SliceOriginal | None = None
    metadata: dict[str, t.Any] | None = None


# Formatting


@dataclass
class FormattedChat:
    text: str
    slices: list[Slice]
    original: "Chat | None" = None
    metadata: dict[str, t.Any] | None = None


@t.runtime_checkable
class ChatParser(t.Protocol):
    def __call__(
        self,
        chat: "FormattedChat",
        /,
    ) -> "Chat": ...


@t.runtime_checkable
class ChatFormatter(t.Protocol):
    def __call__(
        self,
        chat: "Chat",
        /,
    ) -> FormattedChat: ...


# Tokenizing


@dataclass
class TokenizedChat:
    text: str
    tokens: list[int]
    slices: list[Slice]
    original: "Chat | None" = None


@t.runtime_checkable
class Tokenizer(t.Protocol):
    def __call__(self, chat: FormattedChat) -> "TokenizedChat": ...


@t.runtime_checkable
class Encoder(t.Protocol):
    def __call__(self, text: str, /) -> list[int]: ...


@t.runtime_checkable
class Decoder(t.Protocol):
    def __call__(self, tokens: list[int]) -> str: ...


def find_in_tokens(
    target_text: str,
    tokens: list[int],
    decoder: Decoder,
    start_offset: int = 0,
    search_start: int = 0,
) -> tuple[int, int] | None:
    # End-based walk: find a window that contains our target text
    for end_pos in range(search_start + 1, len(tokens) + 1):
        decoded_window = decoder(tokens[search_start:end_pos])
        if target_text not in decoded_window:
            continue

        # Start-based walk: narrow down the start position
        actual_start = search_start
        for start_pos in range(search_start, end_pos):
            decoded_from_start = decoder(tokens[start_pos:end_pos])
            if decoded_from_start.startswith(target_text):
                actual_start = start_pos
                break

        return (start_offset + actual_start, start_offset + end_pos)

    return None


# # Get full tokenization
# messages = [m.to_openai_spec() for m in chat.all]
# full_tokens = tokenizer.apply_chat_template(messages)

# slices: list[TokenSlice] = []
# search_start = 0

# # Process messages in order
# for message in chat.all:
#     # Find this message
#     match = find_in_tokens(message.content, full_tokens, tokenizer, 0, search_start)
#     if not match:
#         print("Warning: Could not find message content in tokens")
#         continue

#     msg_start, msg_end = match

#     # Add message slice
#     slices.append(
#         MessageTokenSlice(
#             message=message,
#             start=msg_start,
#             end=msg_end,
#         ),
#     )

#     # Find parts within this message
#     if message.parts:
#         message_tokens = full_tokens[msg_start:msg_end]
#         part_search_start = 0

#         # Process parts in order
#         for part in message.parts:
#             part_text = message.content[part.slice_]
#             part_match = find_in_tokens(
#                 part_text,
#                 message_tokens,
#                 tokenizer,
#                 msg_start,
#                 part_search_start,
#             )

#             if part_match:
#                 part_start, part_end = part_match
#                 slices.append(
#                     ParsedMessagePartTokenSlice(
#                         message=message,
#                         part=part,
#                         start=part_start,
#                         end=part_end,
#                     ),
#                 )
#                 # Continue searching after this part
#                 part_search_start = part_end - msg_start
#             else:
#                 print(f"Warning: Could not find part text '{part_text[:50]}...' in message tokens")

#     # Continue searching after this message
#     search_start = msg_end

# print(slices)
