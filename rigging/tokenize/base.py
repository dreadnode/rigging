import typing as t
import warnings
from dataclasses import dataclass

from rigging.error import TokenizeWarning
from rigging.transform.base import Transform

if t.TYPE_CHECKING:
    from rigging.chat import Chat


SliceType = t.Literal["message", "tool_call", "tool_response", "model"]
SliceObj = t.Any


@dataclass
class TokenSlice:
    """
    Represents a slice of tokens within a tokenized chat.
    """

    start: int
    """The starting index of the slice in the token list."""
    end: int
    """The ending index of the slice in the token list."""
    type: SliceType
    """The type of the slice (e.g. message, tool_call, etc.)."""
    obj: SliceObj | None = None
    """The original object this slice corresponds to, if any."""
    metadata: dict[str, t.Any] | None = None
    """Additional metadata associated with this slice, if any."""


@dataclass
class TokenizedChat:
    """
    A tokenized representation of a chat, containing the full text,
    token list, and structured slices of tokens.
    """

    text: str
    """The full text of the chat, formatted as a single string."""
    tokens: list[int]
    """The list of tokens representing the chat text."""
    slices: list[TokenSlice]
    """Structured slices of tokens, each representing a part of the chat."""
    obj: "Chat | None" = None
    """The original chat object, if available."""


@t.runtime_checkable
class Tokenizer(t.Protocol):
    def __call__(self, chat: "Chat") -> "TokenizedChat": ...


@t.runtime_checkable
class ChatFormatter(t.Protocol):
    def __call__(self, chat: "Chat") -> str: ...


@t.runtime_checkable
class Encoder(t.Protocol):
    def __call__(self, text: str) -> list[int]: ...


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


async def tokenize_chat(
    chat: "Chat",
    formatter: ChatFormatter,
    encoder: Encoder,
    decoder: Decoder,
    *,
    transform: "Transform | None" = None,
) -> TokenizedChat:
    """
    Transform a chat into a tokenized format with structured slices.

    Args:
        chat: The chat object to tokenize.
        formatter: Function to format the chat into a string.
        encoder: Function to encode strings into tokens.
        decoder: Function to decode tokens back into strings.
        transform: Optional transformation to apply to the chat before tokenization.

    Returns:
        A TokenizedChat object containing the tokenized chat data.
    """
    if transform:
        chat = await chat.transform(transform)

    chat_text = formatter(chat)
    chat_tokens = encoder(chat_text)

    slices: list[TokenSlice] = []
    search_start = 0

    # Process messages in order
    for message in chat.all:
        # Find this message
        if not (match := find_in_tokens(message.content, chat_tokens, decoder, 0, search_start)):
            warnings.warn(
                f"Warning: Could not find message '{message.content[:50]}...' in chat tokens",
                TokenizeWarning,
                stacklevel=2,
            )
            continue

        msg_start, msg_end = match
        msg_metadata = message.metadata or {}
        msg_metadata["role"] = message.role
        if message.tool_call_id:
            msg_metadata["tool_call_id"] = message.tool_call_id

        # Add message slice
        slices.append(
            TokenSlice(
                start=msg_start,
                end=msg_end,
                type="message",
                obj=message,
                metadata=msg_metadata,
            ),
        )

        # Find parts within this message
        message_tokens = chat_tokens[msg_start:msg_end]
        part_search_start = 0

        # Process message slices in order
        for slice_ in message.slices:
            part_text = message.content[slice_.slice_]
            part_match = find_in_tokens(
                part_text,
                message_tokens,
                decoder,
                msg_start,
                part_search_start,
            )
            if not part_match:
                warnings.warn(
                    f"Warning: Could not find part '{part_text[:50]}...' in message tokens",
                    TokenizeWarning,
                    stacklevel=2,
                )
                continue

            part_start, part_end = part_match
            slices.append(
                TokenSlice(
                    start=part_start,
                    end=part_end,
                    type=slice_.type,
                    obj=slice_.obj,
                    metadata=slice_.metadata,
                ),
            )

            # Continue searching after this part
            part_search_start = part_end - msg_start

        # Continue searching after this message
        search_start = msg_end

    return TokenizedChat(
        text=chat_text,
        tokens=chat_tokens,
        slices=slices,
        obj=chat,
    )
