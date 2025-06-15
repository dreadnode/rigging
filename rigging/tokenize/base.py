import typing as t
from dataclasses import dataclass

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
