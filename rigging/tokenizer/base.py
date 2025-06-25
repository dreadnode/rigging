import base64
import contextlib
import typing as t
import warnings
from dataclasses import dataclass
from functools import lru_cache

from pydantic import BaseModel, TypeAdapter

from rigging.error import InvalidTokenizerError, TokenizerWarning

if t.TYPE_CHECKING:
    from rigging.chat import Chat


SliceType = t.Literal["message", "tool_call", "tool_response", "model", "other"]
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
    metadata: dict[str, t.Any] | None = None
    """Additional metadata associated with the tokenized chat, if any."""


@t.runtime_checkable
class LazyTokenizer(t.Protocol):
    def __call__(self) -> type["Tokenizer"]: ...


g_tokenizers: dict[str, type["Tokenizer"] | LazyTokenizer] = {}


class Tokenizer(BaseModel):
    """
    Base class for all rigging tokenizers.

    This class provides common functionality and methods for tokenizing chats.
    """

    model: str
    """The model name to be used by the tokenizer."""

    def _find_in_tokens(
        self,
        target_text: str,
        tokens: list[int],
        start_offset: int = 0,
        search_start: int = 0,
    ) -> tuple[int, int] | None:
        # End-based walk: find a window that contains our target text
        for end_pos in range(search_start + 1, len(tokens) + 1):
            decoded_window = self.decode(tokens[search_start:end_pos])
            if target_text not in decoded_window:
                continue

            # Start-based walk: narrow down the start position
            actual_start = search_start
            for start_pos in range(search_start, end_pos):
                decoded_from_start = self.decode(tokens[start_pos:end_pos])
                if decoded_from_start.startswith(target_text):
                    actual_start = start_pos
                    break

            return (start_offset + actual_start, start_offset + end_pos)

        return None

    def encode(self, text: str) -> list[int]:
        """
        Encodes the given text into a list of tokens.

        Args:
            text: The text to encode.

        Returns:
            A list of tokens representing the encoded text.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.encode() must be implemented by subclasses.",
        )

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of tokens back into a string.

        Args:
            tokens: The list of tokens to decode.

        Returns:
            The decoded string.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.decode() must be implemented by subclasses.",
        )

    def format_chat(self, chat: "Chat") -> str:
        """
        Formats the chat into a string representation.

        Args:
            chat: The chat object to format.

        Returns:
            A string representation of the chat.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.format_chat() must be implemented by subclasses.",
        )

    async def tokenize_chat(self, chat: "Chat") -> TokenizedChat:
        """
        Transform a chat into a tokenized format with structured slices.

        Args:
            chat: The chat object to tokenize.

        Returns:
            A TokenizedChat object containing the tokenized chat data.
        """
        chat_text = self.format_chat(chat)
        chat_tokens = self.encode(chat_text)

        slices: list[TokenSlice] = []
        search_start = 0

        # Process messages in order
        for message in chat.all:
            # Find this message
            if not (match := self._find_in_tokens(message.content, chat_tokens, 0, search_start)):
                warnings.warn(
                    f"Warning: Could not find message '{message.content[:50]}...' in chat tokens",
                    TokenizerWarning,
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
                part_match = self._find_in_tokens(
                    part_text,
                    message_tokens,
                    msg_start,
                    part_search_start,
                )
                if not part_match:
                    warnings.warn(
                        f"Warning: Could not find part '{part_text[:50]}...' in message tokens",
                        TokenizerWarning,
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

        slices.sort(key=lambda s: s.start)

        return TokenizedChat(
            text=chat_text,
            tokens=chat_tokens,
            slices=slices,
            obj=chat,
            metadata=chat.metadata,
        )


@lru_cache(maxsize=128)
def get_tokenizer(identifier: str) -> Tokenizer:
    """
    Get a tokenizer by an identifier string. Uses Transformers by default.

    Identifier strings are formatted like `<provider>!<model>,<**kwargs>`

    (provider is optional and defaults to `transformers` if not specified)

    Examples:
        - "meta-llama/Meta-Llama-3-8B-Instruct" -> `TransformersTokenizer(model="`meta-llama/Meta-Llama-3-8B-Instruct")`
        - "transformers!microsoft/Phi-4-mini-instruct" -> `TransformersTokenizer(model="microsoft/Phi-4-mini-instruct")`

    Args:
        identifier: The identifier string to use to get a tokenizer.

    Returns:
        The tokenizer object.

    Raises:
        InvalidTokenizerError: If the identifier is invalid.
    """

    provider: str = next(iter(g_tokenizers.keys()))
    model: str = identifier

    if not identifier:
        raise InvalidTokenizerError(identifier)

    # Split provider, model, and kwargs

    if "!" in identifier:
        try:
            provider, model = identifier.split("!")
        except Exception as e:
            raise InvalidTokenizerError(identifier) from e

    if provider not in g_tokenizers:
        raise InvalidTokenizerError(identifier)

    if not isinstance(g_tokenizers[provider], type):
        lazy_generator = t.cast("LazyTokenizer", g_tokenizers[provider])
        g_tokenizers[provider] = lazy_generator()

    generator_cls = t.cast("type[Tokenizer]", g_tokenizers[provider])

    kwargs: dict[str, t.Any] = {}
    if "," in model:
        try:
            model, kwargs_str = model.split(",", 1)
            kwargs = dict(arg.split("=", 1) for arg in kwargs_str.split(","))
        except Exception as e:
            raise InvalidTokenizerError(identifier) from e

    # Decode any base64 values if present
    def decode_value(value: str) -> t.Any:
        if value.startswith("base64:"):
            with contextlib.suppress(Exception):
                decoded = base64.b64decode(value[7:])
                return TypeAdapter(t.Any).validate_json(decoded)
        return value

    kwargs = {k: decode_value(v) for k, v in kwargs.items()}

    # Do some subtle type conversion
    for k, v in kwargs.items():
        if not isinstance(v, str):
            continue

        try:
            kwargs[k] = float(v)
            continue
        except ValueError:
            pass

        try:
            kwargs[k] = int(v)
            continue
        except ValueError:
            pass

        if isinstance(v, str) and v.lower() in ["true", "false"]:
            kwargs[k] = v.lower() == "true"

    return generator_cls(model=model, **kwargs)


def register_tokenizer(provider: str, tokenizer_cls: type[Tokenizer] | LazyTokenizer) -> None:
    """
    Register a tokenizer class for a provider id.

    This let's you use [rigging.tokenizer.get_tokenizer][] with a custom tokenizer class.

    Args:
        provider: The name of the provider.
        tokenizer_cls: The tokenizer class to register.

    Returns:
        None
    """
    global g_tokenizers  # noqa: PLW0602
    g_tokenizers[provider] = tokenizer_cls
