import base64
import typing as t
from unittest.mock import patch

import pytest

from rigging import Chat, Message
from rigging.error import InvalidTokenizerError
from rigging.tokenizer import (
    TokenizedChat,
    Tokenizer,
    TokenSlice,
    get_tokenizer,
    register_tokenizer,
)
from rigging.tokenizer.base import g_tokenizers

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001, ARG002


class MockTokenizer(Tokenizer):
    """Mock tokenizer for testing purposes."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def format_chat(self, chat: "Chat") -> str:
        return "\n".join(msg.content for msg in chat.all)


def test_get_tokenizer_default_provider() -> None:
    """Test that default provider is used when no provider specified."""
    # Mock the global tokenizers dict
    with patch.dict(g_tokenizers, {"transformers": MockTokenizer}, clear=True):
        tokenizer = get_tokenizer("test-model")
        assert isinstance(tokenizer, MockTokenizer)
        assert tokenizer.model == "test-model"


def test_get_tokenizer_with_provider() -> None:
    """Test getting tokenizer with explicit provider."""
    with patch.dict(g_tokenizers, {"custom": MockTokenizer}, clear=True):
        tokenizer = get_tokenizer("custom!test-model")
        assert isinstance(tokenizer, MockTokenizer)
        assert tokenizer.model == "test-model"


def test_get_tokenizer_invalid_provider() -> None:
    """Test error when using invalid provider."""
    with (
        patch.dict(g_tokenizers, {"transformers": MockTokenizer}, clear=True),
        pytest.raises(InvalidTokenizerError),
    ):
        get_tokenizer("invalid!test-model")


def test_get_tokenizer_empty_identifier() -> None:
    """Test error when identifier is empty."""
    with pytest.raises(InvalidTokenizerError):
        get_tokenizer("")


def test_get_tokenizer_with_kwargs() -> None:
    """Test getting tokenizer with additional kwargs."""

    class CustomTokenizer(Tokenizer):
        custom_param: str = "default"

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, tokens: list[int]) -> str:
            return "decoded"

        def format_chat(self, chat: "Chat") -> str:
            return "formatted"

    with patch.dict(g_tokenizers, {"custom": CustomTokenizer}, clear=True):
        tokenizer = get_tokenizer("custom!test-model,custom_param=value")
        assert isinstance(tokenizer, CustomTokenizer)
        assert tokenizer.model == "test-model"
        assert tokenizer.custom_param == "value"


def test_get_tokenizer_with_numeric_kwargs() -> None:
    """Test getting tokenizer with numeric kwargs."""

    class NumericTokenizer(Tokenizer):
        max_length: int = 100
        temperature: float = 1.0
        enabled: bool = False

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, tokens: list[int]) -> str:
            return "decoded"

        def format_chat(self, chat: "Chat") -> str:
            return "formatted"

    with patch.dict(g_tokenizers, {"numeric": NumericTokenizer}, clear=True):
        tokenizer = t.cast(
            "NumericTokenizer",
            get_tokenizer(
                "numeric!model,max_length=512,temperature=0.5,enabled=true",
            ),
        )
        assert tokenizer.max_length == 512
        assert tokenizer.temperature == 0.5
        assert tokenizer.enabled is True


def test_get_tokenizer_with_base64_kwargs() -> None:
    """Test getting tokenizer with base64 encoded kwargs."""

    class Base64Tokenizer(Tokenizer):
        config: dict[str, t.Any]

        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

        def decode(self, tokens: list[int]) -> str:
            return "decoded"

        def format_chat(self, chat: "Chat") -> str:
            return "formatted"

    # Encode {"key": "value"} as base64
    encoded_config = base64.b64encode(b'{"key": "value"}').decode()

    with patch.dict(g_tokenizers, {"base64": Base64Tokenizer}, clear=True):
        tokenizer = t.cast(
            "Base64Tokenizer",
            get_tokenizer(f"base64!model,config=base64:{encoded_config}"),
        )
        assert tokenizer.config == {"key": "value"}


def test_get_tokenizer_lazy_tokenizer() -> None:
    """Test getting tokenizer with lazy loading."""

    def lazy_tokenizer() -> type[Tokenizer]:
        return MockTokenizer

    with patch.dict(g_tokenizers, {"lazy": lazy_tokenizer}, clear=True):
        tokenizer = get_tokenizer("lazy!test-model")
        assert isinstance(tokenizer, MockTokenizer)
        assert tokenizer.model == "test-model"
        # Verify that the lazy tokenizer was replaced with the actual class
        assert g_tokenizers["lazy"] == MockTokenizer


def test_get_tokenizer_caching() -> None:
    """Test that get_tokenizer caches results."""
    with patch.dict(g_tokenizers, {"transformers": MockTokenizer}, clear=True):
        tokenizer1 = get_tokenizer("test-model")
        tokenizer2 = get_tokenizer("test-model")
        # Should be the same instance due to caching
        assert tokenizer1 is tokenizer2


def test_register_tokenizer_class() -> None:
    """Test registering a tokenizer class."""
    original_tokenizers = g_tokenizers.copy()
    try:
        register_tokenizer("test", MockTokenizer)
        assert "test" in g_tokenizers
        assert g_tokenizers["test"] == MockTokenizer
    finally:
        g_tokenizers.clear()
        g_tokenizers.update(original_tokenizers)


def test_register_tokenizer_lazy() -> None:
    """Test registering a lazy tokenizer."""

    def lazy_tokenizer() -> type[Tokenizer]:
        return MockTokenizer

    original_tokenizers = g_tokenizers.copy()
    try:
        register_tokenizer("lazy", lazy_tokenizer)
        assert "lazy" in g_tokenizers
        assert g_tokenizers["lazy"] == lazy_tokenizer
    finally:
        g_tokenizers.clear()
        g_tokenizers.update(original_tokenizers)


class TestTokenizer:
    """Test cases for Tokenizer base class."""


def test_tokenizer_abstract_methods() -> None:
    """Test that abstract methods raise NotImplementedError."""
    tokenizer = Tokenizer(model="test")

    with pytest.raises(NotImplementedError):
        tokenizer.encode("test")

    with pytest.raises(NotImplementedError):
        tokenizer.decode([1, 2, 3])

    with pytest.raises(NotImplementedError):
        tokenizer.format_chat(Chat([]))


def test_find_in_tokens() -> None:
    """Test the _find_in_tokens method."""
    tokenizer = MockTokenizer(model="test")

    # Test finding text in tokens
    text = "hello"
    tokens = tokenizer.encode(text)
    result = tokenizer._find_in_tokens(text, tokens)
    assert result == (0, len(tokens))

    # Test finding substring
    full_text = "hello world"
    full_tokens = tokenizer.encode(full_text)
    result = tokenizer._find_in_tokens("world", full_tokens)
    assert result is not None
    start, end = result
    decoded = tokenizer.decode(full_tokens[start:end])
    assert "world" in decoded


def test_find_in_tokens_not_found() -> None:
    """Test _find_in_tokens when text is not found."""
    tokenizer = MockTokenizer(model="test")

    text = "hello"
    tokens = tokenizer.encode(text)
    result = tokenizer._find_in_tokens("notfound", tokens)
    assert result is None


@pytest.mark.asyncio
async def test_tokenize_chat() -> None:
    """Test tokenizing a chat."""
    tokenizer = MockTokenizer(model="test")

    # Create a simple chat
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    chat = Chat(messages)

    # Tokenize the chat
    tokenized = await tokenizer.tokenize_chat(chat)

    assert isinstance(tokenized, TokenizedChat)
    assert tokenized.text == "Hello\nHi there!"
    assert len(tokenized.tokens) > 0
    assert len(tokenized.slices) == 2  # One slice per message
    assert tokenized.obj is chat

    # Check message slices
    for i, slice_ in enumerate(tokenized.slices):
        assert slice_.type == "message"
        assert slice_.obj == messages[i]
        assert slice_.metadata is not None
        assert slice_.metadata["role"] == messages[i].role


@pytest.mark.asyncio
async def test_tokenize_chat_with_tool_calls() -> None:
    """Test tokenizing a chat with tool calls."""
    tokenizer = MockTokenizer(model="test")

    # Create a chat with tool call
    messages = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="I'll check the weather for you.",
            tool_calls=[
                {
                    "id": "call_123",
                    "function": {"name": "get_weather", "arguments": '{"location": "New York"}'},
                },
            ],
        ),
        Message(
            role="tool",
            content="Sunny, 75°F",
            tool_call_id="call_123",
        ),
    ]
    chat = Chat(messages)

    # Tokenize the chat
    tokenized = await chat.to_tokens(tokenizer, transform="json-with-tag")

    assert len(tokenized.slices) == 6

    slice_ = tokenized.slices[0]
    assert slice_.type == "message"
    assert slice_.metadata is not None
    assert slice_.metadata.get("role") == "system"

    slice_ = tokenized.slices[1]
    assert slice_.type == "message"
    assert slice_.obj == messages[0]
    assert slice_.metadata is not None
    assert slice_.metadata.get("role") == "user"

    slice_ = tokenized.slices[2]
    assert slice_.type == "message"
    assert slice_.metadata is not None
    assert slice_.metadata.get("role") == "assistant"

    slice_ = tokenized.slices[3]
    assert slice_.type == "tool_call"
    assert slice_.metadata is not None
    assert slice_.metadata.get("id") == "call_123"

    slice_ = tokenized.slices[4]
    assert slice_.type == "message"
    assert slice_.metadata is not None
    assert slice_.metadata.get("role") == "user"

    slice_ = tokenized.slices[5]
    assert slice_.type == "tool_response"
    assert slice_.metadata is not None
    assert slice_.metadata.get("id") == "call_123"


def test_token_slice_creation() -> None:
    """Test creating a TokenSlice."""
    slice_ = TokenSlice(
        start=0,
        end=10,
        type="message",
        obj=None,
        metadata={"role": "user"},
    )

    assert slice_.start == 0
    assert slice_.end == 10
    assert slice_.type == "message"
    assert slice_.obj is None
    assert slice_.metadata == {"role": "user"}


def test_token_slice_defaults() -> None:
    """Test TokenSlice with default values."""
    slice_ = TokenSlice(start=0, end=10, type="message")

    assert slice_.obj is None
    assert slice_.metadata is None


def test_tokenized_chat_creation() -> None:
    """Test creating a TokenizedChat."""
    slice_ = TokenSlice(start=0, end=5, type="message")
    tokenized = TokenizedChat(
        text="Hello",
        tokens=[1, 2, 3, 4, 5],
        slices=[slice_],
        obj=None,
    )

    assert tokenized.text == "Hello"
    assert tokenized.tokens == [1, 2, 3, 4, 5]
    assert len(tokenized.slices) == 1
    assert tokenized.obj is None


def test_tokenized_chat_defaults() -> None:
    """Test TokenizedChat with default values."""
    tokenized = TokenizedChat(
        text="Hello",
        tokens=[1, 2, 3],
        slices=[],
    )

    assert tokenized.obj is None
