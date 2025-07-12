"""
Tokenizers encode chats and associated message data into tokens for training and inference.
"""

import typing as t

from rigging.tokenizer.base import (
    TokenizedChat,
    Tokenizer,
    TokenSlice,
    get_tokenizer,
    register_tokenizer,
)


def get_transformers_lazy() -> type[Tokenizer]:
    try:
        from rigging.tokenizer.transformers_ import TransformersTokenizer
    except ImportError as e:
        raise ImportError(
            "TransformersTokenizer is not available. Please install `transformers` or use `rigging[extra]`.",
        ) from e

    return TransformersTokenizer


register_tokenizer("transformers", get_transformers_lazy)

__all__ = [
    "TokenSlice",
    "TokenizedChat",
    "Tokenizer",
    "get_tokenizer",
    "register_tokenizer",
]


def __getattr__(name: str) -> t.Any:
    if name == "TransformersTokenizer":
        return get_transformers_lazy()
    raise AttributeError(f"module {__name__} has no attribute {name}")
