"""
Generators produce completions for a given set of messages or text.
"""

import typing as t

from rigging.generator.base import (
    GeneratedMessage,
    GeneratedText,
    GenerateParams,
    Generator,
    StopReason,
    Usage,
    chat,
    complete,
    get_generator,
    get_identifier,
    register_generator,
)
from rigging.generator.http import HTTPGenerator, HttpHook, HttpHookAction, HTTPSpec
from rigging.generator.litellm_ import LiteLLMGenerator

register_generator("litellm", LiteLLMGenerator)
register_generator("http", HTTPGenerator)
register_generator(
    "base",
    Generator,
)  # TODO: Helper while we sort out generators being required so many places.


def get_vllm_lazy() -> type[Generator]:
    try:
        from rigging.generator.vllm_ import VLLMGenerator
    except ImportError as e:
        raise ImportError(
            "VLLMGenerator is not available. Please install `vllm` or use `rigging[llm]`.",
        ) from e

    return VLLMGenerator


register_generator("vllm", get_vllm_lazy)


def get_transformers_lazy() -> type[Generator]:
    try:
        from rigging.generator.transformers_ import TransformersGenerator
    except ImportError as e:
        raise ImportError(
            "TransformersGenerator is not available. Please install `transformers` or use `rigging[llm]`.",
        ) from e

    return TransformersGenerator


register_generator("transformers", get_transformers_lazy)

__all__ = [
    "GenerateParams",
    "GeneratedMessage",
    "GeneratedText",
    "Generator",
    "HTTPGenerator",
    "HTTPGenerator",
    "HTTPSpec",
    "HttpHook",
    "HttpHookAction",
    "LiteLLMGenerator",
    "StopReason",
    "Usage",
    "chat",
    "complete",
    "get_generator",
    "get_generator",
    "get_identifier",
    "register_generator",
]


def __getattr__(name: str) -> t.Any:
    if name == "VLLMGenerator":
        return get_vllm_lazy()
    if name == "TransformersGenerator":
        return get_transformers_lazy()
    raise AttributeError(f"module {__name__} has no attribute {name}")
