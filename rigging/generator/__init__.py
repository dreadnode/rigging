"""
Generators produce completions for a given set of messages or text.
"""

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
from rigging.generator.http import HTTPGenerator
from rigging.generator.litellm_ import LiteLLMGenerator

register_generator("litellm", LiteLLMGenerator)
register_generator("http", HTTPGenerator)
register_generator("base", Generator)  # TODO: Helper while we sort out generators being required so many places.


def get_vllm_lazy() -> type[Generator]:
    try:
        from rigging.generator.vllm_ import VLLMGenerator

        return VLLMGenerator
    except ImportError as e:
        raise ImportError("VLLMGenerator is not available. Please install `vllm` or use `rigging[extra]`.") from e


register_generator("vllm", get_vllm_lazy)


def get_transformers_lazy() -> type[Generator]:
    try:
        from rigging.generator.transformers_ import TransformersGenerator

        return TransformersGenerator
    except ImportError as e:
        raise ImportError(
            "TransformersGenerator is not available. Please install `transformers` or use `rigging[extra]`."
        ) from e


register_generator("transformers", get_transformers_lazy)

__all__ = [
    "get_generator",
    "Generator",
    "GenerateParams",
    "GeneratedMessage",
    "GeneratedText",
    "StopReason",
    "Usage",
    "chat",
    "complete",
    "get_generator",
    "register_generator",
    "get_identifier",
    "LiteLLMGenerator",
    "HTTPGenerator",
    # TODO: We can't add VLLM and Transformers here because they are lazy loaded
]
