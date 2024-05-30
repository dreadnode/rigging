"""
Generators produce completions for a given set of messages or text.
"""

from rigging.generator.base import (
    GeneratedMessage,
    GeneratedText,
    GenerateParams,
    Generator,
    chat,
    complete,
    get_generator,
    get_identifier,
    register_generator,
)
from rigging.generator.litellm_ import LiteLLMGenerator

try:
    from rigging.generator.vllm_ import VLLMGenerator  # noqa: F401
except ImportError:
    pass

try:
    from rigging.generator.transformers_ import TransformersGenerator  # noqa: F401
except ImportError:
    pass

__all__ = [
    "get_generator",
    "Generator",
    "GenerateParams",
    "GeneratedMessage",
    "GeneratedText",
    "chat",
    "complete",
    "get_generator",
    "register_generator",
    "get_identifier",
    "LiteLLMGenerator",
    "VLLMGenerator",
    "TransformersGenerator",
]
