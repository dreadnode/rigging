"""
Generators produce completions for a given set of messages or text.
"""

from rigging.generator.base import (
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

__all__ = [
    "get_generator",
    "Generator",
    "GenerateParams",
    "chat",
    "complete",
    "get_generator",
    "register_generator",
    "get_identifier",
    "LiteLLMGenerator",
    "VLLMGenerator",
]
