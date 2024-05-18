"""
Generators produce completions for a given set of messages or text.
"""

from rigging.generator.base import GenerateParams, Generator, chat, complete, get_generator, register_generator
from rigging.generator.litellm_ import LiteLLMGenerator

__all__ = ["get_generator", "Generator", "GenerateParams", "chat", "complete", "register_generator", "LiteLLMGenerator"]


try:
    import vllm  # noqa: F401

    from rigging.generator.vllm_ import VLLMGenerator  # noqa: F401

    __all__.append("VLLMGenerator")
except ImportError:
    pass
