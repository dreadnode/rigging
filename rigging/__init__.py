from rigging import data, error, model, parsing, watchers
from rigging.chat import Chat, ChatPipeline
from rigging.completion import Completion, CompletionPipeline
from rigging.generator import (
    GeneratedMessage,
    GeneratedText,
    GenerateParams,
    Generator,
    chat,
    complete,
    get_generator,
    register_generator,
)
from rigging.message import Message, MessageDict, Messages
from rigging.model import Model, attr, element, wrapped
from rigging.prompt import Ctx, Prompt, prompt
from rigging.tool import Tool

__version__ = "2.0.0"

__all__ = [
    "get_generator",
    "Message",
    "MessageDict",
    "Messages",
    "Tool",
    "Model",
    "attr",
    "element",
    "wrapped",
    "Chat",
    "ChatPipeline",
    "Generator",
    "GenerateParams",
    "GeneratedMessage",
    "GeneratedText",
    "chat",
    "complete",
    "Completion",
    "CompletionPipeline",
    "register_generator",
    "prompt",
    "Prompt",
    "Ctx",
    "data",
    "watchers",
    "model",
    "error",
    "parsing",
]

from loguru import logger

logger.disable("rigging")
