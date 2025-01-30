from rigging import data, error, generator, model, parsing, watchers
from rigging.chat import Chat, ChatPipeline, MapChatCallback, ThenChatCallback
from rigging.completion import Completion, CompletionPipeline, MapCompletionCallback, ThenCompletionCallback
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
from rigging.interact import interact
from rigging.message import ContentImageUrl, ContentText, Message, MessageDict, Messages
from rigging.model import Model, attr, element, wrapped
from rigging.prompt import Ctx, Prompt, prompt
from rigging.tool import ApiTool, Tool
from rigging.util import await_

__version__ = "2.2.0"

__all__ = [
    "get_generator",
    "Message",
    "MessageDict",
    "Messages",
    "ContentText",
    "ContentImageUrl",
    "Tool",
    "ApiTool",
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
    "await_",
    "interact",
    "ThenChatCallback",
    "MapChatCallback",
    "ThenCompletionCallback",
    "MapCompletionCallback",
    "generator",
]

from loguru import logger

logger.disable("rigging")
