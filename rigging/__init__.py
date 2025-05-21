from rigging import data, error, generator, logging, model, parsing, watchers
from rigging.chat import (
    Chat,
    ChatPipeline,
    MapChatCallback,
    PipelineStep,
    PipelineStepContextManager,
    PipelineStepGenerator,
    ThenChatCallback,
)
from rigging.completion import (
    Completion,
    CompletionPipeline,
    MapCompletionCallback,
    ThenCompletionCallback,
)
from rigging.error import Stop
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
from rigging.message import (
    ContentAudioInput,
    ContentImageUrl,
    ContentText,
    Message,
    MessageDict,
    Messages,
)
from rigging.model import Model, attr, element, wrapped
from rigging.prompt import Ctx, Prompt, prompt
from rigging.tool import Tool, mcp, robopages, tool, tool_method
from rigging.util import await_
from rigging.version import VERSION

__version__ = VERSION

__all__ = [
    "Chat",
    "ChatPipeline",
    "Completion",
    "CompletionPipeline",
    "ContentAudioInput",
    "ContentImageUrl",
    "ContentText",
    "Ctx",
    "GenerateParams",
    "GeneratedMessage",
    "GeneratedText",
    "Generator",
    "MapChatCallback",
    "MapCompletionCallback",
    "Message",
    "MessageDict",
    "Messages",
    "Model",
    "PipelineStep",
    "PipelineStepContextManager",
    "PipelineStepGenerator",
    "Prompt",
    "Stop",
    "ThenChatCallback",
    "ThenCompletionCallback",
    "Tool",
    "attr",
    "await_",
    "chat",
    "complete",
    "data",
    "element",
    "error",
    "generator",
    "get_generator",
    "interact",
    "logging",
    "mcp",
    "model",
    "parsing",
    "prompt",
    "register_generator",
    "robopages",
    "tool",
    "tool_method",
    "watchers",
    "wrapped",
]

from loguru import logger

logger.disable("rigging")
