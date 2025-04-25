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
    "get_generator",
    "Message",
    "MessageDict",
    "Messages",
    "ContentText",
    "ContentImageUrl",
    "ContentAudioInput",
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
    "tool",
    "tool_method",
    "logging",
    "await_",
    "interact",
    "ThenChatCallback",
    "MapChatCallback",
    "ThenCompletionCallback",
    "MapCompletionCallback",
    "PipelineStep",
    "PipelineStepGenerator",
    "PipelineStepContextManager",
    "generator",
    "mcp",
    "robopages",
]

from loguru import logger

logger.disable("rigging")
