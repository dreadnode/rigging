from rigging.chat import Chat, PendingChat
from rigging.completion import Completion, PendingCompletion
from rigging.data import (
    chats_to_df,
    chats_to_elastic,
    chats_to_elastic_data,
    df_to_chats,
    elastic_data_to_chats,
    elastic_to_chats,
    flatten_chats,
    unflatten_chats,
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
from rigging.message import Message, MessageDict, Messages
from rigging.model import Model, attr, element, make_primitive, wrapped
from rigging.tool import Tool

__version__ = "1.1.1"

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
    "PendingChat",
    "Generator",
    "GenerateParams",
    "GeneratedMessage",
    "GeneratedText",
    "chat",
    "complete",
    "Completion",
    "PendingCompletion",
    "register_generator",
    "chats_to_df",
    "df_to_chats",
    "make_primitive",
    "chats_to_elastic",
    "elastic_to_chats",
    "elastic_data_to_chats",
    "chats_to_elastic_data",
    "flatten_chats",
    "unflatten_chats",
]

from loguru import logger

logger.disable("rigging")
