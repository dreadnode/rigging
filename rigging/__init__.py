import os

from rigging.generator import get_generator
from rigging.message import Message, MessageDict, Messages
from rigging.model import CoreModel
from rigging.tool import Tool

__all__ = ["get_generator", "Message", "MessageDict", "Messages", "Tool", "CoreModel"]

from rigging.logging import configure_logging

configure_logging(
    os.getenv("RIGGING_LOG_LEVEL", "INFO"),
    os.getenv("RIGGING_LOG_FILE", None),
    os.getenv("RIGGING_LOG_FILE_LEVEL", "TRACE"),  # type: ignore
)
