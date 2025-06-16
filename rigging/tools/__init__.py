"""
This module defines handles tool interaction with rigging generation.
"""

from rigging.tools.base import FunctionCall, Tool, ToolCall, ToolDefinition, tool, tool_method
from rigging.tools.mcp import mcp
from rigging.tools.robopages import robopages

__all__ = [
    "FunctionCall",
    "Tool",
    "ToolCall",
    "ToolDefinition",
    "mcp",
    "robopages",
    "tool",
    "tool_method",
]
