"""
This module defines handles tool interaction with rigging generation.
"""

from rigging.tool.base import FunctionCall, Tool, ToolCall, ToolDefinition, tool, tool_method
from rigging.tool.mcp import mcp
from rigging.tool.robopages import robopages

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
