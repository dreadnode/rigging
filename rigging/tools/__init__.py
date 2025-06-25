"""
This module defines handles tool interaction with rigging generation.
"""

from rigging.tools.base import (
    FunctionCall,
    FunctionDefinition,
    Tool,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolMode,
    tool,
    tool_method,
)
from rigging.tools.mcp import mcp
from rigging.tools.robopages import robopages

__all__ = [
    "FunctionCall",
    "FunctionDefinition",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "ToolMode",
    "mcp",
    "robopages",
    "tool",
    "tool_method",
]
