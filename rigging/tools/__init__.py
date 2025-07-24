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
from rigging.tools.mcp import as_mcp, mcp
from rigging.tools.robopages import robopages

__all__ = [
    "FunctionCall",
    "FunctionDefinition",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolDefinition",
    "ToolMode",
    "as_mcp",
    "mcp",
    "robopages",
    "tool",
    "tool_method",
]
