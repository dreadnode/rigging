"""
This module defines handles tool interaction with rigging generation.
"""


from rigging.tool.base import Tool, tool, tool_method
from rigging.tool.mcp import mcp
from rigging.tool.robopages import robopages

__all__ = [
    "Tool",
    "robopages",
    "mcp",
    "tool",
    "tool_method",
]
