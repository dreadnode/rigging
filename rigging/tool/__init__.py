"""
This module defines handles tool interaction with rigging generation.
"""

import typing as t

from rigging.tool.api import ApiTool
from rigging.tool.native import Tool

ToolType = t.Literal["api", "native"]

__all__ = [
    "Tool",
    "ApiTool",
    "ToolType",
]
