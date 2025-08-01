"""
Utilities for communicating with MCP servers.
"""

import asyncio
import functools
import inspect
import json
import typing as t
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import typing_extensions as te

from rigging.error import Stop
from rigging.tools.base import Tool
from rigging.util import flatten_list

if t.TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult

    from rigging.message import Content

INITIALIZE_TIMEOUT = 5
DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

Transport = t.Literal["stdio", "sse"]

P = t.ParamSpec("P")
R = t.TypeVar("R")


class StdioConnection(te.TypedDict):
    command: str
    args: list[str]
    cwd: str | Path | None
    env: dict[str, str] | None


class SSEConnection(te.TypedDict):
    url: str
    headers: dict[str, t.Any] | None
    timeout: float
    sse_read_timeout: float


def _convert_mcp_result_to_message_parts(result: "CallToolResult") -> list[t.Any]:
    from mcp.types import TextResourceContents

    from rigging.message import ContentImageUrl, ContentText

    parts: list[Content] = []
    for content in result.content:
        if content.type == "text":
            parts.append(ContentText(text=content.text))
        elif content.type == "image":
            parts.append(ContentImageUrl.from_url(f"data:{content.mimeType};base64,{content.data}"))
        elif content.type == "resource":
            resource = content.resource
            resource_text = (
                resource.text if isinstance(resource, TextResourceContents) else resource.blob
            )
            parts.append(ContentText(text=resource_text))
        else:
            raise ValueError(f"Unknown content type: {content.type}")
    return parts


def _convert_rigging_return_to_mcp(result: t.Any) -> t.Any:
    """
    Converts a return value from a rigging.Tool into a type that
    FastMCP can serialize and send to a client.

    Args:
        result: The return value from a rigging.Tool's execution.

    Returns:
        A value compatible with MCP serialization (JSON types, mcp.Image, etc.).
    """
    from mcp.server.fastmcp import Image

    from rigging.message import ContentImageUrl, ContentText, Message

    if isinstance(result, Stop):
        return f"Tool requested stop: {result.message}"

    if isinstance(result, Message):
        # If the message contains a single content part, we can unwrap it for a
        # cleaner return type. Otherwise, we must serialize the parts together.
        if len(result.content_parts) == 1:
            return _convert_rigging_return_to_mcp(result.content_parts[0])

        return json.dumps([part.model_dump(mode="json") for part in result.content_parts], indent=2)

    if isinstance(result, ContentText):
        return result.text

    if isinstance(result, ContentImageUrl):
        try:
            return Image(data=result.to_bytes())
        except ValueError:
            return result.image_url.url

    return result


def _create_mcp_handler(
    tool: t.Callable[P, t.Any],
) -> t.Callable[P, t.Awaitable[t.Any]]:
    @functools.wraps(tool)
    async def handler(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            result = tool(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Stop as stop:
            result = stop
        return _convert_rigging_return_to_mcp(result)

    return handler


@dataclass
class MCPClient:
    """A client for communicating with MCP servers."""

    transport: Transport
    """The transport to use"""
    connection: StdioConnection | SSEConnection
    """Connection configuration"""
    tools: list[Tool[..., t.Any]]
    """A list of tools available on the server"""

    def __init__(self, transport: Transport, connection: "StdioConnection | SSEConnection") -> None:
        self.transport = transport
        self.connection = connection
        self.tools = []
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def session(self) -> "ClientSession":
        if self._session is None:
            raise RuntimeError("Session not initialized")
        return self._session

    def _make_execute_on_server(self, tool_name: str) -> t.Callable[..., t.Any]:
        async def execute_on_server(**kwargs: t.Any) -> t.Any:
            result = await self.session.call_tool(tool_name, kwargs)
            return _convert_mcp_result_to_message_parts(result)

        return execute_on_server

    async def _load_tools(self) -> None:
        mcp_tool_result = await self.session.list_tools()

        self.tools.clear()

        self.tools = [
            Tool(
                name=tool.name,
                description=tool.description or "",
                parameters_schema=tool.inputSchema,
                fn=self._make_execute_on_server(tool.name),
            )
            for tool in mcp_tool_result.tools
        ]

    async def _connect_via_stdio(self, connection: "StdioConnection") -> "ClientSession":
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(**connection)
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        return await self._exit_stack.enter_async_context(ClientSession(read, write))

    async def _connect_via_sse(self, connection: "SSEConnection") -> "ClientSession":
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        client = sse_client(**connection)
        read, write = await self._exit_stack.enter_async_context(client)
        return await self._exit_stack.enter_async_context(ClientSession(read, write))

    async def _shutdown(self) -> None:
        await self._exit_stack.aclose()
        self._session = None
        self.tools.clear()

    async def __aenter__(self) -> "MCPClient":
        try:
            if self.transport == "stdio":
                self._session = await self._connect_via_stdio(
                    t.cast("StdioConnection", self.connection),
                )
            elif self.transport == "sse":
                self._session = await self._connect_via_sse(
                    t.cast("SSEConnection", self.connection),
                )
            else:
                raise TypeError(  # noqa: TRY301
                    f"Unsupported transport: {self.transport}. Must be 'stdio' or 'sse'",
                )

            await asyncio.wait_for(self.session.initialize(), timeout=INITIALIZE_TIMEOUT)
            await asyncio.wait_for(self._load_tools(), timeout=INITIALIZE_TIMEOUT)
        except Exception:
            await self._shutdown()
            raise

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._shutdown()


@t.overload
def mcp(
    transport: t.Literal["stdio"],
    *,
    command: str,
    args: list[str] | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> MCPClient:
    """
    Create an MCP client that communicates over stdio.

    Args:
        command: The executable to run to start the server.
        args: Command line arguments to pass to the executable.
        env: The environment to use when spawning the process.

    Returns:
        An MCP client context manager.

    Example:
        ```
        with mcp("stdio", command="uv", args=["run", "weather-mcp"]) as mcp:
            chat = (
                await get_generator("gpt-4o")
                .chat("Please tell me the weather")
                .using(*mcp.tools)
                .run()
            )
        ```
    """


@t.overload
def mcp(
    transport: t.Literal["sse"],
    *,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
    sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
) -> MCPClient:
    """
    Create an MCP client that communicates over SSE.

    Args:
        url: The URL of the SSE endpoint to connect to.
        headers: HTTP headers to send to the SSE endpoint.
        timeout: HTTP timeout.
        sse_read_timeout: SSE read timeout.

    Returns:
        An MCP client context manager.

    Example:
        ```
        with mcp("sse", url="http://localhost:8000/weather") as mcp:
            chat = (
                await get_generator("gpt-4o")
                .chat("Please tell me the weather")
                .using(*mcp.tools)
                .run()
            )
        ```
    """


def mcp(transport: Transport, **connection: t.Any) -> MCPClient:
    return MCPClient(transport, t.cast("StdioConnection | SSEConnection", connection))


def as_mcp(
    *tools: t.Any,
    name: str = "Rigging Tools",
) -> "FastMCP":
    """
    Serves a collection of Rigging tools over the Model Context Protocol (MCP).

    This function creates a FastMCP server instance that exposes your
    Rigging tools to any compliant MCP client. It acts as a bridge, handling
    the conversion between Rigging's `Tool` objects and the MCP specification.

    Args:
        tools: Any number of `rigging.Tool` objects, raw Python functions,
               or class instances with `@tool_method` methods.
        name: The name of the MCP server. This is used for identification.

    Example:
        ```python
        # in my_tool_server.py
        import asyncio
        import rigging as rg

        @rg.tool
        def add_numbers(a: int, b: int) -> int:
            \"\"\"Adds two numbers together.\"\"\"
            return a + b

        if __name__ == "__main__":
            rg.as_mcp(add_numbers).run(
                transport="stdio"
            )
        ```
    """
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.tools import Tool as FastMCPTool

    rigging_tools: list[Tool[..., t.Any]] = []
    for tool in flatten_list(list(tools)):
        interior_tools = [
            val
            for _, val in inspect.getmembers(
                tool,
                predicate=lambda x: isinstance(x, Tool),
            )
        ]
        if interior_tools:
            rigging_tools.extend(interior_tools)
        elif not isinstance(tool, Tool):
            rigging_tools.append(Tool.from_callable(tool))
        else:
            rigging_tools.append(tool)

    fastmcp_tools = [
        FastMCPTool.from_function(
            fn=_create_mcp_handler(tool.fn),
            name=tool.name,
            description=tool.description,
        )
        for tool in rigging_tools
    ]
    return FastMCP(name, tools=fastmcp_tools)
