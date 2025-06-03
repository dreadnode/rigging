"""
Utilities for communicating with MCP servers.
"""

import asyncio
import typing as t
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import typing_extensions as te
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextResourceContents

from rigging.tools.base import Tool

if t.TYPE_CHECKING:
    from rigging.message import Content

INITIALIZE_TIMEOUT = 5
DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

Transport = t.Literal["stdio", "sse"]


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


def _convert_mcp_result_to_message_parts(result: CallToolResult) -> list[t.Any]:
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


@dataclass
class MCPClient:
    """A client for communicating with MCP servers."""

    transport: Transport
    """The transport to use"""
    connection: StdioConnection | SSEConnection
    """Connection configuration"""
    tools: list[Tool[..., t.Any]]
    """A list of tools available on the server"""

    def __init__(self, transport: Transport, connection: StdioConnection | SSEConnection) -> None:
        self.transport = transport
        self.connection = connection
        self.tools = []
        self._exit_stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def session(self) -> ClientSession:
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

    async def _connect_via_stdio(self, connection: StdioConnection) -> ClientSession:
        server_params = StdioServerParameters(**connection)
        read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        return await self._exit_stack.enter_async_context(ClientSession(read, write))

    async def _connect_via_sse(self, connection: SSEConnection) -> ClientSession:
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
