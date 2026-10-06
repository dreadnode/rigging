import importlib
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

import rigging as rg
from rigging.tools.base import FunctionCall, ToolCall
from rigging.tools.mcp import DEFAULT_HTTP_TIMEOUT, DEFAULT_SSE_READ_TIMEOUT

# ruff: noqa: S101, SLF001, ARG001, PLR2004

mcp_module = importlib.import_module("rigging.tools.mcp")


@pytest.fixture
def http_server(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(closed=False, options=None)
    session = AsyncMock()
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="web_search",
                description="Search the web",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ],
    )
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="https://example.com: useful search excerpt")],
    )
    state.session = session

    @asynccontextmanager
    async def transport(**kwargs):
        state.options = kwargs
        try:
            yield "read", "write", lambda: "session-id"
        finally:
            state.closed = True

    @asynccontextmanager
    async def client_session(read, write):
        assert (read, write) == ("read", "write")
        yield session

    monkeypatch.setattr("mcp.client.streamable_http.streamablehttp_client", transport)
    monkeypatch.setattr("mcp.ClientSession", client_session)
    return state


async def test_streamable_http_tools(http_server: SimpleNamespace) -> None:
    headers = {"User-Agent": "rigging/test"}
    client = rg.mcp("streamable-http", url="https://example.com/mcp", headers=headers)
    async with client:
        assert http_server.options == {
            "url": "https://example.com/mcp",
            "headers": headers,
            "timeout": timedelta(seconds=DEFAULT_HTTP_TIMEOUT),
            "sse_read_timeout": timedelta(seconds=DEFAULT_SSE_READ_TIMEOUT),
        }
        http_server.session.initialize.assert_awaited_once()
        tool = client.tools[0]
        assert tool.name == "web_search"
        assert tool.parameters_schema["properties"]["query"] == {"type": "string"}
        message, stop = await tool.handle_tool_call(
            ToolCall(
                id="search", function=FunctionCall(name="web_search", arguments='{"query":"MCP"}')
            ),
        )
        assert "useful search excerpt" in message.content
        assert message.tool_call_id == "search"
        assert not stop
        http_server.session.call_tool.assert_awaited_once_with("web_search", {"query": "MCP"})
    assert http_server.closed
    assert client.tools == []
    with pytest.raises(RuntimeError, match="Session not initialized"):
        _ = client.session


async def test_streamable_http_timeouts(http_server: SimpleNamespace) -> None:
    async with rg.mcp(
        "streamable-http", url="https://example.com/mcp", timeout=12, sse_read_timeout=45
    ):
        assert http_server.options["timeout"] == timedelta(seconds=12)
        assert http_server.options["sse_read_timeout"] == timedelta(seconds=45)
        assert http_server.options["headers"] is None


@pytest.mark.parametrize("operation", ["initialize", "list_tools"])
async def test_streamable_http_setup_failure(http_server: SimpleNamespace, operation: str) -> None:
    getattr(http_server.session, operation).side_effect = RuntimeError("server failed")
    client = rg.mcp("streamable-http", url="https://example.com/mcp")
    with pytest.raises(RuntimeError, match="server failed"):
        async with client:
            pass
    assert http_server.closed
    assert client.tools == []
    assert client._session is None


@pytest.mark.parametrize("transport", ["stdio", "sse"])
async def test_existing_transport_selection(
    monkeypatch: pytest.MonkeyPatch, transport: str
) -> None:
    session = AsyncMock()
    session.list_tools.return_value = ListToolsResult(tools=[])
    connect = AsyncMock(return_value=session)
    monkeypatch.setattr(mcp_module.MCPClient, f"_connect_via_{transport}", connect)
    connection = (
        {"command": "server"} if transport == "stdio" else {"url": "https://example.com/sse"}
    )
    client = mcp_module.MCPClient(transport, connection)
    async with client:
        connect.assert_awaited_once_with(connection)
        session.initialize.assert_awaited_once()


async def test_unsupported_transport() -> None:
    with pytest.raises(TypeError, match="streamable-http"):
        async with mcp_module.MCPClient("unknown", {}):
            pass


async def test_streamable_http_cancelled_tool(http_server: SimpleNamespace) -> None:
    import asyncio

    http_server.session.call_tool.side_effect = asyncio.CancelledError
    client = rg.mcp("streamable-http", url="https://example.com/mcp")
    with pytest.raises(asyncio.CancelledError):
        async with client:
            await client.tools[0](query="MCP")
    assert http_server.closed
    assert client._session is None
    assert client.tools == []
