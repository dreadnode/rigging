"""
Utilities for integrating tools from a Robopages server.
"""

import re
import typing as t

import httpx
import requests
from loguru import logger
from pydantic import TypeAdapter

from rigging.tools.base import Tool, ToolDefinition

DEFAULT_HTTP_TIMEOUT = 10


def make_execute_on_server(url: str, tool_name: str) -> t.Callable[..., t.Any]:
    async def execute_on_server(**kwargs: t.Any) -> t.Any:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/process",
                json=[
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": kwargs,
                        },
                    },
                ],
            )
            if response.status_code not in [200, 400]:
                response.raise_for_status()

            if response.status_code == 400:  # noqa: PLR2004
                result = response.content.decode()
            else:
                result = response.json()[0]["content"]

            return result

    return execute_on_server


def robopages(url: str, *, name_filter: str | None = None) -> list[Tool[..., t.Any]]:
    """
    Create a list of tools from a Robopages server.

    Args:
        url: The URL of the Robopages server.
        name_filter: A regular expression to filter the tools by name.

    Returns:
        A list of integrated tools which leverage the Robopages server.

    Example:
        ```
        import rigging as rg

        tools = rg.tool.robopages("http://localhost:8080")

        chat = (
            await rg.get_generator('gpt-4o')
            .chat('Please use tools')
            .using(*tools)
            .run()
        )

        print(chat.conversation)
        ```
    """

    filter_regex = re.compile(name_filter) if name_filter else None

    response = requests.get(url, params={"flavor": "openai"}, timeout=DEFAULT_HTTP_TIMEOUT)
    response.raise_for_status()
    tools_data = response.json()

    adapter = TypeAdapter(list[ToolDefinition])
    tool_definitions = adapter.validate_python(tools_data)

    logger.info(f"Fetched {len(tool_definitions)} functions from Robopages ({url})")

    tools: list[Tool[..., t.Any]] = []
    for definition in tool_definitions:
        function = definition.function

        if filter_regex and not filter_regex.search(function.name):
            logger.debug(f"Skipping function {function.name}")
            continue

        tools.append(
            Tool(
                function.name,
                function.description or "",
                function.parameters or {},
                make_execute_on_server(url, function.name),
            ),
        )

    return tools
