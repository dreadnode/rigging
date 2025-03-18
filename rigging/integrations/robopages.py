import json
import re
import typing as t
from functools import cached_property
from urllib.parse import urlparse

import requests
import typing_extensions as te
from loguru import logger

from rigging.message import Message
from rigging.tool import ApiTool, Tool, ToolType
from rigging.tool.api import FunctionDefinition, ToolCall, ToolDefinition
from rigging.tracing import tracer


class RobopagesApiTool(ApiTool):
    """
    Overload of the ApiTool class to support making tool calls through a Robopages server.
    """

    def __init__(self, url: str, name: str, description: str, parameters: t.Optional[dict[str, t.Any]] = None) -> None:
        self._url = urlparse(url)._replace(path="").geturl()
        self._name = name
        self._description = description
        self._parameters = parameters

    @cached_property
    def name(self) -> str:
        return self._name

    @cached_property
    def description(self) -> str:
        return self._name

    @cached_property
    def schema(self) -> dict[str, t.Any]:
        return self._parameters or {}

    @cached_property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionDefinition(name=self.name, description=self.description, parameters=self._parameters)
        )

    async def execute(self, tool_call: ToolCall) -> Message:
        """Executes a function call on the tool."""

        from rigging.message import Message

        if tool_call.function.name != self.name:
            raise ValueError(f"Function name {tool_call.function.name} does not match {self.name}")

        with tracer.span(f"Robopages Tool {self.name}()", name=self.name, tool_call_id=tool_call.id) as span:
            args = json.loads(tool_call.function.arguments)
            span.set_attribute("arguments", args)

            response = requests.post(
                f"{self._url}/process",
                json=[
                    {
                        "type": "function",
                        "id": tool_call.id,
                        "function": {
                            "name": self._name,
                            "arguments": args,
                        },
                    }
                ],
            )
            if response.status_code not in [200, 400]:
                response.raise_for_status()

            if response.status_code == 400:
                result = response.content.decode()
            else:
                result = response.json()[0]["content"]

            span.set_attribute("result", result)

        return Message(role="tool", tool_call_id=tool_call.id, content=str(result))

    __call__ = execute


class OpenAIFunction(te.TypedDict):
    name: str
    description: str
    parameters: dict[str, t.Any]


class OpenAIFormat(te.TypedDict):
    type: str
    function: OpenAIFunction


class RobopagesFunction(te.TypedDict):
    name: str
    description: str
    parameters: list[dict[str, t.Any]]
    page_name: str
    page_description: str


@t.overload
def robopages(url: str, *, tool_type: t.Literal["api"] = "api") -> list[RobopagesApiTool]:
    ...


@t.overload
def robopages(url: str, *, tool_type: t.Literal["native"]) -> list[Tool]:
    ...


def robopages(
    url: str, *, tool_type: ToolType = "api", name_filter: str | None = None
) -> list[RobopagesApiTool] | list[Tool]:
    """
    Create a list of tools from a Robopages server.

    Args:
        url: The URL of the Robopages server.
        tool_type: The type of tools to fetch.
        name_filter: A regular expression to filter the tools by name.

    Returns:
        A list of tools from the Robopages server.

    Example:

        ```python
        import rigging as rg

        tools = rg.integrations.robopages("http://localhost:8080")

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

    response = requests.get(url, params={"flavor": "rigging" if tool_type == "native" else "openai"})
    response.raise_for_status()
    tools_data = response.json()

    if tool_type == "api":
        openai_tools = t.cast(list[OpenAIFormat], tools_data)

        logger.info(f"Fetched {len(openai_tools)} functions from Robopages ({url})")

        tools: list[RobopagesApiTool] = []
        for tool in openai_tools:
            function = tool["function"]
            if filter_regex and not filter_regex.search(function["name"]):
                logger.debug(f"Skipping function {function['name']}")
                continue
            tools.append(RobopagesApiTool(url, function["name"], function["description"], function.get("parameters")))

        return tools

    raise NotImplementedError("Only API tools are supported right now")
