import typing as t
import uuid
from typing import runtime_checkable

from rigging.generator import GenerateParams
from rigging.message import (
    Message,
    inject_system_content,
)
from rigging.tool.base import FunctionCall, Tool, ToolCall, ToolMode
from rigging.tool.native import (
    TOOL_CALL_TAG,
    JsonInXmlToolDefinition,
    NativeToolCall,
    NativeToolResponse,
    XmlToolDefinition,
    get_native_tool_prompt_part,
)


@runtime_checkable
class PostTransform(t.Protocol):
    def __call__(
        self,
        messages: list[Message],
        params: GenerateParams,
        /,
    ) -> t.Awaitable[tuple[list[Message], GenerateParams]]:
        """
        Passed messages and params to transform.
        """
        ...


@runtime_checkable
class Transform(t.Protocol):
    def __call__(
        self,
        messages: list[Message],
        params: GenerateParams,
        /,
    ) -> t.Awaitable[tuple[list[Message], GenerateParams, PostTransform | None]]:
        """
        Passed messages and params to transform.

        May return an optional post-transform callback to be executed to unwind the transformation.
        """
        ...


def make_native_tool_transform(  # noqa: PLR0915
    tools: list[Tool],
    tool_mode: ToolMode = "xml",
    *,
    add_tool_stop_token: bool = True,
) -> Transform:
    """
    Create a transform that converts tool calls and responses
    to XML or JSON in XML format as needed by any tools which
    use native parsing.
    """

    async def transform_native_tools(  # noqa: PLR0915
        messages: list[Message],
        params: GenerateParams,
    ) -> tuple[list[Message], GenerateParams, PostTransform | None]:
        if not tools or tool_mode not in ["xml", "json-in-xml"]:
            return messages, params, None

        # Render all our existing tool calls and responses

        for message in messages:
            if message.role == "tool":
                message.content = NativeToolResponse(
                    id=message.tool_call_id or "",
                    result=message.content,
                ).to_pretty_xml()
                message.role = "user"

            elif message.tool_calls:
                message.content = "\n".join(
                    [
                        NativeToolCall(
                            name=tool_call.function.name,
                            parameters=tool_call.function.arguments,
                        ).to_pretty_xml()
                        for tool_call in message.tool_calls
                    ],
                )
                message.tool_calls = []  # Clear tool calls after rendering

        # Inject tool definitions into the system prompt

        definitions: list[XmlToolDefinition] | list[JsonInXmlToolDefinition]
        if tool_mode == "xml":
            definitions = [tool.xml_definition for tool in tools]
        else:
            definitions = [tool.json_definition for tool in tools]

        tool_system_prompt = get_native_tool_prompt_part(
            definitions,
            t.cast("t.Literal['xml', 'json-in-xml']", tool_mode),
        )
        messages = inject_system_content(messages, tool_system_prompt)

        # Update generate params

        if add_tool_stop_token:
            params.stop = params.stop or []
            params.stop.append(f"</{TOOL_CALL_TAG}>")

        existing_tool_definitions = params.tools
        params.tools = None
        existing_tool_choice = params.tool_choice
        params.tool_choice = None

        # Build post transform

        async def post_transform_native_tools(
            messages: list[Message],
            params: GenerateParams,
        ) -> tuple[list[Message], GenerateParams]:
            # Re-inject the closing tag if:
            #
            # 1. Are using native tools
            # 2. Set a stop token for the tool calls
            # 3. Hit that stop token

            if (
                add_tool_stop_token
                and tool_mode in ["xml", "json-in-xml"]
                and chat.stop_reason == "stop"
            ):
                for part in chat.last.content_parts:
                    if (
                        part.type == "text"
                        and f"<{TOOL_CALL_TAG}" in part.text
                        and f"</{TOOL_CALL_TAG}>" not in part.text
                    ):
                        part.text += f"</{TOOL_CALL_TAG}>"
                        break

            if not (tool_calls := chat.last.try_parse_set(NativeToolCall)):
                return chat

            # Convert the tool calls and strip them

            chat.last.tool_calls = [
                ToolCall(
                    id=f"rg-{uuid.uuid4().hex[:8]}",
                    function=FunctionCall(
                        name=tool_call.name,
                        arguments=tool_call.parameters,
                    ),
                )
                for tool_call in tool_calls
            ]

            chat.last.strip(NativeToolCall)
            chat.last.content = chat.last.content.strip()

            # Convert any xml calls to json params

            if tool_mode == "xml":
                for tool_call in chat.last.tool_calls:
                    tool = next(
                        (t for t in tools if t.name == tool_call.function.name),
                        None,
                    )
                    if tool is None:
                        continue  # We'll catch this later
                    try:
                        parsed = tool.model.from_text(
                            tool.model.xml_start_tag()
                            + tool_call.function.arguments
                            + tool.model.xml_end_tag(),
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to parse parameters from:\n{tool_call.function.arguments}",
                        ) from e

                    if not parsed:
                        raise ValueError(
                            f"Failed to parse parameters from:\n{tool_call.function.arguments}",
                        )

                    parameters = parsed[0][0]
                    tool_call.function.arguments = parameters.model_dump_json()

            # Convert our tool responses

            for message in chat.all:
                if (tool_response := message.try_parse(NativeToolResponse)) is None:
                    continue

                message.content = tool_response.result
                message.tool_call_id = tool_response.id
                message.role = "tool"

            # Restore the params

            chat.params = chat.params or GenerateParams()
            chat.params.tools = existing_tool_definitions
            chat.params.tool_choice = existing_tool_choice

            # Strip the system message part

            if chat.all and chat.all[0].role == "system":
                chat.all[0].content = (
                    chat.all[0]
                    .content.replace(
                        tool_system_prompt,
                        "",
                    )
                    .strip()
                )

            # If the system message is empty after stripping, remove it

            if chat.messages and chat.messages[0].content == "":
                chat.messages.pop(0)

            return chat

        return messages, params, post_transform_native_tools

    return transform_native_tools
