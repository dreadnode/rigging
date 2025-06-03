import json
import typing as t
import uuid
import warnings
from typing import runtime_checkable

import xmltodict  # type: ignore[import-untyped]

from rigging.error import ToolWarning
from rigging.generator import GenerateParams
from rigging.message import (
    ContentText,
    Message,
    inject_system_content,
)
from rigging.tools.base import FunctionCall, Tool, ToolCall, ToolMode
from rigging.tools.native import (
    TOOL_CALL_TAG,
    NativeToolCall,
    NativeToolResponse,
    get_native_tool_prompt_part,
)

if t.TYPE_CHECKING:
    from rigging.chat import Chat


@runtime_checkable
class PostTransform(t.Protocol):
    def __call__(
        self,
        chat: "Chat",
        /,
    ) -> "t.Awaitable[Chat]":
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
    tools: list[Tool[..., t.Any]],
    tool_mode: ToolMode = "json-in-xml",
    *,
    add_tool_stop_token: bool = True,
) -> Transform:
    """
    Create a transform that converts tool calls and responses
    to XML or JSON in XML format injected and parsed from messages.
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
                message.tool_call_id = None
                # TODO: Would be cleaner to have an injection system for models -> content
                message.try_parse(NativeToolResponse)

            elif message.tool_calls:
                native_tool_calls: list[NativeToolCall] = []
                for tool_call in message.tool_calls:
                    native_call = NativeToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        parameters=tool_call.function.arguments,
                    )

                    if tool_mode != "xml":
                        native_tool_calls.append(native_call)
                        continue

                    xml_parameters: str | None = None

                    # If we still have a reference to the tool that handled this call,
                    # use its model to convert the parameters to XML

                    if tool := next(
                        (t for t in tools if t.name == tool_call.function.name),
                        None,
                    ):
                        try:
                            xml_parameters = (
                                tool.model.model_validate_json(native_call.parameters)
                                .to_pretty_xml()
                                .replace(tool.model.xml_start_tag(), "")
                                .replace(tool.model.xml_end_tag(), "")
                                .strip()
                            )
                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to convert tool call '{tool_call.function.name}' to xml ({e}):\n{tool_call.function.arguments}",
                                ToolWarning,
                                stacklevel=3,
                            )

                    # Fallback to xmltodict as a best-effort if that didn't work

                    if xml_parameters is None:
                        try:
                            xml_parameters = xmltodict.unparse(
                                json.loads(native_call.parameters),
                                pretty=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to convert tool call '{tool_call.function.name}' to xml using xmltodict ({e}):\n{native_call.parameters}",
                                ToolWarning,
                                stacklevel=3,
                            )

                    native_call.parameters = xml_parameters or native_call.parameters
                    native_tool_calls.append(native_call)

                message.content_parts.append(
                    ContentText(
                        text="\n".join(
                            [call.to_pretty_xml() for call in native_tool_calls],
                        ),
                    ),
                )
                message.try_parse_set(NativeToolCall)
                message.tool_calls = None  # Clear tool calls after rendering

        # Inject tool definitions into the system prompt

        definitions = [
            tool.xml_definition if tool_mode == "xml" else tool.json_definition for tool in tools
        ]

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

        async def post_transform_native_tools(chat: "Chat") -> "Chat":  # noqa: PLR0912
            # Re-inject the closing tag if:
            #
            # 1. We are using native tools
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

            # Convert the tool calls and strip them

            for message in [m for m in chat.all if m.role == "assistant"]:
                if not (tool_calls := message.try_parse_set(NativeToolCall)):
                    continue

                message.tool_calls = []

                for native_call in tool_calls:
                    tool_call = ToolCall(
                        id=native_call.id or f"rg-{uuid.uuid4().hex[:8]}",
                        function=FunctionCall(
                            name=native_call.name,
                            arguments=native_call.parameters,
                        ),
                    )

                    # Convert any xml calls to json params

                    if tool_mode == "xml":
                        tool = next(
                            (t for t in tools if t.name == tool_call.function.name),
                            None,
                        )
                        if tool is None:
                            continue  # We'll catch this later

                        arguments_dict: dict[str, t.Any] | None = None

                        try:
                            parsed = tool.model.from_text(
                                tool.model.xml_start_tag()
                                + tool_call.function.arguments
                                + tool.model.xml_end_tag(),
                            )

                            if parsed:
                                arguments_dict = parsed[0][0].model_dump(mode="json")

                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to parse tool call for '{tool_call.function.name}' with arguments ({e}):\n{tool_call.function.arguments}",
                                ToolWarning,
                                stacklevel=3,
                            )

                        # Fallback to xmltodict as a best-effort if that didn't work

                        if arguments_dict is None:
                            try:
                                arguments_dict = xmltodict.parse(
                                    tool_call.function.arguments,
                                )
                            except Exception:  # noqa: BLE001
                                warnings.warn(
                                    f"Failed to parse tool call for '{tool_call.function.name}' with arguments:\n{tool_call.function.arguments}",
                                    ToolWarning,
                                    stacklevel=3,
                                )

                        if arguments_dict is not None:
                            tool_call.function.arguments = json.dumps(arguments_dict)

                    message.tool_calls.append(tool_call)

                message.strip(NativeToolCall)
                message.content = message.content.strip()

            # Convert our tool responses
            # TODO: handle cased where multiple tool responses are present

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
