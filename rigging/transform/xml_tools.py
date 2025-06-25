import inspect
import json
import typing as t
import uuid
import warnings

import xmltodict  # type: ignore[import-untyped]
from pydantic.fields import FieldInfo
from pydantic_xml import attr

from rigging.error import ToolWarning
from rigging.generator import GenerateParams
from rigging.message import (
    Message,
    inject_system_content,
    strip_system_content,
)
from rigging.model import Model
from rigging.tools.base import FunctionCall, Tool, ToolCall, ToolResponse
from rigging.transform.base import PostTransform, Transform

if t.TYPE_CHECKING:
    from rigging.chat import Chat

TOOL_CALL_TAG: t.Literal["rg-tool-call"] = "rg-tool-call"


def _get_field_type_name(annotation: type[t.Any]) -> str:
    """Extract a clean type name from a type annotation."""

    origin = t.get_origin(annotation)
    args = t.get_args(annotation)

    if origin is list or origin is t.List:  # noqa: UP006
        if args:
            item_type = _get_field_type_name(args[0])
            return f"list[{item_type}]"
        return "list"

    if origin is dict or origin is t.Dict:  # noqa: UP006
        if len(args) == 2:  # noqa: PLR2004
            key_type = _get_field_type_name(args[0])
            value_type = _get_field_type_name(args[1])
            return f"dict[{key_type}, {value_type}]"
        return "dict"

    if origin is not None:
        origin_name = origin.__name__.lower()
        args_str = ", ".join(_get_field_type_name(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    if inspect.isclass(annotation):
        return annotation.__name__.lower()

    return str(annotation).replace("class '", "").replace("'", "").split(".")[-1]


def _make_parameter_xml(field_name: str, field: FieldInfo) -> str:
    """Create an XML representation of a parameter."""

    if field.annotation is None:
        raise ValueError(f"Field '{field_name}' has no type annotation")

    type_name = _get_field_type_name(field.annotation)
    description = field.description or ""
    required = field.is_required()

    nested_example = ""
    if field.annotation is list or field.annotation is t.List:  # noqa: UP006
        list_item_type = t.get_args(field.annotation)[0]
        if hasattr(list_item_type, "xml_example"):
            nested_example = list_item_type.xml_example()

    if field.annotation is dict or field.annotation is t.Dict:  # noqa: UP006
        key_type, value_type = t.get_args(field.annotation)
        if hasattr(key_type, "xml_example"):
            nested_example = key_type.xml_example()

    if hasattr(field.annotation, "xml_example"):
        nested_example = field.annotation.xml_example()

    description_part = f' description="{description}"' if description else ""
    required_part = f' required="{str(required).lower()}"'

    if nested_example:
        return f'<param name="{field_name}" type="{type_name}"{description_part}{required_part}>{nested_example}</param>'

    return f'<param name="{field_name}" type="{type_name}"{description_part}{required_part}/>'


class XmlToolDefinition(Model, tag="rg-tool"):
    name: str = attr()
    description: str = attr()
    parameters: str  # don't use element() here, we want to keep the raw xml

    @classmethod
    def from_parameter_model(
        cls,
        model_class: type[Model],
        name: str,
        description: str,
    ) -> "XmlToolDefinition":
        params_xml = "<parameters>"
        for field_name, field in model_class.model_fields.items():
            params_xml += _make_parameter_xml(field_name, field)
        params_xml += "</parameters>"

        return cls(name=name, description=description, parameters=params_xml)


class XmlToolCall(Model, tag=TOOL_CALL_TAG):
    id: str = attr(default="")
    name: str = attr()
    parameters: str

    def __str__(self) -> str:
        return f"<XmlToolCall {self.name}({self.parameters})>"


XML_TOOLS_PREFIX = f"""\
# Tools

You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.

<tools>
**TOOLS**
</tools>

To use a tool, respond with the following format:

<{TOOL_CALL_TAG} name="$tool_name">
<$param_name>argument one</$param_name>
<$param_name>123</$param_name>
</{TOOL_CALL_TAG}>

If a parameter is a primitive list, provide child elements as items:
<numbers>
    <item>1</item>
    <item>2</item>
</numbers>

If a parameter is a list of objects, provide them as named child elements:
<things>
    <thing>
        <foo>bar</foo>
    </thing>
    <thing>
        <foo>baz</foo>
    </thing>
</things>

If a parameter is a dictionary, provide key-value pairs as attributes:
<dict key1="value1" key2="123" />\
"""


def make_tools_to_xml_transform(  # noqa: PLR0915
    tools: list[Tool[..., t.Any]],
    *,
    add_tool_stop_token: bool = True,
) -> Transform:
    """
    Create a transform that converts tool calls and responses
    to Rigging native XML formats.

    This transform will:
    1. Inject tool definitions into the system prompt.
    2. Convert existing tool calls in messages to XML format.
    3. Convert tool responses to XML format.
    4. Optionally add a stop token for tool calls.
    5. Convert tool calls back to native Rigging format after generation.
    6. Handle XML parsing and conversion errors gracefully.

    Args:
        tools: List of Tool instances to convert.
        add_tool_stop_token: Whether to add a stop token for tool calls.

    Returns:
        A transform function that processes messages and generate params,
    """

    async def tools_to_xml_transform(  # noqa: PLR0915
        messages: list[Message],
        params: GenerateParams,
    ) -> tuple[list[Message], GenerateParams, PostTransform | None]:
        # Inject tool definitions into the system prompt

        definitions = [
            XmlToolDefinition.from_parameter_model(
                tool.model,
                tool.name,
                tool.description,
            )
            for tool in tools
        ]

        definitions_str = "\n".join([definition.to_pretty_xml() for definition in definitions])
        tool_system_prompt = XML_TOOLS_PREFIX.replace("**TOOLS**", definitions_str)
        messages = inject_system_content(messages, tool_system_prompt)

        # Render all our existing tool calls and responses

        for message in messages:
            if message.role == "tool":
                message.replace_with_slice(
                    ToolResponse(
                        id=message.tool_call_id or "",
                        result=message.content,
                    ),
                    "tool_response",
                    metadata={"id": message.tool_call_id or ""},
                )
                message.role = "user"
                message.tool_call_id = None

            elif message.tool_calls:
                for tool_call in message.tool_calls:
                    parameters = tool_call.function.arguments

                    # If we still have a reference to the tool that handled this call,
                    # use its model to convert the parameters to XML

                    if tool := next(
                        (t for t in tools if t.name == tool_call.function.name),
                        None,
                    ):
                        try:
                            parameters = (
                                tool.model.model_validate_json(parameters)
                                .to_pretty_xml()
                                .replace(tool.model.xml_start_tag(), "")
                                .replace(tool.model.xml_end_tag(), "")
                                .strip()
                            )
                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to convert tool call '{tool_call.function.name}' to xml ({e}):\n{parameters}",
                                ToolWarning,
                                stacklevel=2,
                            )

                    # Fallback to xmltodict as a best-effort if that didn't work

                    if parameters is None:
                        try:
                            parameters = xmltodict.unparse(
                                json.loads(parameters),
                                pretty=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to convert tool call '{tool_call.function.name}' to xml using xmltodict ({e}):\n{parameters}",
                                ToolWarning,
                                stacklevel=2,
                            )

                    message.append_slice(
                        XmlToolCall(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            parameters=parameters,
                        ),
                        "tool_call",
                        obj=tool_call,
                        metadata={"id": tool_call.id or ""},
                    )

                message.tool_calls = None  # Clear tool calls after rendering

        # Update generate params and save any existing tool params
        existing_stop = params.stop or []
        if add_tool_stop_token:
            params.stop = params.stop or []
            params.stop = list(set(existing_stop) | {f"</{TOOL_CALL_TAG}>"})

        existing_tool_definitions = params.tools
        params.tools = None
        existing_tool_choice = params.tool_choice
        params.tool_choice = None

        async def xml_to_tools_transform(chat: "Chat") -> "Chat":  # noqa: PLR0912
            # Re-inject the closing tag if:
            #
            # 1. We are using native tools
            # 2. Set a stop token for the tool calls
            # 3. Hit that stop token

            if add_tool_stop_token and chat.stop_reason == "stop":
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
                # Restore original tool calls - fast path for efficiency and consistency

                for slice_ in message.slices:
                    if slice_.type == "tool_call" and isinstance(slice_.obj, ToolCall):
                        message.tool_calls = message.tool_calls or []
                        message.tool_calls.append(slice_.obj)
                        message.remove_slices(slice_)

                # Otherwise, find any new tool calls in the content

                if not (tool_calls := message.try_parse_set(XmlToolCall)):
                    continue

                message.tool_calls = []
                for native_call in tool_calls:
                    arguments = native_call.parameters
                    arguments_dict: dict[str, t.Any] | None = None

                    tool = next(
                        (t for t in tools if t.name == native_call.name),
                        None,
                    )
                    if tool is None:
                        warnings.warn(
                            f"Tool call '{native_call.name}' not found in tool definitions, parsing may be incorrect.",
                            ToolWarning,
                            stacklevel=2,
                        )
                    else:
                        try:
                            if parsed := tool.model.from_text(
                                tool.model.xml_start_tag() + arguments + tool.model.xml_end_tag(),
                            ):
                                arguments_dict = parsed[0][0].model_dump(mode="json")
                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to parse tool call for '{native_call.name}' with arguments ({e}):\n{arguments}",
                                ToolWarning,
                                stacklevel=2,
                            )

                    # Fallback to xmltodict as a best-effort if that didn't work

                    if arguments_dict is None:
                        try:
                            arguments_dict = xmltodict.parse(
                                f"<content>{arguments}</content>",
                            )["content"]
                        except Exception as e:  # noqa: BLE001
                            warnings.warn(
                                f"Failed to parse tool call for '{native_call.name}' with arguments using xmltodict ({e}):\n{arguments}",
                                ToolWarning,
                                stacklevel=2,
                            )

                    if arguments_dict is not None:
                        arguments = json.dumps(arguments_dict)

                    message.tool_calls.append(
                        ToolCall(
                            id=native_call.id or f"rg-{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=native_call.name,
                                arguments=arguments,
                            ),
                        ),
                    )

                message.remove_slices(XmlToolCall)

            # Convert our tool responses
            # TODO: handle cased where multiple tool responses are present

            for message in chat.all:
                if (tool_response := message.try_parse(ToolResponse)) is None:
                    continue

                message.content = tool_response.result
                message.tool_call_id = tool_response.id
                message.role = "tool"

            # Restore the params

            chat.params = chat.params or GenerateParams()
            chat.params.tools = existing_tool_definitions
            chat.params.tool_choice = existing_tool_choice
            chat.params.stop = existing_stop

            # Strip the system message part

            chat.messages = strip_system_content(chat.messages, tool_system_prompt)

            return chat

        return messages, params, xml_to_tools_transform

    return tools_to_xml_transform
