import json
import typing as t
import uuid
import warnings

from pydantic_xml import attr, element
from pydantic_xml import create_model as pydantic_xml_create_model

from rigging.error import ToolWarning
from rigging.generator import GenerateParams
from rigging.message import (
    Message,
    inject_system_content,
    strip_system_content,
)
from rigging.model import Model
from rigging.tools.base import FunctionCall, ToolCall, ToolDefinition, ToolResponse
from rigging.transform.base import PostTransform, Transform
from rigging.util import extract_json_objects

if t.TYPE_CHECKING:
    from rigging.chat import Chat

# Types

JsonToolMode = t.Literal["json", "json-in-xml", "json-with-tag"]


@t.runtime_checkable
class ToolPromptCallable(t.Protocol):
    def __call__(
        self,
        tools: list["ToolDefinition"],
        tool_call_tag: str | None,
    ) -> str:
        """
        Callable that generates a tool prompt string from a list of tool definitions and an optional tool call tag.
        """
        ...


# Models for psudo-xml formats


class JsonInXmlToolDefinition(Model, tag="tool"):
    name: str = attr()
    description: str = attr()
    parameters: str = element()


class JsonInXmlToolCall(Model):
    id: str = attr(default="")
    name: str = attr()
    parameters: str

    def __str__(self) -> str:
        return f"<JsonInXmlToolCall id={self.id} {self.name}({self.parameters})>"


class JsonToolCall(Model):
    id: str = attr(default="")
    content: str

    def __str__(self) -> str:
        return f"<JsonToolCall id={self.id} {self.content}>"


# Prompts


def json_tools_prompt(tools: list[ToolDefinition], tool_call_tag: str | None = None) -> str:
    tools_str = "\n".join(tool.model_dump_json() for tool in tools)
    if tool_call_tag:
        tool_call_format = f'<{tool_call_tag}>\n{{"name": <function-name>, "arguments": <args-dict>}}\n</{tool_call_tag}>'
    else:
        tool_call_format = '{"name": <function-name>, "arguments": <args-dict>}'
    return f"""\
# Tools

You may call one or more functions to assist with the user query. \
Don't make assumptions about what values to plug into functions.

<tools>
{tools_str}
</tools>

To call a function, respond with the following format:

{tool_call_format}
"""


def json_in_xml_tools_prompt(
    tools: list[ToolDefinition],
    tool_call_tag: str | None = "tool-call",
) -> str:
    definitions = [
        JsonInXmlToolDefinition(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=json.dumps(tool.function.parameters),
        )
        for tool in tools or []
    ]
    tools_str = "\n".join([definition.to_pretty_xml() for definition in definitions])
    return f"""\
# Tools

You may call one or more functions to assist with the user query. \
Don't make assumptions about what values to plug into functions.

<tools>
{tools_str}
</tools>

To call a function, respond with the following format:

<{tool_call_tag} name="$tool_name">
{{"$param_name": "argument one", "$param_name": 123}}
</{tool_call_tag}>

Arguments should be provided as a valid JSON object between the tags.\
"""


# Transform


def make_tools_to_json_transform(  # noqa: PLR0915
    mode: JsonToolMode = "json-with-tag",
    *,
    system_tool_prompt: ToolPromptCallable | str | None = None,
    tool_responses_as_user_messages: bool = True,
    tool_call_tag: str | None = None,
    tool_response_tag: str | None = None,
) -> Transform:
    """
    Create a transform that converts tool calls and responses to various JSON formats.

    Args:
        mode: The mode of JSON format to use. Options are "json", "json-in-xml", or "json-with-tag".
        system_tool_prompt: A callable or string that generates the system prompt for tools.
        tool_responses_as_user_messages: If True, tool responses will be converted to user messages wrapped in tool response tags.
        tool_call_tag: The tag to use for tool calls in the JSON format.
        tool_response_tag: The tag to use for tool responses in the JSON format.

    Returns:
        A Transform that processes messages to convert tool calls and responses to the specified JSON format.
    """

    match mode:
        case "json":
            system_tool_prompt = system_tool_prompt or json_tools_prompt
        case "json-in-xml":
            system_tool_prompt = system_tool_prompt or json_in_xml_tools_prompt
            tool_call_tag = tool_call_tag or "tool-call"
            tool_response_tag = tool_response_tag or "tool-response"
        case "json-with-tag":
            system_tool_prompt = system_tool_prompt or json_tools_prompt
            tool_call_tag = tool_call_tag or "tool-call"
            tool_response_tag = tool_response_tag or "tool-response"
        case _:
            raise ValueError(f"Invalid mode: {mode}")

    json_tool_call_cls = pydantic_xml_create_model(
        "JsonToolCall",
        __base__=JsonToolCall,
        __cls_kwargs__={"tag": tool_call_tag or "tool-call"},
        __tag__=tool_call_tag or "tool-call",
    )
    json_in_xml_tool_call_cls = pydantic_xml_create_model(
        "JsonInXmlToolCall",
        __base__=JsonInXmlToolCall,
        __cls_kwargs__={"tag": tool_call_tag or "tool-call"},
        __tag__=tool_call_tag or "tool-call",
    )
    tool_response_cls = pydantic_xml_create_model(
        "ToolResponse",
        __base__=ToolResponse,
        __cls_kwargs__={"tag": tool_response_tag or "tool-response"},
        __tag__=tool_response_tag or "tool-response",
    )

    async def tools_to_json_transform(  # noqa: PLR0915
        messages: list[Message],
        params: GenerateParams,
    ) -> tuple[list[Message], GenerateParams, PostTransform | None]:
        # Inject tool definitions into the system prompt

        system_prompt = (
            system_tool_prompt
            if isinstance(system_tool_prompt, str)
            else system_tool_prompt(params.tools or [], tool_call_tag=tool_call_tag)
        )
        messages = inject_system_content(messages, system_prompt)

        # Render all our existing tool calls as JSON in the content

        for message in messages:
            if tool_responses_as_user_messages and message.role == "tool":
                message.replace_with_slice(
                    tool_response_cls(
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
                    content: str | Model
                    match mode:
                        case "json":
                            content = json.dumps(
                                {
                                    "name": tool_call.function.name,
                                    "parameters": json.loads(tool_call.function.arguments),
                                },
                            )
                        case "json-in-xml":
                            content = json_in_xml_tool_call_cls(
                                id=tool_call.id,
                                name=tool_call.function.name,
                                parameters=tool_call.function.arguments,
                            )
                        case "json-with-tag":
                            content = json_tool_call_cls(
                                id=tool_call.id,
                                content=json.dumps(
                                    {
                                        "name": tool_call.function.name,
                                        "arguments": json.loads(tool_call.function.arguments),
                                    },
                                ),
                            )

                    message.append_slice(
                        content,
                        "tool_call",
                        obj=tool_call,
                        metadata={"id": tool_call.id or ""},
                    )

                message.tool_calls = None

        # Save any existing tool params

        existing_tool_definitions = params.tools
        params.tools = None
        existing_tool_choice = params.tool_choice
        params.tool_choice = None

        # Build post transform

        async def json_to_tools_transform(chat: "Chat") -> "Chat":  # noqa: PLR0912
            # Convert the tool calls and strip them

            for message in [m for m in chat.all if m.role == "assistant"]:
                # Restore original tool calls - fast path for efficiency and consistency

                for slice_ in message.slices:
                    if slice_.type == "tool_call" and isinstance(slice_.obj, ToolCall):
                        message.tool_calls = message.tool_calls or []
                        message.tool_calls.append(slice_.obj)
                        message.remove_slices(slice_)

                # Otherwise, find any new tool calls in the content

                if mode == "json":
                    parsed_objects = extract_json_objects(message.content)
                    if not parsed_objects:
                        continue

                    for obj, obj_slice in parsed_objects:
                        if (
                            not isinstance(obj, dict)
                            or "name" not in obj
                            or ("parameters" not in obj and "arguments" not in obj)
                        ):
                            continue

                        message.tool_calls = message.tool_calls or []
                        message.tool_calls.append(
                            ToolCall(
                                id=f"rg-{uuid.uuid4().hex[:8]}",
                                function=FunctionCall(
                                    name=str(obj["name"]),
                                    arguments=json.dumps(
                                        obj.get("parameters", obj.get("arguments", {})),
                                    ),
                                ),
                            ),
                        )
                        message.content = (
                            message.content[: obj_slice.start] + message.content[obj_slice.stop :]
                        )

                elif mode == "json-in-xml":
                    if not (tool_calls := message.try_parse_set(json_in_xml_tool_call_cls)):
                        continue

                    message.tool_calls = []
                    for tool_call in tool_calls:
                        message.tool_calls.append(
                            ToolCall(
                                id=tool_call.id or f"rg-{uuid.uuid4().hex[:8]}",
                                function=FunctionCall(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                ),
                            ),
                        )

                    message.remove_slices(json_in_xml_tool_call_cls)

                elif mode == "json-with-tag":
                    if not (tag_tool_calls := message.try_parse_set(json_tool_call_cls)):
                        continue

                    message.tool_calls = []
                    for tag_tool_call in tag_tool_calls:
                        try:
                            json_native_call = json.loads(tag_tool_call.content)
                            message.tool_calls.append(
                                ToolCall(
                                    id=f"rg-{uuid.uuid4().hex[:8]}",
                                    function=FunctionCall(
                                        name=json_native_call.get("name", ""),
                                        arguments=json.dumps(
                                            json_native_call.get(
                                                "arguments",
                                                json_native_call.get("parameters", {}),
                                            ),
                                        ),
                                    ),
                                ),
                            )
                        except Exception as e:  # noqa: BLE001, PERF203
                            warnings.warn(
                                f"Failed to parse tool call content ({e}):\n{tag_tool_call.content}",
                                ToolWarning,
                                stacklevel=2,
                            )

                    message.strip(json_tool_call_cls)

            # Convert our tool responses

            for message in [m for m in chat.all if m.role == "user"]:
                if (tool_response := message.try_parse(tool_response_cls)) is None:
                    continue

                message.content = tool_response.result
                message.tool_call_id = tool_response.id
                message.role = "tool"

            # Restore the params

            chat.params = chat.params or GenerateParams()
            chat.params.tools = existing_tool_definitions
            chat.params.tool_choice = existing_tool_choice

            # Strip the system prompt content

            chat.messages = strip_system_content(chat.messages, system_prompt)

            return chat

        return messages, params, json_to_tools_transform

    return tools_to_json_transform


tools_to_json_transform = make_tools_to_json_transform(mode="json")
"""
Transform that converts tool calls and responses to a raw JSON format.

Tool calls are represented as JSON objects in the content with `name` and `arguments` fields, and
tool responses are converted to user messages with a "tool_response" type.

See `make_tools_to_json_transform` for more details and more behavior options.
"""

tools_to_json_with_tag_transform = make_tools_to_json_transform(mode="json-with-tag")
"""
Transform that converts tool calls and responses to a JSON format wrapped in a tag for easier identification.

Tool calls are represented as JSON objects in the content with a "tool-call" tag, and
tool responses are converted to user messages with a "tool_response" type.

See `make_tools_to_json_transform` for more details and more behavior options.
"""

tools_to_json_in_xml_transform = make_tools_to_json_transform(mode="json-in-xml")
"""
Transform that converts tool calls and responses to a JSON format for arguments and XML for tool
names and identifiers during calls.

Tool calls are represented as XML elements with a "tool-call" tag containing JSON parameters within
the xml tags, and tool responses are converted to user messages with a "tool_response" type.

See `make_tools_to_json_transform` for more details and more behavior options.
"""
