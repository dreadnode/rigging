"""
Models and utilities for defining and working with native-parsed tools.
"""

import inspect
import typing as t

from pydantic.fields import FieldInfo
from pydantic_xml import attr, element

from rigging.model import Model

TOOL_CALLS_TAG = "rg:tool-calls"

# xml


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


class XmlToolDefinition(Model, tag="tool-def"):
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


class XmlToolCall(Model, tag="invoke"):
    name: str = attr()
    parameters: str


# json-in-xml


class JsonInXmlToolDefinition(Model, tag="tool-def"):
    name: str = attr()
    description: str = attr()
    parameters: str = element()


class JsonInXmlToolCall(Model, tag="invoke"):
    name: str = attr()
    parameters: str


# results


class NativeToolResult(Model, tag="tool-result"):
    name: str = attr()
    result: str


# prompts

XML_CALL_FORMAT = """\
To use a tool, respond with the following format:

<rg:tool-calls>
    <invoke name="$tool_name">
        <$param_name>argument one</$param_name>
        <$param_name>123</$param_name>
    </invoke>
</rg:tool-calls>

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

XML_IN_JSON_CALL_FORMAT = """\
To use a tool, respond with the following format:

<rg:tool-calls>
    <invoke name="$tool_name">
        {"$param_name": "argument one", "$param_name": 123}
    </invoke>
</rg:tool-calls>

Arguments should be provided as a valid JSON object between the tags.\
"""


def tool_description_prompt_part(
    tool_descriptions: list[XmlToolDefinition] | list[JsonInXmlToolDefinition],
    mode: t.Literal["xml", "json-in-xml"],
) -> str:
    call_format = XML_CALL_FORMAT if mode == "xml" else XML_IN_JSON_CALL_FORMAT
    tool_definitions = "\n".join([tool.to_pretty_xml() for tool in tool_descriptions])
    return f"""\
# Tool Use
In this environment you have access to a set of tools you can use.

## Available Tools
<tools>
{tool_definitions}
</tools>

## Tool Call Format
{call_format}

## Tool Use Instructions
- Answer the user's request using the relevant tool(s), if they are available.
- You may issue multiple tools in a single response if needed.
- Check that all the required parameters for each tool call are provided or can reasonably be inferred from context.
- If there are no relevant tools or there are missing values for required parameters, ask the user to supply these values; otherwise proceed with the tool calls.
- Carefully analyze descriptive terms in the request as they may indicate required parameter values that should be included even if not explicitly quoted.
"""
