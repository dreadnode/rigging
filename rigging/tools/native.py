"""
Models and utilities for defining and working with native-parsed tools.
"""

import inspect
import typing as t

from pydantic.fields import FieldInfo
from pydantic_xml import attr, element

from rigging.model import Model

TOOL_CALL_TAG: t.Literal["rg-tool-call"] = "rg-tool-call"
TOOL_RESPONSE_TAG: t.Literal["rg-tool-response"] = "rg-tool-response"

DEFAULT_NATIVE_TOOL_MODE: t.Literal["json-in-xml"] = "json-in-xml"

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


# json-in-xml


class JsonInXmlToolDefinition(Model, tag="rg:tool"):
    name: str = attr()
    description: str = attr()
    parameters: str = element()


# common


class NativeToolCall(Model, tag=TOOL_CALL_TAG):
    id: str = attr(default="")
    name: str = attr()
    parameters: str

    def __str__(self) -> str:
        return f"<NativeToolCall {self.name}({self.parameters})>"


class NativeToolResponse(Model, tag=TOOL_RESPONSE_TAG):
    id: str = attr(default="")
    result: str
