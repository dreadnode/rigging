from rigging.transform.base import PostTransform, Transform
from rigging.transform.json_tools import (
    JsonToolMode,
    make_tools_to_json_transform,
    tools_to_json_in_xml_transform,
    tools_to_json_transform,
    tools_to_json_with_tag_transform,
)
from rigging.transform.xml_tools import make_tools_to_xml_transform


def get_transform(identifier: str) -> Transform:
    """
    Get a well-known transform by its identifier.

    Args:
        identifier: The identifier of the transform to retrieve.

    Returns:
        The corresponding transform callable.
    """
    match identifier:
        case "json":
            return tools_to_json_transform
        case "json-in-xml":
            return tools_to_json_in_xml_transform
        case "json-with-tag":
            return tools_to_json_with_tag_transform
        case _:
            raise ValueError(f"Unknown transform identifier: {identifier}")


__all__ = [
    "JsonToolMode",
    "PostTransform",
    "Transform",
    "make_tools_to_json_transform",
    "make_tools_to_xml_transform",
    "tools_to_json_in_xml_transform",
    "tools_to_json_transform",
    "tools_to_json_with_tag_transform",
]
