import re

from pydantic import alias_generators


def escape_xml(xml_string: str) -> str:
    prepared = re.sub(r"&(?!(?:amp|lt|gt|apos|quot);)", "&amp;", xml_string)

    return prepared


def unescape_xml(xml_string: str) -> str:
    unescaped = re.sub(r"&amp;", "&", xml_string)
    unescaped = re.sub(r"&lt;", "<", unescaped)
    unescaped = re.sub(r"&gt;", ">", unescaped)
    unescaped = re.sub(r"&apos;", "'", unescaped)
    unescaped = re.sub(r"&quot;", '"', unescaped)

    return unescaped


def to_snake(text: str) -> str:
    return alias_generators.to_snake(text).replace("-", "_")


def to_xml_tag(text: str) -> str:
    return to_snake(text).replace("_", "-").strip("-")
