import re
import typing as t
from xml.etree import ElementTree as ET

from pydantic import ValidationError, field_validator
from pydantic.alias_generators import to_snake
from pydantic_xml import BaseXmlModel
from pydantic_xml.element import SearchMode  # type: ignore [attr-defined]
from pydantic_xml.typedefs import NsMap

from rigging.error import MissingModelError

#
# Core XML serializable models for messages
#

CoreModelGeneric = t.TypeVar("CoreModelGeneric", bound="CoreModel")

# TODO: pydantic-xml isn't a great fit for our use case given
# It's strictness for parsing XML and expecting all interior
# content to be escaped. We should probably just write something
# custom for our use case that supports JSON, YAML, and XML


class CoreModel(BaseXmlModel):
    def __init_subclass__(
        cls,
        tag: str | None = None,
        ns: str | None = None,
        nsmap: NsMap | None = None,
        ns_attrs: bool | None = None,
        skip_empty: bool | None = None,
        search_mode: SearchMode | None = None,
        **kwargs: t.Any,
    ):
        super().__init_subclass__(tag, ns, nsmap, ns_attrs, skip_empty, search_mode, **kwargs)
        if cls.__xml_tag__ is None:
            # The default tag is just the class name and the fallback
            # is handled internally, so we'll override it here so we
            # can always assume __xml_tag__ is set to a sane default.
            #
            # Some models appear to do better if the separator is a dash
            # instead of a underscore, and users are free to override
            # as needed.
            cls.__xml_tag__ = to_snake(cls.__name__).replace("_", "-")

    # to_xml() doesn't prettify normally, and extended
    # requirements like lxml seemed like poor form
    def to_pretty_xml(self) -> str:
        tree = self.to_xml_tree()
        ET.indent(tree, "   ")
        pretty_encoded_xml = ET.tostring(tree).decode()

        if self.__class__.is_simple():
            return pretty_encoded_xml.replace("&lt;", "<").replace("&gt;", ">")
        else:
            return pretty_encoded_xml

    # XML parsing gets weird when the interior text contains tags like <br>.
    # Essentially it assumes all the text is valid XML first, then parses.
    # So we'll handle easy cases here and mark the model as "simple"
    # if it only contains a single string field. It makes our parsing
    # much more consistent
    @classmethod
    def is_simple(cls) -> bool:
        field_values = list(cls.model_fields.values())
        return len(field_values) == 1

    @classmethod
    def xml_start_tag(cls) -> str:
        return f"<{cls.__xml_tag__}>"

    @classmethod
    def xml_end_tag(cls) -> str:
        return f"</{cls.__xml_tag__}>"

    @classmethod
    def xml_tags(cls) -> str:
        return cls.xml_start_tag() + cls.xml_end_tag()

    # This can be overridden to provide a more complex example
    # to a model when it's required.
    @classmethod
    def xml_example(cls) -> str:
        return cls.xml_tags()

    # Attempt to extract this object from an arbitrary string
    # which may contain other XML elements or text, returns
    # the object and the string from which is was parsed.
    #
    # The potential complexities here are many, models might
    # return partial tags, multiple copies of our tags, or
    # nested tags inside others. We try to do our best to
    # parse for all the edge cases -> see the note above
    # about migrating from pydantic-xml

    @classmethod
    def extract_xml(cls, content: str) -> tuple[CoreModelGeneric, str]:
        pattern = r"(<([\w-]+)>((.*?)</\2>))"

        matches = re.findall(pattern, content, flags=re.DOTALL)
        matches_with_tag = [m for m in matches if m[1] == cls.__xml_tag__]
        if not matches or not matches_with_tag:
            raise MissingModelError(f"Failed to find '<{cls.__xml_tag__}>' in message")

        # Sort matches_with_tag based on the length of the interior text, longest first
        # this should help us avoid matching the model supplying hollow tags before the
        # actual data.
        sorted_matches = sorted(matches_with_tag, key=lambda m: len(m[3]), reverse=True)

        for i, match in enumerate(sorted_matches):
            full_text, tag, inner_with_end_tag, inner = match
            while f"<{tag}>" in inner_with_end_tag:
                matches = re.findall(r"(<(\w+)>((.*?)</\2>))", inner_with_end_tag, flags=re.DOTALL)
                match = next((m for m in matches if m[1] == cls.__xml_tag__), None)
                if not matches or not match:
                    break
                full_text, tag, inner_with_end_tag, inner = match

            try:
                if cls.is_simple():
                    model = cls(**{next(iter(cls.model_fields)): inner})
                else:
                    model = cls.from_xml(full_text)
                return model, full_text  # type: ignore [return-value]
            except Exception as e:
                if i == len(sorted_matches) - 1:
                    raise e

        raise ValidationError(f"Failed to parse '<{cls.__xml_tag__}>' from message")


#
# Helpers for passing structured errors to models
#


class ErrorModel(CoreModel, tag="error"):
    content: str

    @field_validator("content", mode="before")
    def parse_exception(cls, value: t.Any) -> t.Any:
        if isinstance(value, Exception):
            return str(value)
        return value


class SystemErrorModel(ErrorModel, tag="system_error"):
    content: str


class ValidationErrorModel(ErrorModel, tag="validation_error"):
    content: str
