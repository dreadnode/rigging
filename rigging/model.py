"""
Models are the core datatypes for structured parsing.
"""

from __future__ import annotations

import re
import typing as t
from xml.etree import ElementTree as ET

import xmltodict  # type: ignore
from pydantic import (
    BeforeValidator,
    SerializationInfo,
    ValidationError,
    create_model,
    field_serializer,
    field_validator,
)
from pydantic_xml import BaseXmlModel
from pydantic_xml import attr as attr
from pydantic_xml import element as element
from pydantic_xml import wrapped as wrapped
from pydantic_xml.typedefs import EntityLocation, NsMap

from rigging.error import MissingModelError
from rigging.util import escape_xml, to_xml_tag, unescape_xml

if t.TYPE_CHECKING:
    from pydantic_xml.element import SearchMode  # type: ignore [attr-defined]

#
# Core XML serializable models for messages
#

ModelT = t.TypeVar("ModelT", bound="Model")

# TODO: pydantic-xml isn't a great fit for our use case given
# It's strictness for parsing XML and expecting all interior
# content to be escaped. We should probably just write something
# custom for our use case that supports JSON, YAML, and XML

BASIC_TYPES = [int, str, float, bool]


class XmlTagDescriptor:
    def __get__(self, _: t.Any, owner: t.Any) -> str:
        mro_iter = iter(owner.mro())
        cls = next(mro_iter)
        parent = next(mro_iter)

        # Generics produce names which are difficult
        # to override manually, so we'll just use the
        # parent's tag
        #
        # The altnernative is to use this syntax:
        #
        # class IntModel(Model[int], tag="override"):
        #   ...
        #
        if "[" in cls.__name__:
            return t.cast(str, parent.__xml_tag__)

        return to_xml_tag(cls.__name__)


class Model(BaseXmlModel):
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
        # The default tag is just the class name and the fallback
        # is handled internally, so we'll override it here so we
        # can always assume __xml_tag__ is set to a sane default.
        #
        # Some models appear to do better if the separator is a dash
        # instead of a underscore, and users are free to override
        # as needed.
        super().__init_subclass__(tag, ns, nsmap, ns_attrs, skip_empty, search_mode, **kwargs)
        cls.__xml_tag__ = tag or XmlTagDescriptor()  # type: ignore [assignment]

    # to_xml() doesn't prettify normally, and extended
    # requirements like lxml seemed like poor form for
    # just this feature
    def to_pretty_xml(self) -> str:
        """
        Converts the model to a pretty XML string with indents and newlines.

        Returns:
            The pretty XML representation of the model.
        """
        tree = self.to_xml_tree()
        ET.indent(tree, "  ")
        pretty_encoded_xml = ET.tostring(tree, short_empty_elements=False, encoding="utf-8").decode()

        if self.__class__.is_simple():
            # We only expect to use this in our "simple"
            # models, but I'd like a better long-term solution
            return unescape_xml(pretty_encoded_xml)
        else:
            return pretty_encoded_xml  # type: ignore [no-any-return]

    def __str__(self) -> str:
        return self.to_pretty_xml()

    # XML parsing gets weird when the interior text contains tags like <br>.
    # Essentially it assumes all the text is valid XML first, then parses.
    # So we'll handle easy cases here and mark the model as "simple"
    # if it only contains a single basic field. It makes our parsing
    # much more consistent and is likely the most popular model type.
    #
    # TODO: lxml with the recover option is likely a better approach
    @classmethod
    def is_simple(cls) -> bool:
        """
        Check if the model is "simple", meaning it has a single field with a basic datatype.

        Until we refactor our XML parsing, this helps make the parsing more consistent for models
        which can support it.

        Returns:
            True if the model is simple, False otherwise.
        """
        field_values = list(cls.model_fields.values())
        if len(field_values) != 1:
            return False

        annotation = field_values[0].annotation
        if t.get_origin(annotation) == t.Annotated:
            annotation = t.get_args(annotation)[0]

        return annotation in BASIC_TYPES

    @classmethod
    def xml_start_tag(cls) -> str:
        """Helper method which wrapped the class tag in XML braces."""
        return f"<{cls.__xml_tag__}>"

    @classmethod
    def xml_end_tag(cls) -> str:
        """Helper method which wrapped the class tag in XML braces with a leading slash."""
        return f"</{cls.__xml_tag__}>"

    @classmethod
    def xml_tags(cls) -> str:
        """Helper method which returns the full XML tags for the class."""
        return cls.xml_start_tag() + cls.xml_end_tag()

    # This can be overridden to provide a more complex example
    # to a model when it's required.

    @classmethod
    def xml_example(cls) -> str:
        """
        Returns an example XML representation of the given class.

        Models should typically override this method to provide a more complex example.

        By default, this method returns a hollow XML scaffold one layer deep.

        Returns:
            A string containing the XML representation of the class.
        """
        if cls.is_simple():
            return cls.xml_tags()

        schema = cls.model_json_schema()
        properties = schema["properties"]
        structure = {cls.__xml_tag__: {field: None for field in properties}}
        xml_string = xmltodict.unparse(
            structure, pretty=True, full_document=False, indent="  ", short_empty_elements=True
        )
        return t.cast(str, xml_string)  # Bad type hints in xmltodict

    @classmethod
    def ensure_valid(cls) -> None:
        # Do a sanity check for models with a single
        # attr field, which our parsing currently doesn't support
        #
        # TODO: Add support for <thing attr="value" /> style models

        if len(cls.model_fields) == 1:
            field_info = next(iter(cls.model_fields.values()))
            if hasattr(field_info, "location") and field_info.location == EntityLocation.ATTRIBUTE:
                raise ValueError(f"Model '{cls.__name__}' has a single attr() field which is not supported")

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
    def from_text(cls, content: str) -> list[tuple[ModelT, slice]]:
        """
        The core parsing method which attempts to extract and parse as many
        valid instances of a model from semi-structured text.

        Args:
            content: The text content to parse.

        Returns:
            A list of tuples containing the extracted models and their corresponding slices.

        Raises:
            MissingModelError: If the specified model tags are not found in the message.
            ValidationError: If an error occurs while parsing the content.
        """
        cls.ensure_valid()

        pattern = r"(<([\w-]+).*?>((.*?)</\2>))"
        matches = [m for m in re.finditer(pattern, content, flags=re.DOTALL) if m.group(2) == cls.__xml_tag__]

        if not matches:
            raise MissingModelError(f"Failed to find '{cls.xml_tags()}' in message")

        # Sort matches_with_tag based on the length of the interior text,
        # longest first. This should help us avoid matching the model
        # supplying hollow tags before the actual data.

        sorted_matches = sorted(matches, key=lambda m: len(m.group(4)), reverse=True)

        extracted: list[tuple[ModelT, slice]] = []
        exceptions: list[Exception] = []
        for match in sorted_matches:
            full_text, _, inner_with_end_tag, inner = match.groups()

            # The model might trip up regex by including partial tags
            # in passing before actually using them. We'll continually try
            # to parse the inner text until we can't extract our model anymore.
            #
            # Example: "Sure I'll use <answer> tags: <answer>hello</answer>"
            #
            # TODO: The opposite could be true, and we could greedily parse
            # backwards if we get failures. This is a simple solution for now.

            inner_match: re.Match[str] | None = match
            while inner_match is not None:
                inner_matches = re.finditer(pattern, inner_with_end_tag, flags=re.DOTALL)
                inner_match = next((m for m in inner_matches if m.group(2) == cls.__xml_tag__), None)
                if inner_match is not None:
                    full_text, _, inner_with_end_tag, inner = inner_match.groups()

            try:
                model = (
                    cls(**{next(iter(cls.model_fields)): unescape_xml(inner)})
                    if cls.is_simple()
                    else cls.from_xml(escape_xml(full_text))
                )
                extracted.append((model, slice(match.start(), match.end())))  # type: ignore [arg-type]
            except Exception as e:
                exceptions.append(e)
                continue

        # TODO: This is poor form atm, but the exception stacking
        # and final error should involve some careful thought

        if not extracted:
            raise exceptions[0]

        return extracted

    @classmethod
    def one_from_text(cls, content: str, *, fail_on_many: bool = False) -> tuple[ModelT, slice]:
        """
        Finds and returns a single match from the given text content.

        Args:
            content: The text content to search for matches.
            fail_on_many: If True, raises a ValidationError if multiple matches are found.

        Returns:
            A tuple containing the matched model and the slice indicating the match location.

        Raises:
            ValidationError: If multiple matches are found and fail_on_many is True.
        """
        matches = cls.from_text(content)  # type: ignore [var-annotated]
        if fail_on_many and len(matches) > 1:
            raise ValidationError("Multiple matches found with 'fail_on_many=True'")
        return max(matches, key=lambda x: x[1].stop - x[1].start)


#
# Functional Constructor
#

PrimitiveT = t.TypeVar("PrimitiveT", int, str, float, bool)


class Primitive(Model, t.Generic[PrimitiveT]):
    content: t.Annotated[PrimitiveT, BeforeValidator(lambda x: x.strip() if isinstance(x, str) else x)]


def make_primitive(
    name: str,
    type_: type[PrimitiveT] = str,  # type: ignore [assignment]
    *,
    tag: str | None = None,
    doc: str | None = None,
    validator: t.Callable[[str], str | None] | None = None,
    strip_content: bool = True,
) -> type[Primitive[PrimitiveT]]:
    """
    Helper to create a simple primitive model with an optional content validator.

    Note:
        This API is experimental and may change in the future.

    Args:
        name: The name of the model.
        tag: The XML tag for the model.
        doc: The documentation for the model.
        validator: An optional content validator for the model.
        strip_content: Whether to strip the content string before pydantic validation.

    Returns:
        The primitive model class.
    """

    def _validate(value: str) -> str:
        if validator is not None:
            return validator(value) or value
        return value

    if strip_content:
        type_ = t.Annotated[type_, BeforeValidator(lambda x: x.strip() if isinstance(x, str) else x)]  # type: ignore

    return create_model(
        name,
        __base__=Primitive[type_],  # type: ignore
        __doc__=doc,
        __cls_kwargs__={"tag": tag},
        content=(type_, ...),
        __validators__={"content_validator": field_validator("content")(_validate)} if validator else {},
    )


#
# Helpers for passing structured errors to models
#


class ErrorModel(Model, tag="error"):
    content: str

    @field_validator("content", mode="before")
    def parse_exception(cls, value: t.Any) -> t.Any:
        if isinstance(value, Exception):
            return str(value)
        return value


class SystemErrorModel(ErrorModel, tag="system-error"):
    content: str


class ValidationErrorModel(ErrorModel, tag="validation-error"):
    content: str


# Common structured helpers


class Thinking(Model):
    """Quick model for thinking messages."""

    content: str


class Question(Model):
    """Quick model for questions."""

    content: str


class Answer(Model):
    """Quick model for answers."""

    content: str


class QuestionAnswer(Model):
    """Quick model for question-answer pairs."""

    question: Question = element()
    """The question"""
    answer: Answer = element()
    """The answer"""


class Description(Model):
    """Quick model for descriptions."""

    content: str


class Instructions(Model):
    """Quick model for instructions."""

    content: str


class DelimitedAnswer(Model):
    "Mixed support delimited answer (- | / ,) selected based on most-matches"

    content: str
    _delimiters: t.ClassVar[list[str]] = [",", "-", "/", "|"]

    @property
    def items(self) -> list[str]:
        """Parsed items from the content."""
        split_sizes: dict[str, int] = {}
        for delimiter in self._delimiters:
            split_sizes[delimiter] = len(self.content.split(delimiter))
        delimiter = max(split_sizes, key=split_sizes.get)  # type: ignore [arg-type]
        split = [i.strip(" \"'\t\r\n") for i in self.content.split(delimiter)]
        return [s for s in split if s]

    @field_validator("content", mode="before")
    def parse_str_to_list(cls, v: t.Any) -> t.Any:
        if not isinstance(v, str):
            raise ValueError(f"Cannot parse content as a delimited list: {v}")
        return v


class CommaDelimitedAnswer(DelimitedAnswer):
    "Comma delimited answer (,)"

    _delimiters = [","]


class NewlineDelimitedAnswer(DelimitedAnswer):
    "Newline delimited answer (\n)"

    _delimiters = ["\n"]


class YesNoAnswer(Model):
    "Yes/No answer answer with coercion"

    boolean: bool
    """The boolean value of the answer."""

    @field_validator("boolean", mode="before")
    def parse_str_to_bool(cls, v: t.Any) -> t.Any:
        if isinstance(v, str):
            simple = v.strip().lower()
            if any(simple.startswith(s) for s in ["yes", "true"]):
                return True
            elif any(simple.startswith(s) for s in ["no", "false"]):
                return False
        raise ValueError(f"Cannot parse '{v}' as a boolean")

    @field_serializer("boolean")
    def serialize_bool_to_str(self, v: bool, _info: SerializationInfo) -> str:
        return "yes" if v else "no"
