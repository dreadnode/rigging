import re
import typing as t
from xml.etree import ElementTree as ET

from pydantic import ValidationError, field_validator
from pydantic.alias_generators import to_snake
from pydantic_xml import BaseXmlModel
from pydantic_xml import attr as attr
from pydantic_xml import element as element
from pydantic_xml import wrapped as wrapped
from pydantic_xml.element import SearchMode  # type: ignore [attr-defined]
from pydantic_xml.typedefs import NsMap

from rigging.error import MissingModelError

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
        return to_snake(next(iter(owner.mro())).__name__).replace("_", "-")


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
        cls.__xml_tag__ = XmlTagDescriptor()  # type: ignore [assignment]

    # to_xml() doesn't prettify normally, and extended
    # requirements like lxml seemed like poor form
    def to_pretty_xml(self) -> str:
        tree = self.to_xml_tree()
        ET.indent(tree, "   ")
        pretty_encoded_xml = ET.tostring(tree).decode()

        # TODO: I didn't note why this edge case is here, but it makes
        # me nervous - should investigate and remove if possible
        if self.__class__.is_simple():
            return pretty_encoded_xml.replace("&lt;", "<").replace("&gt;", ">")
        else:
            return pretty_encoded_xml

    # XML parsing gets weird when the interior text contains tags like <br>.
    # Essentially it assumes all the text is valid XML first, then parses.
    # So we'll handle easy cases here and mark the model as "simple"
    # if it only contains a single basic field. It makes our parsing
    # much more consistent and is likely the most popular model type.
    @classmethod
    def is_simple(cls) -> bool:
        field_values = list(cls.model_fields.values())
        return len(field_values) == 1 and field_values[0].annotation in BASIC_TYPES

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
    def from_text(cls, content: str) -> list[tuple[ModelT, slice]]:
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
            inner_match: re.Match[str] | None = match
            while inner_match is not None:
                inner_matches = re.finditer(pattern, inner_with_end_tag, flags=re.DOTALL)
                inner_match = next((m for m in inner_matches if m.group(2) == cls.__xml_tag__), None)
                if inner_match is not None:
                    full_text, _, inner_with_end_tag, inner = inner_match.groups()

            try:
                model = cls(**{next(iter(cls.model_fields)): inner}) if cls.is_simple() else cls.from_xml(full_text)
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
    def one_from_text(cls, content: str, fail_on_many: bool = False) -> tuple[ModelT, slice]:
        matches = cls.from_text(content)  # type: ignore [var-annotated]
        if fail_on_many and len(matches) > 1:
            raise ValidationError("Multiple matches found with 'fail_on_many=True'")
        return max(matches, key=lambda x: x[1].stop - x[1].start)


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


class SystemErrorModel(ErrorModel, tag="system_error"):
    content: str


class ValidationErrorModel(ErrorModel, tag="validation_error"):
    content: str


# Common structured helpers


class Thinking(Model):
    content: str


class Question(Model):
    content: str


class Answer(Model):
    content: str


class QuestionAnswer(Model):
    question: Question
    answer: Answer


class Description(Model):
    content: str


class Instructions(Model):
    content: str


class DelimitedAnswer(Model):
    "Mixed support delimited answer (- | / ,) selected based on most-matches"

    content: str
    _delimiters: t.ClassVar[list[str]] = [",", "-", "/", "|"]

    @property
    def items(self) -> list[str]:
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

    @field_validator("boolean", mode="before")
    def parse_str_to_bool(cls, v: t.Any) -> t.Any:
        if isinstance(v, str):
            if v.strip().lower().startswith("yes"):
                return True
            elif v.strip().lower().startswith("no"):
                return False
        return v
