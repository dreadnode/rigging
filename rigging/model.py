"""
Models are the core datatypes for structured parsing.
"""

import dataclasses
import inspect
import re
import typing as t
from xml.etree import ElementTree as ET  # nosec

import typing_extensions as te
import xmltodict  # type: ignore [import-untyped]
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    SerializationInfo,
    ValidationError,
    field_serializer,
    field_validator,
)
from pydantic import create_model as create_pydantic_model
from pydantic_xml import BaseXmlModel
from pydantic_xml import attr as attr  # noqa: PLC0414
from pydantic_xml import create_model as create_pydantic_xml_model
from pydantic_xml import element as element  # noqa: PLC0414
from pydantic_xml import wrapped as wrapped  # noqa: PLC0414
from pydantic_xml.element import SearchMode  # type: ignore [attr-defined]
from pydantic_xml.model import XmlEntityInfo
from pydantic_xml.typedefs import EntityLocation, NsMap

from rigging.error import MissingModelError
from rigging.util import escape_xml, to_xml_tag, unescape_cdata_tags, unescape_xml

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
            return t.cast("str", parent.__xml_tag__)

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
        super().__init_subclass__(
            tag,
            ns,
            nsmap,
            ns_attrs,
            skip_empty,
            search_mode,
            **kwargs,
        )
        cls.__xml_tag__ = tag or XmlTagDescriptor()  # type: ignore [assignment]

    def _postprocess_with_cdata(self, tree: ET.Element) -> ET.Element:
        # Walk the first elements down and find any that align with str-based fields
        # If so, we should encode them as CDATA to avoid escaping issues

        basic_fields = {
            (
                field_info.path
                if isinstance(field_info, XmlEntityInfo) and field_info.path
                else field_name
            ): field_info
            for field_name, field_info in self.__class__.model_fields.items()
        }

        for elem in [tree, *tree]:
            field_info = basic_fields.get(elem.tag, None)

            # If this is the tree itself, check for an interior primitive field
            if field_info is None and elem is tree:
                field_info = next(
                    (f for f in basic_fields.values() if not isinstance(f, XmlEntityInfo)),
                    None,
                )

            if field_info is None:
                continue

            field_type = field_info.annotation
            if t.get_origin(field_type) == t.Annotated:
                field_type = t.get_args(field_type)[0]

            if field_type not in BASIC_TYPES:
                continue

            # Replace this with an element that uses CDATA
            if elem.text and escape_xml(unescape_xml(elem.text)) != elem.text:
                elem.text = f"<![CDATA[{elem.text}]]>"

        return tree

    # to_xml() doesn't prettify normally, and extended
    # requirements like lxml seemed like poor form for
    # just this feature
    def to_pretty_xml(
        self,
        *,
        skip_empty: bool = False,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        **kwargs: t.Any,
    ) -> str:
        """
        Converts the model to a pretty XML string with indents and newlines.

        Returns:
            The pretty XML representation of the model.
        """
        tree = self.to_xml_tree(
            skip_empty=skip_empty,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
        )
        tree = self._postprocess_with_cdata(tree)

        ET.indent(tree, "  ")
        pretty_encoded_xml = str(
            ET.tostring(
                tree,
                short_empty_elements=False,
                encoding="utf-8",
                **kwargs,
            ).decode(),
        )

        # Now we can go back and safely unescape the XML
        # that we observe between any CDATA tags

        return unescape_cdata_tags(pretty_encoded_xml)

    def to_xml(
        self,
        *,
        skip_empty: bool = False,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        **kwargs: t.Any,
    ) -> str:
        """
        Serializes the object to an xml string.

        Args:
            skip_empty: skip empty elements (elements without sub-elements, attributes and text, Nones)
            exclude_none: exclude `None` values
            exclude_unset: exclude values that haven't been explicitly set
            kwargs: additional xml serialization arguments

        Returns:
            object xml representation
        """

        tree = self.to_xml_tree(
            skip_empty=skip_empty,
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
        )
        tree = self._postprocess_with_cdata(tree)
        xml = ET.tostring(tree, short_empty_elements=False, encoding="utf-8", **kwargs).decode()

        # Now we can go back and safely unescape the XML
        # that we observe between any CDATA tags

        return unescape_cdata_tags(xml)

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

        # If the field is a pydantic-xml wrapper like element()
        # or attr(), we'll assume the user knows what they're doing
        if isinstance(field_values[0], XmlEntityInfo):
            return False

        annotation = field_values[0].annotation
        if t.get_origin(annotation) == t.Annotated:
            annotation = t.get_args(annotation)[0]

        return annotation in BASIC_TYPES

    @classmethod
    def is_simple_with_attrs(cls) -> bool:
        """
        Check if the model would otherwise be marked as "simple", but has other fields which are
        all attributes. If so, we can do some parsing magic below and make sure our non-element
        field is updated with the extracted content properly, while pydantic-xml takes care
        of the attributes.

        Returns:
            True if the model is simple with attrs, False otherwise.
        """
        field_values = list(cls.model_fields.values())

        none_entity_fields = [f for f in field_values if not isinstance(f, XmlEntityInfo)]
        entity_fields = [f for f in field_values if isinstance(f, XmlEntityInfo)]
        attr_fields = [f for f in entity_fields if f.location == EntityLocation.ATTRIBUTE]

        if len(none_entity_fields) != 1 or len(attr_fields) != len(entity_fields):
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
        structure = {cls.__xml_tag__: dict.fromkeys(properties)}
        xml_string = xmltodict.unparse(
            structure,
            pretty=True,
            full_document=False,
            indent="  ",
            short_empty_elements=True,
        )
        return t.cast("str", xml_string)  # Bad type hints in xmltodict

    @classmethod
    def ensure_valid(cls) -> None:
        # Do a sanity check for models with a single
        # attr field, which our parsing currently doesn't support
        #
        # TODO: Add support for <thing attr="value" /> style models

        if len(cls.model_fields) == 1:
            field_info = next(iter(cls.model_fields.values()))
            if hasattr(field_info, "location") and field_info.location == EntityLocation.ATTRIBUTE:
                raise ValueError(
                    f"Model '{cls.__name__}' has a single attr() field which is not supported",
                )

    @classmethod
    def preprocess_with_cdata(cls, content: str) -> str:
        """
        Process the content and attempt to auto-wrap interior
        field content in CDATA tags if they contain unescaped XML entities.

        Args:
            content: The XML content to preprocess.

        Returns:
            The processed XML content with CDATA tags added where necessary.
        """

        # This means our CDATA tags should wrap the entire content
        # as the model is simple enough to not have nested elements.

        if cls.is_simple() or cls.is_simple_with_attrs():
            field_map = {
                (cls.__xml_tag__ or "unknown"): next(
                    info
                    for info in cls.model_fields.values()
                    if not isinstance(info, XmlEntityInfo)
                ),
            }
        else:
            field_map = {
                (
                    field_info.path
                    if isinstance(field_info, XmlEntityInfo) and field_info.path
                    else field_name
                ): field_info
                for field_name, field_info in cls.model_fields.items()
                if isinstance(field_info, XmlEntityInfo)
                and field_info.location == EntityLocation.ELEMENT
            }

        def wrap_with_cdata(match: re.Match[str]) -> str:
            field_name = match.group(1)
            tag_attrs = match.group(2) or ""  # Handle attributes if present
            content = match.group(3)

            if field_name not in field_map:
                return match.group(0)

            field_info = field_map[field_name]
            field_type = field_info.annotation
            if t.get_origin(field_type) == t.Annotated:
                field_type = t.get_args(field_type)[0]

            is_basic_field = field_type in BASIC_TYPES
            is_already_cdata = content.strip().startswith("<![CDATA[")
            needs_escaping = escape_xml(unescape_xml(content)) != content

            if is_basic_field and not is_already_cdata and needs_escaping:
                content = f"<![CDATA[{content}]]>"

            return f"<{field_name}{tag_attrs}>{content}</{field_name}>"

        fields_pattern = "|".join(re.escape(field_name) for field_name in field_map)
        pattern = f"<({fields_pattern})((?:[^>]*?)?)>(.*?)</\\1>"

        return re.sub(pattern, wrap_with_cdata, content, flags=re.DOTALL)

    # Attempt to extract this object from an arbitrary string
    # which may contain other XML elements or text, returns
    # the object and the string from which is was parsed.
    #
    # The potential complexities here are many, models might
    # return partial tags, multiple copies of our tags, or
    # nested tags inside others. We try to do our best to
    # parse for all the edge cases -> see the note above
    # about migrating from pydantic-xml

    @t.overload
    @classmethod
    def from_text(
        cls,
        content: str,
        *,
        return_errors: t.Literal[False] = False,
    ) -> list[tuple[te.Self, slice]]: ...

    @t.overload
    @classmethod
    def from_text(
        cls,
        content: str,
        *,
        return_errors: t.Literal[True],
    ) -> list[tuple[te.Self | Exception, slice]]: ...

    @classmethod
    def from_text(
        cls,
        content: str,
        *,
        return_errors: bool = False,
    ) -> list[tuple[te.Self, slice]] | list[tuple[te.Self | Exception, slice]]:
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

        tag_name = re.escape(cls.__xml_tag__ or "unknown")
        pattern = f"((<({tag_name}).*?>)((.*?)</{tag_name}>))"

        matches = [
            m
            for m in re.finditer(pattern, content, flags=re.DOTALL)
            if m.group(3) == cls.__xml_tag__
        ]

        if not matches:
            raise MissingModelError(f"Failed to find '{cls.xml_tags()}' in message")

        # Sort matches_with_tag based on the length of the interior text,
        # longest first. This should help us avoid matching the model
        # supplying hollow tags before the actual data.

        sorted_matches = sorted(matches, key=lambda m: len(m.group(5)), reverse=True)

        extracted: list[tuple[te.Self | Exception, slice]] = []
        for match in sorted_matches:
            full_text, start_tag, _, inner_with_end_tag, inner = match.groups()

            # The model might trip up regex by including partial tags
            # in passing before actually using them. We'll continually try
            # to parse the inner text until we can't extract our model anymore.
            #
            # Example: "Sure I'll use <answer> tags: <answer>hello</answer>"
            #
            # TODO: The opposite could be true, and we could greedily parse
            # backwards if we get failures. This is a simple solution for now.

            inner_match: re.Match[str] | None = match
            slice_ = slice(match.start(), match.end())

            while inner_match is not None:
                inner_matches = re.finditer(
                    pattern,
                    inner_with_end_tag,
                    flags=re.DOTALL,
                )
                inner_match = next(
                    (m for m in inner_matches if m.group(3) == cls.__xml_tag__),
                    None,
                )
                if inner_match is not None:
                    slice_ = slice(
                        slice_.start + len(start_tag) + inner_match.start(),
                        slice_.start + len(start_tag) + inner_match.end(),
                    )
                    full_text, start_tag, _, inner_with_end_tag, inner = inner_match.groups()

            try:
                model = (
                    cls(**{next(iter(cls.model_fields)): unescape_xml(inner)})
                    if cls.is_simple()
                    else cls.from_xml(
                        cls.preprocess_with_cdata(full_text),
                    )
                )

                # If our model is relatively simple (only attributes and a single non-element field)
                # we should go back and update our non-element field with the extracted content.

                if cls.is_simple_with_attrs():
                    name, field = next(
                        (name, field)
                        for name, field in cls.model_fields.items()
                        if not isinstance(field, XmlEntityInfo)
                    )
                    if field.annotation in BASIC_TYPES:
                        model.__dict__[name] = field.annotation(
                            unescape_xml(inner).strip(),
                        )

                extracted.append((model, slice_))
            except Exception as e:  # noqa: BLE001
                extracted.append((e, slice_))
                continue

        # sort back to original order
        extracted.sort(key=lambda x: x[1].start)

        if not return_errors and (
            first_error := next((m for m, _ in extracted if isinstance(m, Exception)), None)
        ):
            raise first_error

        return (
            extracted
            if return_errors
            else [(m, s) for m, s in extracted if not isinstance(m, Exception)]
        )

    @classmethod
    def one_from_text(
        cls,
        content: str,
        *,
        fail_on_many: bool = False,
    ) -> tuple[te.Self, slice]:
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
        matches = cls.from_text(content)
        if fail_on_many and len(matches) > 1:
            raise ValidationError("Multiple matches found with 'fail_on_many=True'")
        return max(matches, key=lambda x: x[1].stop - x[1].start)


#
# Functional Constructor
#

PrimitiveT = t.TypeVar("PrimitiveT", int, str, float, bool)


class Primitive(Model, t.Generic[PrimitiveT]):
    content: t.Annotated[
        PrimitiveT,
        BeforeValidator(lambda x: x.strip() if isinstance(x, str) else x),
    ]


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
        type_ = t.Annotated[
            type_,
            BeforeValidator(lambda x: x.strip() if isinstance(x, str) else x),
        ]  # type: ignore [assignment]

    return create_pydantic_model(
        name,
        __base__=Primitive[type_],  # type: ignore [valid-type, arg-type]
        __doc__=doc,
        __cls_kwargs__={"tag": tag},
        content=(type_, ...),
        __validators__={"content_validator": field_validator("content")(_validate)}
        if validator
        else {},
    )


#
# Conversion from JSON schemas
#

FieldType = t.Any  # aliases for my sanity
FieldInfo = t.Any


def _process_field(
    field_name: str,
    field_schema: dict[str, t.Any],
) -> tuple[FieldType, FieldInfo]:
    """Process a field schema and return appropriate type and field info."""
    field_info: FieldInfo = {}

    # enums
    if "enum" in field_schema:
        return t.Literal[tuple(field_schema["enum"])], field_info

    # arrays
    if field_schema.get("type") == "array":
        item_schema = field_schema["items"]
        item_type, _ = _process_field(f"{field_name}Item", item_schema)

        if item_schema.get("type") == "object":
            return list[item_type], wrapped(field_name, **field_info)  # type: ignore [valid-type]

        return list[item_type], wrapped(field_name, element("item", **field_info))  # type: ignore [valid-type]

    # objects
    if field_schema.get("type") == "object":
        # dictionaries
        additional_schema = field_schema.get("additionalProperties")
        if additional_schema:
            dict_type, _ = _process_field(f"{field_name}Item", additional_schema)
            return dict[str, dict_type], field_info  # type: ignore [valid-type]

        # Otherwise a nested model
        return make_from_schema(
            field_schema,
            field_schema.get("title"),
            allow_primitive=True,
        ), field_info

    # unions
    for union_type in ["anyOf", "oneOf", "allOf"]:
        if union_type in field_schema:
            types = [
                _process_field(field_name, sub_schema)[0] for sub_schema in field_schema[union_type]
            ]
            return t.Union[tuple(types)], field_info  # noqa: UP007

    type_mapping: dict[str, FieldType] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
        "object": dict,
        "array": list,
    }

    # tuples
    field_type = field_schema.get("type", "string")
    if isinstance(field_type, list):
        types = [type_mapping.get(t, t.Any) for t in field_type]
        return t.Union[tuple(types)], field_info  # noqa: UP007

    # primitives
    return type_mapping.get(field_type, t.Any), field_info


def make_from_schema(
    schema: dict[str, t.Any],
    name: str | None = None,
    *,
    allow_primitive: bool = False,
) -> type[Model]:
    """
    Helper to build a Rigging model dynamically from a JSON schema.

    Note:
        There are plenty of edge cases this doesn't handle, consider this
        very experimental and only suitable for simple schemas.

    Args:
        schema: The JSON schema to build the model from.
        name: The name of the model (otherwise inferred from the schema).
        allow_primitive: If True, allows the model to be a simple primitive

    Returns:
        The Pydantic model class.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, tuple[FieldType, FieldInfo]] = {}

    # If we only have one property and the caller allows it,
    # make the model "simple/primitive" by not using the element() wrapper
    field_cls = Field if len(properties) == 1 and allow_primitive else element

    for field_name, field_schema in properties.items():
        field_type, field_info = _process_field(field_name, field_schema)

        fields[field_name] = (
            field_type,
            field_cls(
                default=... if field_name in required else None,
                description=field_schema.get("description", ""),
                title=field_schema.get("title", ""),
                **field_info,
            )
            if isinstance(field_info, dict)
            else field_info,
        )

    name = name or schema.get("title", "SchemaModel")
    return create_pydantic_xml_model(name, __base__=Model, **fields)  # type: ignore [arg-type]


#
# Conversion from callable signatures
#


def _safe_issubclass(cls: t.Any, class_or_tuple: t.Any) -> bool:
    """Safely check if a class is a subclass of another class or tuple."""
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def _is_complex_type(typ: t.Any) -> bool:
    """Check if a type is a complex type (class-based, not primitive)."""
    try:
        return not _safe_issubclass(typ, (str, int, float, bool, bytes)) and typ is not t.Any
    except TypeError:
        return False


def make_from_signature(
    signature: inspect.Signature,
    name: str | None = None,
) -> type[Model]:
    fields = {}
    for param_name, param in signature.parameters.items():
        param_type = param.annotation
        param_origin = t.get_origin(param_type)
        param_args = t.get_args(param_type)

        # Sanity checks
        for type_ in (param_type, param_origin, *param_args):
            if _safe_issubclass(type_, BaseModel) and not _safe_issubclass(
                type_,
                BaseXmlModel,
            ):
                raise ValueError(
                    f"Function arguments which are Pydantic models must inherit from `BaseXmlModel` ({param_name})",
                )

            if dataclasses.is_dataclass(type_):
                raise ValueError(
                    f"Function arguments which are dataclasses are not supported ({param_name})",
                )

        # Extract description from Annotated types
        description = ""
        if param_origin is t.Annotated:
            param_type = param_args[0]  # The actual type
            param_origin = t.get_origin(param_type)
            if len(param_args) > 1 and isinstance(param_args[1], str):
                description = param_args[1]  # The description
            param_args = t.get_args(param_type)

        # Add default value if available
        default = ... if param.default is inspect.Parameter.empty else param.default
        field = element(default=default, description=description)

        # Handle List types (really just for the wrapped() component)
        if param_origin is list or param_origin is t.List:  # noqa: UP006
            item_tag = "item"

            if (
                param_args
                and _is_complex_type(param_args[0])
                and param_name.endswith("s")
                and (singular := param_name[:-1])
            ):
                item_tag = singular

            field = wrapped(
                param_name,
                element(tag=item_tag, default=default, description=description),
            )

        fields[param_name] = (param_type, field)

    return create_pydantic_xml_model(name, __base__=Model, **fields)  # type: ignore [arg-type]


#
# Helpers for passing structured errors to models
#


class ErrorModel(Model, tag="error"):
    type: str = attr(default="Exception")
    content: str

    @field_validator("content", mode="before")
    @classmethod
    def parse_exception(cls, value: t.Any) -> t.Any:
        if isinstance(value, Exception):
            return str(value)
        return value

    @classmethod
    def from_exception(cls, exception: Exception) -> te.Self:
        """
        Create an ErrorModel instance from an exception.

        Args:
            exception: The exception to convert.

        Returns:
            An instance of ErrorModel with the exception content.
        """
        return cls(content=str(exception), type=type(exception).__name__)


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
    _items: list[str] | None = None

    @property
    def items(self) -> list[str]:
        """Parsed items from the content."""
        if self._items is not None:
            return self._items

        split_sizes: dict[str, int] = {}
        for delimiter in self._delimiters:
            split_sizes[delimiter] = len(self.content.split(delimiter))
        delimiter = max(split_sizes, key=split_sizes.get)  # type: ignore [arg-type]
        split = [i.strip(" \"'\t\r\n") for i in self.content.split(delimiter)]
        self._items = [s for s in split if s]
        return self._items

    @field_validator("content", mode="before")
    @classmethod
    def parse_str_to_list(cls, v: t.Any) -> t.Any:
        if not isinstance(v, str):
            raise TypeError(f"Cannot parse content as a delimited list: {v}")
        return v


class CommaDelimitedAnswer(DelimitedAnswer):
    "Comma delimited answer (,)"

    _delimiters: t.ClassVar = [","]


class NewlineDelimitedAnswer(DelimitedAnswer):
    "Newline delimited answer (\n)"

    _delimiters: t.ClassVar = ["\n"]


class YesNoAnswer(Model):
    "Yes/No answer answer with coercion"

    boolean: bool
    """The boolean value of the answer."""

    @field_validator("boolean", mode="before")
    @classmethod
    def parse_str_to_bool(cls, v: t.Any) -> t.Any:
        if isinstance(v, str):
            simple = v.strip().lower()
            if any(simple.startswith(s) for s in ["yes", "true"]):
                return True
            if any(simple.startswith(s) for s in ["no", "false"]):
                return False
        raise ValueError(f"Cannot parse '{v}' as a boolean")

    @field_serializer("boolean")
    def serialize_bool_to_str(self, v: bool, _info: SerializationInfo) -> str:  # noqa: FBT001
        return "yes" if v else "no"
