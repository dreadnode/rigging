"""
This module covers core message objects and handling.
"""

import copy
import string
import typing as t
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    SerializeAsAny,
    computed_field,
    field_serializer,
    field_validator,
)

from rigging.error import MissingModelError
from rigging.model import Model, ModelT
from rigging.parsing import try_parse_many

Role = t.Literal["system", "user", "assistant"]
"""The role of a message. Can be 'system', 'user', or 'assistant'."""


# Helper type for messages structured
# more similarly to other libraries
class MessageDict(t.TypedDict):
    """
    Helper to represent a [rigging.message.Message][] as a dictionary.
    """

    role: Role
    """The role of the message."""
    content: str
    """The content of the message."""


# Structured portion of a message with
# a slice indicating where is it located
class ParsedMessagePart(BaseModel):
    """
    Represents a parsed message part.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: SerializeAsAny[Model]
    """The rigging/pydantic model associated with the message part."""
    slice_: slice
    """The slice representing the range into the message content."""

    @field_serializer("slice_")
    def serialize_slice(self, slice_: slice, _info: FieldSerializationInfo) -> list[int]:
        return [slice_.start, slice_.stop]

    @field_validator("slice_", mode="before")
    @classmethod
    def validate_slice(cls, value: t.Any) -> slice:
        if isinstance(value, slice):
            return value
        if not isinstance(value, list):
            raise ValueError("slice must be a list or a slice")
        return slice(value[0], value[1])


class Message(BaseModel):
    """
    Represents a message with role, content, and parsed message parts.
    """

    uuid: UUID = Field(default_factory=uuid4, repr=False)
    """The unique identifier for the message."""
    role: Role
    """The role of the message."""
    parts: list[ParsedMessagePart] = Field(default_factory=list)
    """The parsed message parts."""

    _content: str = ""

    def __init__(self, role: Role, content: str, parts: t.Sequence[ParsedMessagePart] | None = None, **kwargs: t.Any):
        super().__init__(role=role, parts=parts or [], **kwargs)
        self._content = content

    def __str__(self) -> str:
        return f"[{self.role}]: {self.content}"

    @computed_field  # type: ignore[misc]
    @property
    def content(self) -> str:
        """The content of the message."""
        # We used to sync the models and content each time it was accessed,
        # hence the getter. Now we just return the stored content.
        # I'll leave it as is for now in case we want to add any
        # logic here in the future.
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        # TODO: Maintain any parsed parts which are
        # still in the content - our move to slices for
        # tracking parsed parts makes this more complicated
        # so fow now I've opted to strip all parts
        # when content is changed.
        self.parts = []
        self._content = value

    # TODO: In general the add/remove/sync_part methods are
    # overly complicated. We should probably just update content,
    # then reparse all the models to get their fresh slices.
    #
    # I don't like all this manual slice recalculation logic, seems brittle.

    def _remove_part(self, part: ParsedMessagePart) -> str:
        removed_length = part.slice_.stop - part.slice_.start
        self._content = self._content[: part.slice_.start] + self._content[part.slice_.stop :]
        self.parts.remove(part)

        # Update slices of any parts that come after the removed part
        for other_part in self.parts:
            if other_part.slice_.start > part.slice_.start:
                other_part.slice_ = slice(
                    other_part.slice_.start - removed_length, other_part.slice_.stop - removed_length
                )

        return self._content

    def _add_part(self, part: ParsedMessagePart) -> None:
        for existing in self.parts:
            if part.slice_ == existing.slice_ and isinstance(part.model, type(existing.model)):
                return  # We clearly already have this part defined
            if max(part.slice_.start, existing.slice_.start) < min(part.slice_.stop, existing.slice_.stop):
                raise ValueError("Incoming part overlaps with an existing part")
        self.parts.append(part)

    # Looks more complicated than it is. We just want to clean all the models
    # in the message content by re-serializing them. As we do so, we'll need
    # to watch for the total size of our message shifting and update the slices
    # of the following parts accordingly. In other words, as A expands, B which
    # follows will have a new start slice and end slice.
    def _sync_parts(self) -> None:
        self.parts = sorted(self.parts, key=lambda p: p.slice_.start)

        shift = 0
        for part in self.parts:
            existing = self._content[part.slice_]

            # Adjust for any previous shifts
            part.slice_ = slice(part.slice_.start + shift, part.slice_.stop + shift)

            # Check if the content has changed
            xml_content = part.model.to_pretty_xml()
            if xml_content == existing:
                continue

            # Otherwise update content, add to shift, and update this slice
            old_length = part.slice_.stop - part.slice_.start
            new_length = len(xml_content)

            self._content = self._content[: part.slice_.start] + xml_content + self._content[part.slice_.stop :]
            part.slice_ = slice(part.slice_.start, part.slice_.start + new_length)

            shift += new_length - old_length

        self.parts = sorted(self.parts, key=lambda p: p.slice_.start)

    def clone(self) -> "Message":
        """Creates a copy of the message."""
        return Message(self.role, self.content, parts=copy.deepcopy(self.parts))

    def apply(self, **kwargs: str) -> "Message":
        """
        Applies the given keyword arguments with string templating to the content of the message.

        Uses [string.Template.safe_substitute](https://docs.python.org/3/library/string.html#string.Template.safe_substitute) underneath.

        Note:
            This call produces a clone of the message, leaving the original message unchanged.

        Args:
            **kwargs: Keyword arguments to substitute in the message content.
        """
        new = self.clone()
        template = string.Template(new.content)
        new.content = template.safe_substitute(**kwargs)
        return new

    def strip(self, model_type: type[Model], *, fail_on_missing: bool = False) -> list[ParsedMessagePart]:
        """
        Removes and returns a list of ParsedMessagePart objects from the message that match the specified model type.

        Args:
            model_type: The type of model to match.
            fail_on_missing: If True, raises a TypeError if no matching model is found.

        Returns:
            A list of removed ParsedMessagePart objects.

        Raises:
            TypeError: If no matching model is found and fail_on_missing is True.
        """
        removed: list[ParsedMessagePart] = []
        for part in self.parts[:]:
            if isinstance(part.model, model_type):
                self._remove_part(part)
                removed.append(part)

        if not removed and fail_on_missing:
            raise TypeError(f"Could not find <{model_type.__xml_tag__}> ({model_type.__name__}) in message")

        return removed

    @property
    def models(self) -> list[Model]:
        """Returns a list of models parsed from the message."""
        return [part.model for part in self.parts]

    # TODO: Many of these functions are duplicates from the parsing
    # module, but here we don't hand back slices and want there
    # to be a convient access model. We should probably consolidate.

    def parse(self, model_type: type[ModelT]) -> ModelT:
        """
        Parses a model from the message content.

        Args:
            model_type: The type of model to parse.

        Returns:
            The parsed model.

        Raises:
            ValueError: If no models of the given type are found and `fail_on_missing` is set to `True`.
        """
        return self.try_parse_many(model_type, fail_on_missing=True)[0]

    def try_parse(self, model_type: type[ModelT]) -> ModelT | None:
        """
        Tries to parse a model from the message content.

        Args:
            model_type: The type of model to search for.

        Returns:
            The first model that matches the given model type, or None if no match is found.
        """
        return next(iter(self.try_parse_many(model_type)), None)

    def parse_set(self, model_type: type[ModelT], minimum: int | None = None) -> list[ModelT]:
        """
        Parses a set of models of the specified identical type from the message content.

        Args:
            model_type: The type of models to parse.
            minimum: The minimum number of models required.

        Returns:
            A list of parsed models.

        Raises:
            MissingModelError: If the minimum number of models is not met.
        """
        return self.try_parse_set(model_type, minimum=minimum, fail_on_missing=True)

    def try_parse_set(
        self, model_type: type[ModelT], minimum: int | None = None, fail_on_missing: bool = False
    ) -> list[ModelT]:
        """
        Tries to parse a set of models from the message content.

        Args:
            model_type: The type of model to parse.
            minimum: The minimum number of models expected.
            fail_on_missing: Whether to raise an exception if models are missing.

        Returns:
            The parsed models.

        Raises:
            MissingModelError: If the number of parsed models is less than the minimum required.
        """
        models = self.try_parse_many(model_type, fail_on_missing=fail_on_missing)
        if minimum is not None and len(models) < minimum:
            raise MissingModelError(f"Expected at least {minimum} {model_type.__name__} in message")
        return models

    def parse_many(self, *types: type[ModelT]) -> list[ModelT]:
        """
        Parses multiple models of the specified non-identical types from the message content.

        Args:
            *types: The types of models to parse.

        Returns:
            A list of parsed models.

        Raises:
            MissingModelError: If any of the models are missing.
        """
        return self.try_parse_many(*types, fail_on_missing=True)

    def try_parse_many(self, *types: type[ModelT], fail_on_missing: bool = False) -> list[ModelT]:
        """
        Tries to parse multiple models from the content of the message.

        Args:
            *types: The types of models to parse.
            fail_on_missing: Whether to raise an exception if a model type is missing.

        Returns:
            A list of parsed models.

        Raises:
            MissingModelError: If a model type is missing and `fail_on_missing` is True.
        """
        model: ModelT
        parsed: list[tuple[ModelT, slice]] = try_parse_many(self.content, *types, fail_on_missing=fail_on_missing)
        for model, slice_ in parsed:
            self._add_part(ParsedMessagePart(model=model, slice_=slice_))
        self._sync_parts()
        return [p[0] for p in parsed]

    @classmethod
    def from_model(
        cls: type["Message"], models: Model | t.Sequence[Model], role: Role = "user", suffix: str | None = None
    ) -> "Message":
        """
        Create a Message object from one or more Model objects.

        Args:
            models: The Model object(s) to convert to a Message.
            role: The role of the Message.
            suffix: A suffix to append to the content.

        Returns:
            The created Message object.
        """
        parts: list[ParsedMessagePart] = []
        content: str = ""
        for model in models if isinstance(models, list) else [models]:
            text_form = model.to_pretty_xml()
            slice_ = slice(len(content), len(content) + len(text_form))
            content += f"{text_form}\n"
            parts.append(ParsedMessagePart(model=model, slice_=slice_))

        if suffix is not None:
            content += f"\n{suffix}"

        return cls(role=role, content=content, parts=parts)

    @classmethod
    def fit_as_list(
        cls, messages: t.Sequence[MessageDict] | t.Sequence["Message"] | MessageDict | "Message" | str
    ) -> list["Message"]:
        """Helper function to convert various common types to a strict list of Message objects."""
        if isinstance(messages, Message | dict | str):
            return [cls.fit(messages)]
        return [cls.fit(message) for message in messages]

    @classmethod
    def fit(cls, message: t.Union["Message", MessageDict, str]) -> "Message":
        """Helper function to convert various common types to a Message object."""
        if isinstance(message, str):
            return cls(role="user", content=message)
        return cls(**message) if isinstance(message, dict) else message

    @classmethod
    def apply_to_list(cls, messages: t.Sequence["Message"], **kwargs: str) -> list["Message"]:
        """Helper function to apply keyword arguments to a list of Message objects."""
        return [message.apply(**kwargs) for message in messages]


Messages = t.Sequence[MessageDict] | t.Sequence[Message]
