"""
This module covers core message objects and handling.
"""

import base64
import copy
import mimetypes
import string
import typing as t
import warnings
from pathlib import Path
from textwrap import dedent
from uuid import UUID, uuid4

import typing_extensions as te
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    WithJsonSchema,
    field_validator,
    model_serializer,
    model_validator,
)

from rigging.error import MissingModelError
from rigging.model import Model, ModelT
from rigging.parsing import try_parse_many
from rigging.tool.api import ApiToolCall
from rigging.util import AudioFormat, identify_audio_format, truncate_string

Role = t.Literal["system", "user", "assistant", "tool"]
"""The role of a message. Can be 'system', 'user', 'assistant', or 'tool'."""

EPHERMAL_CACHE_CONTROL = {"type": "ephemeral"}
"""Cache control entry for ephemeral messages."""


# Helper type for messages structured
# more similarly to other libraries
class MessageDict(t.TypedDict):
    """
    Helper to represent a [rigging.message.Message][] as a dictionary.
    """

    role: Role
    """The role of the message."""
    content: str | list[t.Any]  # TODO: Improve typing here
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
    slice_: t.Annotated[
        slice,
        PlainSerializer(lambda x: [x.start, x.stop], return_type=list[int]),
        WithJsonSchema({"type": "array", "items": {"type": "integer"}}),
    ]
    """The slice representing the range into the message content."""

    @field_validator("slice_", mode="before")
    @classmethod
    def validate_slice(cls, value: t.Any) -> slice:
        if isinstance(value, slice):
            return value
        if not isinstance(value, list):
            raise TypeError("slice must be a list or a slice")
        return slice(value[0], value[1])


class ContentText(BaseModel):
    """A text content part of a message."""

    type: t.Literal["text"] = "text"
    """The type of content (always `text`)."""
    text: str
    """The text content."""
    cache_control: dict[str, str] | None = None
    """Cache control entry for prompt caching."""

    def __str__(self) -> str:
        return self.text


class ContentImageUrl(BaseModel):
    """An image URL content part of a message."""

    class ImageUrl(BaseModel):
        url: str
        """The URL of the image (supports base64-encoded)."""
        detail: t.Literal["auto", "low", "high"] = "auto"
        """The detail level of the image."""

    type: t.Literal["image_url"] = "image_url"
    """The type of content (always `image_url`)."""
    image_url: ImageUrl
    """The image URL content."""
    cache_control: dict[str, str] | None = None
    """Cache control entry for prompt caching."""

    def __str__(self) -> str:
        return f"<ContentImageUrl url='{truncate_string(self.image_url.url, 50)}'>"

    @classmethod
    def from_file(
        cls,
        file: Path | str,
        *,
        mimetype: str | None = None,
        detail: t.Literal["auto", "low", "high"] = "auto",
    ) -> "ContentImageUrl":
        """
        Creates a ContentImageUrl object from a file.

        Args:
            file: The file to create the content from.
            mimetype: The mimetype of the file. If not provided, it will be guessed.

        Returns:
            The created ContentImageUrl object.
        """

        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File '{file}' does not exist")

        if mimetype is None:
            mimetype = mimetypes.guess_type(file)[0]

        if mimetype is None:
            raise ValueError(f"Could not determine mimetype for file '{file}'")

        encoded = base64.b64encode(file.read_bytes()).decode()
        url = f"data:{mimetype};base64,{encoded}"

        return cls(image_url=cls.ImageUrl(url=url, detail=detail))

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        mimetype: str,
        *,
        detail: t.Literal["auto", "low", "high"] = "auto",
    ) -> "ContentImageUrl":
        """
        Creates a ContentImageUrl object from raw bytes.

        Args:
            data: The raw bytes of the image.
            mimetype: The mimetype of the image.
            detail: The detail level of the image.

        Returns:
            The created ContentImageUrl
        """

        encoded = base64.b64encode(data).decode()
        url = f"data:{mimetype};base64,{encoded}"
        return cls(image_url=cls.ImageUrl(url=url, detail=detail))

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        detail: t.Literal["auto", "low", "high"] = "auto",
    ) -> "ContentImageUrl":
        """
        Creates a ContentImageUrl object from a URL.

        Args:
            url: The URL of the image.
            detail: The detail level of the image.

        Returns:
            The created ContentImageUrl object.
        """
        return cls(image_url=cls.ImageUrl(url=url, detail=detail))

    def to_bytes(self) -> bytes:
        """
        Converts the data to bytes (if the URL is base64-encoded).

        Returns:
            The decoded image data.
        """
        if not self.image_url.url.startswith("data:"):
            raise ValueError("Image URL is not base64-encoded")
        return base64.b64decode(self.image_url.url.split(",")[1])

    def save(self, path: Path | str) -> None:
        """
        Saves the data to a file.

        Args:
            path: The path to save the image to.
        """
        data = self.to_bytes()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


# https://platform.openai.com/docs/api-reference/chat/create
ContentAudioFormat = AudioFormat
ContentAudioFormatMimeTypes = ["audio/wav", "audio/mp3", "audio/ogg", "audio/flac"]


class ContentAudioInput(BaseModel):
    """An audio content part of a message."""

    class Audio(BaseModel):
        data: str
        """The base64-encoded audio data."""
        format: str
        """The format of the audio data."""
        transcript: str | None = None
        """The transcript of the audio data (if available)."""

    type: t.Literal["input_audio"] = "input_audio"
    """The type of content (always `input_audio`)."""
    input_audio: Audio
    """The audio URL content."""
    cache_control: dict[str, str] | None = None
    """Cache control entry for prompt caching."""

    def __str__(self) -> str:
        return (
            f"<ContentAudioInput format='{self.input_audio.format}' "
            f"transcript='{self.input_audio.transcript}' "
            f"data='{truncate_string(self.input_audio.data, 50)}'>"
        )

    @classmethod
    def from_file(
        cls,
        file: Path | str,
        *,
        format: ContentAudioFormat | None = None,
        transcript: str | None = None,
    ) -> "ContentAudioInput":
        """
        Creates a ContentAudioInput object from a file.

        Args:
            file: The file to create the content from.
            mimetype: The mimetype of the file. If not provided, it will be guessed.

        Returns:
            The created ContentAudioInput object.
        """

        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File '{file}' does not exist")

        if format is None:
            mimetype = mimetypes.guess_type(file)[0]
            if mimetype is None:
                raise ValueError(
                    f"Could not determine format for file '{file}', please provide one",
                )
            format = t.cast(ContentAudioFormat, mimetype.split("/")[-1])  # noqa: A001

        encoded = base64.b64encode(file.read_bytes()).decode()
        return cls(input_audio=cls.Audio(data=encoded, format=format, transcript=transcript))

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        format: ContentAudioFormat | None = None,
        transcript: str | None = None,
    ) -> "ContentAudioInput":
        """
        Creates a ContentAudioInput object from raw bytes.

        Args:
            data: The raw bytes of the audio.
            format: The format of the audio.

        Returns:
            The created ContentAudioInput
        """
        format = format or identify_audio_format(data) or "unknown"  # type: ignore [assignment] # noqa: A001
        encoded = base64.b64encode(data).decode()
        return cls(input_audio=cls.Audio(data=encoded, format=format, transcript=transcript))

    @property
    def transcript(self) -> str | None:
        """
        Returns the transcript of the audio data.

        Returns:
            The transcript of the audio data.
        """
        return self.input_audio.transcript

    def to_bytes(self) -> bytes:
        """
        Converts the audio data to bytes.

        Returns:
            The decoded audio data.
        """
        return base64.b64decode(self.input_audio.data)

    def save(self, path: Path | str) -> None:
        """
        Saves the audio data to a file.

        Args:
            path: The path to save the audio to.
        """
        data = self.to_bytes()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


Content = ContentText | ContentImageUrl | ContentAudioInput
"""The types of content that can be included in a message."""
ContentTypes = (ContentText, ContentImageUrl, ContentAudioInput)


class Message(BaseModel):
    """
    Represents a message with role, content, and parsed message parts.

    Note:
        Historically, `content` was a string, but multi-modal LLMs
        require us to have a more structured content representation.

        For interface stability, `content` will remain a property
        accessor for the text of a message, but the "real" content
        is available in `content_parts`. During serialization, we rename
        `content_parts` to `content` for compatibility.
    """

    uuid: UUID = Field(default_factory=uuid4, repr=False)
    """The unique identifier for the message."""
    role: Role
    """The role of the message."""
    parts: list[ParsedMessagePart] = Field(default_factory=list)
    """The parsed message parts."""
    content_parts: list[Content] = Field([], repr=False)
    """Interior str content or structured content parts."""
    tool_calls: list[ApiToolCall] | None = Field(None)
    """The tool calls associated with the message."""
    tool_call_id: str | None = Field(None)
    """Associated call id if this message is a response to a tool call."""

    def __init__(
        self,
        role: Role,
        content: str | t.Sequence[str | Content] | None = None,
        parts: t.Sequence[ParsedMessagePart] | None = None,
        tool_calls: t.Sequence[ApiToolCall] | t.Sequence[dict[str, t.Any]] | None = None,
        tool_call_id: str | None = None,
        cache_control: t.Literal["ephemeral"] | dict[str, str] | None = None,
        **kwargs: t.Any,
    ):
        # TODO: We default to an empty string, but this technically isn't
        # correct. APIs typically support the concept of a null-content msg
        if content is None:
            content = ""

        content = [content] if isinstance(content, str) else content
        content_parts = [
            ContentText(text=dedent(part)) if isinstance(part, str) else part for part in content
        ]

        if tool_calls is not None and not all(isinstance(call, ApiToolCall) for call in tool_calls):
            tool_calls = [
                ApiToolCall.model_validate(call) if isinstance(call, dict) else call
                for call in tool_calls
            ]

        if cache_control is not None and content_parts:
            content_parts[-1].cache_control = (
                cache_control if isinstance(cache_control, dict) else EPHERMAL_CACHE_CONTROL
            )

        super().__init__(
            role=role,
            content_parts=content_parts,
            parts=parts or [],
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            **kwargs,
        )

    def __str__(self) -> str:
        formatted = f"[{self.role}]:"

        if len(self.content_parts) == 1 and isinstance(self.content_parts[0], ContentText):
            formatted += f" {self.content_parts[0].text}"
        else:
            formatted += "\n |- " + "\n |- ".join(str(content) for content in self.content_parts)

        for tool_call in self.tool_calls or []:
            formatted += f"\n |- {tool_call}"

        return formatted

    def __len__(self) -> int:
        return len(self.content)

    def __repr_args__(self) -> t.Iterable[tuple[str | None, t.Any]]:
        # We want our content property to be in repr, but we can't
        # mark as a computed_field and also exclude it from dumps.
        yield from super().__repr_args__()
        yield "content", self.content

    @model_serializer(mode="wrap")
    def rename_content(self, handler: SerializerFunctionWrapHandler) -> t.Any:
        serialized = handler(self)
        if "content_parts" in serialized:
            serialized["content"] = serialized.pop("content_parts")

            # Some backwards compatibility for single text content
            # which we'll load straight into the content value as opposed
            # to a list of content parts.
            if (
                len(serialized["content"]) == 1
                and list(serialized["content"][0].keys()) == ["type", "text"]
                and serialized["content"][0].get("type") == "text"
            ):
                serialized["content"] = serialized["content"][0]["text"]

        return serialized

    @property
    @te.deprecated(".all_content is deprecated, use .content_parts instead", category=None)
    def all_content(self) -> str | list[Content]:
        """
        Returns all content parts of the message or the single text content part as a string.

        Deprecated - Use `.content_parts` instead
        """
        warnings.warn(
            ".all_content is deprecated, use .content_parts instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(self.content_parts) == 1 and isinstance(self.content_parts[0], ContentText):
            return self.content_parts[0].text
        return self.content_parts

    @property
    def content(self) -> str:
        """
        The content of the message as a string.

        If you need to access the structured content parts, use `.content_parts`.
        """
        return "\n".join(
            [content.text for content in self.content_parts if isinstance(content, ContentText)],
        )

    @content.setter
    def content(self, value: str) -> None:
        """
        Updates the content of the message.

        Warning:
            This will remove all parsed parts from the message.
        """
        # TODO: Maintain any parsed parts which are
        # still in the content - our move to slices for
        # tracking parsed parts makes this more complicated
        # so fow now I've opted to strip all parts
        # when content is changed.
        self.parts = []

        text_parts = [c for c in self.content_parts if isinstance(c, ContentText)]

        # If we have a single text part, we can just update it
        if len(text_parts) == 1:
            text_parts[0].text = value
            return

        # Otherwise we need to remove text parts without
        # removing other content parts
        other_parts = [c for c in self.content_parts if not isinstance(c, ContentText)]
        self.content_parts = [*other_parts, ContentText(text=value)]

    @model_validator(mode="after")
    def validate_parts(self) -> "Message":
        from rigging.model import Model

        # TODO: For now, we don't want to keep parts
        # under a generic Model class. This can result
        # from deserialization and will break our
        # overlapping part check later.
        #
        # We'll remove them from our parsed parts list,
        # but keep them in the content for reparsing.

        for part in self.parts[:]:
            if part.model.__class__ == Model:
                self.parts.remove(part)
        return self

    def to_openai_spec(self) -> dict[str, t.Any]:
        """
        Converts the message to the OpenAI-compatible JSON format. This should
        be the primary way to serialize a message for use with APIs.

        Returns:
            The serialized message.
        """
        # `content_parts` will be moved to `content`
        obj = self.model_dump(
            include={"role", "content_parts", "tool_calls", "tool_call_id"},
            exclude_none=True,
        )

        # Walk content parts and add a `\n` to the end of any text parts
        # which are followed by another text part (if not already present).
        #
        # This prevents model API behaviors from just concatenating the text
        # parts together without any separation which confuses the model.

        for i, current in enumerate(obj.get("content", [])):
            if i == len(obj["content"]) - 1:
                break

            next_ = obj["content"][i + 1]

            if (
                isinstance(current, dict)
                and current.get("type") == "text"
                and next_.get("type") == "text"
                and not current.get("text", "").endswith("\n")
            ):
                current["text"] += "\n"

        # Strip any transcript parts from audio input

        for part in obj.get("content", []):
            if isinstance(part, dict) and part.get("type") == "input_audio":
                part.get("input_audio", {}).pop("transcript", None)

        return obj

    # TODO: In general the add/remove/sync_part methods are
    # overly complicated. We should probably just update content,
    # then reparse all the models to get their fresh slices.
    #
    # I don't like all this manual slice recalculation logic, seems brittle.

    def _remove_part(self, part: ParsedMessagePart) -> None:
        text_parts = [p for p in self.content_parts if isinstance(p, ContentText)]
        if len(text_parts) > 1:
            raise NotImplementedError(
                "Managing parsed parts in messages with multiple content text parts is not supported",
            )

        if len(text_parts) == 0:
            raise ValueError("No text content to remove part from")

        text_part = text_parts[0]

        removed_length = part.slice_.stop - part.slice_.start
        text_part.text = text_part.text[: part.slice_.start] + text_part.text[part.slice_.stop :]
        self.parts.remove(part)

        # Update slices of any parts that come after the removed part
        for other_part in self.parts:
            if other_part.slice_.start > part.slice_.start:
                other_part.slice_ = slice(
                    other_part.slice_.start - removed_length,
                    other_part.slice_.stop - removed_length,
                )

    def _add_part(self, part: ParsedMessagePart) -> None:
        for existing in self.parts:
            if (
                part.slice_ == existing.slice_
                and part.model.xml_tags() == existing.model.xml_tags()
            ):
                return  # We clearly already have this part defined
            if max(part.slice_.start, existing.slice_.start) < min(
                part.slice_.stop,
                existing.slice_.stop,
            ):
                raise ValueError("Incoming part overlaps with an existing part")
        self.parts.append(part)

    # Looks more complicated than it is. We just want to clean all the models
    # in the message content by re-serializing them. As we do so, we'll need
    # to watch for the total size of our message shifting and update the slices
    # of the following parts accordingly. In other words, as A expands, B which
    # follows will have a new start slice and end slice.

    def _sync_parts(self) -> None:
        text_parts = [p for p in self.content_parts if isinstance(p, ContentText)]
        if len(text_parts) > 1:
            raise NotImplementedError(
                "Managing parsed parts in messages with multiple content text parts is not supported",
            )

        if len(text_parts) == 0:
            raise ValueError("No text content to remove part from")

        text_part = text_parts[0]

        self.parts = sorted(self.parts, key=lambda p: p.slice_.start)

        shift = 0
        for part in self.parts:
            existing = text_part.text[part.slice_]

            # Adjust for any previous shifts
            part.slice_ = slice(part.slice_.start + shift, part.slice_.stop + shift)

            # Check if the content has changed
            xml_content = part.model.to_pretty_xml()
            if xml_content == existing:
                continue

            # Otherwise update content, add to shift, and update this slice
            old_length = part.slice_.stop - part.slice_.start
            new_length = len(xml_content)

            text_part.text = (
                text_part.text[: part.slice_.start]
                + xml_content
                + text_part.text[part.slice_.stop :]
            )
            part.slice_ = slice(part.slice_.start, part.slice_.start + new_length)

            shift += new_length - old_length

        self.parts = sorted(self.parts, key=lambda p: p.slice_.start)

    def clone(self) -> "Message":
        """Creates a copy of the message."""
        return Message(
            self.role,
            copy.deepcopy(self.content_parts),
            parts=copy.deepcopy(self.parts),
        )

    def cache(self, cache_control: dict[str, str] | bool = True) -> "Message":  # noqa: FBT002
        """
        Update cache control settings for this message.

        Args:
            cache_control: The cache control settings to
                apply to the message. If `False`, all cache
                control settings will be removed. If `True`,
                the default ephemeral cache control will be applied.
                If a dictionary, it will be applied as the cache control settings.

        Returns:
            The updated message.
        """

        for part in self.content_parts:
            part.cache_control = None

        if cache_control is False:
            return self

        if cache_control is True:
            cache_control = EPHERMAL_CACHE_CONTROL

        if not isinstance(cache_control, dict):
            raise TypeError(f"Invalid cache control: {cache_control}")

        self.content_parts[-1].cache_control = cache_control

        return self

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

    def strip(
        self,
        model_type: type[Model],
        *,
        fail_on_missing: bool = False,
    ) -> list[ParsedMessagePart]:
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
            raise TypeError(
                f"Could not find <{model_type.__xml_tag__}> ({model_type.__name__}) in message",
            )

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
        self,
        model_type: type[ModelT],
        minimum: int | None = None,
        fail_on_missing: bool = False,  # noqa: FBT001, FBT002 (historical)
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
        parsed: list[tuple[ModelT, slice]] = try_parse_many(
            self.content,
            *types,
            fail_on_missing=fail_on_missing,
        )
        for model, slice_ in parsed:
            self._add_part(ParsedMessagePart(model=model, slice_=slice_))
        self._sync_parts()
        return [p[0] for p in parsed]

    @classmethod
    def from_model(
        cls: type["Message"],
        models: Model | t.Sequence[Model],
        role: Role = "user",
        suffix: str | None = None,
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
        cls,
        messages: t.Sequence[MessageDict]
        | t.Sequence["Message"]
        | MessageDict
        | "Message"
        | Content
        | str,
    ) -> list["Message"]:
        """Helper function to convert various common types to a strict list of Message objects."""
        if isinstance(messages, (Message, dict, str, *ContentTypes)):
            return [cls.fit(messages)]
        return [cls.fit(message) for message in messages]

    @classmethod
    def fit(cls, message: t.Union["Message", MessageDict, Content, str]) -> "Message":
        """Helper function to convert various common types to a Message object."""
        if isinstance(message, (str, *ContentTypes)):
            return cls(role="user", content=[message])
        return cls(**message) if isinstance(message, dict) else message.model_copy(deep=True)

    @classmethod
    def apply_to_list(cls, messages: t.Sequence["Message"], **kwargs: str) -> list["Message"]:
        """Helper function to apply keyword arguments to a list of Message objects."""
        return [message.apply(**kwargs) for message in messages]


Messages = t.Sequence[MessageDict] | t.Sequence[Message]
