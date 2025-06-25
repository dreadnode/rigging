"""
This module covers core message objects and handling.
"""

import base64
import copy
import mimetypes
import re
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
    PrivateAttr,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from rigging.error import MessageWarning, MissingModelError
from rigging.model import Model, ModelT
from rigging.parsing import try_parse_many
from rigging.tools.base import ToolCall
from rigging.util import AudioFormat, identify_audio_format, shorten_string, truncate_string

Role = t.Literal["system", "user", "assistant", "tool"]
"""The role of a message. Can be 'system', 'user', 'assistant', or 'tool'."""

EPHERMAL_CACHE_CONTROL = {"type": "ephemeral"}
"""Cache control entry for ephemeral messages."""


# Helper type for messages structured more similarly to other libraries
class MessageDict(t.TypedDict):
    """
    Helper to represent a [rigging.message.Message][] as a dictionary.
    """

    role: Role
    """The role of the message."""
    content: str | list[t.Any]  # TODO: Improve typing here
    """The content of the message."""


SliceType = t.Literal["tool_call", "tool_response", "model", "other"]
SliceObj = t.Any


class MessageSlice(BaseModel):
    """
    Represents a slice content within a message.

    This can be a tool call, tool response, or model output. You can associate
    metadata with the slice to add rich information like scores, confidence levels,
    or reward information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: SliceType
    """The type of the slice."""
    obj: SerializeAsAny[SliceObj] | None = Field(default=None, repr=False)
    """The model, tool call, or other object associated with the slice."""
    start: int
    """The start index of the slice."""
    stop: int
    """The stop index of the slice."""
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    """Metadata associated with the slice."""

    _message: "Message | None" = PrivateAttr(None)

    @property
    def slice_(self) -> slice:
        """Returns the slice representing the range into the message content."""
        return slice(self.start, self.stop)

    @property
    def content(self) -> str:
        """Get the content text for this slice from the parent message."""
        if self._message is None:
            return "[detached]"
        return self._message.content[self.start : self.stop]

    @content.setter
    def content(self, value: str) -> None:
        """Set the content text for this slice in the parent message."""
        if self._message is None:
            warnings.warn(
                "Setting content on a detached MessageSlice, this will not update the message.",
                MessageWarning,
                stacklevel=2,
            )
            return

        self._message.content = (
            self._message.content[: self.start] + value + self._message.content[self.stop :]
        )
        self.stop = self.start + len(value)

    def __len__(self) -> int:
        """Returns the length of the slice."""
        return self.stop - self.start

    def __str__(self) -> str:
        """Returns a string representation of the slice."""
        content_preview = self.content if self._message else "[detached]"
        return f"<MessageSlice type='{self.type}' start={self.start} stop={self.stop} obj={self.obj.__class__.__name__ if self.obj else None} content='{shorten_string(content_preview, 50)}'>"

    def clone(self) -> "MessageSlice":
        """
        Creates a deep copy of the MessageSlice.

        Returns:
            A new MessageSlice instance with the same properties.
        """
        return MessageSlice(
            type=self.type,
            obj=self.obj,
            start=self.start,
            stop=self.stop,
            metadata=copy.deepcopy(self.metadata),
            _message=self._message,  # Keep the reference to the original message
        )


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
        return f"<ContentImageUrl url='{shorten_string(self.image_url.url, 50)}'>"

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
            f"data='{shorten_string(self.input_audio.data, 50)}'>"
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
            format: The format of the audio. If not provided, it will be guessed based on the file extension.
            transcript: The transcript of the audio data (if available).

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
            format = t.cast("ContentAudioFormat", mimetype.split("/")[-1])  # noqa: A001

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

CompatibilityFlag = t.Literal["content_as_str", "skip_tools"]


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

    model_config = ConfigDict(serialize_by_alias=True)

    uuid: UUID = Field(default_factory=uuid4, repr=False)
    """The unique identifier for the message."""
    role: Role
    """The role of the message."""
    content_parts: list[Content] = Field([], repr=False)
    """Interior str content or structured content parts."""
    tool_calls: list[ToolCall] | None = Field(None)
    """The tool calls associated with the message."""
    tool_call_id: str | None = Field(None)
    """Associated call id if this message is a response to a tool call."""
    metadata: dict[str, t.Any] = Field(default_factory=dict, repr=False)
    """Metadata associated with the message."""
    compatibility_flags: set[CompatibilityFlag] = Field(default_factory=set, repr=False)
    """Compatibility flags to be applied when conversions occur."""

    slice_refs: list[MessageSlice] = Field(default_factory=list, repr=False, alias="slices")

    def __init__(
        self,
        role: Role,
        content: str | t.Sequence[str | Content] | None = None,
        slices: t.Sequence[MessageSlice] | None = None,
        tool_calls: t.Sequence[ToolCall] | t.Sequence[dict[str, t.Any]] | None = None,
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

        if cache_control is not None and content_parts:
            content_parts[-1].cache_control = (
                cache_control if isinstance(cache_control, dict) else EPHERMAL_CACHE_CONTROL
            )

        super().__init__(
            role=role,
            content_parts=content_parts,
            slices=slices or [],
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            **kwargs,
        )

        for slice_ in self.slice_refs:
            slice_._message = self  # noqa: SLF001

    def __str__(self) -> str:
        formatted = (
            f"[{self.role}:{self.tool_call_id}]:" if self.tool_call_id else f"[{self.role}]:"
        )

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
        The content of the message as a string. If multiple text parts are present,
        they will be concatenated together with newlines in between.

        This is considered the ground truth for slices of this message. In other words,
        slices do not take into account any structured content parts like images or audio.

        If you need to access the structured content parts, use `.content_parts`.
        """
        text_parts = [
            content.text
            for content in self.content_parts
            if isinstance(content, ContentText) and content.text.strip()
        ]
        return "\n".join(text_parts)

    @content.setter
    def content(self, value: str) -> None:
        """
        Updates the text content of the message. If the message has multiple text parts,
        it will replace all text parts with a single new `ContentText` part containing the value.
        If the message has a single text part, it will update that part's text directly.
        Any parts that are not text will remain unchanged.
        """
        other_parts = [c for c in self.content_parts if not isinstance(c, ContentText)]

        # Find slices that still exist in the new content
        preserved_slices: list[MessageSlice] = []
        for slice_obj in self.slices:
            slice_content = slice_obj.content
            slice_start = value.find(slice_content)

            if slice_start != -1:
                # Update the existing slice object's positions
                slice_obj.start = slice_start
                slice_obj.stop = slice_start + len(slice_content)
                preserved_slices.append(slice_obj)

        # Handle empty content
        if value.strip() == "":
            value = ""

        self.content_parts = [*other_parts, ContentText(text=value)]
        self.slices = preserved_slices

    @property
    def slices(self) -> list[MessageSlice]:
        """The slices of the message content."""
        return self.slice_refs

    @slices.setter
    def slices(self, value: list[MessageSlice]) -> None:
        for slice_ in value:
            slice_._message = self  # noqa: SLF001
        self.slice_refs = value

    def _add_slice(self, slice_obj: MessageSlice) -> MessageSlice:
        """Add a slice to the message."""
        for existing in self.slices:
            if (  # avoid obvious duplicates
                existing.start == slice_obj.start
                and existing.stop == slice_obj.stop
                and existing.type == slice_obj.type
            ):
                return existing
        slice_obj._message = self  # noqa: SLF001
        self.slices.append(slice_obj)
        return slice_obj

    def _remove_slice(self, slice_: MessageSlice) -> MessageSlice:
        """Remove a slice and update content."""
        if slice_ not in self.slices:
            raise ValueError(f"Slice {slice_} not found in message slices")

        self.content = self.content[: slice_.start] + self.content[slice_.stop :]
        slice_._message = None  # Detach from message  # noqa: SLF001
        return slice_

    def append_slice(
        self,
        content: str | Model,
        slice_type: SliceType | None = None,
        *,
        obj: SliceObj | None = None,
        metadata: dict[str, t.Any] | None = None,
    ) -> MessageSlice:
        """
        Add content to the end of the message (with newline separator) and create a slice tracking it.

        Type defaults to 'model' for Model objects, 'other' for strings.

        Args:
            content: The content to append. This can be a string or a Model instance.
            slice_type: The type of slice to create, inferred from content type if not provided.
            obj: The object associated with the slice
            metadata: Additional metadata for the slice

        Returns:
            The created MessageSlice
        """
        if isinstance(content, Model):
            content_str = content.to_pretty_xml()
            slice_type = slice_type or "model"
            obj = obj or content
        else:
            content_str = content
            slice_type = slice_type or "other"

        start_pos = len(self.content) + (
            1 if self.content_parts else 0  # +1 for newline if not empty
        )
        self.content_parts.append(ContentText(text=content_str))

        return self._add_slice(
            MessageSlice(
                type=slice_type,
                obj=obj,
                start=start_pos,
                stop=len(self.content),
                metadata=metadata or {},
            ),
        )

    def replace_with_slice(
        self,
        content: str | Model,
        slice_type: SliceType | None = None,
        *,
        obj: SliceObj | None = None,
        metadata: dict[str, t.Any] | None = None,
    ) -> MessageSlice:
        """
        Replace all message content and create a slice tracking the new content.

        Type defaults to 'model' for Model objects, 'other' for strings.

        Args:
            content: The content to replace with. This can be a string or a Model instance.
            slice_type: The type of slice to create, inferred from content type if not provided.
            obj: The object associated with the slice
            metadata: Additional metadata for the slice

        Returns:
            The created MessageSlice
        """
        # Clear existing content and slices
        self.content_parts = []
        self.slices = []

        return self.append_slice(
            content,
            slice_type=slice_type,
            obj=obj,
            metadata=metadata,
        )

    @t.overload
    def mark_slice(
        self,
        target: str | tuple[int, int] | t.Literal[-1] | re.Pattern[str] | type[Model],
        slice_type: SliceType | None = None,
        *,
        obj: SliceObj | None = None,
        metadata: dict[str, t.Any] | None = None,
        select: t.Literal["first", "last"] = "first",
        case_sensitive: bool = True,
    ) -> MessageSlice | None: ...

    @t.overload
    def mark_slice(
        self,
        target: str | tuple[int, int] | t.Literal[-1] | re.Pattern[str] | type[Model],
        slice_type: SliceType | None = None,
        *,
        obj: SliceObj | None = None,
        metadata: dict[str, t.Any] | None = None,
        select: t.Literal["all"],
        case_sensitive: bool = True,
    ) -> list[MessageSlice]: ...

    def mark_slice(  # noqa: PLR0912
        self,
        target: str | tuple[int, int] | t.Literal[-1] | re.Pattern[str] | type[Model],
        slice_type: SliceType | None = None,
        *,
        obj: SliceObj | None = None,
        metadata: dict[str, t.Any] | None = None,
        select: t.Literal["first", "last", "all"] = "first",
        case_sensitive: bool = True,
    ) -> MessageSlice | list[MessageSlice] | None:
        """
        Mark existing content as slices without modifying content.

        Args:
            target: What to mark as a slice:
                - str: Find this text in content
                - tuple[int, int]: Mark this exact range
                - "*" or -1: Mark entire message content
                - re.Pattern: Find matches of this pattern
                - type[Model]: Parse and mark instances of this model type
            slice_type: The type of slice to create
            obj: The object associated with the slice
            metadata: Additional metadata for the slice
            select: Which matches to return - 'first', 'last', or 'all'
            case_sensitive: Whether string search should be case sensitive

        Returns:
            If select='first'/'last': MessageSlice or None if no matches, otherwise if select='all': list[MessageSlice] (empty if no matches)
        """
        matches: list[tuple[int, int]] = []
        objects: list[SliceObj] = []
        content = self.content

        # Mark entire content
        if content and (target in (-1, "*")):
            matches = [(0, len(content))]

        # Direct range specification - validate bounds
        elif isinstance(target, tuple):
            start, stop = target
            if not (0 <= start < len(content) and start < stop <= len(content)):
                warnings.warn(
                    f"Invalid range ({start}, {stop}) for content length {len(content)}",
                    MessageWarning,
                    stacklevel=2,
                )
                matches = []
            else:
                matches = [(start, stop)]

        elif isinstance(target, str):
            # Handle empty string case
            if not target:
                warnings.warn("Empty string target provided", MessageWarning, stacklevel=2)
                matches = []

            # Find all occurrences of the string (case insensitive by default)
            else:
                search_content = content.lower() if not case_sensitive else content
                search_target = target.lower() if not case_sensitive else target
                start = 0
                while True:
                    pos = search_content.find(search_target, start)
                    if pos == -1:
                        break
                    matches.append((pos, pos + len(target)))
                    start = pos + 1

        # Find all regex matches
        elif isinstance(target, re.Pattern):
            matches = [(match.start(), match.end()) for match in target.finditer(content)]

        # Parse and mark instances of this model type from content
        elif isinstance(target, type) and issubclass(target, Model):
            try:
                parsed_models = try_parse_many(content, target)
                for model, slice_range in parsed_models:
                    matches.append((slice_range.start, slice_range.stop))
                    objects.append(model)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"Failed to parse {target.__name__} from content: {e}",
                    MessageWarning,
                    stacklevel=2,
                )
                matches = []

        if not objects:
            objects = [obj] * len(matches)

        # Create base slices for storage
        created_slices = []
        for (start, stop), obj_ in zip(matches, objects, strict=True):
            base_slice = MessageSlice(
                type=slice_type
                or ("model" if isinstance(target, type) and issubclass(target, Model) else "other"),
                obj=obj_,
                start=start,
                stop=stop,
                metadata=metadata or {},
            )
            created_slices.append(self._add_slice(base_slice))

        if select == "first":
            return created_slices[0] if created_slices else None

        if select == "last":
            return created_slices[-1] if created_slices else None

        return created_slices

    def find_slices(
        self,
        slice_type: SliceType | None = None,
        filter_fn: t.Callable[[MessageSlice], bool] | None = None,
        *,
        reverse: bool = False,
    ) -> list[MessageSlice]:
        """
        Find slices with simple filtering.

        Args:
            slice_type: Filter by slice type
            filter_fn: Custom filter function called for each slice

        Returns:
            List of matching slices
        """
        results = []
        for slice_obj in self.iter_slices(slice_type=slice_type, reverse=reverse):
            if filter_fn is not None and not filter_fn(slice_obj):
                continue
            results.append(slice_obj)

        return results

    def get_slice(
        self,
        slice_type: SliceType | None = None,
        *,
        select: t.Literal["first", "last"] = "first",
    ) -> MessageSlice | None:
        """
        Get a single slice of the message, optionally filtering by type.

        Args:
            slice_type: Optional type or string to filter slices by.
            select: Which slice to return - 'first' or 'last'.

        Returns:
            The requested MessageSlice or None if not found.
        """
        return next(self.iter_slices(slice_type=slice_type, reverse=(select == "last")), None)

    def iter_slices(
        self,
        slice_type: SliceType | t.Iterable[SliceType] | None = None,
        *,
        reverse: bool = False,
    ) -> t.Iterator[MessageSlice]:
        """
        Iterate over slices of the message, optionally filtering by type.

        Args:
            slice_type: Optional type or iterable of types to filter slices by.
            reverse: If True, iterate in reverse order.

        Returns:
            An iterator over MessageSlice objects.
        """
        sorted_slices = sorted(self.slices, key=lambda s: s.start, reverse=reverse)
        if slice_type is None:
            return iter(sorted_slices)

        slice_type = [slice_type] if isinstance(slice_type, str) else slice_type

        if isinstance(slice_type, (list, tuple)):
            return (s for s in sorted_slices if s.type in slice_type)

        return (s for s in sorted_slices if s.type == slice_type)

    def remove_slices(
        self,
        *slices: MessageSlice | str | SliceType | type[t.Any],
    ) -> list[MessageSlice]:
        """
        Removes and returns slices from the message that match the given object.

        If the object is a string, it will find slices that match the string content.
        If the object is a `SliceType`, it will find slices of that type.
        If the object is a type, it will find slices that have an `obj` of that type.
        If the object is a `MessageSlice`, it will remove that slice exactly.

        Args:
            *slices: The slices to remove. Can be a `MessageSlice`, a string, a `SliceType`, or a type.

        Returns:
            The removed `MessageSliceRef` objects.
        """
        removed: list[MessageSlice] = []

        matching_slices: list[MessageSlice] = []
        for slice_ in slices:
            for existing in self.slices:
                if (
                    (
                        isinstance(slice_, str)
                        and slice_ in t.get_args(SliceType)
                        and existing.type == slice_
                    )
                    or (
                        isinstance(slice_, str)
                        and slice_ not in t.get_args(SliceType)
                        and self.content[existing.slice_].lower() == slice_.lower()
                    )
                    or (
                        isinstance(slice_, type)
                        and existing.obj
                        and isinstance(existing.obj, slice_)
                    )
                    or (isinstance(slice_, MessageSlice) and existing == slice_)
                ):
                    matching_slices.append(existing)  # noqa: PERF401

        removed = [
            self._remove_slice(matched)
            for matched in sorted(matching_slices, key=lambda s: s.start, reverse=True)
        ]
        self.content = self.content.strip()

        return removed

    @te.deprecated(".strip() is deprecated, use .remove_slice() instead", category=None)
    def strip(self, obj: SliceType | type[t.Any]) -> list[MessageSlice]:
        """
        Removes and returns all slices of the specified type from the message.

        This is a deprecated method, use `remove_slice()` instead.

        Args:
            obj: The type of slice to remove. Can be a `SliceType` or a model class.
                If a model class is provided, it will remove all slices
                that have a model of that type.

        Returns:
            A list of removed slices.
        """
        warnings.warn(
            ".strip() is deprecated, use .remove_slice() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.remove_slices(obj)

    def clone(self) -> "Message":
        """Creates a copy of the message."""
        return Message(
            role=self.role,
            content=copy.deepcopy(self.content_parts),
            slices=copy.deepcopy(self.slices),
            tool_calls=copy.deepcopy(self.tool_calls),
            tool_call_id=self.tool_call_id,
            metadata=copy.deepcopy(self.metadata),
            compatibility_flags=copy.deepcopy(self.compatibility_flags),
        )

    @te.deprecated(".to_openai_spec() is deprecated, use .to_openai() instead.", category=None)
    def to_openai_spec(self) -> dict[str, t.Any]:
        """
        Converts the message to the OpenAI-compatible JSON format. This should
        be the primary way to serialize a message for use with APIs.

        Deprecated - Use `.to_openai` instead
        """
        warnings.warn(
            ".to_openai_spec() is deprecated, use .to_openai() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_openai()

    def to_openai(
        self,
        *,
        compatibility_flags: set[CompatibilityFlag] | None = None,
    ) -> dict[str, t.Any]:
        """
        Converts the message to the OpenAI-compatible JSON format. This should
        be the primary way to serialize a message for use with APIs.

        Returns:
            The serialized message.
        """
        compatibility_flags = compatibility_flags or self.compatibility_flags
        include_fields = {"role", "content_parts"}
        if "skip_tools" not in compatibility_flags:
            include_fields.add("tool_calls")
            include_fields.add("tool_call_id")

        obj = self.model_dump(
            include=include_fields,
            exclude_none=True,
        )

        # Some backwards compatibility for single text content
        # which we'll load straight into the content value as opposed
        # to a list of content parts.

        if (
            len(obj["content"]) == 1
            and list(obj["content"][0].keys()) == ["type", "text"]
            and obj["content"][0].get("type") == "text"
        ):
            compatibility_flags.add("content_as_str")

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
                and not str(current.get("text", "")).endswith("\n")
            ):
                current["text"] += "\n"

        # Strip any transcript parts from audio input

        for part in obj.get("content", []):
            if isinstance(part, dict) and part.get("type") == "input_audio":
                part.get("input_audio", {}).pop("transcript", None)

        # If enabled, we need to convert our content to a flat
        # string for API compatibility. Groq is an example of an API
        # which will complain for some roles if we send a list of content parts.

        if "content_as_str" in compatibility_flags:
            obj["content"] = "".join(
                part["text"]
                for part in obj["content"]
                if isinstance(part, dict) and part.get("type") == "text"
            )

        return obj

    def meta(self, **kwargs: t.Any) -> "Message":
        """
        Updates the metadata of the message with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated message.
        """
        self.metadata.update(kwargs)
        return self

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

    def truncate(self, max_length: int, suffix: str = "\n[truncated]") -> "Message":
        """
        Truncates the message content to a maximum length.

        Args:
            max_length: The maximum length of the message content.

        Returns:
            The truncated message.
        """
        new = self.clone()
        new.content = truncate_string(new.content, max_length, suf=suffix)
        return new

    @property
    def models(self) -> list[Model]:
        """
        Returns a list of all models available in slices of the message.
        """
        return [slice_.obj for slice_ in self.slices if isinstance(slice_.obj, Model)]

    @property
    @te.deprecated(".parts is deprecated, iterate through .slices instead", category=None)
    def parts(self) -> list[t.Any]:
        """
        Deprecated - iterate through .slices instead
        """
        warnings.warn(
            ".parts is deprecated, iterate through .slices instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    # TODO: Many of these functions are duplicates from the parsing
    # module, but here we don't hand back slices and want there
    # to be a convenient access model. We should probably consolidate.

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
            self._add_slice(
                MessageSlice(
                    type="model",
                    obj=model,
                    start=slice_.start,
                    stop=slice_.stop,
                    metadata={"model_type": model.__class__.__name__},
                ),
            )
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
        slices_: list[MessageSlice] = []
        content: str = ""
        for model in models if isinstance(models, list) else [models]:
            model_xml = model.to_pretty_xml()
            slice_ = slice(len(content), len(content) + len(model_xml))
            content += f"{model_xml}\n"
            slices_.append(
                MessageSlice(
                    type="model",
                    obj=model,
                    start=slice_.start,
                    stop=slice_.stop,
                    metadata={"model_type": model.__class__.__name__},
                ),
            )

        if suffix is not None:
            content += f"\n{suffix}"

        return cls(role=role, content=content, slices=slices_)

    @classmethod
    def fit_as_list(
        cls,
        messages: "t.Sequence[MessageDict] | t.Sequence[Message] | MessageDict | Message | Content | str",
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
        return (
            cls.model_validate(message)
            if isinstance(message, dict)
            else message.model_copy(deep=True)
        )

    @classmethod
    def apply_to_list(cls, messages: t.Sequence["Message"], **kwargs: str) -> list["Message"]:
        """Helper function to apply keyword arguments to a list of Message objects."""
        return [message.apply(**kwargs) for message in messages]


Messages = t.Sequence[MessageDict] | t.Sequence[Message]


def inject_system_content(messages: list[Message], content: str) -> list[Message]:
    """
    Injects content into a list of messages as a system message.

    Note:
        If the message list is empty or the first message is not a system message,
        a new system message with the given content is inserted at the beginning of the list.
        If the first message is a system message, the content is appended to it.

    Args:
        messages: The list of messages to modify.
        content: The content to be injected.

    Returns:
        The modified list of messages
    """
    if content.strip() == "":
        return messages
    if len(messages) == 0 or messages[0].role != "system":
        messages.insert(0, Message(role="system", content=content))
    elif messages[0].role == "system" and content not in messages[0].content:
        messages[0].content += "\n\n" + content
    return messages


def strip_system_content(messages: list[Message], content: str) -> list[Message]:
    """
    Strips the system message from a list of messages.

    Args:
        messages: The list of messages to modify.

    Returns:
        The modified list of messages without the system message.
    """
    if content.strip() == "":
        return messages

    if not messages or messages[0].role != "system":
        return messages

    system_message = messages[0]
    system_message.content = system_message.content.replace(content, "").strip()

    if system_message.content == "":
        return messages[1:]

    return messages
