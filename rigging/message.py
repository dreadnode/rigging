import string
import typing as t

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

Role = t.Literal["system", "user", "assistant"]


# Helper type for messages structured
# more similarly to other libraries
class MessageDict(t.TypedDict):
    role: Role
    content: str


# Structured portion of a message with
# a slice indicating where is it located
class ParsedMessagePart(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: SerializeAsAny[Model]
    slice_: slice

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
    role: Role
    parts: list[ParsedMessagePart] = Field(default_factory=list, exclude=True)

    _content: str = ""

    def __init__(self, role: Role, content: str, parts: t.Sequence[ParsedMessagePart] | None = None):
        super().__init__(role=role, parts=parts if parts is not None else [])
        self._content = content

    def __str__(self) -> str:
        return f"[{self.role}]: {self.content}"

    def _remove_part(self, part: ParsedMessagePart) -> str:
        self._content = self._content[: part.slice_.start] + self._content[part.slice_.stop :]
        self.parts.remove(part)
        return self._content

    def _add_part(self, part: ParsedMessagePart) -> None:
        for existing in self.parts:
            if max(part.slice_.start, existing.slice_.start) < min(part.slice_.stop, existing.slice_.stop):
                raise ValueError("Incoming part overlaps with an existing part")
        self.parts.append(part)

    def _sync_parts(self) -> None:
        for part in self.parts:
            xml_content = part.model.to_pretty_xml()
            self._content = self._content[: part.slice_.start] + xml_content + self._content[part.slice_.stop :]
            part.slice_ = slice(part.slice_.start, part.slice_.start + len(xml_content))

    @computed_field  # type: ignore[misc]
    @property
    def content(self) -> str:
        self._sync_parts()
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

    def apply(self, **kwargs: str) -> None:
        template = string.Template(self.content)
        self.content = template.safe_substitute(**kwargs)

    def strip(self, model_type: type[Model], fail_on_missing: bool = False) -> list[ParsedMessagePart]:
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
        return [part.model for part in self.parts]

    def parse(self, model_type: type[ModelT]) -> ModelT:
        for model in self.models:
            if isinstance(model, model_type):
                return model
        return self.try_parse_many([model_type], fail_on_missing=True)[0]

    def try_parse(self, model_type: type[ModelT]) -> ModelT | None:
        for model in self.models:
            if isinstance(model, model_type):
                return model
        return next(iter(self.try_parse_many([model_type])), None)

    def parse_set(self, model_type: type[ModelT], minimum: int | None = None) -> list[ModelT]:
        return self.try_parse_set(model_type, minimum=minimum, fail_on_missing=True)

    def try_parse_set(
        self, model_type: type[ModelT], minimum: int | None = None, fail_on_missing: bool = False
    ) -> list[ModelT]:
        models = self.try_parse_many([model_type], fail_on_missing=fail_on_missing)
        if minimum is not None and len(models) < minimum:
            raise MissingModelError(f"Expected at least {minimum} {model_type.__name__} in message")
        return models

    def parse_many(self, types: t.Sequence[type[ModelT]]) -> list[ModelT]:
        return self.try_parse_many(types, fail_on_missing=True)

    def try_parse_many(self, types: t.Sequence[type[ModelT]], fail_on_missing: bool = False) -> list[ModelT]:
        model: ModelT
        parsed: list[ModelT] = []
        for model_class in types:
            try:
                for model, slice_ in model_class.from_text(self.content):
                    self._add_part(ParsedMessagePart(model=model, slice_=slice_))
                    parsed.append(model)
            except MissingModelError as e:
                if fail_on_missing:
                    raise e

        return parsed  # type: ignore [return-value]

    @classmethod
    def from_model(
        cls: type["Message"], models: Model | t.Sequence[Model], role: Role = "user", suffix: str | None = None
    ) -> "Message":
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
    def fit_list(cls, messages: t.Sequence["Message"] | t.Sequence[MessageDict]) -> list["Message"]:
        return [cls.fit(message) for message in messages]

    @classmethod
    def fit(cls, message: t.Union["Message", MessageDict]) -> "Message":
        return cls(**message) if isinstance(message, dict) else message


Messages = t.Sequence[MessageDict] | t.Sequence[Message]
