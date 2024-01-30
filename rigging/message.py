import string
import typing as t

from pydantic import BaseModel, Field, SerializeAsAny, computed_field

from rigging.error import MissingModelError
from rigging.model import CoreModel, CoreModelGeneric

Role = t.Literal["system", "user", "assistant"]


class MessageDict(t.TypedDict):
    role: Role
    content: str


# Structured portion of a message with
# the reference locator text
class ParsedMessagePart(BaseModel):
    model: SerializeAsAny[CoreModel]
    ref: str


class Message(BaseModel):
    role: Role
    parts: list[ParsedMessagePart] = Field(default_factory=list, exclude=True)

    _content: str = ""

    def __init__(self, role: Role, content: str, parts: list[ParsedMessagePart] | None = None):
        super().__init__(role=role, parts=parts if parts is not None else [])
        self._content = content

    def __str__(self) -> str:
        return f"[{self.role}]: {self.content}"

    @computed_field  # type: ignore[misc]
    @property
    def content(self) -> str:
        for part in self.parts:
            if part.ref not in self._content:
                raise ValueError(
                    f"Could not find '{part.ref}' in message content for model '{part.model.__class__.__name__}'"
                )
            xml_content = part.model.to_pretty_xml()
            self._content = self._content.replace(part.ref, xml_content)
            part.ref = xml_content
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        # TODO: Verify this behavior. We're stripping out
        # any parts that don't apply to the new content, but
        # we might want to just trigger a reparse against the
        # new content instead with our old models
        self.parts = [p for p in self.parts if p.ref in value]
        self._content = value

    def apply(self, **kwargs: str) -> None:
        template = string.Template(self.content)
        self.content = template.safe_substitute(**kwargs)

    def strip(self, model_type: type[CoreModel], fail_on_missing: bool = False) -> None:
        for part in self.parts:
            if isinstance(part.model, model_type):
                self.content = self.content.replace(part.ref, "")
                return

        if fail_on_missing:
            raise TypeError(f"Could not find <{model_type.__xml_tag__}> ({model_type.__name__}) in message")

    @property
    def models(self) -> list[CoreModel]:
        return [part.model for part in self.parts]

    def parse(self, model_type: type[CoreModelGeneric]) -> CoreModelGeneric:
        for model in self.models:
            if isinstance(model, model_type):
                return model
        return self.try_parse_many([model_type], fail_on_missing=True)[0]  # type: ignore [return-value]

    def try_parse(self, model_type: type[CoreModelGeneric]) -> CoreModelGeneric | None:
        for model in self.models:
            if isinstance(model, model_type):
                return model
        return next(iter(self.try_parse_many([model_type])), None)

    def parse_many(self, types: list[type[CoreModelGeneric]]) -> list[CoreModelGeneric]:
        return self.try_parse_many(types, fail_on_missing=True)

    def try_parse_many(
        self, types: list[type[CoreModelGeneric]], fail_on_missing: bool = False
    ) -> list[CoreModelGeneric]:
        parts: list[ParsedMessagePart] = []

        model: CoreModelGeneric
        for model_class in types:
            try:
                model, text = model_class.extract_xml(self.content)
                parts.append(ParsedMessagePart(model=model, ref=text))
            except MissingModelError as e:
                if fail_on_missing:
                    raise e
        self.parts = parts
        return self.models  # type: ignore [return-value]

    @classmethod
    def from_model(cls: type["Message"], models: CoreModel | list[CoreModel], role: Role = "user") -> "Message":
        parts: list[ParsedMessagePart] = []
        for model in models if isinstance(models, list) else [models]:
            text_form = model.to_pretty_xml()
            parts.append(ParsedMessagePart(model=model, ref=text_form))

        return cls(role=role, content="\n".join([part.ref for part in parts]), parts=parts)

    @classmethod
    def fit_list(cls: ["Message"], messages: list["Message"] | list[MessageDict]) -> list["Message"]:
        return [cls.fit(message) for message in messages]

    @classmethod
    def fit(cls: type["Message"], message: t.Union["Message", MessageDict]) -> "Message":
        return cls(**message) if isinstance(message, dict) else message


Messages = list[MessageDict] | list[Message]
