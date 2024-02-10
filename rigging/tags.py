import typing as t
from pydantic import field_validator

from rigging.model import CoreModel


class Thinking(CoreModel):
    content: str


class Answer(CoreModel):
    content: str


class CommaDelimitedAnswer(CoreModel, tag="delimited-answer"):
    "Comma delimited answer, with a field validator to parse strings to lists"

    content: str

    @property
    def items(self) -> list[str]:
        return [i.strip(" \"'\t\r\n") for i in self.content.split(",")]

    @field_validator("content", mode="before")
    def parse_str_to_list(cls, v: t.Any) -> t.Any:
        if not isinstance(v, str) or "," not in v:
            raise ValueError(f"Cannot parse content as a comma delimited list: {v}")
        return v


class Description(CoreModel):
    content: str


class Instructions(CoreModel):
    content: str


class YesNoAnswer(CoreModel, tag="yes-no-answer"):
    "Yes/No answer, with a field validator to parse strings to booleans"

    boolean: bool

    @field_validator("boolean", mode="before")
    def parse_str_to_bool(cls, v: t.Any) -> t.Any:
        if isinstance(v, str):
            if v.strip().lower().startswith("yes"):
                return True
            elif v.strip().lower().startswith("no"):
                return False
        return v
