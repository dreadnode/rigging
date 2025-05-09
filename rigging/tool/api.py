"""
Models and utilities for working with API-based function calling tools.
"""

import typing as t

from pydantic import BaseModel, field_validator


class ApiNamedFunction(BaseModel):
    name: str


class ApiToolChoiceDefinition(BaseModel):
    type: t.Literal["function"] = "function"
    function: ApiNamedFunction


# I want to avoid making ToolChoice too specific as
# different providers interpret it differently.

# ToolChoice = t.Union[t.Literal["none"], t.Literal["auto"], ToolChoiceDefinition]
ApiToolChoice = str | dict[str, t.Any]


class ApiFunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, t.Any] | None = None

    # Logic here is a bit hacky, but in general
    # we want to handle cases where pydantic has
    # generated an empty object schema for a function
    # with no parameters, and convert it to None.
    #
    # This seems to be the most well-supported way
    # across providers to indicate a function with
    # no arguments.
    #
    # TODO: I've also seen cases where keys like additionalProperties
    # have special handling, but we'll assume LiteLLM will
    # take care of things like that for now.
    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, value: t.Any) -> t.Any:
        if not isinstance(value, dict):
            return value

        if value.get("type") == "object" and value.get("properties") == {}:
            return None

        return value


class ApiToolDefinition(BaseModel):
    type: t.Literal["function"] = "function"
    function: ApiFunctionDefinition


class ApiFunctionCall(BaseModel):
    name: str
    arguments: str


class ApiToolCall(BaseModel):
    id: str
    type: t.Literal["function"] = "function"
    function: ApiFunctionCall

    def __str__(self) -> str:
        return f"<ToolCall {self.function.name}({self.function.arguments})>"

    @property
    def name(self) -> str:
        return self.function.name
