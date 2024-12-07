"""
This module handles tools provided externally through APIs (OpenAI, Anthropic, Groq, etc.).
"""
from __future__ import annotations

import inspect
import json
import typing as t
import warnings
from functools import cached_property

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, TypeAdapter

if t.TYPE_CHECKING:
    from rigging.message import Message


class NamedFunction(BaseModel):
    name: str


class ToolChoiceDefinition(BaseModel):
    type: t.Literal["function"] = "function"
    function: NamedFunction


# I want to avoid making ToolChoice too specific as
# different providers interpret it differently.

# ToolChoice = t.Union[t.Literal["none"], t.Literal["auto"], ToolChoiceDefinition]
ToolChoice = str | dict[str, t.Any]


class FunctionDefinition(BaseModel):
    name: str
    description: t.Optional[str] = None
    parameters: t.Optional[dict[str, t.Any]] = None


class ToolDefinition(BaseModel):
    type: t.Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: t.Literal["function"] = "function"
    function: FunctionCall

    def __str__(self) -> str:
        return f"{self.function.name}({self.function.arguments})"


class ApiTool:
    def __init__(self, fn: t.Callable[..., t.Any]) -> None:
        self.fn = fn
        self.signature = inspect.signature(fn)
        self.type_adapter: TypeAdapter[t.Any] = TypeAdapter(self.fn)
        _ = self.schema  # Ensure schema is valid

    @cached_property
    def name(self) -> str:
        return self.fn.__name__

    @cached_property
    def description(self) -> str:
        return self.fn.__doc__ or ""

    @cached_property
    def schema(self) -> dict[str, t.Any]:
        schema = to_strict_json_schema(self.type_adapter)

        # Maintain the behavior of Annotated[<type>, "<description>"] by walking
        # adjusting the schema manually using signature annotations.

        for name in schema.get("properties", {}).keys():
            if name not in self.signature.parameters:
                continue

            param = self.signature.parameters[name]
            if t.get_origin(param.annotation) != t.Annotated:
                continue

            annotation_args = t.get_args(param.annotation)
            if len(annotation_args) != 2 or not isinstance(annotation_args[1], str):
                continue

            schema["properties"][name]["description"] = annotation_args[1]

        return schema

    @cached_property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionDefinition(name=self.name, description=self.description, parameters=self.schema)
        )

    async def execute(self, tool_call: ToolCall) -> Message:
        """Executes a function call on the tool."""

        from rigging.message import ContentTypes, Message

        if tool_call.function.name != self.fn.__name__:
            raise ValueError(f"Function name {tool_call.function.name} does not match {self.fn.__name__}")

        args = json.loads(tool_call.function.arguments)

        # For some reason, this will throw a coroutine unawaited warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.type_adapter.validate_python(args)

        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(**args)
        else:
            result = self.fn(**args)

        message = Message(role="tool", tool_call_id=tool_call.id)

        if isinstance(result, Message):
            message.all_content = result.all_content
        elif isinstance(result, ContentTypes):
            message.all_content = [result]
        elif isinstance(result, list) and all(isinstance(item, ContentTypes) for item in result):
            message.all_content = result
        else:
            message.all_content = str(result)

        return message

    __call__ = execute
