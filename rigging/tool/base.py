"""
Core types and functions for defining tools and handling tool calls.
"""

import functools
import inspect
import json
import typing as t
from dataclasses import dataclass, field
from functools import cached_property

from pydantic import TypeAdapter

from rigging.error import ToolDefinitionError
from rigging.model import Model, make_from_schema, make_from_signature
from rigging.tool.api import ApiFunctionDefinition, ApiToolCall, ApiToolDefinition
from rigging.tool.native import (
    JsonInXmlToolCall,
    JsonInXmlToolDefinition,
    NativeToolResult,
    XmlToolCall,
    XmlToolDefinition,
)
from rigging.tracing import tracer
from rigging.util import deref_json

if t.TYPE_CHECKING:
    from rigging.message import Message

ToolMode = t.Literal["auto", "api", "xml", "json-in-xml"]
"""
How tool calls are handled.

- `auto`: The method is chosed based on support (api > xml).
- `api`: Tool calls are delegated to api-provided function calling.
- `xml`: Tool calls are parsed in nested XML format.
- `json-in-xml`: Tool calls are parsed as raw JSON inside XML tags.
"""


@dataclass
class Tool:
    """Base class for representing a tool to a generator."""

    name: str
    """The name of the tool."""
    description: str
    """A description of the tool."""
    parameters_schema: dict[str, t.Any]
    """The JSON schema for the tool's parameters."""
    fn: t.Callable[..., t.Any]
    """The function to call."""

    _signature: inspect.Signature | None = field(default=None, init=False, repr=False)
    _type_adapter: TypeAdapter[t.Any] | None = field(default=None, init=False, repr=False)
    _model: type[Model] | None = field(default=None, init=False, repr=False)

    # In general we are split between 2 strategies for handling the data translations:
    #
    # 1. TypeAdapter applied straight to a callable (`api` and `json-in-xml` modes)
    # 2. Dynamic Model class built from the signature (`xml` mode)
    #
    # TODO: I'd like to unify these and pick which strategy works best for us. I'm inclined
    # to trust the TypeAdapter approach as it's less custom code, but we lose some capacity
    # to handle complex xml structures. Even with something like `xmltodict`, there is a lot
    # of ambiguity about homogeneous lists and nested structures.

    @classmethod
    def from_callable(
        cls,
        fn: t.Callable[..., t.Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> "Tool":
        from rigging.prompt import Prompt

        fn_for_signature = fn

        # We need to do some magic here because our Prompt object and
        # associated run function lack the context needed to construct
        # the schema at runtime - so we pass in the wrapped function for
        # attribute access and the top level Prompt.run for actual execution

        if isinstance(fn, Prompt):
            fn_for_signature = fn.func  # type: ignore [assignment]
            fn = fn.run

        # In the case that we are recieving a bound function which is tracking
        # an originating prompt, unwrap the prompt and use it's function for
        # the signature. Be sure to error for cases where we aren't dealing
        # with the singular Prompt.run, as we don't currently have logic to
        # handle any Concatenate use on the ParamSpec.

        elif hasattr(fn, "__rg_prompt__") and isinstance(fn.__rg_prompt__, Prompt):
            if fn.__name__ in ["run_many", "run_over"]:
                raise ValueError(
                    "Only the singular Prompt.run (Prompt.bind) is supported when using prompt objects inside API tools",
                )
            fn_for_signature = fn.__rg_prompt__.func  # type: ignore [assignment]

        signature = inspect.signature(fn_for_signature, eval_str=True)

        # Passing a function to a TypeAdapter results in an internal
        # call being made to the function after during validation.
        # We want to manage the call ourselves, so we pass in a dummy function
        # that does nothing, with a mirrored signature of the original
        # to ensure arguments are still validated properly.
        #
        # We pull a small trick here by returning the kwargs from the
        # dummy function, which will reflect all of the validation
        # logic applied from the TypeAdapter. We can use these
        # parsed arguments to call the original function.

        @functools.wraps(fn_for_signature)
        def empty_func(*args, **kwargs):  # type: ignore [no-untyped-def] # noqa: ARG001
            return kwargs

        # We'll also reconstruct __annotations__ from the signature
        # manually in case we are working with an object that
        # has manually set __signature__ and the __annotations__
        # might not be accurate. This is a bit of a hack, but
        # we can't control how pydantic resolves annotations

        annotations: dict[str, t.Any] = {}
        for param_name, param in signature.parameters.items():
            if param.annotation is not inspect.Parameter.empty:
                annotations[param_name] = param.annotation

        if signature.return_annotation is not inspect.Parameter.empty:
            annotations["return"] = signature.return_annotation

        empty_func.__annotations__ = annotations

        type_adapter: TypeAdapter[t.Any] = TypeAdapter(empty_func)

        schema = type_adapter.json_schema()

        # Maintain the behavior of Annotated[<type>, "<description>"] by walking
        # adjusting the schema manually using signature annotations.

        for prop_name in schema.get("properties", {}):
            if prop_name is None or prop_name not in signature.parameters:
                continue

            param = signature.parameters[prop_name]
            if t.get_origin(param.annotation) != t.Annotated:
                continue

            annotation_args = t.get_args(param.annotation)
            if len(annotation_args) != 2 or not isinstance(annotation_args[1], str):  # noqa: PLR2004
                continue

            schema["properties"][prop_name]["description"] = annotation_args[1]

        # Deref and flatten the schema for consistency

        schema = deref_json(schema, is_json_schema=True)

        self = cls(
            name=name or fn_for_signature.__name__,
            description=description or fn_for_signature.__doc__ or "",
            parameters_schema=schema,
            fn=fn,
        )

        self._signature = signature

        # For handling API calls, we'll use the type adapter to validate
        # the arguments before calling the function

        self._type_adapter = type_adapter

        return self

    @cached_property
    def api_definition(self) -> ApiToolDefinition:
        return ApiToolDefinition(
            function=ApiFunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.parameters_schema,
            ),
        )

    @property
    def model(self) -> type[Model]:
        # Do this lazily in case our our xml-based model doesn't
        # support some of the argument types. We don't want to
        # break api-based tools because of this.

        if self._model is None:
            try:
                self._model = (
                    make_from_signature(self._signature, "params")
                    if self._signature
                    else make_from_schema(self.parameters_schema, "params")
                )
            except Exception as e:  # noqa: BLE001
                raise ToolDefinitionError(
                    f"Failed to create model for tool '{self.name}'. "
                    "This is likely due to constraints on arguments when the `xml` tool mode is used.",
                ) from e
        return self._model

    @cached_property
    def xml_definition(self) -> XmlToolDefinition:
        return XmlToolDefinition.from_parameter_model(self.model, self.name, self.description)

    @cached_property
    def json_definition(self) -> JsonInXmlToolDefinition:
        return JsonInXmlToolDefinition(
            name=self.name,
            description=self.description,
            parameters=json.dumps(self.parameters_schema),
        )

    async def handle_tool_call(
        self,
        tool_call: ApiToolCall | XmlToolCall | JsonInXmlToolCall,
    ) -> tuple["Message", bool]:
        """
        Handle an incoming tool call from a generator.

        Args:
            tool_call: The tool call to handle.

        Returns:
            The message to send back to the generator or None if tool calling should not proceed.
        """

        from rigging.message import ContentText, ContentTypes, Message

        tool_call_parameters = (
            tool_call.function.arguments
            if isinstance(tool_call, ApiToolCall)
            else tool_call.parameters
        )

        with tracer.span(f"Tool {self.name}()", name=self.name) as span:
            if tool_call.name != self.name:
                raise ValueError(
                    f"Requested function name '{tool_call.name}' does not match '{self.name}'",
                )

            if isinstance(tool_call, ApiToolCall):
                span.set_attribute("tool_call_id", tool_call.id)

            # Load + validate arguments

            args: dict[str, t.Any]
            if isinstance(tool_call, ApiToolCall | JsonInXmlToolCall):
                args = json.loads(tool_call_parameters)

                if self._type_adapter is not None:
                    args = self._type_adapter.validate_python(args)

            elif isinstance(tool_call, XmlToolCall):
                parsed = self.model.from_text(
                    self.model.xml_start_tag() + tool_call_parameters + self.model.xml_end_tag(),
                )
                if not parsed:
                    raise ValueError("Failed to parse parameters")

                parameters = parsed[0][0]

                # As opposed to a model_dump, we want to retain the
                # argument object instances. We'll just flatten the
                # model into a dictionary for the function call.

                args = {
                    field_name: getattr(parameters, field_name, None)
                    for field_name in self.model.model_fields
                }

            span.set_attribute("arguments", args)

            # Call the function

            result = self.fn(**args)
            if inspect.isawaitable(result):
                result = await result

            span.set_attribute("result", result)

        message = (
            Message(role="tool", tool_call_id=tool_call.id)
            if isinstance(tool_call, ApiToolCall)
            else Message("user")
        )

        # If the tool returns nothing back to us, we'll assume that
        # they do not want to proceed with additional tool calling

        if result is None:
            message.content_parts = [ContentText(text="<none>")]
            return message, False

        # If the tool gave us back anything that looks like a message, we'll
        # just pass it along. Otherwise we need to box up the result.

        if isinstance(result, Message):
            message.content_parts = result.content_parts
        elif isinstance(result, ContentTypes):
            message.content_parts = [result]
        elif isinstance(result, list) and all(isinstance(item, ContentTypes) for item in result):
            message.content_parts = result
        else:
            message.content_parts = [ContentText(text=str(result))]

        # If this is a native tool call, we should wrap up our
        # result in a NativeToolResult object to provide clarity to the
        # generator. Otherwise we can rely on the `tool` role and associated
        # tool_call_id to provide context.
        #
        # TODO: It would be great to have some kind of identifier here to let
        # the model know what result is associated with what tool call when
        # we aren't working with api calls
        #
        # (we'd likely have to insert the shared identifier upstream in the call)

        if (
            len(message.content_parts) == 1
            and isinstance(message.content_parts[0], ContentText)
            and isinstance(tool_call, XmlToolCall | JsonInXmlToolCall)
        ):
            message.content_parts[0].text = NativeToolResult(
                name=self.name,
                result=message.content_parts[0].text,
            ).to_pretty_xml()

        return message, True


# Decorator


@t.overload
def tool(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> t.Callable[[t.Callable[..., t.Any]], Tool]:
    ...


@t.overload
def tool(
    func: t.Callable[..., t.Any],
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    ...


def tool(
    func: t.Callable[..., t.Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> t.Callable[[t.Callable[..., t.Any]], Tool] | Tool:
    """
    Decorator for creating a Tool, useful for overriding a name or description.

    Args:
        func: The function to wrap.
        name: The name of the tool.
        description: The description of the tool.

    Returns:
        The decorated Tool object.
    """

    def make_tool(func: t.Callable[..., t.Any]) -> Tool:
        return Tool.from_callable(func, name=name, description=description)

    if func is not None:
        return make_tool(func)

    return make_tool
