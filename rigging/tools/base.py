"""
Core types and functions for defining tools and handling tool calls.
"""

import functools
import inspect
import json
import re
import typing as t
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import typing_extensions as te
from pydantic import BaseModel, TypeAdapter, ValidationError, field_validator
from pydantic_xml import attr

from rigging.error import Stop, ToolDefinitionError, ToolWarning
from rigging.model import (
    ErrorModel,
    Model,
    SystemErrorModel,
    make_from_schema,
    make_from_signature,
)
from rigging.tracing import tracer
from rigging.util import deref_json

if t.TYPE_CHECKING:
    from rigging.message import Message

TOOL_STOP_TAG = "rg-stop"

P = t.ParamSpec("P")
R = t.TypeVar("R")

ToolMode = t.Literal["auto", "api", "xml", "json", "json-in-xml", "json-with-tag"]
"""
How tool calls are handled.

- `auto`: The method is chosen based on support (api w/ fallback to json-in-xml).
- `api`: Tool calls are delegated to api-provided function calling.
- `xml`: Tool calls are parsed in a nested XML format which is native to Rigging.
- `json`: Tool calls are parsed as raw name/arg JSON anywhere in assistant message content.
- `json-in-xml`: Tool calls are parsed using JSON for arguments, and XML for everything else.
- `json-with-tag`: Tool calls are parsed as name/arg JSON structures inside an XML tag to identify it.
"""


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
        return f"<ToolCall id={self.id} {self.function.name}({self.function.arguments})>"

    @property
    def name(self) -> str:
        return self.function.name

    @property
    def arguments(self) -> str:
        return self.function.arguments


class ToolResponse(Model):
    id: str = attr(default="")
    result: str


def _is_unbound_method(func: t.Any) -> bool:
    is_method = (
        inspect.ismethod(func)
        and not isinstance(func, staticmethod)
        and not isinstance(func, classmethod)
    )
    return is_method is not hasattr(func, "__self__")


@dataclass
class Tool(t.Generic[P, R]):
    """Base class for representing a tool to a generator."""

    name: str
    """The name of the tool."""
    description: str
    """A description of the tool."""
    parameters_schema: dict[str, t.Any]
    """The JSON schema for the tool's parameters."""
    fn: t.Callable[P, R]
    """The function to call."""
    catch: bool | set[type[Exception]] = False
    """
    Whether to catch exceptions and return them as messages.

    - `False`: Do not catch exceptions.
    - `True`: Catch all exceptions.
    - `set[type[Exception]]`: Catch only the specified exceptions.
    """
    truncate: int | None = None
    """If set, the maximum number of characters to truncate any tool output to."""

    _signature: inspect.Signature | None = field(default=None, init=False, repr=False)
    _type_adapter: TypeAdapter[t.Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
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
        fn: t.Callable[P, R],
        *,
        name: str | None = None,
        description: str | None = None,
        catch: bool | t.Iterable[type[Exception]] = False,
        truncate: int | None = None,
    ) -> te.Self:
        from rigging.prompt import Prompt

        fn_for_signature = fn

        # We need to do some magic here because our Prompt object and
        # associated run function lack the context needed to construct
        # the schema at runtime - so we pass in the wrapped function for
        # attribute access and the top level Prompt.run for actual execution

        if isinstance(fn, Prompt):
            fn_for_signature = fn.func  # type: ignore [assignment]
            fn = fn.run  # type: ignore [assignment]

        # In the case that we are receiving a bound function which is tracking
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

        description = inspect.cleandoc(description or fn_for_signature.__doc__ or "")
        description = re.sub(r"(?![\r\n])(\b\s+)", " ", description)

        self = cls(
            name=name or fn_for_signature.__name__,
            description=description,
            parameters_schema=schema,
            fn=fn,
            catch=catch if isinstance(catch, bool) else set(catch),
            truncate=truncate,
        )

        self._signature = signature
        self.__signature__ = signature  # type: ignore [attr-defined]
        self.__name__ = self.name  # type: ignore [attr-defined]
        self.__doc__ = self.description

        # For handling API calls, we'll use the type adapter to validate
        # the arguments before calling the function

        self._type_adapter = type_adapter

        return self

    @cached_property
    def definition(self) -> ToolDefinition:
        """
        Returns the tool definition for this tool.
        This is used for API calls and should be used
        to construct the tool call in the generator.
        """
        return ToolDefinition(
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters=self.parameters_schema,
            ),
        )

    @cached_property
    def api_definition(self) -> ToolDefinition:
        return self.definition

    @property
    def model(self) -> type[Model]:
        # Usually, we only dynamically construct a model when we are
        # using `xml` tool calls (noted above). We'll do this lazily
        # to avoid overhead and exceptions.

        # We use the signature if we have it as it's more accurate,
        # but fallback to using just the schema if we don't.

        if self._model is None:
            try:
                self._model = (
                    make_from_signature(self._signature, "params")
                    if self._signature
                    else make_from_schema(self.parameters_schema, "params")
                )
            except Exception as e:
                raise ToolDefinitionError(
                    f"Failed to create model for tool '{self.name}'. "
                    "This is likely due to constraints on arguments when the `xml` tool mode is used.",
                ) from e
        return self._model

    async def handle_tool_call(  # noqa: PLR0912
        self,
        tool_call: ToolCall,
    ) -> tuple["Message", bool]:
        """
        Handle an incoming tool call from a generator.

        Args:
            tool_call: The tool call to handle.

        Returns:
            A tuple containing the message to send back to the generator and a
            boolean indicating whether tool calling should stop.
        """

        from rigging.message import ContentText, ContentTypes, Message

        with tracer.span(f"Tool {self.name}()", name=self.name) as span:
            if tool_call.name != self.name:
                warnings.warn(
                    f"Tool call name mismatch: {tool_call.name} != {self.name}",
                    ToolWarning,
                    stacklevel=2,
                )
                return Message.from_model(SystemErrorModel(content="Invalid tool call.")), True

            if hasattr(tool_call, "id") and isinstance(tool_call.id, str):
                span.set_attribute("tool_call_id", tool_call.id)

            result: t.Any
            stop = False

            # Load + validate arguments

            try:
                kwargs = json.loads(tool_call.function.arguments)
                if self._type_adapter is not None:
                    kwargs = self._type_adapter.validate_python(kwargs)
                span.set_attribute("arguments", kwargs)
            except (json.JSONDecodeError, ValidationError) as e:
                result = ErrorModel.from_exception(e)

            # Call the function

            else:
                try:
                    result = self.fn(**kwargs)  # type: ignore [call-arg]
                    if inspect.isawaitable(result):
                        result = await result

                    if isinstance(result, Stop):
                        raise result  # noqa: TRY301
                except Stop as e:
                    result = f"<{TOOL_STOP_TAG}>{e.message}</{TOOL_STOP_TAG}>"
                    span.set_attribute("stop", True)
                    stop = True
                except Exception as e:
                    if self.catch is True or (
                        not isinstance(self.catch, bool) and isinstance(e, tuple(self.catch))
                    ):
                        result = ErrorModel.from_exception(e)
                    else:
                        raise

            span.set_attribute("result", result)

        message = Message(role="tool", tool_call_id=tool_call.id)

        # If the tool gave us back anything that looks like a message, we'll
        # just pass it along. Otherwise we need to box up the result.

        if isinstance(result, Message):
            message.content_parts = result.content_parts
        elif isinstance(result, ContentTypes):
            message.content_parts = [result]
        elif (
            isinstance(result, list)
            and result
            and all(isinstance(item, ContentTypes) for item in result)
        ):
            message.content_parts = result
        else:
            message.content_parts = [ContentText(text=str(result))]

        if self.truncate:
            message = message.truncate(self.truncate)

        return message, stop

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)


@t.overload
def tool(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] = False,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], Tool[P, R]]: ...


@t.overload
def tool(
    func: t.Callable[P, R],
    /,
) -> Tool[P, R]: ...


def tool(
    func: t.Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] = False,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[P, R]], Tool[P, R]] | Tool[P, R]:
    """
    Decorator for creating a Tool, useful for overriding a name or description.

    Args:
        func: The function to wrap.
        name: The name of the tool.
        description: The description of the tool.
        catch: Whether to catch exceptions and return them as messages.
            - `False`: Do not catch exceptions.
            - `True`: Catch all exceptions.
            - `list[type[Exception]]`: Catch only the specified exceptions.
        truncate: If set, the maximum number of characters to truncate any tool output to.

    Returns:
        The decorated Tool object.

    Example:
        ```
        @tool(name="add_numbers", description="This is my tool")
        def add(x: int, y: int) -> int:
            return x + y
        ```
    """

    def make_tool(func: t.Callable[..., t.Any]) -> Tool[P, R]:
        # TODO: Improve consistency of detection here before enabling this warning
        # if _is_unbound_method(func):
        #     warnings.warn(
        #         "Passing a class method to @tool improperly handles the 'self' argument, use @tool_method instead.",
        #         SyntaxWarning,
        #         stacklevel=3,
        #     )

        return Tool.from_callable(
            func,
            name=name,
            description=description,
            catch=catch,
            truncate=truncate,
        )

    if func is not None:
        return make_tool(func)

    return make_tool


# Special code for handling tool decorators on class methods


class ToolMethod(Tool[P, R]):
    """A Tool wrapping a class method."""

    def __get__(self, instance: t.Any, owner: t.Any) -> "Tool[P, R]":
        if instance is None:
            return self

        bound_method = self.fn.__get__(instance, owner)
        bound_tool = Tool[P, R](
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
            fn=bound_method,
            catch=self.catch,
        )

        bound_tool.__signature__ = self.__signature__  # type: ignore [attr-defined]
        bound_tool._signature = self._signature  # noqa: SLF001
        bound_tool._type_adapter = self._type_adapter  # noqa: SLF001
        bound_tool._model = self._model  # noqa: SLF001

        return bound_tool


@t.overload
def tool_method(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] = False,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], ToolMethod[P, R]]: ...


@t.overload
def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R],
    /,
) -> ToolMethod[P, R]: ...


def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] = False,
    truncate: int | None = None,
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], ToolMethod[P, R]] | ToolMethod[P, R]:
    """
    Decorator for creating a Tool from a class method.

    The tool produced from this method will be incomplete until it is called
    from an instantiated class. If you don't require any active state from
    a `self` argument, consider wrapping a method first with `@staticmethod`
    and then using `@tool` to create a tool from it.

    See `@tool` for more details.

    Example:
        ```
        class Thing:
            delta: int = 5

            # Use @tool_method to create a tool from a class method with a `self` argument.
            @tool_method(name="add_numbers_with_delta", description="This is my tool")
            def delta_add(self, x: int, y: int) -> int:
                return x + y + self.delta

            # Use @staticmethod + @tool to avoid the 'self' argument altogether
            @tool(name="add_numbers", description="This is my tool")
            @staticmethod
            def static_add(x: int, y: int) -> int:
                return x + y
        ```
    """

    def make_tool(func: t.Callable[..., t.Any]) -> ToolMethod[P, R]:
        # TODO: Improve consistency of detection here before enabling this warning
        # if not _is_unbound_method(func):
        #     warnings.warn(
        #         "Passing a regular function to @tool_method improperly handles the 'self' argument, use @tool instead.",
        #         SyntaxWarning,
        #         stacklevel=3,
        #     )

        # Strip the `self` argument from the function signature so
        # our schema generation doesn't include it under the hood.

        @functools.wraps(func)
        def wrapper(self: t.Any, *args: P.args, **kwargs: P.kwargs) -> R:
            return func(self, *args, **kwargs)  # type: ignore [no-any-return]

        wrapper.__signature__ = inspect.signature(func).replace(  # type: ignore [attr-defined]
            parameters=tuple(
                param
                for param in inspect.signature(func).parameters.values()
                if param.name != "self"
            ),
        )

        return ToolMethod.from_callable(
            wrapper,  # type: ignore [arg-type]
            name=name,
            description=description,
            catch=catch,
            truncate=truncate,
        )

    if func is not None:
        return make_tool(func)

    return make_tool
