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

P = t.ParamSpec("P")
R = t.TypeVar("R")

ToolMode = t.Literal["auto", "api", "xml", "json-in-xml"]
"""
How tool calls are handled.

- `auto`: The method is chosed based on support (api > xml).
- `api`: Tool calls are delegated to api-provided function calling.
- `xml`: Tool calls are parsed in nested XML format.
- `json-in-xml`: Tool calls are parsed as raw JSON inside XML tags.
"""


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
    - `list[type[Exception]]`: Catch only the specified exceptions.
    """

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

        description = inspect.cleandoc(description or fn_for_signature.__doc__ or "")
        description = re.sub(r"(?![\r\n])(\b\s+)", " ", description)

        self = cls(
            name=name or fn_for_signature.__name__,
            description=description,
            parameters_schema=schema,
            fn=fn,
            catch=catch if isinstance(catch, bool) else set(catch),
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
        return XmlToolDefinition.from_parameter_model(
            self.model,
            self.name,
            self.description,
        )

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
            The message to send back to the generator or `None` if iterative tool calling should not proceed any further.
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

            kwargs: dict[str, t.Any]
            if isinstance(tool_call, ApiToolCall | JsonInXmlToolCall):
                kwargs = json.loads(tool_call_parameters)

                if self._type_adapter is not None:
                    kwargs = self._type_adapter.validate_python(kwargs)

            elif isinstance(tool_call, XmlToolCall):
                try:
                    parsed = self.model.from_text(
                        self.model.xml_start_tag()
                        + tool_call_parameters
                        + self.model.xml_end_tag(),
                    )
                except Exception as e:  # noqa: BLE001
                    raise ValueError(
                        f"Failed to parse parameters from:\n{tool_call_parameters}",
                    ) from e

                if not parsed:
                    raise ValueError(
                        f"Failed to parse parameters from:\n{tool_call_parameters}",
                    )

                parameters = parsed[0][0]

                # As opposed to a model_dump, we want to retain the
                # argument object instances. We'll just flatten the
                # model into a dictionary for the function call.

                kwargs = {
                    field_name: getattr(parameters, field_name, None)
                    for field_name in self.model.model_fields
                }

            span.set_attribute("arguments", kwargs)

            # Call the function

            try:
                result: t.Any = self.fn(**kwargs)  # type: ignore [call-arg]
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:  # noqa: BLE001
                if self.catch is True or (
                    not isinstance(self.catch, bool) and isinstance(e, tuple(self.catch))
                ):
                    result = f'<error type="{e.__class__.__name__}">{e}</error>'
                else:
                    raise

            span.set_attribute("result", result)

        message = (
            Message(role="tool", tool_call_id=tool_call.id)
            if isinstance(tool_call, ApiToolCall)
            else Message("user")
        )

        # If the tool returns nothing back to us, we'll assume that
        # they do not want to proceed with additional tool calling

        should_continue = result is not None

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

        return message, should_continue

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
) -> t.Callable[[t.Callable[P, R]], Tool[P, R]]:
    ...


@t.overload
def tool(
    func: t.Callable[P, R],
    /,
) -> Tool[P, R]:
    ...


def tool(
    func: t.Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] = False,
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
        if _is_unbound_method(func):
            warnings.warn(
                "Passing a class method to @tool improperly handles the 'self' argument, use @tool_method instead.",
                SyntaxWarning,
                stacklevel=3,
            )

        return Tool.from_callable(func, name=name, description=description, catch=catch)

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
) -> t.Callable[[t.Callable[t.Concatenate[t.Any, P], R]], ToolMethod[P, R]]:
    ...


@t.overload
def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R],
    /,
) -> ToolMethod[P, R]:
    ...


def tool_method(
    func: t.Callable[t.Concatenate[t.Any, P], R] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] = False,
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

            @tool_method(name="add_numbers_with_delta", description="This is my tool")
            def delta_add(self, x: int, y: int) -> int:
                return x + y + self.delta

            @tool(name="add_numbers", description="This is my tool")
            @staticmethod
            def static_add(x: int, y: int) -> int:
                return x + y
        ```
    """

    def make_tool(func: t.Callable[..., t.Any]) -> ToolMethod[P, R]:
        if not _is_unbound_method(func):
            warnings.warn(
                "Passing a regular function to @tool_method improperly handles the 'self' argument, use @tool instead.",
                SyntaxWarning,
                stacklevel=3,
            )

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
        )

    if func is not None:
        return make_tool(func)

    return make_tool
