"""
Core types and functions for defining tools and handling tool calls.
"""

import contextlib
import functools
import inspect
import json
import re
import typing as t

import typing_extensions as te
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic_xml import attr

from rigging.error import Stop, ToolDefinitionError
from rigging.model import (
    ErrorModel,
    Model,
    make_from_schema,
    make_from_signature,
)
from rigging.util import deref_json, shorten_string

if t.TYPE_CHECKING:
    from rigging.message import Message

TOOL_STOP_TAG = "rg-stop"

P = t.ParamSpec("P")
R = t.TypeVar("R")

ToolMode = t.Literal["auto", "api", "xml", "json", "json-in-xml", "json-with-tag", "pythonic"]
"""
How tool calls are handled.

- `auto`: The method is chosen based on support (api w/ fallback to json-in-xml).
- `api`: Tool calls are delegated to api-provided function calling.
- `xml`: Tool calls are parsed in a nested XML format which is native to Rigging.
- `json`: Tool calls are parsed as raw name/arg JSON anywhere in assistant message content.
- `json-in-xml`: Tool calls are parsed using JSON for arguments, and XML for everything else.
- `json-with-tag`: Tool calls are parsed as name/arg JSON structures inside an XML tag to identify it.
- `pythonic`: Tool calls are parsed as pythonic function call syntax.
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
    model_config = ConfigDict(json_schema_extra={"rigging.type": "tool_call"})

    id: str
    type: t.Literal["function"] = "function"
    function: FunctionCall

    def __str__(self) -> str:
        arguments = shorten_string(self.function.arguments, max_length=50)
        return f"ToolCall({self.function.name}({arguments}), id='{self.id}')"

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


DEFAULT_CATCH_EXCEPTIONS: set[type[Exception]] = {json.JSONDecodeError, ValidationError}


class Tool(BaseModel, t.Generic[P, R]):
    """Base class for representing a tool to a generator."""

    name: str
    """The name of the tool."""
    description: str
    """A description of the tool."""
    parameters_schema: dict[str, t.Any]
    """The JSON schema for the tool's parameters."""
    fn: t.Callable[P, R] = Field(  # type: ignore [assignment]
        default_factory=lambda: lambda *args, **kwargs: None,  # noqa: ARG005
        exclude=True,
    )
    """The function to call."""
    catch: bool | set[type[Exception]] = set(DEFAULT_CATCH_EXCEPTIONS)
    """
    Whether to catch exceptions and return them as messages.

    - `False`: Do not catch exceptions.
    - `True`: Catch all exceptions.
    - `set[type[Exception]]`: Catch only the specified exceptions.

    By default, catches `json.JSONDecodeError` and `ValidationError`.
    """
    truncate: int | None = None
    """If set, the maximum number of characters to truncate any tool output to."""

    _signature: inspect.Signature | None = PrivateAttr(default=None, init=False)
    _type_adapter: TypeAdapter[t.Any] | None = PrivateAttr(default=None, init=False)
    _model: type[Model] | None = PrivateAttr(default=None, init=False)

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
        catch: bool | t.Iterable[type[Exception]] | None = None,
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
            catch=catch or DEFAULT_CATCH_EXCEPTIONS,
            truncate=truncate,
        )

        self._signature = signature
        self.__signature__ = signature  # type: ignore [misc]
        self.__name__ = self.name  # type: ignore [attr-defined]
        self.__doc__ = self.description

        # For handling API calls, we'll use the type adapter to validate
        # the arguments before calling the function

        self._type_adapter = type_adapter

        return self

    @property
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

    @property
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
        import dreadnode as dn

        from rigging.message import ContentText, ContentTypes, Message

        with dn.task_span(
            self.name,
            tags=["rigging/tool"],
            attributes={"tool_name": self.name, "rigging.type": "tool"},
        ) as task:
            dn.log_input("tool_call", tool_call)

            if hasattr(tool_call, "id") and isinstance(tool_call.id, str):
                task.set_attribute("tool_call_id", tool_call.id)

            result: t.Any
            stop = False

            try:
                # Load + Validate args

                kwargs = json.loads(tool_call.function.arguments)
                if self._type_adapter is not None:
                    kwargs = self._type_adapter.validate_python(kwargs)
                kwargs = kwargs or {}

                dn.log_inputs(**kwargs)

                # Call the function

                result = self.fn(**kwargs)  # type: ignore [call-arg]
                if inspect.isawaitable(result):
                    result = await result

                if isinstance(result, Stop):
                    raise result  # noqa: TRY301
            except Stop as e:
                result = f"<{TOOL_STOP_TAG}>{e.message}</{TOOL_STOP_TAG}>"
                task.set_attribute("stop", True)
                stop = True
            except Exception as e:
                if self.catch is True or (
                    not isinstance(self.catch, bool) and isinstance(e, tuple(self.catch))
                ):
                    task.set_exception(e)
                    result = ErrorModel.from_exception(e)
                else:
                    raise

            dn.log_output("output", result)

        message = Message(role="tool", tool_call_id=tool_call.id)

        # If this is being gracefully handled as an ErrorModel,
        # wwe will construct it explicitly so it can attach
        # metadata about the failure.

        if isinstance(result, ErrorModel):
            message = Message.from_model(
                result,
                role="tool",
                tool_call_id=tool_call.id,
            )

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
        elif isinstance(result, Model):
            message.content_parts = [ContentText(text=result.to_pretty_xml())]
        else:
            with contextlib.suppress(Exception):
                if type(result) not in [str, int, float, bool]:
                    result = TypeAdapter(t.Any).dump_json(result).decode(errors="replace")
            message.content_parts = [ContentText(text=str(result))]

        if self.truncate:
            # Use shorten instead of truncate to try and preserve
            # the most context possible.
            message = message.shorten(self.truncate)

        return message, stop

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.fn(*args, **kwargs)

    def clone(self) -> "Tool[P, R]":
        """
        Create a clone of this tool with the same parameters.
        Useful for creating tools with the same signature but different names.
        """
        new = Tool[P, R](
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
            fn=self.fn,
            catch=self.catch,
            truncate=self.truncate,
        )

        new._signature = self._signature
        new.__signature__ = self.__signature__  # type: ignore [misc]
        new._type_adapter = self._type_adapter
        new.__name__ = self.name  # type: ignore [attr-defined]
        new.__doc__ = self.description

        return new

    def with_(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        catch: bool | t.Iterable[type[Exception]] | None = None,
        truncate: int | None = None,
    ) -> "Tool[P, R]":
        """
        Create a new tool with updated parameters.
        Useful for creating tools with the same signature but different names or descriptions.

        Args:
            name: The name of the tool.
            description: The description of the tool.
            catch: Whether to catch exceptions and return them as messages.
                - `False`: Do not catch exceptions.
                - `True`: Catch all exceptions.
                - `list[type[Exception]]`: Catch only the specified exceptions.
                - `None`: By default, catches `json.JSONDecodeError` and `ValidationError
            truncate: If set, the maximum number of characters to truncate any tool output to.

        Returns:
            A new tool with the updated parameters.
        """
        new = self.clone()
        new.name = name or self.name
        new.description = description or self.description
        new.catch = (
            catch if isinstance(catch, bool) else self.catch if catch is None else set(catch)
        )
        new.truncate = truncate if truncate is not None else self.truncate
        return new


@t.overload
def tool(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
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
    catch: bool | t.Iterable[type[Exception]] | None = None,
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
            - `None`: By default, catches `json.JSONDecodeError` and `ValidationError`.
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


class ToolMethod(property, t.Generic[P, R]):
    """
    A descriptor that acts as a factory for creating bound Tool instances.

    It inherits from `property` to be ignored by pydantic's `ModelMetaclass`
    during field inspection. This prevents validation errors which would
    otherwise treat the descriptor as a field and stop tool_method decorators
    from being applied in BaseModel classes.
    """

    def __init__(
        self,
        fget: t.Callable[..., t.Any],
        name: str,
        description: str,
        parameters_schema: dict[str, t.Any],
        catch: bool | t.Iterable[type[Exception]] | None,
        truncate: int | None,
        signature: inspect.Signature,
        type_adapter: TypeAdapter[t.Any],
    ):
        super().__init__(fget)
        self.tool_name = name
        self.tool_description = description
        self.tool_parameters_schema = parameters_schema
        self.tool_catch = catch
        self.tool_truncate = truncate
        self._tool_signature = signature
        self._tool_type_adapter = type_adapter

    @te.overload  # type: ignore [override, unused-ignore]
    def __get__(self, instance: None, owner: type[object] | None = None) -> Tool[P, R]: ...

    @te.overload
    def __get__(self, instance: object, owner: type[object] | None = None) -> Tool[P, R]: ...

    @te.override
    def __get__(self, instance: object | None, owner: type[object] | None = None) -> Tool[P, R]:
        if self.fget is None:
            raise AttributeError(
                f"Tool '{self.tool_name}' is not defined on instance of {owner.__name__ if owner else 'unknown'}.",
            )

        # Class access: return an unbound Tool for inspection.
        if instance is None:
            tool = Tool[P, R](
                fn=self.fget,
                name=self.tool_name,
                description=self.tool_description,
                parameters_schema=self.tool_parameters_schema,
                catch=self.tool_catch or DEFAULT_CATCH_EXCEPTIONS,
                truncate=self.tool_truncate,
            )
            tool._signature = self._tool_signature  # noqa: SLF001
            tool._type_adapter = self._tool_type_adapter  # noqa: SLF001
            return tool

        # Instance access: return a new Tool bound to the instance.
        bound_method = self.fget.__get__(instance, owner)
        bound_tool = Tool[P, R](
            fn=bound_method,
            name=self.tool_name,
            description=self.tool_description,
            parameters_schema=self.tool_parameters_schema,
            catch=self.tool_catch or DEFAULT_CATCH_EXCEPTIONS,
            truncate=self.tool_truncate,
        )
        bound_tool._signature = self._tool_signature  # noqa: SLF001
        bound_tool._type_adapter = self._tool_type_adapter  # noqa: SLF001
        return bound_tool


@t.overload
def tool_method(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    catch: bool | t.Iterable[type[Exception]] | None = None,
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
    catch: bool | t.Iterable[type[Exception]] | None = None,
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

    def make_tool(f: t.Callable[t.Concatenate[t.Any, P], R]) -> ToolMethod[P, R]:
        # This logic is specialized from `Tool.from_callable` to correctly
        # handle the `self` parameter in method signatures.

        signature = inspect.signature(f)
        params_without_self = [p for p_name, p in signature.parameters.items() if p_name != "self"]
        schema_signature = signature.replace(parameters=params_without_self)

        @functools.wraps(f)
        def empty_func(*_: t.Any, **kwargs: t.Any) -> t.Any:
            return kwargs

        empty_func.__signature__ = schema_signature  # type: ignore [attr-defined]
        type_adapter: TypeAdapter[t.Any] = TypeAdapter(empty_func)
        schema = deref_json(type_adapter.json_schema(), is_json_schema=True)

        tool_name = name or f.__name__
        tool_description = inspect.cleandoc(description or f.__doc__ or "")

        return ToolMethod(
            fget=f,
            name=tool_name,
            description=tool_description,
            parameters_schema=schema,
            catch=catch,
            truncate=truncate,
            signature=schema_signature,
            type_adapter=type_adapter,
        )

    if func is not None:
        return make_tool(func)
    return make_tool
