import abc
import inspect
import typing as t

from pydantic import Field, computed_field, model_validator
from pydantic_xml import attr, element, wrapped

from rigging.model import Model

SUPPORTED_TOOL_ARGUMENT_TYPES = int | float | str | bool
ToolArgumentTypesCast = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
}


#
# 1 - Inbound models from LLM -> functions
#


class ToolCallParameter(Model):
    name: str = attr()
    attr_value: SUPPORTED_TOOL_ARGUMENT_TYPES | None = attr("value", default=None, exclude=True)
    text_value: SUPPORTED_TOOL_ARGUMENT_TYPES | None = Field(default=None, exclude=True)

    @computed_field
    def value(self) -> SUPPORTED_TOOL_ARGUMENT_TYPES:
        return self.attr_value or self.text_value or ""

    @model_validator(mode="after")
    def validate_value(self) -> "ToolCallParameter":
        if self.value is None:
            raise ValueError("Missing parameter value")
        return self


class ToolCall(Model, tag="call"):
    tool: str = attr()
    function: str = attr()
    parameters: list[ToolCallParameter] = element(tag="parameter")


#
# 2 - Inbound function calls from a model
#


class ToolCalls(Model, tag="tool_calls"):
    calls: list[ToolCall] = element()

    # This can be used in prompts to teach the model
    # the particular XML structure we're looking for
    #
    # TODO: We should consider building a base model
    # interface for both simple tags (<thing></thing>)
    # and full examples will filled in template vars
    @classmethod
    def xml_example(cls) -> str:
        return cls(
            calls=[
                ToolCall(
                    tool="$TOOL_A",
                    function="$FUNCTION_A",
                    parameters=[ToolCallParameter(name="$PARAMETER_NAME", value="$PARAMETER_VALUE")],
                ),
                ToolCall(
                    tool="$TOOL_B",
                    function="$FUNCTION_B",
                    parameters=[ToolCallParameter(name="$PARAMETER_NAME", value="$PARAMETER_VALUE")],
                ),
            ]
        ).to_pretty_xml()


#
# 3 - Outbound models from functions -> LLM
#


# Description of a single tool parameter
class ToolParameter(Model, tag="parameter"):
    name: str = attr()
    type: str = attr()
    description: str = attr()


# Description of a single tool function
class ToolFunction(Model, tag="function"):
    name: str = attr()
    description: str = attr()
    parameters: list[ToolParameter] = wrapped("parameters", element())


# Description of an entire tool
class ToolDescription(Model, tag="tool"):
    name: str = attr()
    description: str = attr()
    functions: list[ToolFunction] = wrapped("functions", element())


# A list of tools to present to the model
class ToolDescriptionList(Model, tag="tools"):
    tools: list[ToolDescription] = element()


# A single result from a tool call
class ToolResult(Model, tag="result"):
    tool: str = attr()
    function: str = attr()
    error: bool = attr()
    content: str


# How we pass back results from a set of calls
class ToolResults(Model, tag="tool_results"):
    results: list[ToolResult] = element()


#
# 4 - Base class for implementing tools
#


class Tool(abc.ABC):
    # TODO: I don't love having these defined as property getters,
    # I would prefer to have them as class attributes, but I'm not
    # sure how we can hint/enforce that to derived classes
    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        ...

    # TODO: We could alternatively use the get_description()
    # object and check against that (or even cast into it first)
    #
    # NOTE: We assume some sanity checks have already been performed
    def _execute(self, call: ToolCall) -> str:
        tool_description = self.get_description()

        if call.function not in [f.name for f in tool_description.functions]:
            raise ValueError(f"Function '{call.function}' does not exist on '{self.name}'")

        function_description = next(f for f in tool_description.functions if f.name == call.function)

        # TODO: The casting here is terrible, we should probably
        # be exposing the raw types on the description, but I
        # need to make sure they aren't serialized into the model
        arguments: dict[str, SUPPORTED_TOOL_ARGUMENT_TYPES] = {}
        for parameter in call.parameters:
            if parameter.name not in [p.name for p in function_description.parameters]:
                raise ValueError(f"Parameter '{parameter.name}' does not exist on '{self.name}.{call.function}'")

            parameter_description = next(p for p in function_description.parameters if p.name == parameter.name)

            arguments[parameter.name] = ToolArgumentTypesCast[parameter_description.type](parameter.value)

        function = getattr(self, call.function)
        result = function(**arguments)

        # Cast back to string for simplicity despite us likely
        # having more complex types underneath (we want them castable to str)
        return str(result)

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            content = self._execute(call)
            return ToolResult(tool=call.tool, function=call.function, error=False, content=content)
        except Exception as e:
            return ToolResult(tool=call.tool, function=call.function, error=True, content=str(e))

    __call__ = execute

    # Lots of sanity checks and validation, but we essentially
    # want to use the class def, functions, params, etc. and
    # build a ToolDescription object that can be serialized
    # and passed to a model
    def get_description(self) -> ToolDescription:
        functions: list[ToolFunction] = []
        for method_name, method in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if not method.__qualname__.startswith(self.__class__.__name__):
                continue

            if method_name.startswith("_"):
                continue

            signature = inspect.signature(method)

            if signature.return_annotation is inspect.Signature.empty:
                raise TypeError(f"Functions must have return type hints ({method_name})")

            if signature.return_annotation != str:
                raise TypeError(f"Functions must return strings ({method_name})")

            parameters: list[ToolParameter] = []
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                formatted_name = f"{method.__name__}#{param_name}"

                if param.kind not in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    raise TypeError(f"Parameters must be positional or keyword ({formatted_name})")

                if param.annotation is inspect.Parameter.empty:
                    raise TypeError(f"Parameters must have type hints ({formatted_name})")

                if t.get_origin(param.annotation) != t.Annotated:
                    raise TypeError(
                        f'Parameters must be annotated like Annotated[<type>, "<description>"] ({formatted_name})'
                    )

                annotation_args = t.get_args(param.annotation)

                if len(annotation_args) != 2 or not isinstance(annotation_args[1], str):
                    raise TypeError(
                        f'Parameters must be annotated like Annotated[<type>, "<description>"] ({formatted_name})'
                    )

                if annotation_args[0] not in SUPPORTED_TOOL_ARGUMENT_TYPES.__args__:  # type: ignore
                    raise TypeError(f"Parameters must be annotated with supported types ({formatted_name})")

                type_name = annotation_args[0].__name__
                description = annotation_args[1]

                parameters.append(ToolParameter(name=param_name, type=type_name, description=description))

            functions.append(
                ToolFunction(
                    name=method_name,
                    description=method.__doc__ if method.__doc__ else "",
                    parameters=parameters,
                )
            )

        return ToolDescription(name=self.name, description=self.description, functions=functions)
