"""
Treat empty function signatures as prompts for structured chat interfaces.
"""

import asyncio
import dataclasses
import inspect
import re
import typing as t
from collections import OrderedDict

from jinja2 import Environment, StrictUndefined, meta
from pydantic import ValidationError
from typing_extensions import Concatenate, ParamSpec  # noqa: UP035

from rigging.chat import (
    Chat,
    ChatPipeline,
    MapChatCallback,
    PipelineStepContextManager,
    ThenChatCallback,
    WatchChatCallback,
)
from rigging.generator.base import GenerateParams, Generator, get_generator
from rigging.message import Message
from rigging.model import Model, SystemErrorModel, ValidationErrorModel, make_primitive
from rigging.tools import Tool
from rigging.tracing import tracer
from rigging.util import escape_xml, get_qualified_name, to_snake, to_xml_tag

DEFAULT_DOC = "Convert the following inputs to outputs ({func_name})."
"""Default docstring if none is provided to a prompt function."""

DEFAULT_MAX_PARSE_ROUNDS = 3
"""Default maximum number of recursive output parsing attempts to allow."""
DEFAULT_MAX_TOOL_ROUNDS = 20
"""Default maximum number of recursive tool calls to allow."""

P = ParamSpec("P")
R = t.TypeVar("R")

# Annotation


@dataclasses.dataclass
class Ctx:
    """
    Used in type annotations to provide additional context for the prompt construction.

    You can use this annotation on inputs and outputs to prompt functions.

    Example:
        ```
        tag_override = Annotated[str, Ctx(tag="custom_tag", ...)]
        ```
    """

    tag: str | None = None
    prefix: str | None = None
    example: str | Model | None = None


# Utilities


def unwrap_annotated(annotation: t.Any) -> tuple[t.Any, Ctx | None]:
    if t.get_origin(annotation) is t.Annotated:
        base_type, *meta = t.get_args(annotation)
        for m in meta:
            if isinstance(m, Ctx):
                return base_type, m
        return base_type, None
    return annotation, None


def get_undeclared_variables(template: str) -> set[str]:
    env = Environment()  # noqa: S701 # nosec
    parsed_template = env.parse(template)
    return meta.find_undeclared_variables(parsed_template)


def make_parameter(
    annotation: t.Any,
    *,
    name: str = "nested",
    kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD,
) -> inspect.Parameter:
    return inspect.Parameter(name=name, kind=kind, annotation=annotation)


# Function Inputs


@dataclasses.dataclass
class Input:
    name: str
    context: Ctx

    @property
    def tag(self) -> str:
        return self.context.tag or to_xml_tag(self.name)

    def _prefix(self, xml: str) -> str:
        if self.context.prefix:
            return f"{self.context.prefix}\n{xml}"
        return xml

    def to_str(self, value: t.Any) -> str:
        raise NotImplementedError

    def to_xml(self, value: t.Any) -> str:
        value_str = self.to_str(value)
        if "\n" in value_str:
            value_str = f"\n{value_str}\n"
        return self._prefix(f"<{self.tag}>{escape_xml(value_str)}</{self.tag}>")


@dataclasses.dataclass
class BasicInput(Input):
    def to_str(self, value: t.Any) -> str:
        if not isinstance(value, int | float | str | bool):
            raise TypeError(f"Value must be a basic type, got: {type(value)}")
        return str(value)


@dataclasses.dataclass
class ModelInput(Input):
    def to_str(self, value: t.Any) -> str:
        if not isinstance(value, Model):
            raise TypeError(f"Value must be a Model instance, got: {type(value)}")
        return value.to_pretty_xml()

    def to_xml(self, value: t.Any) -> str:
        return self._prefix(self.to_str(value))


@dataclasses.dataclass
class ListInput(Input):
    interior: Input

    def to_str(self, value: list[t.Any]) -> str:
        return "\n\n".join(self.interior.to_str(v) for v in value)


@dataclasses.dataclass
class DictInput(Input):
    interior: Input

    def to_str(self, value: t.Any) -> str:
        if not isinstance(value, dict):
            raise TypeError(f"Value must be a dictionary, got: {type(value)}")
        if not all(isinstance(k, str) for k in value):
            raise ValueError("Dictionary keys must be strings")
        return "\n".join(f"<{k}>{self.interior.to_str(v)}</{k}>" for k, v in value.items())


def parse_parameter(param: inspect.Parameter, error_name: str) -> Input:
    if param.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise TypeError(f"Parameters must be keyword compatible {error_name}")

    if param.annotation in [None, inspect.Parameter.empty]:
        raise TypeError(f"All parameters require type annotations {error_name}")

    annotation, context = unwrap_annotated(param.annotation)

    if annotation in [int, float, str, bool]:
        return BasicInput(param.name, context or Ctx())

    if t.get_origin(annotation) is list:
        if param.name == "nested":
            raise TypeError(f"Nested list parameters are not supported: {error_name}")

        args = t.get_args(annotation)
        if not args:
            raise TypeError(f"List param must be fully typed: {error_name}")

        arg_type, arg_context = unwrap_annotated(args[0])
        return ListInput(
            param.name,
            arg_context or context or Ctx(),
            parse_parameter(make_parameter(arg_type), error_name),
        )

    if t.get_origin(annotation) is dict:
        if param.name == "nested":
            raise TypeError(f"Nested dict parameters are not supported: {error_name}")

        args = t.get_args(annotation)
        if not args or len(args) != 2:  # noqa: PLR2004
            raise TypeError(f"Dict param must be fully typed: {error_name}")
        if args[0] is not str:
            raise TypeError(f"Dict param keys must be strings: {error_name}")

        return DictInput(
            param.name,
            context or Ctx(),
            parse_parameter(make_parameter(args[1]), error_name),
        )

    if inspect.isclass(annotation) and issubclass(annotation, Model):
        return ModelInput(param.name, context or Ctx())

    raise TypeError(f"Unsupported parameter type: {error_name}")


# Function Outputs


@dataclasses.dataclass
class Output:
    id: str
    context: Ctx

    @property
    def tag(self) -> str:
        return self.context.tag or to_xml_tag(self.id)

    def _prefix(self, xml: str) -> str:
        if self.context.prefix:
            return f"{self.context.prefix}\n{xml}"
        return xml

    def guidance(self) -> str:
        return "Produce the following output (use xml tags):"

    def to_format(self) -> str:
        tag = self.context.tag or self.tag
        if isinstance(self.context.example, Model):
            raise NotImplementedError("Model examples are not supported for this output")
        return self._prefix(f"<{tag}>{escape_xml(self.context.example or '')}</{tag}>")

    def from_chat(self, chat: Chat) -> t.Any:
        raise NotImplementedError


@dataclasses.dataclass
class ChatOutput(Output):
    def from_chat(self, chat: Chat) -> t.Any:
        return chat


@dataclasses.dataclass
class BasicOutput(Output):
    type_: type[t.Any]  # TODO: We should be able to scope this down

    def from_chat(self, chat: Chat) -> t.Any:
        model_cls = make_primitive("Model", self.type_, tag=self.context.tag or self.tag)
        return chat.last.parse(model_cls).content


@dataclasses.dataclass
class BasicListOutput(BasicOutput):
    def guidance(self) -> str:
        return "Produce the following output for each item (use xml tags):"

    def from_chat(self, chat: Chat) -> t.Any:
        model_cls = make_primitive("Model", self.type_, tag=self.context.tag or self.tag)
        return [m.content for m in chat.last.parse_set(model_cls)]


@dataclasses.dataclass
class ModelOutput(Output):
    type_: type[Model]

    def to_format(self) -> str:
        if isinstance(self.context.example, Model):
            return self.context.example.to_pretty_xml()
        return self.type_.xml_example()

    def from_chat(self, chat: Chat) -> t.Any:
        return chat.last.parse(self.type_)


@dataclasses.dataclass
class ModelListOutput(ModelOutput):
    def guidance(self) -> str:
        return "Produce the following output for each item (use xml tags):"

    def from_chat(self, chat: Chat) -> t.Any:
        return chat.last.parse_set(self.type_)


@dataclasses.dataclass
class TupleOutput(Output):
    interiors: list[Output]

    @property
    def real_interiors(self) -> list[Output]:
        return [i for i in self.interiors if not isinstance(i, ChatOutput)]

    @property
    def wrapped(self) -> bool:
        # Handles cases where we are using a tuple just to
        # capture a Chat along with a real output, in this
        # case we should fall through for most of the work
        #
        # () -> tuple[Chat, ...]
        return len(self.real_interiors) == 1

    def guidance(self) -> str:
        if self.wrapped:
            return self.real_interiors[0].guidance()
        return "Produce the following outputs (use xml tags):"

    def to_format(self) -> str:
        if self.wrapped:
            return self.real_interiors[0].to_format()
        return self._prefix("\n\n".join(i.to_format() for i in self.real_interiors))

    def from_chat(self, chat: Chat) -> t.Any:
        return tuple(i.from_chat(chat) for i in self.interiors)


@dataclasses.dataclass
class DataclassOutput(TupleOutput):
    type_: type[t.Any]

    def from_chat(self, chat: Chat) -> t.Any:
        return self.type_(*super().from_chat(chat))


def parse_output(annotation: t.Any, error_name: str, *, allow_nested: bool = True) -> Output:  # noqa: PLR0912
    from rigging.chat import Chat

    if annotation in [None, inspect.Parameter.empty]:
        raise TypeError(f"Return type annotation is required ({error_name})")

    # Unwrap any annotated types
    annotation, context = unwrap_annotated(annotation)

    if annotation == Chat:
        # Use a special subclass here -> args don't matter
        return ChatOutput(id="chat", context=context or Ctx())

    if annotation in [int, float, str, bool]:
        return BasicOutput(id=annotation.__name__, context=context or Ctx(), type_=annotation)

    if t.get_origin(annotation) is list:
        if not allow_nested:
            raise TypeError(f"Nested list outputs are not supported ({error_name})")

        args = t.get_args(annotation)
        if not args:
            raise TypeError(f"List return type must be fully specified ({error_name})")

        arg_type, arg_context = unwrap_annotated(args[0])

        if arg_type in [int, float, str, bool]:
            return BasicListOutput(
                id=arg_type.__name__,
                context=arg_context or context or Ctx(),
                type_=arg_type,
            )

        if inspect.isclass(arg_type) and issubclass(arg_type, Model):
            return ModelListOutput(
                id=arg_type.__name__,
                context=arg_context or context or Ctx(),
                type_=arg_type,
            )

    if t.get_origin(annotation) is tuple:
        if not allow_nested:
            raise TypeError(f"Nested tuple outputs are not supported ({error_name})")

        args = t.get_args(annotation)
        if not args:
            raise TypeError(f"Tuple return type must be fully specified ({error_name})")

        tuple_interiors = [parse_output(arg, error_name, allow_nested=False) for arg in args]

        if len({i.tag for i in tuple_interiors}) != len(tuple_interiors):
            raise TypeError(
                f"Tuple return annotations must have unique internal types\n"
                "or use Annotated[..., Ctx(tag=...)] overrides to\n"
                f"make them differentiable ({error_name})",
            )

        return TupleOutput(id="tuple", context=context or Ctx(), interiors=tuple_interiors)

    if dataclasses.is_dataclass(annotation) and type(annotation) is type:
        interior_annotations: list[t.Any] = []
        for field in dataclasses.fields(annotation):
            field_annotation, field_context = unwrap_annotated(field.type)

            ctx_dict = dataclasses.asdict(field_context) if field_context else {}
            if ctx_dict.get("tag") is None:
                ctx_dict["tag"] = to_xml_tag(field.name)

            interior_annotations.append(t.Annotated[field_annotation, Ctx(**ctx_dict)])

        dataclass_interiors: list[Output] = []
        for field, field_annotation in zip(
            dataclasses.fields(annotation),
            interior_annotations,
            strict=True,
        ):
            interior = parse_output(
                field_annotation,
                f"{error_name}#{field.name}",
                allow_nested=False,
            )
            if interior is None:
                raise TypeError(f"Dataclass field type is invalid ({error_name}#{field.name}")
            dataclass_interiors.append(interior)

        if len({i.tag for i in dataclass_interiors}) != len(dataclass_interiors):
            raise TypeError(
                f"Dataclass return annotations must have unique internal types\n"
                "or use Annotated[..., Ctx(tag=...)] overrides to\n"
                f"make them differentiable ({error_name})",
            )

        return DataclassOutput(
            id=annotation.__name__,
            type_=annotation,
            context=context or Ctx(),
            interiors=dataclass_interiors,
        )

    # This has to come after our list/tuple checks as they pass isclass
    if inspect.isclass(annotation) and issubclass(annotation, Model):
        return ModelOutput(id=annotation.__name__, context=context or Ctx(), type_=annotation)

    raise TypeError(f"Unsupported return type: {error_name}")


# Prompt


@dataclasses.dataclass
class Prompt(t.Generic[P, R]):
    """
    Prompts wrap hollow functions and create structured chat interfaces for
    passing inputs into a ChatPipeline and parsing outputs.
    """

    func: t.Callable[P, t.Coroutine[t.Any, t.Any, R]] | None = None
    """The function that the prompt was derived from."""

    max_parsing_rounds: int = DEFAULT_MAX_PARSE_ROUNDS
    """The maximum number of recursive output parsing attempts to allow."""
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    """The maximum number of recursive tool calls to allow."""

    inputs: list[Input] = dataclasses.field(default_factory=list)
    """The structured input handlers for the prompt."""
    output: Output = dataclasses.field(default_factory=lambda: ChatOutput(id="chat", context=Ctx()))
    """The structured output handler for the prompt."""

    watch_callbacks: "list[WatchChatCallback]" = dataclasses.field(default_factory=list)
    """Callbacks to be passed any chats produced while executing this prompt."""
    then_callbacks: "list[ThenChatCallback]" = dataclasses.field(default_factory=list)
    """Callbacks to be executed for every generated chat (see ChatPipeline.then)."""
    map_callbacks: "list[MapChatCallback]" = dataclasses.field(default_factory=list)
    """Callbacks to be executed for every generated chat (see ChatPipeline.map)."""

    params: GenerateParams | None = None
    """The parameters to be used when generating chats for this prompt."""
    tools: list[Tool[..., t.Any]] = dataclasses.field(default_factory=list)
    """The API tools to be made available when generating chats for this prompt."""
    system_prompt: str | None = None
    """A system prompt fragment to be injected into the messages before generation."""

    _generator_id: str | None = None
    _generator: Generator | None = None
    _pipeline: ChatPipeline | None = None

    _docstring: str | None = None

    def __post_init__(self) -> None:
        if self.func is None:
            return

        # We include eval_str here to help with __future__ annotations
        # usage, but it may cause issues and I haven't had time to fully
        # dig into implications.

        signature = inspect.signature(self.func, eval_str=True)
        undeclared = get_undeclared_variables(self.docstring)

        for param in signature.parameters.values():
            if param.name in undeclared:
                continue
            error_name = f"{self.func.__name__}({param})"
            self.inputs.append(parse_parameter(param, error_name))

        if len({i.tag for i in self.inputs}) != len(self.inputs):
            raise TypeError("All input parameters must have unique names/tags")

        error_name = f"{self.func.__name__}() -> {signature.return_annotation}"
        self.output = parse_output(signature.return_annotation, error_name)

        self.__signature__ = signature
        self.__name__ = self.func.__name__
        self.__doc__ = self.docstring

    @property
    def docstring(self) -> str:
        """The docstring for the prompt function."""
        if self._docstring is None:
            # Guidance is taken from https://github.com/outlines-dev/outlines/blob/main/outlines/prompts.py
            docstring = self.func.__doc__ or DEFAULT_DOC.format(
                func_name=self.func.__name__ if self.func else "function",
            )
            docstring = inspect.cleandoc(docstring)
            self._docstring = re.sub(r"(?![\r\n])(\b\s+)", " ", docstring)
        return self._docstring

    @property
    def template(self) -> str:
        """The dynamic jinja2 template for the prompt function."""
        text = f"{self.docstring}\n"

        for input_ in self.inputs:
            text += "\n{{ " + to_snake(input_.tag) + " }}\n"

        if self.output is None or isinstance(self.output, ChatOutput):
            return text

        text += f"\n{self.output.guidance()}\n"
        text += f"\n{self.output.to_format()}\n"

        return text

    @property
    def pipeline(self) -> ChatPipeline | None:
        """If available, the resolved Chat Pipeline for the prompt."""
        if self._pipeline is not None:
            return self._pipeline

        if self._generator is None and self._generator_id is not None:
            self._generator = get_generator(self._generator_id)

        if self._generator is not None:
            self._pipeline = self._generator.chat()
            return self._pipeline

        return None

    def _resolve_to_pipeline(self, other: ChatPipeline | Generator | Chat | str) -> ChatPipeline:
        if isinstance(other, ChatPipeline):
            return other
        if isinstance(other, Generator):
            return other.chat()
        if isinstance(other, Chat):
            return other.restart(include_all=True)
        if isinstance(other, str):
            return get_generator(other).chat()
        raise ValueError(f"Invalid type for binding: {type(other)}")

    async def _then_parse(self, chat: Chat) -> PipelineStepContextManager | None:
        if self.output is None or isinstance(self.output, ChatOutput):
            return None

        next_pipeline = chat.restart(include_all=True)

        try:
            # A bit weird, but we need from_chat to properly handle
            # wrapping Chat output types inside lists/dataclasses
            self.output.from_chat(chat)
        except ValidationError as e:
            next_pipeline.add(
                Message.from_model(
                    ValidationErrorModel(content=str(e)),
                    suffix="Rewrite your entire message with all of the required xml elements.",
                ),
            )
        except Exception as e:  # noqa: BLE001
            next_pipeline.add(
                Message.from_model(
                    SystemErrorModel(content=str(e)),
                    suffix="Rewrite your entire message with all of the required xml elements.",
                ),
            )
        else:  # parsed successfully
            return None

        return next_pipeline.step()

    def clone(self, *, include_callbacks: bool = True) -> "Prompt[P, R]":
        """
        Creates a deep copy of this prompt.

        Args:
            include_callbacks: Whether to skip copying the watch callbacks.

        Returns:
            A new instance of the prompt.
        """
        new = Prompt(
            func=self.func,
            _pipeline=self.pipeline,
            params=self.params.model_copy() if self.params is not None else None,
            max_parsing_rounds=self.max_parsing_rounds,
            max_tool_rounds=self.max_tool_rounds,
            system_prompt=self.system_prompt,
        )
        if not include_callbacks:
            new.watch_callbacks = self.watch_callbacks.copy()
            new.then_callbacks = self.then_callbacks.copy()
        return new

    def watch(
        self,
        *callbacks: WatchChatCallback,
        allow_duplicates: bool = False,
    ) -> "Prompt[P, R]":
        """
        Registers a callback to monitor any chats produced for this prompt

        See ChatPipeline.watch for more details.

        Args:
            *callbacks: The callback functions to be executed.
            allow_duplicates: Whether to allow duplicate callbacks.

        Returns:
            The updated prompt instance.

        Example:
            ```
            async def log(chats: list[Chat]) -> None:
                ...

            @rg.prompt()
            async def summarize(text: str) -> str:
                ...

            summarize.watch(log)(...)
            ```
        """
        for callback in callbacks:
            if not allow_duplicates and callback in self.watch_callbacks:
                raise ValueError(
                    f"Callback '{get_qualified_name(callback)}' is already registered.",
                )

        self.watch_callbacks.extend(callbacks)
        return self

    def then(
        self,
        *callbacks: ThenChatCallback,
        allow_duplicates: bool = False,
    ) -> "Prompt[P, R]":
        """
        Registers one or many callbacks to be executed during the prompt run

        See ChatPipeline.then for more details.

        Args:
            callbacks: The callback functions to be added.
            allow_duplicates: Whether to allow duplicate callbacks.

        Returns:
            The updated prompt.

        Example:
            ```
            async def score_summary(chat: Chat) -> Chat:
                ...

            @rg.prompt()
            async def summarize(text: str) -> str:
                ...

            summarize.then(score_summary)(...)
            ```
        """
        for callback in callbacks:
            if not asyncio.iscoroutinefunction(callback):
                raise TypeError(
                    f"Callback '{get_qualified_name(callback)}' must be an async function",
                )

            if allow_duplicates:
                continue

            if callback in self.then_callbacks:
                raise ValueError(
                    f"Callback '{get_qualified_name(callback)}' is already registered.",
                )

        self.then_callbacks.extend(callbacks)
        return self

    def map(
        self,
        *callbacks: MapChatCallback,
        allow_duplicates: bool = False,
    ) -> "Prompt[P, R]":
        """
        Registers a callback to be executed for each chat produced during the prompt run.

        See ChatPipeline.map for more details.

        Args:
            callbacks: The callback function to be executed.
            allow_duplicates: Whether to allow duplicate callbacks.

        Returns:
            The updated pipeline.

        Example:
            ```
            async def summarize_chats(chats: list[Chat]) -> list[Chat]:
                ...

            @rg.prompt()
            async def summarize(text: str) -> str:
                ...

            summarize.map(summarize_chats).bind_many()(10, ...)
            ```
        """
        for callback in callbacks:
            if not asyncio.iscoroutinefunction(callback):
                raise TypeError(
                    f"Callback '{get_qualified_name(callback)}' must be an async function",
                )

            if allow_duplicates:
                continue

            if callback in self.map_callbacks:
                raise ValueError(
                    f"Callback '{get_qualified_name(callback)}' is already registered.",
                )

        self.map_callbacks.extend(callbacks)
        return self

    def with_(self, params: GenerateParams | None = None, **kwargs: t.Any) -> "Prompt[P, R]":
        """
        Assign specific generation parameter overloads for this prompt.

        Args:
            params: The parameters to set for the underlying chat pipeline.
            **kwargs: An alternative way to pass parameters as keyword arguments.

        Returns:
            Self
        """
        self.params = params if params is not None else GenerateParams(**kwargs)
        return self

    # We could put these params into the decorator, but it makes it
    # less flexible when we want to build gateway interfaces into
    # creating a prompt from other code.

    def set_(
        self,
        max_parsing_rounds: int | None = None,
        max_tool_rounds: int | None = None,
    ) -> "Prompt[P, R]":
        """
        Helper to allow updates to the parsing configuration.

        Args:
            max_parsing_rounds: The maximum number of recursive output parsing attempts to allow.
            max_tool_rounds: The maximum number of recursive tool calls to allow.

        Returns:
            Self
        """
        self.max_parsing_rounds = max_parsing_rounds or self.max_parsing_rounds
        self.max_tool_rounds = max_tool_rounds or self.max_tool_rounds
        return self

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> t.OrderedDict[str, t.Any]:
        if self.func is None:
            return OrderedDict()

        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments

    def render(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """
        Pass the arguments to the jinja2 template and render the full prompt.
        """

        env = Environment(  # noqa: S701 # nosec
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        jinja_template = env.from_string(self.template)

        if self.func is None:
            return jinja_template.render()

        bound_args = self._bind_args(*args, **kwargs)

        for input_ in self.inputs:
            bound_args[to_snake(input_.tag)] = input_.to_xml(bound_args[input_.name])

        return jinja_template.render(**bound_args)

    def process(self, chat: Chat) -> R:
        """
        Attempt to parse the output from a chat into the expected return type.
        """
        return self.output.from_chat(chat)  # type: ignore [no-any-return]

    def bind(
        self,
        other: ChatPipeline | Generator | Chat | str,
    ) -> t.Callable[P, t.Coroutine[t.Any, t.Any, R]]:
        """
        Binds the prompt to a pipeline, generator, or chat and returns a scoped run callable.

        Args:
            other: The pipeline, generator, generator id, or chat to bind to.

        Returns:
            A callable for executing this prompt

        Example:
            ```
            @rg.prompt
            def say_hello(name: str) -> str:
                \"""Say hello to {{ name }}\"""

            await say_hello.bind("gpt-3.5-turbo")("the world")
            ```
        """
        pipeline = self._resolve_to_pipeline(other)
        if pipeline.on_failed == "skip":
            raise NotImplementedError(
                "pipeline.on_failed='skip' cannot be used for prompt methods that return one object",
            )

        async def run(*args: P.args, **kwargs: P.kwargs) -> R:
            results = await self.bind_many(pipeline)(1, *args, **kwargs)
            return results[0]

        run.__signature__ = self.__signature__  # type: ignore [attr-defined]
        run.__name__ = self.__name__
        run.__doc__ = self.__doc__
        run.__rg_prompt__ = self  # type: ignore [attr-defined]

        return run

    def bind_many(
        self,
        other: ChatPipeline | Generator | Chat | str,
    ) -> t.Callable[Concatenate[int, P], t.Coroutine[t.Any, t.Any, list[R]]]:
        """
        Binds the prompt to a pipeline, generator, or chat and returns a scoped run_many callable.

        Args:
            other: The pipeline, generator, generator id, or chat to bind to.

        Returns:
            A callable for executing this prompt.

        Example:
            ```
            @rg.prompt
            def say_hello(name: str) -> str:
                \"""Say hello to {{ name }}\"""

            await say_hello.bind("gpt-3.5-turbo")(5, "the world")
            ```
        """
        pipeline = self._resolve_to_pipeline(other)
        if pipeline.on_failed == "include" and not isinstance(self.output, ChatOutput):
            raise NotImplementedError(
                "pipeline.on_failed='include' cannot be used with prompts that process outputs",
            )

        async def run_many(count: int, /, *args: P.args, **kwargs: P.kwargs) -> list[R]:
            name = get_qualified_name(self.func) if self.func else "<generated>"
            with tracer.span(
                f"Prompt {name}()" if count == 1 else f"Prompt {name}() (x{count})",
                count=count,
                name=name,
                arguments=self._bind_args(*args, **kwargs),
            ) as span:
                content = self.render(*args, **kwargs)
                _pipeline = (
                    pipeline.fork(content)
                    .using(*self.tools, max_depth=self.max_tool_rounds)
                    .then(self._then_parse, max_depth=self.max_parsing_rounds)
                    .then(*self.then_callbacks)
                    .map(*self.map_callbacks)
                    .watch(*self.watch_callbacks)
                    .with_(self.params)
                )

                if self.system_prompt:
                    _pipeline.chat.inject_system_content(self.system_prompt)

                chats = await _pipeline.run_many(count)

                # TODO: I can't remember why we don't just pass the watch_callbacks to the pipeline
                # Maybe it has something to do with uniqueness and merging?

                def wrap_watch_callback(callback: "WatchChatCallback") -> "WatchChatCallback":
                    async def traced_watch_callback(chats: list[Chat]) -> None:
                        callback_name = get_qualified_name(callback)
                        with tracer.span(
                            f"Watch with {callback_name}()",
                            callback=callback_name,
                            chat_count=len(chats),
                            chat_ids=[str(c.uuid) for c in chats],
                        ):
                            await callback(chats)

                    return traced_watch_callback

                coros = [
                    wrap_watch_callback(watch)(chats)
                    for watch in self.watch_callbacks
                    if watch not in pipeline.watch_callbacks
                ]
                await asyncio.gather(*coros)

                results = [self.process(chat) for chat in chats]
                span.set_attribute("results", results)
                return results

        run_many.__rg_prompt__ = self  # type: ignore [attr-defined]

        return run_many

    def bind_over(
        self,
        other: ChatPipeline | Generator | Chat | str | None = None,
    ) -> t.Callable[
        Concatenate[t.Sequence[Generator | str], P],
        t.Coroutine[t.Any, t.Any, list[R]],
    ]:
        """
        Binds the prompt to a pipeline, generator, or chat and returns a scoped run_over callable.

        Args:
            other: The pipeline, generator, generator id, or chat to bind to.

        Returns:
            A callable for executing this prompt.

        Example:
            ```
            @rg.prompt
            def say_hello(name: str) -> str:
                \"""Say hello to {{ name }}\"""

            await say_hello.bind("gpt-3.5-turbo")(["gpt-4o", "gpt-4"], "the world")
            ```
        """
        include_original = other is not None

        if other is None:
            pipeline = (
                get_generator("base!base").chat().catch(on_failed="skip")
            )  # TODO: Clean this up
        else:
            pipeline = self._resolve_to_pipeline(other)

        if pipeline.on_failed == "include" and not isinstance(self.output, ChatOutput):
            raise NotImplementedError(
                "pipeline.on_failed='include' cannot be used with prompts that process outputs",
            )

        async def run_over(
            generators: t.Sequence[Generator | str],
            /,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> list[R]:
            content = self.render(*args, **kwargs)
            _pipeline = (
                pipeline.fork(content)
                .using(*self.tools, max_depth=self.max_tool_rounds)
                .then(self._then_parse, max_depth=self.max_parsing_rounds)
                .then(*self.then_callbacks)
                .map(*self.map_callbacks)
                .watch(*self.watch_callbacks)
                .with_(self.params)
            )

            if self.system_prompt:
                _pipeline.chat.inject_system_content(self.system_prompt)

            chats = await _pipeline.run_over(*generators, include_original=include_original)

            coros = [
                watch(chats)
                for watch in self.watch_callbacks
                if watch not in pipeline.watch_callbacks
            ]
            await asyncio.gather(*coros)

            return [self.process(chat) for chat in chats]

        run_over.__rg_prompt__ = self  # type: ignore [attr-defined]

        return run_over

    async def run_many(self, count: int, /, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        """
        Use the prompt to run the function multiple times with the provided arguments and return the output.

        Args:
            count: The number of times to run the prompt.
            *args: The positional arguments for the prompt function.
            **kwargs: The keyword arguments for the prompt function.

        Returns:
            The outputs of the prompt function.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Prompt cannot be executed as a standalone function without being assigned a pipeline or generator",
            )
        return await self.bind_many(self.pipeline)(count, *args, **kwargs)

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Use the prompt to run the function with the provided arguments and return the output.

        Args:
            *args: The positional arguments for the prompt function.
            **kwargs: The keyword arguments for the prompt function.

        Returns:
            The output of the prompt function.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Prompt cannot be executed as a standalone function without being assigned a pipeline or generator",
            )
        return await self.bind(self.pipeline)(*args, **kwargs)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self.run(*args, **kwargs)

    async def run_over(
        self,
        generators: t.Sequence[Generator | str],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[R]:
        """
        Executes the prompt process across multiple generators.

        For each generator, a pipeline is created and the generator is replaced
        before the run call. All callbacks and parameters are preserved.

        If this prompt has a pipeline assigned, it will be included in the run.

        Warning:
            The implementation currently skips any failed chats and only
            processes successful chats. This may change in the future.

        Args:
            generators: A sequence of generators to be used for the generation process.

        Returns:
            A list of generatated Chats.
        """
        return await self.bind_over(self.pipeline)(generators, *args, **kwargs)


# Decorator


@t.overload
def prompt(
    func: None = None,
    /,
    *,
    pipeline: ChatPipeline | None = None,
    generator: Generator | None = None,
    generator_id: str | None = None,
    tools: list[Tool[..., t.Any] | t.Callable[..., t.Any]] | None = None,
    system_prompt: str | None = None,
) -> t.Callable[[t.Callable[P, t.Coroutine[t.Any, t.Any, R]] | t.Callable[P, R]], Prompt[P, R]]: ...


@t.overload
def prompt(
    func: t.Callable[P, t.Coroutine[t.Any, t.Any, R]],
    /,
    *,
    pipeline: ChatPipeline | None = None,
    generator: Generator | None = None,
    generator_id: str | None = None,
    tools: list[Tool[..., t.Any] | t.Callable[..., t.Any]] | None = None,
    system_prompt: str | None = None,
) -> Prompt[P, R]: ...


@t.overload
def prompt(
    func: t.Callable[P, R],
    /,
    *,
    pipeline: ChatPipeline | None = None,
    generator: Generator | None = None,
    generator_id: str | None = None,
    tools: list[Tool[..., t.Any] | t.Callable[..., t.Any]] | None = None,
    system_prompt: str | None = None,
) -> Prompt[P, R]: ...


def prompt(
    func: t.Callable[P, t.Coroutine[t.Any, t.Any, R]] | t.Callable[P, R] | None = None,
    /,
    *,
    pipeline: ChatPipeline | None = None,
    generator: Generator | None = None,
    generator_id: str | None = None,
    tools: list[Tool[..., t.Any] | t.Callable[..., t.Any]] | None = None,
    system_prompt: str | None = None,
) -> (
    t.Callable[[t.Callable[P, t.Coroutine[t.Any, t.Any, R]] | t.Callable[P, R]], Prompt[P, R]]
    | Prompt[P, R]
):
    """
    Convert a hollow function into a Prompt, which can be called directly or passed a
    chat pipeline to execute the function and parse the outputs.

    Note:
        A docstring is not required, but this can be used to provide guidance to the model, or
        even handle any number of input transformations. Any input parameter which is not
        handled inside the docstring will be automatically added and formatted internally.

    Note:
        Output parameters can be basic types, dataclasses, rigging models, lists, or tuples.
        Internal inspection will attempt to ensure your output types are valid, but there is
        no guarantee of complete coverage/safety. It's recommended to check
        [rigging.prompt.Prompt.template][] to inspect the generated jinja2 template.

    Note:
        If you annotate the return value of the function as a [rigging.chat.Chat][] object,
        then no output parsing will take place and you can parse objects out manually.

        You can also use Chat in any number of type annotation inside tuples or dataclasses.
        All instances will be filled with the final chat object transparently.

    Note:
        All input parameters and output types can be annotated with the [rigging.prompt.Ctx][] annotation
        to provide additional context for the prompt. This can be used to override the xml tag, provide
        a prefix string, or example content which will be placed inside output xml tags.

        In the case of output parameters, especially in tuples, you might have xml tag collisions
        between the same basic types. Manually annotating xml tags with [rigging.prompt.Ctx][] is
        recommended.

    Args:
        func: The function to convert into a prompt.
        pipeline: An optional pipeline to use for the prompt.
        generator: An optional generator to use for the prompt.
        generator_id: An optional generator id to use for the prompt.
        tools: An optional list of tools to make available during generation (can be other prompts).
        system_prompt: An optional system prompt fragment to inject into the messages before generation.

    Returns:
        A prompt instance or a function that can be used to create a prompt.

    Example:
        ```
        from dataclasses import dataclass
        import rigging as rg

        @dataclass
        class ExplainedJoke:
            chat: rg.Chat
            setup: str
            punchline: str
            explanation: str

        @rg.prompt(generator_id="gpt-3.5-turbo")
        async def write_joke(topic: str) -> ExplainedJoke:
            \"""Write a joke.\"""
            ...

        await write_joke("programming")
        ```
    """
    if sum(arg is not None for arg in (pipeline, generator, generator_id)) > 1:
        raise ValueError("Only one of pipeline, generator, or generator_id can be provided")

    def make_prompt(
        func: t.Callable[P, t.Coroutine[t.Any, t.Any, R]] | t.Callable[P, R],
    ) -> Prompt[P, R]:
        return Prompt[P, R](
            func=func,  # type: ignore [arg-type]
            _generator_id=generator_id,
            _pipeline=pipeline,
            _generator=generator,
            system_prompt=system_prompt,
            tools=[tool if isinstance(tool, Tool) else Tool.from_callable(tool) for tool in tools]
            if tools
            else [],
        )

    if func is not None:
        return make_prompt(func)
    return make_prompt


@t.overload
def make_prompt(
    content: str,
    return_type: type[R],
    *,
    ctx: Ctx | None = None,
) -> Prompt[..., R]: ...


@t.overload
def make_prompt(
    content: str,
    return_type: None = None,
    *,
    ctx: Ctx | None = None,
) -> Prompt[..., str]: ...


def make_prompt(
    content: str,
    return_type: type[R] | None = None,
    *,
    ctx: Ctx | None = None,
) -> Prompt[..., R] | Prompt[..., str]:
    """
    Create a prompt at runtime from a basic string and return type (experimental).

    Note:
        Adding input parameters is not currently supported. Instead use
        the [rigging.prompt.prompt][] decorator.

    Args:
        content: The docstring content for the prompt.
        return_type: The return type of the prompt function.
        ctx: Context for the return type (Use this instead of Annotated for better type hints).

    Returns:
        The constructed Prompt

    Example:
        ```
        import rigging as rg

        write_joke = rg.make_prompt("Write a joke.", ctx=rg.Ctx(tag="joke"))

        await write_joke.bind("gpt-4o-mini")()
        ```
    """
    return_type = return_type or str  # type: ignore [assignment]
    output = parse_output(
        t.Annotated[return_type, ctx] if ctx is not None else return_type,
        "make_prompt(<return>)",
    )
    return Prompt(output=output, _docstring=content)
