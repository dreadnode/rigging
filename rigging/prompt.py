from __future__ import annotations

import inspect
import typing as t

from jinja2 import Environment, meta
from pydantic import BaseModel, computed_field, model_validator
from typing_extensions import ParamSpec

from rigging.generator.base import GenerateParams, Generator, get_generator
from rigging.model import Model

if t.TYPE_CHECKING:
    from rigging.chat import ChatPipeline

DEFAULT_DOC = "You will convert the following inputs to outputs."

P = ParamSpec("P")
R = t.TypeVar("R")

BASIC_TYPES = [int, float, str, bool, list, dict, set, tuple, type(None)]
# Utilities


def get_undefined_values(template: str) -> set[str]:
    env = Environment()
    parsed_template = env.parse(template)
    return meta.find_undeclared_variables(parsed_template)


def format_parameter(param: inspect.Parameter, value: t.Any) -> str:
    name = param.name

    if isinstance(value, str):
        if "\n" in value:
            value = f"\n{value.strip()}\n"
        return f"<{name}>{value}</{name}>"

    if isinstance(value, (int, float)):
        return f"<{name}>{value}</{name}>"

    if isinstance(value, bool):
        return f"<{name}>{'true' if value else 'false'}</{name}>"

    if isinstance(value, Model):
        return value.to_pretty_xml()

    if isinstance(value, (list, set)):
        type_args = t.get_args(param.annotation)

        xml = f"<{name}>\n"
        for item in value:
            pass

    raise ValueError(f"Unsupported parameter: {param}: '{value}'")


def check_valid_function(func: t.Callable[P, t.Coroutine[None, None, R]]) -> None:
    signature = inspect.signature(func)

    for param in signature.parameters.values():
        error_name = f"{func.__name__}({param})"

        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"Parameters must be positional or keyword {error_name}")

        if param.annotation in [None, inspect.Parameter.empty]:
            raise TypeError(f"All parameters require type annotations {error_name}")

        origin = t.get_origin(param.annotation)
        if origin is not None:
            # Check for a dict[str, str]
            if origin is dict:
                if param.annotation.__args__[0] != (str, str):
                    raise TypeError(f"Dicts must have str keys {error_name}")
            raise TypeError(f"Parameters cannot be generic, lists, sets, or dicts {error_name}")

        if param.annotation in [int, float, str, bool]:
            continue

        if issubclass(param.annotation, Model):
            continue

        raise TypeError(
            f"Invalid parameter type: {param.annotation}, must be one of int, bool, str, float or rg.Model ({func.__name__}#{param.name})"
        )

    if signature.return_annotation in [None, inspect.Parameter.empty]:
        raise TypeError(f"Return type annotation is required ({func.__name__})")

    if not isinstance(signature.return_annotation, tuple):
        return


def build_template(func: t.Callable) -> str:
    docstring = func.__doc__ or DEFAULT_DOC
    docstring = inspect.cleandoc(docstring)

    signature = inspect.signature(func)


# Prompt


class Prompt(BaseModel, t.Generic[P, R]):
    _func: t.Callable[P, t.Coroutine[None, None, R]]

    _generator_id: str | None = None
    _generator: Generator | None = None
    _pipeline: ChatPipeline | None = None
    _params: GenerateParams | None = None

    @model_validator(mode="after")
    def check_valid_function(self) -> Prompt[P, R]:
        check_valid_function(self._func)
        return self

    @computed_field  # type: ignore [misc]
    @property
    def template(self) -> str:
        return ""

    @property
    def pipeline(self) -> ChatPipeline | None:
        if self._pipeline is not None:
            return self._pipeline.with_(params=self._params)

        if self._generator is None:
            if self._generator_id is None:
                raise ValueError(
                    "You cannot execute this prompt ad-hoc. No pipeline, generator, or generator_id was provided."
                )

            self._generator = get_generator(self._generator_id)

        return self._generator.chat(params=self._params)

    def clone(self) -> Prompt[P, R]:
        return Prompt(_func=self._func, _pipeline=self.pipeline)

    def with_(self, params: t.Optional[GenerateParams] = None, **kwargs: t.Any) -> Prompt[P, R]:
        if params is None:
            params = GenerateParams(**kwargs)

        if self._params is not None:
            new = self.clone()
            new._params = self._params.merge_with(params)
            return new

        self.params = params
        return self

    def render(self, *args: P.args, **kwargs: P.kwargs) -> str:
        pass

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        pass

    async def run_many(self, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        pass

    __call__ = run


def prompt(
    *, pipeline: ChatPipeline | None = None, generator: Generator | None = None, generator_id: str | None = None
) -> t.Callable[[t.Callable[P, t.Coroutine[None, None, R]]], Prompt[P, R]]:
    if sum(arg is not None for arg in (pipeline, generator, generator_id)) > 1:
        raise ValueError("Only one of pipeline, generator, or generator_id can be provided")

    def decorator(func: t.Callable[P, t.Coroutine[None, None, R]]) -> Prompt[P, R]:
        return Prompt[P, R](_func=func, _generator_id=generator_id, _pipeline=pipeline, _generator=generator)

    return decorator


@prompt()
async def testing() -> None:
    pass
