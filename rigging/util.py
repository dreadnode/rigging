import asyncio
import functools
import typing as t

from typing_extensions import ParamSpec

P = ParamSpec("P")
R = t.TypeVar("R")


def make_sync_entrypoint(callable_: t.Callable[P, t.Coroutine[t.Any, t.Any, R]]) -> t.Callable[P, R]:
    @functools.wraps(callable_, assigned=("__doc__", "__name__", "__qualname__", "__module__"))
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop is not None:
            raise RuntimeError(
                "Cannot call an synchronous entrypoint while an async event loop is running. "
                "You might be running in a notebook or inside async code. "
                f"Use the asyncronous version of the function instead: {callable_.__name__}"
            )

        return asyncio.run(callable_(*args, **kwargs))

    if wrapper.__name__.startswith("a"):
        wrapper.__name__ = wrapper.__name__[1:]

    if wrapper.__qualname__.startswith("a"):
        wrapper.__qualname__ = wrapper.__qualname__[1:]

    return wrapper
