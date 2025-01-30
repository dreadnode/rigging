"""
We try to avoid creating custom exceptions unless they are necessary.

We use the built-in and pydantic exceptions as much as possible.
"""

import functools
import typing as t

import typing_extensions as te

if t.TYPE_CHECKING:
    from rigging.message import Message


class UnknownToolError(Exception):
    """
    Raised when the an api tool call is made for an unknown tool.
    """

    def __init__(self, tool_name: str):
        super().__init__(f"Unknown tool call was requested for '{tool_name}'")
        self.tool_name = tool_name
        """The name of the tool which was unknown."""


class ExhaustedMaxRoundsError(Exception):
    """
    Raised when the maximum number of rounds is exceeded while generating.
    """

    def __init__(self, max_rounds: int):
        super().__init__(f"Exhausted max rounds ({max_rounds}) while generating")
        self.max_rounds = max_rounds
        """The number of rounds which was exceeded."""


class MessagesExhaustedMaxRoundsError(ExhaustedMaxRoundsError):
    """
    Raised when the maximum number of rounds is exceeded while generating messages.
    """

    def __init__(self, max_rounds: int, messages: list["Message"]):
        super().__init__(max_rounds)
        self.messages = messages
        """The messages which were being generated when the exception occured."""


class CompletionExhaustedMaxRoundsError(ExhaustedMaxRoundsError):
    """
    Raised when the maximum number of rounds is exceeded while generating completions.
    """

    def __init__(self, max_rounds: int, completion: str):
        super().__init__(max_rounds)
        self.completion = completion
        """The completion which was being generated when the exception occured."""


class InvalidModelSpecifiedError(Exception):
    """
    Raised when an invalid identifier is specified when getting a generator.
    """

    def __init__(self, model: str):
        super().__init__(f"Invalid model specified: {model}")


class MissingModelError(Exception):
    """
    Raised when a model is missing when parsing a message.
    """

    def __init__(self, content: str):
        super().__init__(content)


class ProcessingError(Exception):
    """
    Raised when an error occurs during internal generator processing.
    """

    def __init__(self, content: str):
        super().__init__(content)


P = te.ParamSpec("P")
R = t.TypeVar("R")


def raise_as(error_type: type[Exception], message: str) -> t.Callable[[t.Callable[P, R]], t.Callable[P, R]]:
    "When the wrapped function raises an exception, `raise ... from` with the new error type."

    def _raise_as(func: t.Callable[P, R]) -> t.Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = error_type(message)
                raise error from e

        if wrapper.__doc__ is None:
            wrapper.__doc__ = ""

        wrapper.__doc__ += f"\n\nRaises:\n    {error_type.__name__}{': ' + message}"

        return wrapper

    return _raise_as
