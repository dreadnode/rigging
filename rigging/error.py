"""
We try to avoid creating custom exceptions unless they are necessary.

We use the built-in and pydantic exceptions as much as possible.
"""

import functools
import typing as t

import typing_extensions as te

if t.TYPE_CHECKING:
    from rigging.chat import PipelineStep
    from rigging.message import Message


# User Throwable Exceptions


class Stop(Exception):  # noqa: N818
    """
    Raise inside a pipeline to indicate a stopping condition.

    Example:
        ```
        import rigging as rg

        async def read_file(path: str) -> str:
            "Read the contents of a file."

            if no_more_files(path):
                raise rg.Stop("There are no more files to read.")

            ...

        chat = await pipeline.using(read_file).run()
        ```
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        """The message associated with the stop."""


# Warnings


class PipelineWarning(Warning):
    """
    Base class for all pipeline warnings.

    This is used to indicate that something unexpected happened during the pipeline execution,
    but it is not critical enough to stop the execution.
    """


class ToolWarning(Warning):
    """
    Base class for all tool warnings.

    This is used to indicate that something unexpected happened during the tool execution,
    but it is not critical enough to stop the execution.
    """


class MessageWarning(Warning):
    """
    Base class for all message warnings.

    This is used to indicate that something unexpected happened during the message processing,
    but it is not critical enough to stop the execution.
    """


class TokenizerWarning(Warning):
    """
    Base class for all tokenization warnings.

    This is used to indicate that something unexpected happened during the tokenization process,
    but it is not critical enough to stop the execution.
    """


# System Exceptions


class UnknownToolError(Exception):
    """
    Raised when the an api tool call is made for an unknown tool.
    """

    def __init__(self, tool_name: str):
        super().__init__(f"Unknown tool call was requested for '{tool_name}'")
        self.tool_name = tool_name
        """The name of the tool which was unknown."""


class ToolDefinitionError(Exception):
    """
    Raised when a tool cannot be properly defined.
    """

    def __init__(self, message: str):
        super().__init__(message)


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
        """The messages which were being generated when the exception occurred."""


class CompletionExhaustedMaxRoundsError(ExhaustedMaxRoundsError):
    """
    Raised when the maximum number of rounds is exceeded while generating completions.
    """

    def __init__(self, max_rounds: int, completion: str):
        super().__init__(max_rounds)
        self.completion = completion
        """The completion which was being generated when the exception occurred."""


class MaxDepthError(Exception):
    """
    Raised when the maximum depth is exceeded while generating.
    """

    def __init__(self, max_depth: int, step: "PipelineStep", reference: str):
        super().__init__(f"Exceeded max depth ({max_depth}) while generating ('{reference}')")
        self.max_depth = max_depth
        """The maximum depth of nested pipeline generations which was exceeded."""
        self.step = step
        """The pipeline step which cause the depth error."""


class InvalidGeneratorError(Exception):
    """
    Raised when an invalid identifier is specified when getting a generator.
    """

    def __init__(self, model: str):
        super().__init__(f"Invalid model specified: {model}")


class InvalidTokenizerError(Exception):
    """
    Raised when an invalid tokenizer is specified.
    """

    def __init__(self, tokenizer: str):
        super().__init__(f"Invalid tokenizer specified: {tokenizer}")
        self.tokenizer = tokenizer
        """The name of the tokenizer which was invalid."""


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


def raise_as(
    error_type: type[Exception],
    message: str,
) -> t.Callable[[t.Callable[P, R]], t.Callable[P, R]]:
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
