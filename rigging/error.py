"""
We try to avoid creating custom exceptions unless they are necessary.

We use the built-in and pydantic exceptions as much as possible.
"""


class ExhaustedMaxRoundsError(Exception):
    """
    Raised when the maximum number of rounds is exceeded while generating.
    """

    def __init__(self, max_rounds: int):
        super().__init__(f"Exhausted max rounds ({max_rounds}) while generating")
        self.max_rounds = max_rounds


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
