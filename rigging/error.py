class ExhaustedMaxRoundsError(Exception):
    """Raised when the maximum number of rounds is exceeded while generating.
    """
    def __init__(self, max_rounds: int):
        """Initializes the exception with the maximum number of rounds that was exceeded.
        """
        super().__init__(f"Exhausted max rounds ({max_rounds}) while generating")
        self.max_rounds = max_rounds


class InvalidModelSpecifiedError(Exception):
    """Raised when an invalid model is specified.
    """
    def __init__(self, model: str):
        super().__init__(f"Invalid model specified: {model}")


class MissingModelError(Exception):
    """Raised when a model is missing.
    """
    def __init__(self, content: str):
        super().__init__(content)
