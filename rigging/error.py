class ExhaustedMaxRoundsError(Exception):
    def __init__(self, max_rounds: int):
        super().__init__(f"Exhausted max rounds ({max_rounds}) while generating")
        self.max_rounds = max_rounds


class InvalidModelSpecifiedError(Exception):
    def __init__(self, model: str):
        super().__init__(f"Invalid model specified: {model}")


class MissingModelError(Exception):
    def __init__(self, content: str):
        super().__init__(content)
