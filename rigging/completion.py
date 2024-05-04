"""
Completions work with isolated strings of text pre and post generation.
"""

import asyncio
import string
import typing as t
from copy import deepcopy
from datetime import datetime
from uuid import UUID, uuid4

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

from rigging.error import ExhaustedMaxRoundsError
from rigging.model import (
    Model,
    ModelT,
)
from rigging.parsing import parse_many

if t.TYPE_CHECKING:
    from rigging.generator import GenerateParams, Generator

DEFAULT_MAX_ROUNDS = 5


class Completion(BaseModel):
    """
    Represents a completed text generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uuid: UUID = Field(default_factory=uuid4)
    """The unique identifier."""
    timestamp: datetime = Field(default_factory=datetime.now, repr=False)
    """The timestamp when the completion was created."""
    text: str
    """The original text."""
    generated: str
    """The generated text."""

    pending: t.Optional["PendingCompletion"] = Field(None, exclude=True, repr=False)
    """The pending completion associated with this completion."""

    @computed_field(repr=False)
    def generator_id(self) -> str | None:
        """The identifier of the generator used to create the completion"""
        if self.pending is not None:
            return self.pending.generator.to_identifier(self.pending.params)
        return None

    def __init__(
        self,
        text: str,
        generated: str,
        pending: t.Optional["PendingCompletion"] = None,
        **kwargs: t.Any,
    ):
        """
        Initialize a Chat object.

        Args:
            text: The original text.
            generated: The generated text.
            pending: The pending completion associated with this completion
            **kwargs: Additional keyword arguments (typically used for serialization).
        """
        from rigging.generator import get_generator

        if "generator_id" in kwargs and pending is None:
            generator = get_generator(kwargs.pop("generator_id"))
            pending = generator.complete(text)

        super().__init__(
            text=text,
            generated=generated,
            pending=pending,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.text) + len(self.generated)

    @property
    def all(self) -> str:
        """Returns both the text and the generation."""
        return self.text + self.generated

    def restart(self, *, generator: t.Optional["Generator"] = None, include_all: bool = False) -> "PendingCompletion":
        """
        Attempt to convert back to a PendingCompletion for further generation.

        Args:
            generator: The generator to use for the restarted chat. Otherwise
                the generator from the original PendingCompletion will be used.
            include_all: Whether to include the generation before the next round.

        Returns:
            The restarted completion.

        Raises:
            ValueError: If the completion was not created with a PendingCompletion and no generator is provided.
        """

        text = self.all if include_all else self.text
        if generator is not None:
            return generator.complete(text)
        elif self.pending is None:
            raise ValueError("Cannot restart Completion that was not created with a PendingCompletion")
        return PendingCompletion(self.pending.generator, text, self.pending.params)

    def fork(self, text: str) -> "PendingCompletion":
        """
        Forks the completion by creating calling [rigging.completion.Completion.restart][] and appends the specified text.

        Args:
            text: The text to append.

        Returns:
            A new instance of a pending competion with the specified messages added.
        """
        return self.restart().add(text)

    def clone(self) -> "Completion":
        """Creates a deep copy of the chat."""
        return Completion(self.text, self.generated, self.pending)


# Passed the next message, returns whether or not to continue
# and an optional list of messages to append before continuing
UntilCompletionCallback = t.Callable[[str], bool]


class PendingCompletion:
    """
    Represents a pending completion that can be modified and executed.
    """

    def __init__(self, generator: "Generator", text: str, params: t.Optional["GenerateParams"] = None):
        self.generator: "Generator" = generator
        """The generator object responsible for generating the completion."""
        self.text = text
        """The text to be completed."""
        self.params = params
        """The parameters for generating the completion."""
        self.metadata: dict[str, t.Any] = {}
        """Additional metadata associated with the completion."""

        # (callback, all_text, max_rounds)
        self.until_callbacks: list[tuple[UntilCompletionCallback, bool, int]] = []
        self.until_types: list[type[Model]] = []

    def overload(self, **kwargs: t.Any) -> "PendingCompletion":
        """
        Overloads the current completion with the given parameters.

        This is a convenience method for calling `with_params(GenerateParams(**kwargs))`.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            **kwargs: Keyword arguments representing the parameters to be overloaded.

        Returns:
            A new instance of PendingCompletion with the overloaded parameters.
        """
        from rigging.generator import GenerateParams

        return self.with_params(GenerateParams(**kwargs))

    def with_params(self, params: "GenerateParams") -> "PendingCompletion":
        """
        Sets the generation parameter overloads for the completion.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            params: The parameters to set for the completion.

        Returns:
            A new instance of PendingCompletion with the updated parameters.
        """
        if self.params is not None:
            new = self.clone()
            new.params = params
            return new

        self.params = params
        return self

    def add(self, text: str) -> "PendingCompletion":
        """
        Appends new text to the internal text before generation.

        Args:
            text: The text to be added to the completion.

        Returns:
            The updated PendingCompletion object.
        """
        self.text += text
        return self

    def fork(self, text: str) -> "PendingCompletion":
        """
        Creates a new instance of `PendingCompletion` by forking the current completion and adding the specified text.

        This is a convenience method for calling `clone().add(text)`.

        Args:
            text: The text to be added to the new completion.

        Returns:
            A new instance of `PendingCompletion` with the specified text added.
        """
        return self.clone().add(text)

    def clone(self, *, only_text: bool = False) -> "PendingCompletion":
        """
        Creates a clone of the current `PendingCompletion` instance.

        Args:
            only_text: If True, only the text will be cloned.
                If False (default), the entire `PendingCompletion` instance will be cloned
                including until callbacks and types.

        Returns:
            A new instance of `PendingCompletion` that is a clone of the current instance.
        """
        new = PendingCompletion(self.generator, self.text, self.params)
        if not only_text:
            new.until_callbacks = self.until_callbacks.copy()
            new.until_types = self.until_types.copy()
            new.metadata = deepcopy(self.metadata)
        return new

    def meta(self, **kwargs: t.Any) -> "PendingCompletion":
        """
        Updates the metadata of the completion with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated completion object.
        """
        self.metadata.update(kwargs)
        return self

    def apply(self, **kwargs: str) -> "PendingCompletion":
        """
        Applies keyword arguments to the text using string template substitution.

        Args:
            **kwargs: Keyword arguments to be applied to the text.

        Returns:
            A new instance of PendingCompletion with the applied arguments.
        """
        new = self.clone()
        template = string.Template(self.text)
        new.text = template.safe_substitute(**kwargs)
        return new

    def until(
        self,
        callback: UntilCompletionCallback,
        *,
        use_all_text: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> "PendingCompletion":
        """
        Registers a callback to participate in validating the generation process.

        ```python
        # Takes the generated text, and returns whether or not to retry generation.

        def callback(text: str) -> bool:
            if is_valid(text):
                return False
            else:
                return True

        pending.until(callback).run()
        ```

        Args:
            callback: The callback function to be executed.
            use_all_text: Whether to pass the entire text (including prompt) to the callback.

            max_rounds: The maximum number of rounds to attempt generation + callbacks
                before giving up.

        Returns:
            The current instance of the completion.
        """
        self.until_callbacks.append((callback, use_all_text, max_rounds))
        return self

    def until_parsed_as(
        self,
        *types: type[ModelT],
        use_all_text: bool = False,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> "PendingCompletion":
        """
        Adds the specified types to the list of types which should successfully parse
        before the generation process completes.

        Args:
            *types: The type or types of models to wait for.
            use_all_text: Whether to pass the entire text (including prompt) to the parser.

            max_rounds: The maximum number of rounds to try to parse
                successfully.

        Returns:
            The updated PendingCompletion object.
        """
        self.until_types += types
        if next((c for c in self.until_callbacks if c[0] == self._until_parse_callback), None) is None:
            self.until_callbacks.append((self._until_parse_callback, use_all_text, max_rounds))

        return self

    def _until_parse_callback(self, text: str) -> bool:
        try:
            parse_many(text, *self.until_types)
        except Exception:
            return True
        return False

    def _execute(self) -> t.Generator[str, str, str]:
        # If there are no until_callbacks, we can just yield the text
        if not self.until_callbacks:
            generated = yield self.text
            return generated

        # It's opaque exactly how we should blend multiple
        # until callbacks together, so here is the current implementation:
        #
        # - We take the lowest max_rounds from all until_callbacks
        # - Each loop, we let every callback run, if any tell us to retry, we do
        # - If we leave the loop with should_retry still True, we raise an error
        # - Assuming every should_retry is False, we break out of the loop and return

        lowest_max_rounds = min((c[2] for c in self.until_callbacks), default=1)

        current_round = 0
        should_retry = True
        while should_retry and current_round < lowest_max_rounds:
            current_round += 1
            generated = yield self.text
            for callback, use_all_text, _ in self.until_callbacks:
                should_retry = callback(self.text + generated if use_all_text else generated)
                if should_retry:
                    continue

        if should_retry:
            logger.warning(f"Exhausted lowest max rounds ({lowest_max_rounds})")
            raise ExhaustedMaxRoundsError(lowest_max_rounds)

        return generated

    @t.overload
    def run(self, count: t.Literal[None] = None) -> Completion:
        ...

    @t.overload
    def run(self, count: int) -> list[Completion]:
        ...

    def run(self, count: int | None = None) -> Completion | list[Completion]:
        """
        Execute the generation process to produce the final completion.

        If `count` is provided, `run_many` will be called instead.

        Args:
            count: The number of times to generate using the same inputs.

        Returns:
            The completion object or a list of completion objects, depending on the value of `count`.
        """
        if count is not None:
            return self.run_many(count)

        executor = self._execute()
        outbound = next(executor)

        try:
            while True:
                inbound = self.generator.generate_text(outbound, self.params)
                outbound = executor.send(inbound)
        except StopIteration as stop:
            outbound = t.cast(str, stop.value)

        return Completion(self.text, outbound, pending=self)

    def run_many(self, count: int) -> list[Completion]:
        """
        Executes the generation process multiple times with the same inputs.

        Parameters:
            count: The number of times to execute the generation process.

        Returns:
            A list of Completion objects representing the results of each execution.
        """
        return [self.run() for _ in range(count)]

    __call__ = run

    @t.overload
    async def arun(self, count: t.Literal[None] = None) -> Completion:
        ...

    @t.overload
    async def arun(self, count: int) -> list[Completion]:
        ...

    async def arun(self, count: int | None = None) -> Completion | list[Completion]:
        """async variant of the [rigging.completion.PendingCompletion.run][] method."""
        if count is not None:
            return await self.arun_many(count)

        executor = self._execute()
        outbound = next(executor)

        try:
            while True:
                inbound = await self.generator.agenerate_text(outbound, self.params)
                outbound = executor.send(inbound)
        except StopIteration as stop:
            outbound = t.cast(str, stop.value)

        return Completion(self.text, outbound, pending=self)

    async def arun_many(self, count: int) -> list[Completion]:
        """async variant of the [rigging.completion.PendingCompletion.run_many][] method."""
        chats = await asyncio.gather(*[self.arun() for _ in range(count)])
        return list(chats)
