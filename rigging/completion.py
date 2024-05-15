"""
Completions work with isolated strings of text pre and post generation.
"""

import asyncio
import string
import typing as t
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import runtime_checkable
from uuid import UUID, uuid4

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
)

from rigging.error import ExhaustedMaxRoundsError
from rigging.generator import GenerateParams, Generator, get_generator
from rigging.model import (
    Model,
    ModelT,
)
from rigging.parsing import parse_many

DEFAULT_MAX_ROUNDS = 5

# TODO: Chats and Completions share a lot of structure and code.
# Ideally we should build out a base class which they both inherit from.


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
    metadata: dict[str, t.Any] = Field(default_factory=dict)
    """Additional metadata for the completion."""

    generator: t.Optional["Generator"] = Field(None, exclude=True, repr=False)
    """The generator associated with the completion."""
    params: t.Optional["GenerateParams"] = Field(None, exclude=True, repr=False)
    """Any additional generation params used for this completion."""

    @computed_field(repr=False)  # type: ignore[misc]
    @property
    def generator_id(self) -> str | None:
        """The identifier of the generator used to create the completion"""
        if self.generator is not None:
            return self.generator.to_identifier(self.params)
        return None

    def __init__(
        self,
        text: str,
        generated: str,
        generator: t.Optional["Generator"] = None,
        **kwargs: t.Any,
    ):
        """
        Initialize a Completion object.

        Args:
            text: The original text.
            generated: The generated text.
            generator: The generator associated with this completion.
            **kwargs: Additional keyword arguments (typically used for serialization).
        """
        if "generator_id" in kwargs and generator is None:
            # TODO: Should we move params to self.params?
            generator = get_generator(kwargs.pop("generator_id"))

        super().__init__(
            text=text,
            generated=generated,
            generator=generator,
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
            generator: The generator to use for the restarted completion. Otherwise
                the generator from the original PendingCompletion will be used.
            include_all: Whether to include the generation before the next round.

        Returns:
            The restarted completion.

        Raises:
            ValueError: If the completion was not created with a PendingCompletion and no generator is provided.
        """

        text = self.all if include_all else self.generated
        if generator is None:
            generator = self.generator
        if generator is None:
            raise ValueError("Cannot restart a completion without an associated generator")
        return generator.complete(text, self.params)

    def fork(self, text: str, *, include_all: bool = False) -> "PendingCompletion":
        """
        Forks the completion by creating calling [rigging.completion.Completion.restart][] and appends the specified text.

        Args:
            text: The text to append.

        Returns:
            A new instance of a pending competion with the specified messages added.
        """
        return self.restart(include_all=include_all).add(text)

    def continue_(self, text: str) -> "PendingCompletion":
        """Alias for the [rigging.completion.Completion.fork][] with `include_all=True`."""
        return self.fork(text, include_all=True)

    def clone(self, *, only_messages: bool = False) -> "Completion":
        """Creates a deep copy of the completion."""
        new = Completion(self.text, self.generated, self.generator)
        if not only_messages:
            new.metadata = deepcopy(self.metadata)
        return new

    def meta(self, **kwargs: t.Any) -> "Completion":
        """
        Updates the metadata of the completion with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated completion object.
        """
        new = self.clone()
        new.metadata.update(kwargs)
        return new


# Callbacks


@runtime_checkable
class UntilCompletionCallback(t.Protocol):
    def __call__(self, text: str) -> bool:
        """
        A callback function that takes the generated text and returns whether or not to retry generation.
        """
        ...


@runtime_checkable
class ThenCompletionCallback(t.Protocol):
    def __call__(self, completion: Completion) -> Completion | None:
        """
        Passed a finalized completion to process and can return a new completion to replace it.
        """
        ...


@runtime_checkable
class AsyncThenCompletionCallback(t.Protocol):
    async def __call__(self, completion: Completion) -> Completion | None:
        """
        async variant of the [rigging.completion.ThenCompletionCallback][] protocol.
        """
        ...


@runtime_checkable
class MapCompletionCallback(t.Protocol):
    def __call__(self, completions: list[Completion]) -> list[Completion]:
        """
        Passed a finalized completion to process. Can replace completions in the pipeline by returning
        a new completion object.
        """
        ...


@runtime_checkable
class AsyncMapCompletionCallback(t.Protocol):
    async def __call__(self, completions: list[Completion]) -> list[Completion]:
        """
        async variant of the [rigging.completion.MapCompletionCallback][] protocol.
        """
        ...


ThenCompletionCallbacks = ThenCompletionCallback | AsyncThenCompletionCallback
MapCompletionCallbacks = MapCompletionCallback | AsyncMapCompletionCallback


@dataclass
class RunState:
    text: str
    params: "GenerateParams"
    processor: t.Generator[None, str, str]
    completion: Completion | None = None
    done: bool = False


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
        self.then_callbacks: list[ThenCompletionCallbacks] = []
        self.map_callbacks: list[MapCompletionCallbacks] = []

    def __len__(self) -> int:
        return len(self.text)

    def with_(self, params: t.Optional["GenerateParams"] = None, **kwargs: t.Any) -> "PendingCompletion":
        """
        Assign specific generation parameter overloads for this completion.

        Note:
            This will trigger a `clone` if overload params have already been set.

        Args:
            params: The parameters to set for the completion.
            **kwargs: An alternative way to pass parameters as keyword arguments.

        Returns:
            The current (or cloned) instance of the completion.
        """
        if params is None:
            params = GenerateParams(**kwargs)

        if self.params is not None:
            new = self.clone()
            new.params = self.params.merge_with(params)
            return new

        self.params = params
        return self

    def then(self, callback: ThenCompletionCallback) -> "PendingCompletion":
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            Returning a Completion object from the callback will replace the current completion.
            for the remainder of the callbacks + return value of `run()`.

        ```
        def process(completion: Completion) -> Completion | None:
            ...

        pending.then(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the pending completion.
        """
        self.then_callbacks.append(callback)
        return self

    def map(self, callback: MapCompletionCallback | AsyncMapCompletionCallback) -> "PendingCompletion":
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            You must return a list of completion objects from the callback which will
            represent the state of completions for the remainder of the callbacks and return.

        Warning:
            If you implement an async callback, you must use the async variant of the
            run methods when executing the generation process.

        ```
        def process(completions: list[Completion]) -> list[Completion]:
            ...

        pending.map(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the completion.
        """
        self.map_callbacks.append(callback)
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

        Note:
            This produces a clone of the PendingCompletion, leaving the original unchanged.

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

        ```py
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

    def _post_run(self, completions: list[Completion]) -> list[Completion]:
        for map_callback in self.map_callbacks:
            if asyncio.iscoroutinefunction(map_callback):
                raise ValueError(
                    f"Cannot use async map() callbacks inside a non-async run call: {map_callback.__name__}"
                )
            completions = map_callback(completions)  # type: ignore

        for then_callback in self.then_callbacks:
            if asyncio.iscoroutinefunction(then_callback):
                raise ValueError(
                    f"Cannot use async then() callbacks inside a non-async run call: {then_callback.__name__}"
                )
            updated = [then_callback(completion) for completion in completions]
            completions = [updated[i] or completion for i, completion in enumerate(completions)]  # type: ignore

        return completions

    async def _apost_run(self, completions: list[Completion]) -> list[Completion]:
        for map_callback in self.map_callbacks:
            if not asyncio.iscoroutinefunction(map_callback):
                raise ValueError(
                    f"Cannot use non-async map() callbacks inside an async run call: {map_callback.__call__.__name__}"
                )
            completions = await map_callback(completions)

        for then_callback in self.then_callbacks:
            if not asyncio.iscoroutinefunction(then_callback):
                raise ValueError(
                    f"Cannot use non-async then() callbacks inside an async run call: {then_callback.__call__.__name__}"
                )
            updated = [await then_callback(completion) for completion in completions]
            completions = [updated[i] or completion for i, completion in enumerate(completions)]

        return completions

    def _fit_params(
        self, count: int, params: t.Sequence[t.Optional["GenerateParams"] | None] | None = None
    ) -> list["GenerateParams"]:
        params = [None] * count if params is None else list(params)
        if len(params) != count:
            raise ValueError(f"The number of params must be {count}")
        if self.params is not None:
            params = [self.params.merge_with(p) for p in params]
        return [(p or GenerateParams()) for p in params]

    # TODO: It's opaque exactly how we should blend multiple
    # until callbacks together, so here is the current implementation:
    #
    # - We take the lowest max_rounds from all until_callbacks
    # - Each loop, we let every callback run, if any tell us to retry, we do
    # - If we leave the loop with should_retry still True, we raise an error
    # - Assuming every should_retry is False, we break out of the loop and return

    def _process(self) -> t.Generator[None, str, str]:
        # If there are no until_callbacks, we can just yield the text
        if not self.until_callbacks:
            generated = yield
            return generated

        lowest_max_rounds = min((c[2] for c in self.until_callbacks), default=1)

        current_round = 0
        should_retry = True
        while should_retry and current_round < lowest_max_rounds:
            current_round += 1
            generated = yield
            for callback, use_all_text, _ in self.until_callbacks:
                should_retry = callback(self.text + generated if use_all_text else generated)
                if should_retry:
                    continue

        if should_retry:
            logger.warning(f"Exhausted lowest max rounds ({lowest_max_rounds})")
            raise ExhaustedMaxRoundsError(lowest_max_rounds)

        return generated

    def run(self) -> Completion:
        """
        Execute the generation process to produce the final completion.

        Returns:
            The generated Completion.
        """
        return self.run_many(1)[0]

    async def arun(self) -> Completion:
        """async variant of the [rigging.completion.PendingCompletion.run][] method."""
        return (await self.arun_many(1))[0]

    __call__ = run

    # Many messages

    def run_many(
        self,
        count: int,
        *,
        params: t.Sequence[t.Optional["GenerateParams"]] | None = None,
        skip_failed: bool = False,
    ) -> list[Completion]:
        """
        Executes the generation process multiple times with the same inputs.

        Parameters:
            count: The number of times to execute the generation process.
            params: A sequence of parameters to be used for each execution.
            skip_failed: Enable to ignore any max rounds errors and return only successful completions.

        Returns:
            A list of generatated Completions.
        """
        states: list[RunState] = [RunState(self.text, p, self._process()) for p in self._fit_params(count, params)]
        _ = [next(state.processor) for state in states]

        pending_states = states
        while pending_states:
            inbounds = self.generator.generate_texts(
                [s.text for s in pending_states], [s.params for s in pending_states]
            )

            for inbound, state in zip(inbounds, pending_states, strict=True):
                try:
                    state.processor.send(inbound)
                except StopIteration as stop:
                    state.done = True
                    state.completion = Completion(
                        self.text,
                        t.cast(str, stop.value),
                        generator=self.generator,
                        params=state.params,
                        metadata=self.metadata,
                    )
                except ExhaustedMaxRoundsError:
                    if not skip_failed:
                        raise
                    state.done = True

            pending_states = [s for s in pending_states if not s.done]

        return self._post_run([s.completion for s in states if s.completion is not None])

    async def arun_many(
        self,
        count: int,
        *,
        params: t.Sequence[t.Optional["GenerateParams"]] | None = None,
        skip_failed: bool = False,
    ) -> list[Completion]:
        """async variant of the [rigging.completion.PendingCompletion.run_many][] method."""
        states: list[RunState] = [RunState(self.text, p, self._process()) for p in self._fit_params(count, params)]
        _ = [next(state.processor) for state in states]

        pending_states = states
        while pending_states:
            inbounds = await self.generator.agenerate_texts(
                [s.text for s in pending_states], [s.params for s in pending_states]
            )

            for inbound, state in zip(inbounds, pending_states, strict=True):
                try:
                    state.processor.send(inbound)
                except StopIteration as stop:
                    state.done = True
                    state.completion = Completion(
                        self.text,
                        t.cast(str, stop.value),
                        generator=self.generator,
                        params=state.params,
                        metadata=self.metadata,
                    )
                except ExhaustedMaxRoundsError:
                    if not skip_failed:
                        raise
                    state.done = True

            pending_states = [s for s in pending_states if not s.done]

        return self._post_run([s.completion for s in states if s.completion is not None])

    # Batch completions

    def run_batch(
        self,
        many: t.Sequence[str],
        params: t.Sequence[t.Optional["GenerateParams"]] | None = None,
        *,
        skip_failed: bool = False,
    ) -> list[Completion]:
        """
        Executes the generation process accross multiple input messages.

        Note:
            Anything already in this pending completion will be used as the `prefix` parameter
            to [rigging.generator.Generator.generate_messages][].

        Parameters:
            many: A sequence of texts to generate with.
            params: A sequence of parameters to be used for each text.
            skip_failed: Enable to ignore any max rounds errors and return only successful completions.

        Returns:
            A list of generatated Completions.
        """
        params = self._fit_params(len(many), params)
        states: list[RunState] = [RunState(m, p, self._process()) for m, p in zip(many, params, strict=True)]
        _ = [next(state.processor) for state in states]

        pending_states = states
        while pending_states:
            inbounds = self.generator.generate_texts(
                [s.text for s in pending_states],
                [s.params for s in pending_states],
                prefix=self.text,
            )

            for inbound, state in zip(inbounds, pending_states, strict=True):
                try:
                    state.processor.send(inbound)
                except StopIteration as stop:
                    state.done = True
                    state.completion = Completion(
                        self.text,
                        t.cast(str, stop.value),
                        generator=self.generator,
                        params=state.params,
                        metadata=self.metadata,
                    )
                except ExhaustedMaxRoundsError:
                    if not skip_failed:
                        raise
                    state.done = True

            pending_states = [s for s in pending_states if not s.done]

        return self._post_run([s.completion for s in states if s.completion is not None])

    async def arun_batch(
        self,
        many: t.Sequence[str],
        params: t.Sequence[t.Optional["GenerateParams"]] | None = None,
        *,
        skip_failed: bool = False,
    ) -> list[Completion]:
        """async variant of the [rigging.completion.PendingCompletion.run_batch][] method."""
        params = self._fit_params(len(many), params)
        states: list[RunState] = [RunState(m, p, self._process()) for m, p in zip(many, params, strict=True)]
        _ = [next(state.processor) for state in states]

        pending_states = states
        while pending_states:
            inbounds = await self.generator.agenerate_texts(
                [s.text for s in pending_states],
                [s.params for s in pending_states],
                prefix=self.text,
            )

            for inbound, state in zip(inbounds, pending_states, strict=True):
                try:
                    state.processor.send(inbound)
                except StopIteration as stop:
                    state.done = True
                    state.completion = Completion(
                        self.text,
                        t.cast(str, stop.value),
                        generator=self.generator,
                        params=state.params,
                        metadata=self.metadata,
                    )
                except ExhaustedMaxRoundsError:
                    if not skip_failed:
                        raise
                    state.done = True

            pending_states = [s for s in pending_states if not s.done]

        return self._post_run([s.completion for s in states if s.completion is not None])
