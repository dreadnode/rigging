"""
Completions work with isolated strings of text pre and post generation.
"""

from __future__ import annotations

import asyncio
import string
import typing as t
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import runtime_checkable
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field

from rigging.error import CompletionExhaustedMaxRoundsError
from rigging.generator import GenerateParams, Generator, get_generator
from rigging.generator.base import GeneratedText, StopReason, Usage  # noqa: TCH001
from rigging.parsing import parse_many
from rigging.tracing import Span, tracer
from rigging.util import get_qualified_name

if t.TYPE_CHECKING:
    from rigging.chat import FailMode
    from rigging.model import Model, ModelT

CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])


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

    stop_reason: StopReason = Field(default="unknown")
    """The reason the generation stopped."""
    usage: t.Optional[Usage] = Field(None, repr=False)
    """The usage statistics for the generation if available."""
    extra: dict[str, t.Any] = Field(default_factory=dict, repr=False)
    """Any additional information from the generation."""

    generator: t.Optional[Generator] = Field(None, exclude=True, repr=False)
    """The generator associated with the completion."""
    params: t.Optional[GenerateParams] = Field(None, exclude=True, repr=False)
    """Any additional generation params used for this completion."""

    error: t.Optional[Exception] = Field(None, exclude=True, repr=False)
    """Holds any exception that was caught during the generation pipeline."""
    failed: bool = Field(False, exclude=False, repr=False)
    """
    Indicates whether conditions during generation were not met.
    This is typically used for graceful error handling when parsing.
    """

    @computed_field(repr=False)  # type: ignore [prop-decorator]
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
        generator: t.Optional[Generator] = None,
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

    def restart(self, *, generator: t.Optional[Generator] = None, include_all: bool = False) -> CompletionPipeline:
        """
        Attempt to convert back to a CompletionPipeline for further generation.

        Args:
            generator: The generator to use for the restarted completion. Otherwise
                the generator from the original CompletionPipeline will be used.
            include_all: Whether to include the generation before the next round.
        Returns:
            The restarted completion.

        Raises:
            ValueError: If the completion was not created with a CompletionPipeline and no generator is provided.
        """

        text = self.all if include_all else self.generated
        if generator is None:
            generator = self.generator
        if generator is None:
            raise ValueError("Cannot restart a completion without an associated generator")
        return generator.complete(text, self.params)

    def fork(self, text: str, *, include_all: bool = False) -> CompletionPipeline:
        """
        Forks the completion by creating calling [rigging.completion.Completion.restart][] and appends the specified text.

        Args:
            text: The text to append.

        Returns:
            A new instance of the pipeline with the specified messages added.
        """
        return self.restart(include_all=include_all).add(text)

    def continue_(self, text: str) -> CompletionPipeline:
        """Alias for the [rigging.completion.Completion.fork][] with `include_all=True`."""
        return self.fork(text, include_all=True)

    def clone(self, *, only_messages: bool = False) -> Completion:
        """Creates a deep copy of the completion."""
        new = Completion(self.text, self.generated, self.generator)
        if not only_messages:
            new.metadata = deepcopy(self.metadata)
            new.stop_reason = self.stop_reason
            new.usage = self.usage.model_copy() if self.usage is not None else self.usage
            new.extra = deepcopy(self.extra)
            new.params = self.params.model_copy() if self.params is not None else self.params
            new.failed = self.failed
        return new

    def meta(self, **kwargs: t.Any) -> Completion:
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
    def __call__(self, text: str, /) -> bool:
        """
        A callback function that takes the generated text and returns whether or not to retry generation.
        """
        ...


@runtime_checkable
class ThenCompletionCallback(t.Protocol):
    def __call__(self, completion: Completion, /) -> t.Awaitable[Completion | None]:
        """
        Passed a finalized completion to process and can return a new completion to replace it.
        """
        ...


@runtime_checkable
class MapCompletionCallback(t.Protocol):
    def __call__(self, completions: list[Completion], /) -> t.Awaitable[list[Completion]]:
        """
        Passed a finalized completion to process.

        This callback can replace, remove, or extend completions
        in the pipeline.
        """
        ...


@runtime_checkable
class WatchCompletionCallback(t.Protocol):
    def __call__(self, completions: list[Completion], /) -> t.Awaitable[None]:
        """
        Passed any created completion objects for monitoring/logging.
        """
        ...


@dataclass
class RunState:
    text: str
    params: GenerateParams
    processor: t.Generator[None, str, str]
    completion: Completion | None = None
    watched: bool = False


class CompletionPipeline:
    """
    Pipeline to manipulate and produce completions.
    """

    def __init__(
        self,
        generator: Generator,
        text: str,
        *,
        params: t.Optional[GenerateParams] = None,
        watch_callbacks: t.Optional[list[WatchCompletionCallback]] = None,
    ):
        self.generator: Generator = generator
        """The generator object responsible for generating the completion."""
        self.text = text
        """The text to be completed."""
        self.params = params
        """The parameters for generating the completion."""
        self.metadata: dict[str, t.Any] = {}
        """Additional metadata associated with the completion."""
        self.errors_to_fail_on: set[type[Exception]] = set()
        """
        The list of exceptions to catch during generation if you are including or skipping failures.

        ExhuastedMaxRounds is implicitly included.
        """
        self.on_failed: FailMode = "raise"
        """How to handle failures in the pipeline unless overriden in calls."""

        # (callback, all_text, max_rounds)
        self.until_callbacks: list[tuple[UntilCompletionCallback, bool, int]] = []
        self.until_types: list[type[Model]] = []
        self.then_callbacks: list[ThenCompletionCallback] = []
        self.map_callbacks: list[MapCompletionCallback] = []
        self.watch_callbacks: list[WatchCompletionCallback] = watch_callbacks or []

    def __len__(self) -> int:
        return len(self.text)

    def with_(self, params: t.Optional[GenerateParams] = None, **kwargs: t.Any) -> CompletionPipeline:
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

    def catch(self, *errors: type[Exception], on_failed: FailMode | None = None) -> CompletionPipeline:
        """
        Adds exceptions to catch during generation when including or skipping failures.

        Args:
            *errors: The exception types to catch.
            on_failed: How to handle failures in the pipeline unless overriden in calls.

        Returns:
            The updated CompletionPipeline object.
        """
        self.errors_to_fail_on.update(errors)
        self.on_failed = on_failed or self.on_failed
        return self

    def watch(self, *callbacks: WatchCompletionCallback, allow_duplicates: bool = False) -> CompletionPipeline:
        """
        Registers a callback to monitor any completions produced.

        Args:
            *callbacks: The callback functions to be executed.
            allow_duplicates: Whether to allow (seemingly) duplicate callbacks to be added.

        ```
        async def log(completions: list[Completion]) -> None:
            ...

        await pipeline.watch(log).run()
        ```

        Returns:
            The current instance.
        """
        for callback in callbacks:
            if allow_duplicates or callback not in self.watch_callbacks:
                self.watch_callbacks.append(callback)
        return self

    def then(self, callback: ThenCompletionCallback) -> CompletionPipeline:
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            Returning a Completion object from the callback will replace the current completion.
            for the remainder of the callbacks + return value of `run()`.

        ```
        async def process(completion: Completion) -> Completion | None:
            ...

        await pipeline.then(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the pipeline.
        """
        self.then_callbacks.append(callback)
        return self

    def map(self, callback: MapCompletionCallback) -> CompletionPipeline:
        """
        Registers a callback to be executed after the generation process completes.

        Note:
            You must return a list of completion objects from the callback which will
            represent the state of completions for the remainder of the callbacks and return.

        ```
        async def process(completions: list[Completion]) -> list[Completion]:
            ...

        await pipeline.map(process).run()
        ```

        Args:
            callback: The callback function to be executed.

        Returns:
            The current instance of the completion.
        """
        self.map_callbacks.append(callback)
        return self

    def add(self, text: str) -> CompletionPipeline:
        """
        Appends new text to the internal text before generation.

        Args:
            text: The text to be added to the completion.

        Returns:
            The updated CompletionPipeline object.
        """
        self.text += text
        return self

    def fork(self, text: str) -> CompletionPipeline:
        """
        Creates a new instance of `CompletionPipeline` by forking the current completion and adding the specified text.

        This is a convenience method for calling `clone().add(text)`.

        Args:
            text: The text to be added to the new completion.

        Returns:
            A new instance of `CompletionPipeline` with the specified text added.
        """
        return self.clone().add(text)

    def clone(self, *, only_text: bool = False) -> CompletionPipeline:
        """
        Creates a clone of the current `CompletionPipeline` instance.

        Args:
            only_text: If True, only the text will be cloned.
                If False (default), the entire `CompletionPipeline` instance will be cloned
                including until callbacks, types, and metadata.

        Returns:
            A new instance of `CompletionPipeline` that is a clone of the current instance.
        """
        new = CompletionPipeline(
            self.generator,
            self.text,
            params=self.params.model_copy() if self.params is not None else None,
            watch_callbacks=self.watch_callbacks,
        )
        if not only_text:
            new.until_callbacks = self.until_callbacks.copy()
            new.until_types = self.until_types.copy()
            new.metadata = deepcopy(self.metadata)
            new.then_callbacks = self.then_callbacks.copy()
            new.map_callbacks = self.map_callbacks.copy()
        return new

    def meta(self, **kwargs: t.Any) -> CompletionPipeline:
        """
        Updates the metadata of the completion with the provided key-value pairs.

        Args:
            **kwargs: Key-value pairs representing the metadata to be updated.

        Returns:
            The updated completion object.
        """
        self.metadata.update(kwargs)
        return self

    def apply(self, **kwargs: str) -> CompletionPipeline:
        """
        Applies keyword arguments to the text using string template substitution.

        Note:
            This produces a clone of the CompletionPipeline, leaving the original unchanged.

        Args:
            **kwargs: Keyword arguments to be applied to the text.

        Returns:
            A new instance of CompletionPipeline with the applied arguments.
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
    ) -> CompletionPipeline:
        """
        Registers a callback to participate in validating the generation process.

        ```py
        # Takes the generated text, and returns whether or not to retry generation.

        def callback(text: str) -> bool:
            if is_valid(text):
                return False
            else:
                return True

        await pipeline.until(callback).run()
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
    ) -> CompletionPipeline:
        """
        Adds the specified types to the list of types which should successfully parse
        before the generation process completes.

        Args:
            *types: The type or types of models to wait for.
            use_all_text: Whether to pass the entire text (including prompt) to the parser.
            max_rounds: The maximum number of rounds to try to parse successfully.

        Returns:
            The updated CompletionPipeline object.
        """
        self.until_types += types
        if next((c for c in self.until_callbacks if c[0] == self._until_parse_callback), None) is None:
            self.until_callbacks.append((self._until_parse_callback, use_all_text, max_rounds))

        return self

    def wrap(self, func: t.Callable[[CallableT], CallableT]) -> CompletionPipeline:
        """
        Helper for [rigging.generator.base.Generator.wrap][].

        Args:
            func: The function to wrap the calls with.

        Returns:
            The current instance of the pipeline.
        """
        self.generator = self.generator.wrap(func)
        return self

    def _until_parse_callback(self, text: str) -> bool:
        try:
            parse_many(text, *self.until_types)
        except Exception:
            return True
        return False

    async def _watch_callback(self, completions: list[Completion]) -> None:
        def wrap_watch_callback(callback: WatchCompletionCallback) -> WatchCompletionCallback:
            async def traced_watch_callback(completions: list[Completion]) -> None:
                callback_name = get_qualified_name(callback)
                with tracer.span(
                    f"Watch with {callback_name}()",
                    callback=callback_name,
                    competion_count=len(completions),
                    completion_ids=[str(c.uuid) for c in completions],
                ):
                    await callback(completions)

            return traced_watch_callback

        coros = [wrap_watch_callback(callback)(completions) for callback in self.watch_callbacks]
        await asyncio.gather(*coros)

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
            raise CompletionExhaustedMaxRoundsError(lowest_max_rounds, generated)

        return generated

    async def _post_run(self, completions: list[Completion], on_failed: FailMode) -> list[Completion]:
        if on_failed == "skip":
            completions = [c for c in completions if not c.failed]

        # These have to be sequenced to support the concept of
        # a pipeline where future then/map calls can depend on
        # previous calls being ran.

        for map_callback in self.map_callbacks:
            callback_name = get_qualified_name(map_callback)
            with tracer.span(
                f"Map with {callback_name}()",
                callback=callback_name,
                completion_count=len(completions),
                completion_ids=[str(c.uuid) for c in completions],
            ):
                completions = await map_callback(completions)
                if not all(isinstance(c, Completion) for c in completions):
                    raise ValueError(f".map() callback must return a Completion object or None ({callback_name})")

        def wrap_then_callback(callback: ThenCompletionCallback) -> ThenCompletionCallback:
            callback_name = get_qualified_name(callback)

            async def traced_then_callback(completion: Completion) -> Completion | None:
                with tracer.span(
                    f"Then with {callback_name}()", callback=callback_name, completion_id=str(completion.uuid)
                ):
                    return await callback(completion)

            return traced_then_callback

        for then_callback in self.then_callbacks:
            coros = [wrap_then_callback(then_callback)(completion) for completion in completions]
            new_completions = await asyncio.gather(*coros)
            if not all(isinstance(c, Completion) or c is None for c in new_completions):
                raise ValueError(
                    f".then() callback must return a Completion object or None ({get_qualified_name(then_callback)})"
                )

            completions = [new or completion for new, completion in zip(new_completions, completions)]

        return completions

    def _create_completion(
        self, state: RunState, output: str, inbound: GeneratedText, failed: bool = False, error: Exception | None = None
    ) -> Completion:
        return Completion(
            self.text,
            output,
            generator=self.generator,
            params=state.params,
            metadata=self.metadata,
            stop_reason=inbound.stop_reason,
            usage=inbound.usage,
            extra=inbound.extra,
            failed=failed,
            error=error,
        )

    def _create_failed_completion(self, state: RunState, error: Exception) -> Completion:
        return Completion(
            state.text,
            "",
            generator=self.generator,
            params=state.params,
            metadata=self.metadata,
            failed=True,
            error=error,
        )

    def _fit_params(
        self, count: int, params: t.Sequence[t.Optional[GenerateParams] | None] | None = None
    ) -> list[GenerateParams]:
        params = [None] * count if params is None else list(params)
        if len(params) != count:
            raise ValueError(f"The number of params must be {count}")
        if self.params is not None:
            params = [self.params.merge_with(p) for p in params]
        return [(p or GenerateParams()) for p in params]

    def _initialize_states(
        self, count: int, params: t.Sequence[t.Optional[GenerateParams]] | None = None
    ) -> list[RunState]:
        states = [RunState(self.text, p, self._process()) for p in self._fit_params(count, params)]
        for state in states:
            next(state.processor)
        return states

    async def _run(
        self, span: Span, states: list[RunState], on_failed: FailMode, batch_mode: bool = False
    ) -> list[Completion]:
        pending_states = states
        while pending_states:
            try:
                inbounds = await self.generator.generate_texts(
                    [(self.text + s.text) if batch_mode else s.text for s in pending_states],
                    [s.params for s in pending_states],
                )
            except Exception as e:
                if on_failed == "raise" or not any(isinstance(e, t) for t in self.errors_to_fail_on):
                    raise

                span.set_attribute("failed", True)
                span.set_attribute("error", e)

                for state in pending_states:
                    state.completion = self._create_failed_completion(state, e)
            else:
                for inbound, state in zip(inbounds, pending_states):
                    output: str = ""
                    failed: bool = False
                    error: Exception | None = None

                    try:
                        state.processor.send(inbound.text)
                        continue
                    except StopIteration as stop:
                        output = t.cast(str, stop.value)
                    except CompletionExhaustedMaxRoundsError as exhausted:
                        if on_failed == "raise":
                            raise
                        output = exhausted.completion
                        failed = True
                        error = exhausted
                    except Exception as e:
                        if on_failed == "raise" or not any(isinstance(e, t) for t in self.errors_to_fail_on):
                            raise
                        failed = True
                        error = e

                    if error is not None:
                        span.set_attribute("failed", True)
                        span.set_attribute("error", error)

                    state.completion = self._create_completion(state, output, inbound, failed, error)

            pending_states = [s for s in pending_states if s.completion is None]
            to_watch_states = [s for s in states if s.completion is not None and not s.watched]

            await self._watch_callback([s.completion for s in to_watch_states if s.completion is not None])

            for state in to_watch_states:
                state.watched = True

        completions = await self._post_run([s.completion for s in states if s.completion is not None], on_failed)
        span.set_attribute("completions", completions)
        return completions

    async def run(self, *, allow_failed: bool = False, on_failed: FailMode | None = None) -> Completion:
        """
        Execute the generation process to produce the final chat.

        Parameters:
            allow_failed: Ignore any errors and potentially
                return the chat in a failed state.
            on_failed: The behavior when a message fails to generate.
                (this is used as an alternative to allow_failed)

        Returns:
            The generated Completion.
        """
        if on_failed is None:
            on_failed = "include" if allow_failed else self.on_failed

        if on_failed == "skip":
            raise ValueError(
                "Cannot use 'skip' mode with single completion generation (pass allow_failed=True or on_failed='include'/'raise')"
            )

        on_failed = on_failed or self.on_failed
        states = self._initialize_states(1)

        with tracer.span(
            f"Completion with {self.generator.to_identifier()}",
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            return (await self._run(span, states, on_failed))[0]

    __call__ = run

    # Many messages

    async def run_many(
        self,
        count: int,
        *,
        params: t.Sequence[t.Optional[GenerateParams]] | None = None,
        on_failed: FailMode | None = None,
    ) -> list[Completion]:
        """
        Executes the generation process multiple times with the same inputs.

        Parameters:
            count: The number of times to execute the generation process.
            params: A sequence of parameters to be used for each execution.
            on_failed: How to handle failures in the pipeline unless overriden in calls.

        Returns:
            A list of generatated Completions.
        """
        on_failed = on_failed or self.on_failed
        states = self._initialize_states(count, params)

        with tracer.span(
            f"Completion with {self.generator.to_identifier()} (x{count})",
            count=count,
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            return await self._run(span, states, on_failed)

    # Batch completions

    async def run_batch(
        self,
        many: t.Sequence[str],
        params: t.Sequence[t.Optional[GenerateParams]] | None = None,
        *,
        on_failed: FailMode = "raise",
    ) -> list[Completion]:
        """
        Executes the generation process accross multiple input messages.

        Note:
            Anything already in this pending completion will be prepended to the text.

        Parameters:
            many: A sequence of texts to generate with.
            params: A sequence of parameters to be used for each text.
            on_failed: How to handle failures in the pipeline unless overriden in calls.

        Returns:
            A list of generatated Completions.
        """
        on_failed = on_failed or self.on_failed
        params = self._fit_params(len(many), params)

        states: list[RunState] = [RunState(m, p, self._process()) for m, p in zip(many, params)]
        for state in states:
            next(state.processor)

        with tracer.span(
            f"Completion batch with {self.generator.to_identifier()} ({len(states)})",
            count=len(states),
            generator_id=self.generator.to_identifier(),
            params=self.params.to_dict() if self.params is not None else {},
        ) as span:
            return await self._run(span, states, on_failed, batch_mode=True)

    # Generator iteration

    async def run_over(
        self, *generators: Generator | str, include_original: bool = True, on_failed: FailMode | None = None
    ) -> list[Completion]:
        """
        Executes the generation process across multiple generators.

        For each generator, this pipeline is cloned and the generator is replaced
        before the run call. All callbacks and parameters are preserved.

        Parameters:
            *generators: A sequence of generators to be used for the generation process.
            include_original: Whether to include the original generator in the list of runs.
            on_failed: The behavior when a message fails to generate.

        Returns:
            A list of generatated Completions.
        """
        on_failed = on_failed or self.on_failed

        _generators: list[Generator] = [g if isinstance(g, Generator) else get_generator(g) for g in generators]
        if include_original:
            _generators.append(self.generator)

        coros: list[t.Coroutine[t.Any, t.Any, Completion]] = []
        for generator in _generators:
            sub = self.clone()
            sub.generator = generator
            coros.append(sub.run(allow_failed=(on_failed != "raise")))

        with tracer.span(f"Completion over {len(coros)} generators", count=len(coros)):
            completions = await asyncio.gather(*coros)
            return await self._post_run(completions, on_failed)
