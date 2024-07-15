"""
Parsing helpers for extracting rigging models from text
"""

from __future__ import annotations

import typing as t

from rigging.error import MissingModelError

if t.TYPE_CHECKING:
    from rigging.model import ModelT


def parse(text: str, model_type: type[ModelT]) -> tuple[ModelT, slice]:
    """
    Parses a single model from text.

    Args:
        text: The content to parse.
        model_type: The type of model to parse.

    Returns:
        The parsed model.

    Raises:
        ValueError: If no models of the given type are found and `fail_on_missing` is set to `True`.
    """
    return try_parse_many(text, model_type, fail_on_missing=True)[0]


def try_parse(text: str, model_type: type[ModelT]) -> tuple[ModelT, slice] | None:
    """
    Tries to parse a model from text.

    Args:
        text: The content to parse.
        model_type: The type of model to search for.

    Returns:
        The first model that matches the given model type, or None if no match is found.
    """
    return next(iter(try_parse_many(text, model_type)), None)


def parse_set(text: str, model_type: type[ModelT], *, minimum: int | None = None) -> list[tuple[ModelT, slice]]:
    """
    Parses a set of models with the specified identical type from text.

    Args:
        text: The content to parse.
        model_type: The type of models to parse.
        minimum: The minimum number of models required.

    Returns:
        A list of parsed models.

    Raises:
        MissingModelError: If the minimum number of models is not met.
    """
    return try_parse_set(text, model_type, minimum=minimum, fail_on_missing=True)


def try_parse_set(
    text: str, model_type: type[ModelT], *, minimum: int | None = None, fail_on_missing: bool = False
) -> list[tuple[ModelT, slice]]:
    """
    Tries to parse a set of models with the specified identical type from text.

    Args:
        text: The content to parse.
        model_type: The type of model to parse.
        minimum: The minimum number of models expected.
        fail_on_missing: Whether to raise an exception if models are missing.

    Returns:
        The parsed models.

    Raises:
        MissingModelError: If the number of parsed models is less than the minimum required.
    """
    models = try_parse_many(text, model_type, fail_on_missing=fail_on_missing)
    if minimum is not None and len(models) < minimum:
        raise MissingModelError(f"Expected at least {minimum} {model_type.__name__} in message")
    return models


def parse_many(text: str, *types: type[ModelT]) -> list[tuple[ModelT, slice]]:
    """
    Parses multiple models of the specified non-identical types from text.

    Args:
        text: The content to parse.
        *types: The types of models to parse.

    Returns:
        A list of parsed models.

    Raises:
        MissingModelError: If any of the models are missing.
    """
    return try_parse_many(text, *types, fail_on_missing=True)


def try_parse_many(text: str, *types: type[ModelT], fail_on_missing: bool = False) -> list[tuple[ModelT, slice]]:
    """
    Tries to parses multiple models of the specified non-identical types from text.

    Args:
        text: The content to parse.
        *types: The types of models to parse.
        fail_on_missing: Whether to raise an exception if a model type is missing.

    Returns:
        A list of parsed models.

    Raises:
        MissingModelError: If a model type is missing and `fail_on_missing` is True.
        Exception: If the model is malformed and `fail_on_missing` is True.
    """
    model: ModelT
    parsed: list[tuple[ModelT, slice]] = []
    for model_class in types:
        try:
            for model, slice_ in model_class.from_text(text):
                parsed.append((model, slice_))
        except Exception as e:
            if fail_on_missing:
                raise e

    return parsed
