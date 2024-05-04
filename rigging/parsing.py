"""
Parsing helpers for extracting rigging models from text
"""

from rigging.error import MissingModelError
from rigging.model import ModelT


def parse(text: str, model_type: type[ModelT]) -> tuple[ModelT, slice]:
    """
    Parses a single model from text.

    Args:
        text (str): The content to parse.
        model_type (type): The type of model to parse.

    Returns:
        ModelT: The parsed model.

    Raises:
        ValueError: If no models of the given type are found and `fail_on_missing` is set to `True`.
    """
    return try_parse_many(text, model_type, fail_on_missing=True)[0]


def try_parse(text: str, model_type: type[ModelT]) -> tuple[ModelT, slice] | None:
    """
    Tries to parse a model from text.

    Args:
        text (str): The content to parse.
        model_type (type[ModelT]): The type of model to search for.

    Returns:
        ModelT | None: The first model that matches the given model type, or None if no match is found.
    """
    # for model in self.models:
    #     if isinstance(model, model_type):
    #         return model
    return next(iter(try_parse_many(text, model_type)), None)


def parse_set(text: str, model_type: type[ModelT], *, minimum: int | None = None) -> list[tuple[ModelT, slice]]:
    """
    Parses a set of models with the specified identical type from text.

    Args:
        text (str): The content to parse.
        model_type (type[ModelT]): The type of models to parse.
        minimum (int | None, optional): The minimum number of models required. Defaults to None.

    Returns:
        list[tuple[ModelT, slice]]: A list of parsed models.

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
        text (str): The content to parse.
        model_type (type[ModelT]): The type of model to parse.
        minimum (int | None, optional): The minimum number of models expected. Defaults to None.
        fail_on_missing (bool, optional): Whether to raise an exception if models are missing. Defaults to False.

    Returns:
        list[tuple[ModelT, slice]]: The parsed models.

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
        text (str): The content to parse.
        *types (type[ModelT]): The types of models to parse.

    Returns:
        list[tuple[ModelT, slice]]: A list of parsed models.

    Raises:
        MissingModelError: If any of the models are missing.
    """
    return try_parse_many(text, *types, fail_on_missing=True)


def try_parse_many(text: str, *types: type[ModelT], fail_on_missing: bool = False) -> list[tuple[ModelT, slice]]:
    """
    Tries to parses multiple models of the specified non-identical types from text.

    Args:
        text (str): The content to parse.
        *types (type[ModelT]): The types of models to parse.
        fail_on_missing (bool, optional): Whether to raise an exception if a model type is missing. Defaults to False.

    Returns:
        list[tuple[ModelT, slice]]: A list of parsed models.

    Raises:
        MissingModelError: If a model type is missing and `fail_on_missing` is True.
    """
    model: ModelT
    parsed: list[tuple[ModelT, slice]] = []
    for model_class in types:
        try:
            for model, slice_ in model_class.from_text(text):
                parsed.append((model, slice_))
        except MissingModelError as e:
            if fail_on_missing:
                raise e

    return parsed
