import importlib.util
import typing as t

if importlib.util.find_spec("transformers") is None:
    raise ModuleNotFoundError("Please install the `transformers` package to use this module.")


from transformers import AutoTokenizer

from rigging.tokenize.base import Decoder


def get_tokenizer(
    model: str | t.Any,
    **tokenizer_kwargs: t.Any,
) -> AutoTokenizer:
    """
    Get the tokenizer from transformers model identifier, or from an already loaded tokenizer.

    Args:
        model: The model identifier (string) or an already loaded tokenizer.
        tokenizer_kwargs: Additional keyword arguments for the tokenizer initialization.

    Returns:
        An instance of `AutoTokenizer`.
    """
    if isinstance(model, str):
        return AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
    return model


def find_in_tokens(
    target_text: str,
    tokens: list[int],
    decoder: Decoder,
    start_offset: int = 0,
    search_start: int = 0,
) -> tuple[int, int] | None:
    # End-based walk: find a window that contains our target text
    for end_pos in range(search_start + 1, len(tokens) + 1):
        decoded_window = decoder(tokens[search_start:end_pos])
        if target_text not in decoded_window:
            continue

        # Start-based walk: narrow down the start position
        actual_start = search_start
        for start_pos in range(search_start, end_pos):
            decoded_from_start = decoder(tokens[start_pos:end_pos])
            if decoded_from_start.startswith(target_text):
                actual_start = start_pos
                break

        return (start_offset + actual_start, start_offset + end_pos)

    return None
