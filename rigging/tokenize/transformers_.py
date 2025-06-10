import importlib.util

if importlib.util.find_spec("transformers") is None:
    raise ModuleNotFoundError("Please install the `transformers` package to use this module.")

import typing as t

from transformers import AutoTokenizer

from rigging.tokenize.base import ChatFormatter, Decoder, Encoder

if t.TYPE_CHECKING:
    from rigging.chat import Chat


def make_tokenizers_from_transformers(
    model: str | t.Any,
    *,
    apply_chat_template_kwargs: dict[str, t.Any] | None = None,
    encode_kwargs: dict[str, t.Any] | None = None,
    decode_kwargs: dict[str, t.Any] | None = None,
    **tokenizer_kwargs: t.Any,
) -> tuple[ChatFormatter, Encoder, Decoder]:
    """
    Get the chat formatter, encoder, and decoder from transformers model
    identifier, or from an already loaded tokenizer.

    Examples:
        ```
        import rigging as rg
        from rigging.tokenize.transformers_ import make_tokenizers_from_transformers

        formatter, encoder, decoder = make_tokenizers_from_transformers("Qwen/Qwen3-8B")
        transform = rg.transform.tools_to_json_with_tag_transform

        tokenized = await rg.tokenize.tokenize_chat(
            chat,
            formatter,
            encoder,
            decoder,
            transform=transform
        )
        ```

    Args:
        model: The model identifier (string) or an already loaded tokenizer.
        apply_chat_template_kwargs: Additional keyword arguments for applying the chat template.
        encode_kwargs: Additional keyword arguments for encoding text.
        decode_kwargs: Additional keyword arguments for decoding tokens.
        tokenizer_kwargs: Additional keyword arguments for the tokenizer initialization.

    Returns:
        A tuple containing the chat formatter, encoder, and decoder.
    """
    if isinstance(model, str):
        tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
    else:
        tokenizer = model

    apply_chat_template_kwargs = {
        "tokenize": False,
        **(apply_chat_template_kwargs or {}),
    }
    encode_kwargs = {
        **(encode_kwargs or {}),
    }
    decode_kwargs = {
        "clean_up_tokenization_spaces": False,
        **(decode_kwargs or {}),
    }

    def chat_formatter(chat: "Chat") -> str:
        messages = [m.to_openai(compatibility_flags={"content_as_str"}) for m in chat.all]
        tools = chat.params.tools if chat.params else None
        return tokenizer.apply_chat_template(messages, tools=tools, **apply_chat_template_kwargs)  # type: ignore [no-any-return]

    def encoder(text: str) -> list[int]:
        return tokenizer.encode(text, **encode_kwargs)  # type: ignore [no-any-return]

    def decoder(tokens: list[int]) -> str:
        return tokenizer.decode(tokens, **decode_kwargs)  # type: ignore [no-any-return]

    return chat_formatter, encoder, decoder
