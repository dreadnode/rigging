import typing as t

from pydantic import Field
from transformers import AutoTokenizer

from rigging.tokenizer.base import Tokenizer

if t.TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from rigging.chat import Chat


class TransformersTokenizer(Tokenizer):
    """
    A tokenizer implementation using Hugging Face Transformers.

    This class provides tokenization capabilities for chat conversations
    using transformers models and their associated tokenizers.
    """

    apply_chat_template_kwargs: dict[str, t.Any] = Field(default_factory=dict)
    """Additional keyword arguments for applying the chat template."""

    encode_kwargs: dict[str, t.Any] = Field(default_factory=dict)
    """Additional keyword arguments for encoding text."""

    decode_kwargs: dict[str, t.Any] = Field(default_factory=dict)
    """Additional keyword arguments for decoding tokens."""

    _tokenizer: "PreTrainedTokenizer | None" = None

    @property
    def tokenizer(self) -> "PreTrainedTokenizer":
        """The underlying `PreTrainedTokenizer` instance."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self._tokenizer

    @classmethod
    def from_obj(cls, tokenizer: "PreTrainedTokenizer") -> "TransformersTokenizer":
        """
        Create a new instance of TransformersTokenizer from an already loaded tokenizer.

        Args:
            tokenizer: The tokenizer associated with the model.

        Returns:
            The TransformersTokenizer instance.
        """
        return cls(model=str(tokenizer), _tokenizer=tokenizer)

    def encode(self, text: str) -> list[int]:
        """
        Encodes the given text into a list of tokens.

        Args:
            text: The text to encode.

        Returns:
            A list of tokens representing the encoded text.
        """
        return self.tokenizer.encode(text, **self.encode_kwargs)  # type: ignore [no-any-return]

    def decode(self, tokens: list[int]) -> str:
        decode_kwargs = {
            "clean_up_tokenization_spaces": False,
            **self.decode_kwargs,
        }
        return self.tokenizer.decode(tokens, **decode_kwargs)

    def format_chat(self, chat: "Chat") -> str:
        messages = [m.to_openai(compatibility_flags={"content_as_str"}) for m in chat.all]
        tools = (
            [tool.model_dump() for tool in chat.params.tools]
            if chat.params and chat.params.tools
            else None
        )

        apply_chat_template_kwargs = {
            "tokenize": False,
            **self.apply_chat_template_kwargs,
        }

        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tools=tools,  # type: ignore [arg-type]
                **apply_chat_template_kwargs,
            ),
        )
