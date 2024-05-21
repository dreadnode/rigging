import typing as t

from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TextGenerationPipeline,
    pipeline,
)

from rigging.generator.base import GenerateParams, Generator, register_generator, trace_messages, trace_str
from rigging.message import Message

DEFAULT_MAX_TOKENS = 1024
"""Lifting the default max tokens from transformers"""


class TransformersGenerator(Generator):
    """
    Generator backed by the Transformers library for local model loading.

    Warning:
        The use of Transformers requires the `transformers` package to be installed directly or by
        installing rigging as `rigging[all]`.

    Warning:
        The `transformers` library is expansive with many different models, tokenizers,
        options, constructors, etc. We do our best to implement a consistent interface,
        but there may be limitations. Where needed, use
        [`.from_obj()`][rigging.generator.transformers_.TransformersGenerator.from_obj].

    Note:
        The async methods currently just call synchronous variants for compatibility.

    Note:
        The model load into memory will occur lazily when the first generation is requested.
        If you'd want to force this to happen earlier, you can use the
        [`.load()`][rigging.generator.transformers_.TransformersGenerator.load] method.
    """

    torch_dtype: str = "auto"
    """Torch dtype passed to [`AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/v4.41.0/en/model_doc/auto)"""
    device_map: str = "auto"
    """Device map passed to [`AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/v4.41.0/en/model_doc/auto)"""
    trust_remote_code: bool = False
    """Trust remote code passed to [`AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/v4.41.0/en/model_doc/auto)"""
    load_in_8bit: bool = False
    """Load in 8 bit passed to [`AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/v4.41.0/en/model_doc/auto)"""
    load_in_4bit: bool = False
    """Load in 4 bit passed to [`AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/v4.41.0/en/model_doc/auto)"""

    _llm: AutoModelForCausalLM | None = None
    _tokenizer: AutoTokenizer | None = None
    _pipeline: TextGenerationPipeline | None = None

    @property
    def llm(self) -> AutoModelForCausalLM:
        """The underlying `AutoModelForCausalLM` instance."""
        # Lazy initialization
        if self._llm is None:
            llm_kwargs = self.model_dump(
                exclude_unset=True,
                include={"torch_dtype", "device_map", "trust_remote_code", "load_in_8bit", "load_in_4bit"},
            )
            self._llm = AutoModelForCausalLM.from_pretrained(self.model, **llm_kwargs)
        return self._llm

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """The underlying `AutoTokenizer` instance."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self._tokenizer

    @property
    def pipeline(self) -> TextGenerationPipeline:
        """The underlying `TextGenerationPipeline` instance."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-generation",
                return_full_text=False,
                model=self.llm,
                tokenizer=self.tokenizer,
            )
        return self._pipeline

    @classmethod
    def from_obj(
        cls,
        model: str,
        llm: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        *,
        pipeline: TextGenerationPipeline | None = None,
        params: GenerateParams | None = None,
    ) -> "TransformersGenerator":
        """
        Create a new instance of TransformersGenerator from an already loaded model and tokenizer.

        Args:
            model: The loaded model for text generation.
            tokenizer : The tokenizer associated with the model.
            pipeline: The text generation pipeline. Defaults to None.

        Returns:
            The TransformersGenerator instance.
        """
        instance = cls(model=model, params=params or GenerateParams())
        instance._llm = model
        instance._tokenizer = tokenizer
        instance._pipeline = pipeline
        return instance

    def load(self) -> "TransformersGenerator":
        """Load the model into memory."""
        _ = self.pipeline
        return self

    def _generate(
        self,
        inputs: t.Sequence[str] | t.Sequence[Message],
        params: t.Sequence[GenerateParams],
    ) -> list[str]:
        param_set = {p.model_dump_json() for p in params}
        if len(param_set) != 1:
            raise ValueError("All GenerateParams must be identical for this generator")

        # Generation Args + Fixups
        if self.params.max_tokens is None:
            self.params.max_tokens = DEFAULT_MAX_TOKENS

        kwargs = self.params.merge_with(params[0]).to_dict()
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        if any(k in kwargs for k in ["temperature", "top_k", "top_p"]):
            kwargs["do_sample"] = True

        outputs = self.pipeline(inputs, **kwargs)
        return [output[-1]["generated_text"].strip() for output in outputs]

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[Message]:
        message_dicts = [[m.model_dump(include={"role", "content"}) for m in _messages] for _messages in messages]
        texts = self.tokenizer.apply_chat_template(message_dicts, add_generation_prompt=True, tokenize=False)
        outputs = self._generate(texts, params)
        generated = [Message(role="assistant", content=output) for output in outputs]

        for i, (in_messages, out_message) in enumerate(zip(messages, generated, strict=True)):
            trace_messages(in_messages, f"Messages {i+1}/{len(in_messages)}")
            trace_messages([out_message], f"Response {i+1}/{len(in_messages)}")

        return generated

    async def agenerate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[Message]:
        return self.generate_messages(messages, params)

    def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[str]:
        generated = self._generate(list(texts), params=params)

        for i, (text, response) in enumerate(zip(texts, generated, strict=True)):
            trace_str(text, f"Text {i+1}/{len(texts)}")
            trace_str(response, f"Generated {i+1}/{len(texts)}")

        return generated

    async def agenerate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[str]:
        return self.generate_texts(texts, params)


register_generator("transformers", TransformersGenerator)
