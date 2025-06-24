import gc
import typing as t

import torch
import transformers  # type: ignore [import-untyped, unused-ignore]
from transformers import (  # type: ignore [attr-defined]
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)

from rigging.generator.base import (
    GeneratedMessage,
    GeneratedText,
    GenerateParams,
    Generator,
    trace_messages,
    trace_str,
)

if t.TYPE_CHECKING:
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
        This generator doesn't leverage any async capabilities.

    Note:
        The model load into memory will occur lazily when the first generation is requested.
        If you'd want to force this to happen earlier, you can use the
        [`.load()`][rigging.generator.Generator.load] method.

        To unload, call [`.unload()`][rigging.generator.Generator.unload].
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
                include={
                    "torch_dtype",
                    "device_map",
                    "trust_remote_code",
                    "load_in_8bit",
                    "load_in_4bit",
                },
            )
            self._llm = AutoModelForCausalLM.from_pretrained(self.model, **llm_kwargs)  # type: ignore [no-untyped-call]
        return self._llm

    @property
    def tokenizer(self) -> AutoTokenizer:
        """The underlying `AutoTokenizer` instance."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        return self._tokenizer

    @property
    def pipeline(self) -> TextGenerationPipeline:
        """The underlying `TextGenerationPipeline` instance."""
        if self._pipeline is None:
            self._pipeline = transformers.pipeline(  # type: ignore [attr-defined, assignment]
                "text-generation",
                return_full_text=False,
                model=self.llm,  # type: ignore [arg-type]
                tokenizer=self.tokenizer,  # type: ignore [arg-type]
            )
        return self._pipeline  # type: ignore [return-value]

    @classmethod
    def from_obj(
        cls,
        model: t.Any,
        tokenizer: AutoTokenizer,
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
        _ = self.pipeline
        return self

    def unload(self) -> "TransformersGenerator":
        del self._pipeline
        del self._llm
        del self._tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def _generate(
        self,
        inputs: t.Sequence[str] | t.Sequence[t.Sequence[dict[str, str]]],
        params: t.Sequence[GenerateParams],
    ) -> list[GeneratedText]:
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

        # TODO: We do strip() here as it's often needed, but I think
        # we should return and standardize this behavior.
        return [GeneratedText(text=o["generated_text"].strip()) for o in outputs]

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence["Message"]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        message_dicts = [
            [m.to_openai(compatibility_flags={"content_as_str"}) for m in _messages]
            for _messages in messages
        ]
        outputs = self._generate(message_dicts, params)
        generated = [o.to_generated_message() for o in outputs]

        for i, (in_messages, out_message) in enumerate(zip(messages, generated, strict=False)):
            trace_messages(in_messages, f"Messages {i + 1}/{len(in_messages)}")
            trace_messages([out_message], f"Response {i + 1}/{len(in_messages)}")

        return generated

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        generated = self._generate(texts, params)

        for i, (text, response) in enumerate(zip(texts, generated, strict=False)):
            trace_str(text, f"Text {i + 1}/{len(texts)}")
            trace_str(response, f"Generated {i + 1}/{len(texts)}")

        return generated
