import gc
import inspect
import typing as t

import torch
import vllm

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

# Any batch over this size will trigger a dedicated
# cache warmup step
CACHE_TRIGGER = 8


class VLLMGenerator(Generator):
    """
    Generator backed by the vLLM library for local model loading.

    Find more information about supported models and formats [in their docs.](https://docs.vllm.ai/en/latest/index.html)

    Warning:
        The use of VLLM requires the `vllm` package to be installed directly or by
        installing rigging as `rigging[all]`.

    Note:
        This generator doesn't leverage any async capabilities.

    Note:
        The model load into memory will occur lazily when the first generation is requested.
        If you'd want to force this to happen earlier, you can use the
        [`.load()`][rigging.generator.Generator.load] method.

        To unload, call [`.unload()`][rigging.generator.Generator.unload].
    """

    dtype: str = "auto"
    """Tensor dtype passed to [`vllm.LLM`](https://docs.vllm.ai/en/latest/offline_inference/llm.html)"""
    quantization: str | None = None
    """Quantiziation passed to [`vllm.LLM`](https://docs.vllm.ai/en/latest/offline_inference/llm.html)"""
    gpu_memory_utilization: float = 0.9
    """Memory utilization passed to [`vllm.LLM`](https://docs.vllm.ai/en/latest/offline_inference/llm.html)"""
    enforce_eager: bool = False
    """Eager enforcement passed to [`vllm.LLM`](https://docs.vllm.ai/en/latest/offline_inference/llm.html)"""
    trust_remote_code: bool = False
    """Trust remote code passed to [`vllm.LLM`](https://docs.vllm.ai/en/latest/offline_inference/llm.html)"""

    # TODO: We should look at leveraging the AsyncLLMEngine or an
    # async alternative to the LLM class to allow for async generation

    _llm: vllm.LLM | None = None

    @property
    def llm(self) -> vllm.LLM:
        """The underlying [`vLLM model`](https://docs.vllm.ai/en/latest/offline_inference/llm.html) instance."""
        # Lazy initialization
        if self._llm is None:
            self._llm = vllm.LLM(
                self.model,
                dtype=self.dtype,
                quantization=self.quantization,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
                trust_remote_code=self.trust_remote_code,
            )
        return self._llm

    @classmethod
    def from_obj(
        cls,
        model: str,
        llm: vllm.LLM,
        *,
        params: GenerateParams | None = None,
    ) -> "VLLMGenerator":
        """Create a generator from an existing vLLM instance.

        Args:
            llm: The vLLM instance to create the generator from.

        Returns:
            The VLLMGenerator instance.
        """
        generator = cls(model=model, params=params or GenerateParams())
        generator._llm = llm
        return generator

    def load(self) -> "VLLMGenerator":
        _ = self.llm
        return self

    def unload(self) -> "VLLMGenerator":
        del self._llm
        gc.collect()
        torch.cuda.empty_cache()
        return self

    def _generate(
        self,
        texts: list[str],
        params: t.Sequence[GenerateParams],
    ) -> list[GeneratedText]:
        sampling_params_args = list(
            inspect.signature(vllm.SamplingParams.__init__).parameters.keys(),
        )
        sampling_params = (
            [
                vllm.SamplingParams(
                    **{
                        k: v
                        for k, v in self.params.merge_with(p).to_dict().items()
                        if k in sampling_params_args
                    },
                )
                for p in params
            ]
            if params
            else None
        )

        # Do a cache warmup step if we have a lot of texts
        if len(texts) > CACHE_TRIGGER:
            self.llm.generate(
                prompts=min(texts, key=len),
                use_tqdm=False,
            )

        outputs = self.llm.generate(
            prompts=texts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [
            GeneratedText(
                text=o.outputs[-1].text,
                stop_reason=o.outputs[-1].finish_reason,
                extra={
                    "request_id": o.request_id,
                    "metrics": o.metrics,
                    "stop_token": o.outputs[-1].stop_reason,
                },
            )
            for o in outputs
        ]

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence["Message"]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        message_dicts = [[m.to_openai() for m in _messages] for _messages in messages]
        tokenizer = self.llm.get_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "The tokenizer does not support the apply_chat_template method.",
            )

        texts = tokenizer.apply_chat_template(
            message_dicts,
            add_generation_prompt=True,
            tokenize=False,
        )
        generated_texts = self._generate(t.cast("list[str]", texts), params=params)
        generated = [g.to_generated_message() for g in generated_texts]

        for i, (in_messages, out_message) in enumerate(zip(messages, generated, strict=False)):
            trace_messages(in_messages, f"Messages {i + 1}/{len(in_messages)}")
            trace_messages([out_message], f"Response {i + 1}/{len(in_messages)}")

        return generated

    async def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedText]:
        generated = self._generate(list(texts), params=params)

        for i, (text, response) in enumerate(zip(texts, generated, strict=False)):
            trace_str(text, f"Text {i + 1}/{len(texts)}")
            trace_str(response, f"Generated {i + 1}/{len(texts)}")

        return generated
