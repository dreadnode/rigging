import inspect
import typing as t

import vllm

from rigging.generator.base import GenerateParams, Generator, register_generator, trace_messages, trace_str
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
        The async methods currently just call synchronous variants for compatibility.

    Note:
        The model load into memory will occur lazily when the first generation is requested.
        If you'd want to force this to happen earlier, you can access the
        [`.llm`][rigging.generator.vllm_.VLLMGenerator.llm] property on this class before first use.
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

    def _generate(
        self,
        *,
        texts: list[str] | None = None,
        tokens: list[list[int]] | None = None,
        params: t.Sequence[GenerateParams] | None = None,
    ) -> list[str]:
        if texts is None and tokens is None:
            raise ValueError("Either texts or tokens must be provided")

        sampling_params_args = list(inspect.signature(vllm.SamplingParams.__init__).parameters.keys())
        sampling_params = (
            [
                vllm.SamplingParams(
                    **{k: v for k, v in self.params.merge_with(p).to_dict().items() if k in sampling_params_args}
                )
                for p in params
            ]
            if params
            else None
        )

        # Do a cache warmup step if we have a lot of texts
        if len(texts or tokens or []) > CACHE_TRIGGER:
            self.llm.generate(
                prompts=min(texts, key=len) if texts else None,
                prompt_token_ids=[min(tokens, key=len)] if tokens else None,
                use_tqdm=False,
            )

        outputs = self.llm.generate(
            prompts=texts,
            prompt_token_ids=tokens,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[-1].text for output in outputs]

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[Message]:
        message_dicts = [[m.model_dump(include={"role", "content"}) for m in _messages] for _messages in messages]
        tokens = self.llm.get_tokenizer().apply_chat_template(message_dicts, add_generation_prompt=True)
        outputs = self._generate(tokens=tokens, params=params)
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
        generated = self._generate(texts=list(texts), params=params)

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


register_generator("vllm", VLLMGenerator)
