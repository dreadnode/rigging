import inspect
import typing as t

import vllm

from rigging.generator.base import GenerateParams, Generator, register_generator
from rigging.message import (
    Message,
)


class VLLMGenerator(Generator):
    """
    Generator backed by the vLLM library.

    Note:
        Find more information about supported models and formats [in their docs.](https://docs.vllm.ai/en/latest/index.html).

    Note:
        The async methods currently just map to the sync variants for compatibility.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None,
        params: GenerateParams,
        dtype: str = "auto",
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
    ):
        super().__init__(model=model, api_key=api_key, params=params)

        self._llm = vllm.LLM(
            model,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[Message]:
        message_dicts = [[m.model_dump(include={"role", "content"}) for m in _messages] for _messages in messages]
        texts = self._llm.get_tokenizer().apply_chat_template(message_dicts)
        outputs = self.generate_texts(texts, params)
        return [Message(role="assistant", content=output) for output in outputs]

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
        sampling_params_args = list(inspect.signature(vllm.SamplingParams.__init__).parameters.keys())
        sampling_params = [
            vllm.SamplingParams(
                **{k: v for k, v in self.params.merge_with(p).to_dict().items() if k in sampling_params_args}
            )
            for p in params
        ]
        outputs = self._llm.generate(list(texts), sampling_params)
        return [output.outputs[-1].text for output in outputs]

    async def agenerate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[str]:
        return self.generate_texts(texts, params)


register_generator("vllm", VLLMGenerator)
