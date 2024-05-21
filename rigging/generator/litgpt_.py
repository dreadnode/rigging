import typing as t
from pathlib import Path

import lightning as L
from litgpt.model import GPT
from litgpt.model import Config as ModelConfig
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint

import rigging as rg

g_fabric: L.Fabric | None = None


def setup_fabric(
    devices: list[int] | str | int = "auto", precision: str | None = None, seed: int | None = None
) -> L.Fabric:
    global g_fabric

    if g_fabric is not None:
        return g_fabric

    precision = precision or get_default_supported_precision(training=False)
    g_fabric = L.Fabric(devices=devices, precision=precision)  # type: ignore
    g_fabric.seed_everything(seed)
    g_fabric.launch()

    return g_fabric


class LitGPTGenerator(rg.Generator):
    def __init__(self, model: str, api_key: str | None, params: rg.GenerateParams):
        super().__init__(model=model, api_key=api_key, params=params)

        fabric = setup_fabric()

        checkpoint_dir = Path(model)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory '{model}' not found")

        check_valid_checkpoint_dir(model)
        model_config = ModelConfig.from_file(checkpoint_dir / "model_config.yaml")

        self._tokenizer = Tokenizer(checkpoint_dir)
        self._prompt_stype = (
            load_prompt_style(checkpoint_dir)
            if has_prompt_style(checkpoint_dir)
            else PromptStyle.from_config(model_config)
        )

        with fabric.init_module(empty_init=True):
            self._model = GPT(model_config)
            self._model.set_kv_cache(batch_size=1)

        self._model.eval()  # Disable dropout

        load_checkpoint(fabric, model, checkpoint_dir / "lit_model.pth")

        self._context_size = self._model.config.block_size  # TODO: Is this right?

    def generate_texts(
        self,
        texts: t.Sequence[str],
        params: t.Sequence[rg.GenerateParams],
        *,
        prefix: str | None = None,
    ) -> t.Sequence[str]:
        raise NotImplementedError("`generate_texts` is not supported by this generator.")
