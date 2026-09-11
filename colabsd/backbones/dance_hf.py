"""SeqDance and ESMDance — `ChaoHou/SeqDance`, `ChaoHou/ESMDance`.

Both checkpoints are the same wrapper around ESM2-35M: an `EsmModel` trunk plus
per-residue and pairwise heads trained on molecular-dynamics ensembles. The
wrapper class comes from `colabsd.engine.formats.make_seqdance_class`
so the parameter names keep matching the published weights; only the
HuggingFace loading and the pooling path are ours.

What is worth pooling differs between the two, and this is the trap:

* SeqDance  — `feature="res_emb"`, the 480-dim hidden states of an ESM2-35M that
  was re-initialised and retrained from scratch on dynamics.
* ESMDance  — `feature="res_pred"`, the **50-dim** output of `res_pred_nn`, the
  predicted per-residue dynamics properties. Its trunk is the frozen pretrained
  ESM2-35M, so the hidden states hold nothing that ESM2-35M does not already
  give you; the prediction head is the model. `embed_dim` is therefore 50, which
  is what this adapter reports regardless of the encoder hidden size — the
  registry says the same thing and the tests fail if the two ever drift apart.

Tokenizer — the two repos ship weights only (`config.json` there holds the two
wrapper arguments, not a transformers config), so the ESM2-35M tokenizer is
loaded instead. It emits one token per residue between `<cls>` and `<eos>`.
SeqDance was trained with a 2048-token window; sequences are not truncated here,
so a longer protein raises in `HFAdapterBase` rather than silently losing its
C-terminus.

LoRA coverage — the trunk is a plain `EsmModel` under the `esm2.` prefix:

    esm2.encoder.layer.<i>.attention.self.{query,key,value}
    esm2.encoder.layer.<i>.attention.output.dense

so upstream's grouping sees one complete q/k/v/o group per layer and
`validate_qkvo_coverage` passes for all 12 layers (48 projections). The
prediction heads hold no attention projections and are left untouched. ESMDance
ships its trunk with `requires_grad=False`; LoRA still trains, because
`LoRALinear` adds its own trainable parameters around the frozen base weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from colabsd.backbones.base import HFAdapterBase
from colabsd.errors import BackboneError

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

ESM2_TOKENIZER_ID = "facebook/esm2_t12_35M_UR50D"

DANCE_SPECS: dict[str, dict[str, str | int]] = {
    "SeqDance": {"model_select": "seqdance", "feature": "res_emb", "embed_dim": 480},
    "ESMDance": {"model_select": "esmdance", "feature": "res_pred", "embed_dim": 50},
}


class DanceAdapter(HFAdapterBase):
    """SeqDance and ESMDance as a `SequenceAdapter`; the family entry picks the feature."""

    family: ClassVar[str] = "SeqDance"

    def __init__(
        self,
        model_name: str,
        *,
        pooling_positions_0based: Sequence[int],
        wt_3di: str | None = None,
        dtype: str = "float32",
        hf_id: str | None = None,
    ) -> None:
        spec = DANCE_SPECS.get(model_name)
        if spec is None:
            raise BackboneError(f"Unknown dynamics backbone '{model_name}'. Pick one of: {', '.join(DANCE_SPECS)}.")
        super().__init__(
            model_name,
            pooling_positions_0based=pooling_positions_0based,
            wt_3di=wt_3di,
            dtype=dtype,
            hf_id=hf_id,
        )
        self.model_select = str(spec["model_select"])
        self.feature = str(spec["feature"])
        self.embed_dim = int(spec["embed_dim"])

    def tokenizer(self) -> Any:
        """Return the ESM2-35M tokenizer; the dynamics repos ship no tokenizer of their own."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = self._guarded(lambda: AutoTokenizer.from_pretrained(ESM2_TOKENIZER_ID), "tokenizer")
        return self._tokenizer

    def _load_hf_model(self) -> nn.Module:
        from colabsd.engine.formats import make_seqdance_class

        wrapper = make_seqdance_class()
        model = self._guarded(
            lambda: wrapper.from_pretrained(
                self.hf_id,
                esm2_select="model_35M",
                model_select=self.model_select,
            ),
            "weights",
        )
        return model.to(self.torch_dtype())

    def encode(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        return model(inputs, feature=self.feature)
