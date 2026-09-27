"""ESM2 backbones through `transformers.EsmModel`.

No fair-esm: the HuggingFace port of ESM2 is weight-identical and its tokenizer
emits exactly one token per residue between `<cls>` and `<eos>`, so the residue
axis lines up with 1-based protein coordinates after stripping special tokens.

LoRA coverage — `EsmModel` names its attention projections

    encoder.layer.<i>.attention.self.query
    encoder.layer.<i>.attention.self.key
    encoder.layer.<i>.attention.self.value
    encoder.layer.<i>.attention.output.dense

which upstream's `target_roles` maps onto query/key/value/output inside the group
`encoder.layer.<i>.attention`, so `validate_qkvo_coverage` passes for every layer.
The model is loaded without the pooling head so no unused `pooler.dense` is
carried around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from colabsd.backbones.base import HFAdapterBase

if TYPE_CHECKING:
    import torch.nn as nn


class ESM2Adapter(HFAdapterBase):
    """ESM2-8M/35M/150M/650M as a `SequenceAdapter`."""

    family: ClassVar[str] = "ESM2"

    def _load_hf_model(self) -> nn.Module:
        from transformers import EsmModel

        return self._from_pretrained(EsmModel, add_pooling_layer=False)
