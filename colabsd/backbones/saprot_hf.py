"""SaProt backbones through `transformers.EsmModel`.

SaProt reads a structure-aware vocabulary of amino-acid + 3Di pairs: the input
string is twice as long as the protein, but the tokenizer emits one token per
*pair*, so the residue axis is still `len(sequence)` after stripping `<cls>` and
`<eos>`. The interleaving itself is
`colabsd.engine.formats.saprot_interleaved_sequence`.

Every variant of a display library shares the wild-type structure, so one
wild-type 3Di string is paired with all of them — the same assumption upstream's
fair-esm SaProt adapter makes.

LoRA coverage — SaProt is an ESM2 architecture, so `EsmModel` names its attention
projections

    encoder.layer.<i>.attention.self.{query,key,value}
    encoder.layer.<i>.attention.output.dense

and `validate_qkvo_coverage` passes for every layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from colabsd.backbones.base import HFAdapterBase
from colabsd.errors import BackboneError

if TYPE_CHECKING:
    import torch.nn as nn


class SaProtAdapter(HFAdapterBase):
    """SaProt-35M/650M/1.3B as a `SequenceAdapter`; requires `wt_3di`."""

    family: ClassVar[str] = "SaProt"

    def __init__(
        self,
        model_name: str,
        *,
        pooling: str,
        pooling_positions_0based: list[int] | None = None,
        wt_3di: str | None = None,
        dtype: str = "float32",
        hf_id: str | None = None,
    ) -> None:
        super().__init__(
            model_name,
            pooling=pooling,
            pooling_positions_0based=pooling_positions_0based,
            wt_3di=wt_3di,
            dtype=dtype,
            hf_id=hf_id,
        )
        if not self.wt_3di:
            raise BackboneError(
                f"'{model_name}' is structure-aware and needs the wild-type 3Di string. Build one with "
                "colabsd.structure.three_di_from_structure(<AlphaFold PDB/CIF>) or "
                "colabsd.structure.three_di_from_esmfold(wt_sequence), then pass it as wt_3di=."
            )

    def _load_hf_model(self) -> nn.Module:
        from transformers import EsmModel

        return self._from_pretrained(EsmModel, add_pooling_layer=False)

    def format_sequences(self, sequences: list[str]) -> list[str]:
        from colabsd.engine.formats import saprot_interleaved_sequence

        assert self.wt_3di is not None
        try:
            return [saprot_interleaved_sequence(sequence, self.wt_3di) for sequence in sequences]
        except ValueError as exc:
            lengths = sorted({len(sequence) for sequence in sequences})
            raise BackboneError(
                f"The wild-type 3Di string is {len(self.wt_3di)} characters but the variants are "
                f"{lengths} residues long. They must match one-to-one: rebuild the 3Di string from a "
                "structure of the full-length wild type."
            ) from exc
