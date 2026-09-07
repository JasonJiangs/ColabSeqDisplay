"""Ankh-large through `transformers.T5EncoderModel`.

`ElnaggarLab/ankh-large` is an encoder-decoder checkpoint; only the encoder is
loaded, matching upstream's extraction path and the
`config/best/Ankh-large_*.yaml` entries.

Tokenizer — Ankh is the mirror image of ProtT5 in two ways. Its vocabulary holds
bare residue characters, so the sequence must *not* be space separated (a space
tokenizes to `<unk>` and doubles the token count); `U/Z/O/B` are still mapped to
`X`, as in upstream's `_format_sequence`. And the repo ships `tokenizer.json`
but no `spiece.model`, so `AutoTokenizer` builds `T5TokenizerFast` straight from
that file — checked with `transformers.utils.is_sentencepiece_available()` forced
false, which is why this adapter does *not* pre-require the package the way
`prott5_hf` has to. Refusing to load without it would make a working backbone
unavailable, so the install hint (the `prott5` extra, which is what pulls
`sentencepiece` in) is appended only when the load actually fails *and* the
package is missing.

Residue axis — the Hub tokenizer appends `</s>` and prepends nothing: 33 residues
give 34 tokens (checked against transformers 4.48). Upstream's loader drops one
extra leading token, which is correct for a local sentencepiece copy of Ankh
(that one emits a leading `_` piece) and wrong for this checkpoint.
`HFAdapterBase` strips by special-token mask and requires exactly
`len(sequence)` surviving tokens, so a checkpoint whose tokenizer adds a
non-special leading piece raises instead of silently frameshifting every pooling
position by one residue.

LoRA coverage — same T5 layout as ProtT5:

    encoder.block.<i>.layer.0.SelfAttention.{q,k,v,o}

Upstream's alias table maps the single-letter leaves onto query/key/value/output
inside the group `encoder.block.<i>.layer.0.selfattention`, so
`validate_qkvo_coverage` passes for every block. Ankh-large is 48 encoder blocks
of `d_model=1536`, so that is 192 LoRA-wrapped projections. Its feed-forward is
the gated variant (`layer.1.DenseGatedActDense.{wi_0,wi_1,wo}`), which upstream's
`target_roles` does not match — no path segment of those names holds "attention".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from colabsd.backbones.base import HFAdapterBase
from colabsd.errors import BackboneError

if TYPE_CHECKING:
    import torch.nn as nn


class AnkhAdapter(HFAdapterBase):
    """Ankh-large as a `SequenceAdapter`."""

    family: ClassVar[str] = "Ankh"

    def _load_hf_model(self) -> nn.Module:
        from transformers import T5EncoderModel

        return self._from_pretrained(T5EncoderModel)

    def tokenizer(self) -> Any:
        """Load the fast tokenizer, naming `sentencepiece` only if its absence is what broke."""
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            return super().tokenizer()
        except BackboneError as exc:
            from importlib.util import find_spec

            if find_spec("sentencepiece") is not None:
                raise
            raise BackboneError(
                f"{exc} This checkpoint normally loads from its own tokenizer.json without "
                "`sentencepiece`; if transformers fell back to the slow T5 tokenizer here, run "
                "`pip install 'colabseqdisplay[prott5]'` (the extra that ships `sentencepiece`) "
                "and restart the runtime."
            ) from exc

    def format_sequences(self, sequences: list[str]) -> list[str]:
        from colabsd.engine.formats import format_ankh_sequence

        return [format_ankh_sequence(sequence) for sequence in sequences]
