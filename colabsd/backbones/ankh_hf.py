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
false, which is why this adapter does *not* pre-require the packages the way
`prott5_hf` has to: this tokenizer is fast and needs neither `sentencepiece` nor
`protobuf`, and refusing to load without them would make a working backbone
unavailable. So the install hint is a diagnosis of a failure that has already
happened rather than a precondition — appended only when the load really did
fail *and* one of the two packages is missing, and naming whichever that is.
It used to ask about `sentencepiece` alone, which left out the one combination
where transformers genuinely does fall back to the slow T5 tokenizer and fail:
a runtime carrying `sentencepiece` and not `protobuf` got the loader's generic
five-cause error, which names no package at all.

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

from colabsd.backbones.base import HFAdapterBase, missing_t5_tokenizer_packages
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
        """Load the fast tokenizer, naming a T5 package only if its absence is what broke.

        Both packages are asked about, through the pair `prott5_hf` prechecks with. This guard
        read `find_spec("sentencepiece")` alone, so the one runtime the hint exists for — the one
        that has `sentencepiece`, lacks `protobuf`, and therefore really can fall back to a slow
        T5 tokenizer that cannot parse its vocabulary — was handed the loader's five-cause error
        instead, which names neither package. `missing_t5_tokenizer_packages` also spells protobuf
        as the module it installs rather than the name you type, which is why a runtime that has
        it is no longer told to install it.
        """
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            return super().tokenizer()
        except BackboneError as exc:
            missing = missing_t5_tokenizer_packages()
            if not missing:
                raise
            raise BackboneError(
                f"{exc} This checkpoint normally loads from its own `tokenizer.json` without "
                f"{' or '.join(f'`{name}`' for name in missing)}; if transformers fell back to the slow "
                f"T5 tokenizer here, run `pip install {' '.join(missing)}` and restart the runtime."
            ) from exc

    def format_sequences(self, sequences: list[str]) -> list[str]:
        from colabsd.engine.formats import format_ankh_sequence

        return [format_ankh_sequence(sequence) for sequence in sequences]
