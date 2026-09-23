"""ProtT5-XL through `transformers.T5EncoderModel`.

`Rostlab/prot_t5_xl_uniref50` is an encoder-decoder checkpoint; only the encoder
is loaded, which is what upstream's cached embeddings and the tuned
`config/best/ProtT5-XL_*.yaml` hyperparameters were produced with.

Tokenizer — the sentencepiece vocabulary stores `_A`-style pieces, so residues
have to be space separated (`"M K T ..."`) and `U/Z/O/B` mapped to `X` first;
upstream's `format_prott5_sequence` does both. Feeding the unspaced string is not
an error, it just silently produces a different and shorter token sequence. The
repo ships `spiece.model` but no `tokenizer.json`, and the slow-to-fast conversion
raises a bare `Exception("You're trying to run a `Unigram` model but you're file
was trained with a different algorithm")` that `HFAdapterBase._guarded` would not
even catch, so the slow tokenizer (`use_fast=False`) is mandatory — and with it
two packages `transformers` does not install by itself: `sentencepiece`, which
reads the vocabulary, and `protobuf`, which the slow T5 tokenizer parses
`spiece.model` through. Neither is a required dependency of this project; both
are declared in its `prott5` extra. Without them both `use_fast` settings fail
with `ValueError: Converting from Tiktoken failed`, which `_guarded` would report
as a network or checkpoint-path problem, so the precheck below runs first and
names whichever of the two is actually missing instead. Unlike Ankh, this one
really is mandatory.

Residue axis — the tokenizer appends `</s>` and prepends nothing, so a 33-residue
sequence gives 34 tokens (checked against transformers 4.48). `HFAdapterBase`
strips by special-token mask and refuses to continue unless exactly
`len(sequence)` tokens survive, so residue *i* (1-based) is column *i-1* of the
pooled tensor.

LoRA coverage — `T5EncoderModel` names its attention projections

    encoder.block.<i>.layer.0.SelfAttention.{q,k,v,o}

and upstream's alias table maps those single-letter leaves onto
query/key/value/output within the group `encoder.block.<i>.layer.0.selfattention`,
so `validate_qkvo_coverage` passes with one complete group per block. The
feed-forward `layer.1.DenseReluDense.{wi,wo}` leaves are left alone, and the
relative-attention bias is an `nn.Embedding` rather than an `nn.Linear`, so it is
not a target either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from colabsd.backbones.base import HFAdapterBase, missing_t5_tokenizer_packages
from colabsd.errors import BackboneError

if TYPE_CHECKING:
    import torch.nn as nn


class ProtT5Adapter(HFAdapterBase):
    """ProtT5-XL as a `SequenceAdapter`."""

    family: ClassVar[str] = "ProtT5"

    def _load_hf_model(self) -> nn.Module:
        from transformers import T5EncoderModel

        return self._from_pretrained(T5EncoderModel)

    def tokenizer(self) -> Any:
        """Return the slow sentencepiece tokenizer; the fast conversion fails for this repo."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            # Two packages, not one, and the pair is spelled once in `base` because `ankh_hf` asks
            # the same question from the other end. A runtime that had `sentencepiece` and not
            # `protobuf` fell through this precheck into the loader's generic five-cause error,
            # which names neither, so a reader was told to clear their HuggingFace cache over a
            # missing pip package. Both are asked for here, so the message names what is missing.
            missing = missing_t5_tokenizer_packages()
            if missing:
                packages = " ".join(missing)
                raise BackboneError(
                    f"'{self.model_name}' reads a T5 sentencepiece vocabulary, and transformers cannot "
                    f"build its tokenizer without {' and '.join(f'`{name}`' for name in missing)}. Run "
                    f"`pip install {packages}` and restart the runtime."
                )
            self._tokenizer = self._guarded(
                lambda: AutoTokenizer.from_pretrained(self.hf_id, do_lower_case=False, use_fast=False),
                "tokenizer",
            )
        return self._tokenizer

    def format_sequences(self, sequences: list[str]) -> list[str]:
        from colabsd.engine.formats import format_prott5_sequence

        return [format_prott5_sequence(sequence) for sequence in sequences]
