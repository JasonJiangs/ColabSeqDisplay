"""ESM-C 300M/600M through the EvolutionaryScale `esm` SDK — tier `extra`.

ESM-C has no `transformers` port, so this is the one adapter that does not load
its weights with `AutoModel`: the SDK names `esmc_300m` / `esmc_600m` resolve to
the licence-gated HuggingFace repos `EvolutionaryScale/esmc-300m-2024-12` and
`EvolutionaryScale/esmc-600m-2024-12`. The SDK is imported lazily and a missing
install raises an actionable error — note that the forbidden `fair-esm` package
installs a *different* module under the same `esm` name, so "already installed"
is not the same as "usable".

Attention implementation — `ESMC.from_pretrained` builds `FlashMultiHeadAttention`
(the SDK builders default to `use_flash_attn=True`), which is unusable here in
all three configurations this package cares about: its Triton rotary kernel
refuses CPU tensors, `flash_attn_varlen_qkvpacked_func` accepts only fp16/bf16 so
it rejects our default `dtype="float32"` even on a GPU, and when `flash-attn` is
not installed at all the SDK still constructs that class and its forward raises
`NameError`. The contract wants `load_model()` to hand back a CPU module the
caller moves and runs itself, so this adapter asks the SDK's own builder for the
plain `MultiHeadAttention` variant (`use_flash_attn=False`). Same parameters,
same state dict, `F.scaled_dot_product_attention` instead of the varlen kernel.

Tokenizer — `EsmSequenceTokenizer` is built in code from a hard-coded vocabulary,
so it costs no download, and it wraps the sequence as `<cls> ... <eos>`: 33
residues give 35 tokens. It renames its first model input to `sequence_tokens`,
which is why `encode` cannot use the base implementation; the batch itself still
comes back under `input_ids`.

Residue axis — one token per residue between the two specials, stripped by
special-token mask in `HFAdapterBase`, so residue *i* (1-based) is column *i-1*.

LoRA coverage — attention projections are

    transformer.blocks.<i>.attn.layernorm_qkv.1     (fused q/k/v)
    transformer.blocks.<i>.attn.out_proj

Upstream's `_FUSED_QKV_ALIASES` reads a `layernorm_qkv` path segment as
query+key+value and `out_proj` as output, both inside the group
`transformer.blocks.<i>.attn`, so `validate_qkvo_coverage` passes: 30 blocks for
300M, 36 for 600M. `FlashMultiHeadAttention` subclasses `MultiHeadAttention` and
keeps both names, so the coverage does not depend on which variant is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from colabsd.backbones.base import HFAdapterBase
from colabsd.errors import BackboneError

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

SDK_MODEL_NAMES: dict[str, str] = {
    "ESMC-300M": "esmc_300m",
    "ESMC-600M": "esmc_600m",
}


def import_esm_sdk() -> tuple[Any, Any]:
    """Return `(ESMC, EsmSequenceTokenizer)` from the EvolutionaryScale SDK."""
    try:
        from esm.models.esmc import ESMC
        from esm.tokenization import EsmSequenceTokenizer
    except ImportError as exc:
        raise BackboneError(
            "ESM-C needs the EvolutionaryScale SDK, which is not importable in this runtime. "
            "Run `pip install esm` (or `pip install 'colabseqdisplay[esmc]'`) and restart the runtime. "
            "Beware of the older `fair-esm` package: it installs a different module under the same `esm` "
            f"name and has no ESM-C, so uninstall it first if it is present. Original error: {exc}"
        ) from exc
    return ESMC, EsmSequenceTokenizer


def plain_attention_builder(sdk_model_name: str) -> Any | None:
    """Return the SDK builder for *sdk_model_name* when it accepts `use_flash_attn`."""
    from inspect import signature

    try:
        from esm.pretrained import LOCAL_MODEL_REGISTRY
    except ImportError:
        return None
    builder = LOCAL_MODEL_REGISTRY.get(sdk_model_name)
    if builder is None or "use_flash_attn" not in signature(builder).parameters:
        return None
    return builder


class ESMCAdapter(HFAdapterBase):
    """ESM-C 300M/600M as a `SequenceAdapter`."""

    family: ClassVar[str] = "ESMC"

    def sdk_model_name(self) -> str:
        """Return the SDK identifier the ESM-C builders expect."""
        try:
            return SDK_MODEL_NAMES[self.model_name]
        except KeyError:
            raise BackboneError(
                f"'{self.model_name}' is not an ESM-C model. Pick one of: {', '.join(SDK_MODEL_NAMES)}. "
                "The ESM-C weights load through the EvolutionaryScale SDK, so hf_id= cannot redirect them "
                "to another checkpoint."
            ) from None

    def tokenizer(self) -> Any:
        """Return the SDK's in-code tokenizer; the gated repos hold weights, not tokenizer files."""
        if self._tokenizer is None:
            _, esm_sequence_tokenizer = import_esm_sdk()
            self._tokenizer = esm_sequence_tokenizer()
        return self._tokenizer

    def _load_hf_model(self) -> nn.Module:
        import torch

        esmc, _ = import_esm_sdk()
        self._require_sdk_checkpoint()
        sdk_name = self.sdk_model_name()
        builder = plain_attention_builder(sdk_name)
        cpu = torch.device("cpu")
        try:
            model = builder(device=cpu, use_flash_attn=False) if builder else esmc.from_pretrained(sdk_name, device=cpu)
        except OSError as exc:
            raise BackboneError(
                f"Could not fetch the ESM-C weights for '{self.model_name}' ({self.hf_id}): {exc}. "
                "The EvolutionaryScale repos are licence-gated: accept the licence on huggingface.co, run "
                "`huggingface-cli login` in this runtime, then retry."
            ) from exc
        self._require_plain_attention(model)
        self._require_matching_width(model)
        return model.to(self.torch_dtype())

    def _require_sdk_checkpoint(self) -> None:
        """`hf_id` cannot redirect ESM-C, so refuse it rather than loading something else."""
        registered = self.entry.hf_id if self.entry else ""
        if registered and self.hf_id != registered:
            raise BackboneError(
                f"'{self.model_name}' loads through the EvolutionaryScale SDK, which resolves the checkpoint "
                f"from its own name '{self.sdk_model_name()}' and ignores hf_id='{self.hf_id}'. Drop hf_id= so "
                f"the weights come from {registered}, or use a transformers-native backbone for a local copy."
            )

    def _require_plain_attention(self, model: nn.Module) -> None:
        """Flash attention here is a silent trap, so fail loudly if the SDK built it anyway."""
        if not getattr(model, "_use_flash_attn", False):
            return
        raise BackboneError(
            f"The esm SDK built '{self.model_name}' with flash attention, which cannot run on CPU (its rotary "
            "embedding is a Triton kernel) and rejects float32 even on a GPU. This adapter asks for the "
            "plain-attention variant, but the SDK in this runtime exposes no `use_flash_attn` switch on its "
            "esmc builders, and uninstalling `flash-attn` does not help — the SDK builds that class either "
            "way. Install a release of the `esm` package whose esm.pretrained builders take use_flash_attn."
        )

    def _require_matching_width(self, model: nn.Module) -> None:
        """ESM-C models carry no `config`, so check the embedding width the registry promised."""
        hidden = getattr(getattr(model, "embed", None), "embedding_dim", None)
        if hidden is not None and int(hidden) != self.embed_dim:
            raise BackboneError(
                f"'{self.model_name}' reports embed_dim {self.embed_dim} but the SDK checkpoint "
                f"'{self.sdk_model_name()}' is {int(hidden)}-dimensional. The registry entry and the "
                "EvolutionaryScale weights disagree; pick a matching backbone name."
            )

    def encode(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return model(sequence_tokens=batch["input_ids"]).embeddings
