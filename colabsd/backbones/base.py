"""Shared machinery for the HuggingFace backbone adapters.

Every adapter satisfies the `colabsd.engine.adapters.SequenceAdapter`
protocol structurally (`model_name`, `embed_dim`, `load_model`, `pooled_forward`).
The science lives in `colabsd.engine`: pooling comes from `colabsd.engine.pooling.pool`
and LoRA injection from `colabsd.engine.lora`.

`pooling_positions_0based` is the one thing an adapter has to be told beyond its
name: the residues to average, as 1-based protein coordinates minus one. It comes
from the library spec's mutated sites, and every adapter pools the same way, so
there is no strategy to name or dispatch on.

`torch`, `transformers` and the engine modules are imported inside methods so that
importing this module costs nothing in a notebook cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from colabsd.backbones.registry import BACKBONES, BackboneEntry
from colabsd.errors import BackboneError, ColabSDError

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

DTYPES = ("float32", "float16", "bfloat16")


class HFAdapterBase:
    """Tokenize, run a HuggingFace encoder, strip special tokens, pool.

    Subclasses override `_load_hf_model` and, when the backbone consumes something
    other than the raw amino-acid string, `format_sequences`.
    """

    family: ClassVar[str] = ""

    def __init__(
        self,
        model_name: str,
        *,
        pooling_positions_0based: Sequence[int],
        wt_3di: str | None = None,
        dtype: str = "float32",
        hf_id: str | None = None,
    ) -> None:
        self.entry: BackboneEntry | None = BACKBONES.get(model_name)
        if self.entry is None and hf_id is None:
            raise BackboneError(
                f"Unknown backbone '{model_name}'. Pick one of: {', '.join(BACKBONES)}, "
                "or pass hf_id= to point at a HuggingFace repo or a local checkpoint directory."
            )
        if dtype not in DTYPES:
            raise BackboneError(f"Unsupported dtype '{dtype}'. Pick one of: {', '.join(DTYPES)}.")

        self.model_name = model_name
        self.hf_id = hf_id or (self.entry.hf_id if self.entry else "")
        self.dtype = dtype
        self.pooling_positions_0based = [int(position) for position in pooling_positions_0based]
        self.wt_3di = "".join(wt_3di.split()).lower() if wt_3di else None
        self._tokenizer: Any = None

        if not self.pooling_positions_0based:
            raise BackboneError(
                f"'{model_name}' was given an empty pooling_positions_0based, so there would be nothing to "
                "average. Pass the library's mutated sites as 1-based protein coordinates minus one, e.g. "
                "pooling_positions_0based=spec.positions_0based()."
            )
        self.embed_dim = self._resolve_embed_dim()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_name={self.model_name!r}, "
            f"n_pooled_positions={len(self.pooling_positions_0based)}, hf_id={self.hf_id!r})"
        )

    # -- construction helpers -------------------------------------------------

    def _resolve_embed_dim(self) -> int:
        if self.entry is not None:
            return int(self.entry.embed_dim)
        config = self._load_hf_config()
        for field in ("hidden_size", "d_model", "embed_dim"):
            value = getattr(config, field, None)
            if value:
                return int(value)
        raise BackboneError(
            f"Could not read an embedding dimension from the config of '{self.hf_id}'. "
            "Use a registered backbone name instead."
        )

    def _load_hf_config(self) -> Any:
        from transformers import AutoConfig

        return self._guarded(lambda: AutoConfig.from_pretrained(self.hf_id), "configuration")

    def _guarded(self, call: Any, what: str) -> Any:
        """Turn every `from_pretrained` failure into one actionable error.

        `OSError` covers a missing repo, a missing file and a dead connection, and
        `ValueError` a malformed repo id, but transformers does not confine itself to
        those: a repo whose config and tokenizer are cached while its weights are not —
        an interrupted download, or `HF_HUB_OFFLINE` on a half-warm cache — resolves the
        checkpoint to `None` and dies inside `load_state_dict` with
        `AttributeError: 'NoneType' object has no attribute 'endswith'`. That reaches a
        notebook as a stack trace through library code nobody can act on, so every
        exception is caught here. A `ColabSDError` raised deliberately inside `call`
        (an adapter's own "pip install this" message) passes through untouched.
        """
        try:
            return call()
        except ColabSDError:
            raise
        except Exception as exc:
            raise BackboneError(
                f"Could not load the {what} for '{self.model_name}' from '{self.hf_id}': "
                f"{type(exc).__name__}: {exc}. Four things cause this. The repo may be missing or "
                "renamed. The runtime may have no internet access. A previous download may have "
                "stopped part-way and left the config and tokenizer cached without the weights — "
                "clear ~/.cache/huggingface/hub and retry. Or an expired HF_TOKEN may be turning a "
                "public repo into a 401 'Repository Not Found', which hides a cache that is "
                "actually complete — clear the token, or set HF_HUB_OFFLINE=1 to read the cache "
                "directly. You can also download the repo once and pass hf_id=<local directory>, "
                "which must hold the checkpoint's own config.json, weights and tokenizer files."
            ) from exc

    # -- model ----------------------------------------------------------------

    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype named by `self.dtype`."""
        import torch

        return getattr(torch, self.dtype)

    def _from_pretrained(self, factory: Any, **kwargs: Any) -> nn.Module:
        return self._guarded(
            lambda: factory.from_pretrained(self.hf_id, torch_dtype=self.torch_dtype(), **kwargs),
            "weights",
        )

    def _load_hf_model(self) -> nn.Module:
        from transformers import AutoModel

        return self._from_pretrained(AutoModel)

    def load_model(self) -> nn.Module:
        """Load the backbone on CPU in eval mode; the caller moves it to a device."""
        model = self._load_hf_model()
        model.eval()
        hidden = getattr(getattr(model, "config", None), "hidden_size", None) or getattr(
            getattr(model, "config", None), "d_model", None
        )
        if hidden is not None and int(hidden) != self.embed_dim:
            raise BackboneError(
                f"'{self.model_name}' reports embed_dim {self.embed_dim} but '{self.hf_id}' has hidden size "
                f"{int(hidden)}. The registry entry and the checkpoint disagree; pick a matching hf_id."
            )
        return model

    def tokenizer(self) -> Any:
        """Return the cached HuggingFace tokenizer for this backbone."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = self._guarded(lambda: AutoTokenizer.from_pretrained(self.hf_id), "tokenizer")
        return self._tokenizer

    # -- forward --------------------------------------------------------------

    def format_sequences(self, sequences: list[str]) -> list[str]:
        """Turn amino-acid sequences into the strings the tokenizer expects."""
        return list(sequences)

    def tokenize(self, sequences: list[str], device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
        """Tokenize a batch of equal-length sequences, returning the residue count too."""
        from colabsd.engine.sequences import require_uniform_sequence_length

        n_residues = require_uniform_sequence_length(sequences)
        batch = self.tokenizer()(
            self.format_sequences(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        moved = {key: value.to(device) for key, value in batch.items()}
        self._check_residue_axis(moved, sequences, n_residues)
        return moved, n_residues

    @staticmethod
    def _residue_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the `(B, L_tokens)` mask selecting real residue tokens."""
        return batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()

    def _check_residue_axis(
        self,
        batch: dict[str, torch.Tensor],
        sequences: list[str],
        n_residues: int,
    ) -> None:
        """Refuse a batch whose token axis would not line up with the residues one for one."""
        import torch

        keep = self._residue_mask(batch)
        counts = keep.sum(dim=1)
        if not bool(torch.all(counts == n_residues)):
            row = int(torch.nonzero(counts != n_residues)[0, 0])
            raise BackboneError(
                f"'{self.model_name}' tokenized a {n_residues}-residue sequence into {int(counts[row])} residue "
                "tokens, so pooling positions would not line up with 1-based protein coordinates. The sequence "
                "probably contains characters this tokenizer splits differently (X, B, Z, U, O or whitespace)."
            )

        unknown = self.tokenizer().unk_token_id
        if unknown is None:
            return
        bad = batch["input_ids"].eq(unknown) & keep
        if not bool(bad.any()):
            return
        row, column = (int(value) for value in torch.nonzero(bad)[0])
        residue_1based = int(keep[row, :column].sum()) + 1
        residue = sequences[row][residue_1based - 1]
        raise BackboneError(
            f"'{self.model_name}' has no vocabulary entry for residue {residue_1based} ('{residue}') of sequence "
            f"{row} in this batch, so it would be encoded as <unk> and that residue's identity would be silently "
            "lost. SaProt covers only the 20 standard amino acids; replace X/B/Z/U/O in the library (or in the "
            "wild-type sequence), or pick a backbone whose tokenizer covers them, such as ESM2."
        )

    def encode(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the encoder and return the full token axis `(B, L_tokens, D)`."""
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state

    def _strip_special_tokens(
        self,
        hidden: torch.Tensor,
        batch: dict[str, torch.Tensor],
        n_residues: int,
    ) -> torch.Tensor:
        import torch

        keep = self._residue_mask(batch)
        if not bool(torch.all(keep.sum(dim=1) == n_residues)):
            raise BackboneError(
                f"'{self.model_name}' produced a token axis that does not hold exactly {n_residues} residue "
                "tokens, so pooling positions would not line up with 1-based protein coordinates."
            )
        return torch.stack([hidden[row][keep[row]] for row in range(hidden.shape[0])], dim=0)

    def residue_representations(
        self,
        model: nn.Module,
        sequences: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-residue embeddings `(B, len(sequence), D)` with special tokens removed."""
        batch, n_residues = self.tokenize(sequences, device)
        hidden = self.encode(model, batch)
        reps = self._strip_special_tokens(hidden, batch, n_residues)
        assert reps.shape[1] == n_residues, f"residue axis {reps.shape[1]} != sequence length {n_residues}"
        return reps

    def pooled_forward(self, model: nn.Module, sequences: list[str], device: torch.device) -> torch.Tensor:
        """Return pooled features `(B, embed_dim)`."""
        from colabsd.engine.pooling import pool

        reps = self.residue_representations(model, sequences, device)
        try:
            return pool(reps, self.pooling_positions_0based)
        except ValueError as exc:
            raise BackboneError(self._pooling_failure(int(reps.shape[1]), exc)) from exc

    def _pooling_failure(self, n_residues: int, exc: Exception) -> str:
        out_of_range = sorted(
            position for position in self.pooling_positions_0based if position < 0 or position >= n_residues
        )
        return (
            f"'{self.model_name}' could not pool a {n_residues}-residue sequence: {exc!r}. "
            f"pooling_positions_0based must be 1-based protein coordinates minus one, all within 0..{n_residues - 1}"
            + (f"; these are outside that range: {out_of_range[:10]}." if out_of_range else ".")
        )

    # -- LoRA -----------------------------------------------------------------

    def lora_module_names(self, model: nn.Module) -> list[str]:
        """Return the attention projections upstream's `inject_lora` would match."""
        from torch.nn import Linear

        from colabsd.engine.lora import target_roles

        return [
            name for name, module in model.named_modules() if name and isinstance(module, Linear) and target_roles(name)
        ]

    def check_lora_coverage(self, model: nn.Module) -> list[str]:
        """Fail early when a backbone's attention blocks are not fully LoRA-targetable."""
        from colabsd.engine.lora import validate_qkvo_coverage

        names = self.lora_module_names(model)
        try:
            validate_qkvo_coverage(names)
        except RuntimeError as exc:
            raise BackboneError(
                f"'{self.model_name}' is not LoRA-compatible as loaded: {exc}. Use a different backbone, "
                "or run it frozen instead of fine-tuned."
            ) from exc
        return names
