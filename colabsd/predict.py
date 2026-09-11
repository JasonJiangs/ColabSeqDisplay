"""Score new variants with a saved bundle.

Rebuilds the backbone adapter named in the manifest over the mutated sites the
bundled spec lists, re-injects the saved LoRA weights and head, and runs upstream's
batched prediction. Sorting, filtering and top-N export belong to the notebook, not
here.

Sequences are built by `colabsd.data.build_sequences`, the same function the training path
uses, so a residue the loader would have rejected cannot slip into a prediction table.
`build_sequences` here is nothing but a spec-first alias of it: one implementation, in
`colabsd.data`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from colabsd.data import build_sequences as _build_sequences
from colabsd.data import require_unique_columns
from colabsd.errors import BackboneError, BundleError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import torch

    from colabsd.bundle import Bundle
    from colabsd.spec import LibrarySpec


def score_variants(
    bundle: Bundle,
    variants_df: pd.DataFrame,
    *,
    device: str = "cuda",
    batch_size: int = 8,
) -> pd.DataFrame:
    """Predict every condition for each row of *variants_df*.

    Returns the spec's mutation columns, one column per condition, and the
    `pred_mean` / `pred_min` summaries a screening decision is usually made on.
    """
    import pandas as pd

    spec = bundle.spec
    columns = list(spec.mutation_columns)
    conditions = list(spec.condition_columns)
    missing = [column for column in columns if column not in variants_df.columns]
    if missing:
        raise DataError(
            f"variants_df is missing the mutation columns {missing}. It must carry the same columns as the "
            f"training library: {columns}."
        )
    if len(variants_df) == 0:
        raise DataError("variants_df has no rows; nothing to score.")
    require_unique_columns(variants_df, columns, source="variants_df")
    clashing = sorted(set(columns) & set(conditions))
    if clashing:
        raise DataError(
            f"The bundled spec uses {clashing} as both a mutation column and a condition column, so the "
            "predictions would overwrite the residues they were made from. Re-train with distinct column names."
        )

    sequences = build_sequences(spec, variants_df)
    predictions = score_sequences(bundle, sequences, device=device, batch_size=batch_size)

    frame = variants_df.loc[:, columns].reset_index(drop=True).copy()
    for index, condition in enumerate(conditions):
        frame[condition] = predictions[:, index]
    frame["pred_mean"] = predictions.mean(axis=1)
    frame["pred_min"] = predictions.min(axis=1)
    return pd.DataFrame(frame)


def build_sequences(spec: LibrarySpec, variants_df: pd.DataFrame) -> list[str]:
    """Spec-first alias of `colabsd.data.build_sequences(frame, spec)`, which does the work.

    Scoring starts from a bundle, so the spec reads first at this call site. The argument
    order is the only difference: there is no second implementation and no second residue table.
    """
    return _build_sequences(variants_df, spec)


def score_sequences(
    bundle: Bundle,
    sequences: Sequence[str],
    *,
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """Return an `(N, T)` prediction array on the assay's own scale when a scaler was bundled."""
    from colabsd.engine.train_config import predict as engine_predict

    sequences = [str(item) for item in sequences]
    length = len(bundle.spec.wt_sequence)
    wrong = {len(sequence) for sequence in sequences}.difference({length})
    if wrong:
        raise DataError(
            f"Sequences of length {sorted(wrong)} were built for a {length}-residue wild type. "
            "Every variant must be the full-length sequence with substitutions."
        )
    adapter, model, head, resolved = load_for_inference(bundle, device=device)
    predictions = engine_predict(
        adapter=adapter,
        model=model,
        head=head,
        sequences=sequences,
        indices=list(range(len(sequences))),
        batch_size=max(1, int(batch_size)),
        device=resolved,
    )
    return _inverse_scale(bundle, np.asarray(predictions, dtype=np.float32))


def load_for_inference(bundle: Bundle, *, device: str = "cuda") -> tuple[Any, torch.nn.Module, torch.nn.Module, Any]:
    """Rebuild the adapter, the LoRA-injected backbone and the head from a bundle."""
    from colabsd.engine.heads import MLPHead
    from colabsd.engine.lora import inject_lora

    resolved = _resolve_device(device)
    adapter = _create_adapter(bundle)
    model = adapter.load_model().to(resolved)
    for parameter in model.parameters():
        parameter.requires_grad = False
    params = bundle.params
    try:
        matched = inject_lora(
            model,
            rank=int(params["lora_rank"]),
            alpha=int(params["lora_alpha"]),
            dropout=float(params.get("lora_dropout", 0.0)),
        )
    except KeyError as exc:
        raise BundleError(
            f"The bundle manifest has no {exc} hyperparameter, so the LoRA layers cannot be rebuilt. "
            "Re-create the bundle from the training run."
        ) from exc
    if not matched:
        raise BackboneError(
            f"No LoRA query/key/value/output projection matched in {bundle.adapter_name}, so the bundled adapter "
            "weights cannot be re-injected. The installed backbone is not the one the bundle was trained on."
        )
    _load_lora(model, bundle.lora_state)

    head = MLPHead().build_model(adapter.embed_dim, n_outputs=len(bundle.condition_columns)).to(resolved)
    try:
        head.load_state_dict({name: value.to(resolved) for name, value in bundle.head_state.items()})
    except RuntimeError as exc:
        raise BundleError(
            f"The bundled head does not fit a {adapter.embed_dim}-dimensional {bundle.adapter_name} backbone "
            f"with {len(bundle.condition_columns)} conditions ({exc}). The bundle and the backbone disagree; "
            "re-create the bundle."
        ) from exc
    model.eval()
    head.eval()
    return adapter, model, head, resolved


def _create_adapter(bundle: Bundle) -> Any:
    try:
        from colabsd.backbones.registry import create_adapter
    except ImportError as exc:  # pragma: no cover - registry is a hard dependency
        raise BackboneError(
            f"colabsd.backbones.registry is unavailable, so {bundle.adapter_name} cannot be rebuilt ({exc})."
        ) from exc
    optional = {
        name: value
        for name, value in (("dtype", bundle.adapter_dtype), ("hf_id", bundle.adapter_hf_id))
        if value
    }
    try:
        return create_adapter(
            bundle.adapter_name,
            pooling_positions_0based=bundle.pooling_positions_0based,
            wt_3di=bundle.spec.wt_3di,
            **optional,
        )
    except (KeyError, BackboneError) as exc:
        raise BackboneError(
            f"The bundle names backbone {bundle.adapter_name!r}, which this colabsd cannot build ({exc}). "
            "Upgrade colabsd, or re-train with a backbone listed by colabsd.backbones.registry.available()."
        ) from exc


def _load_lora(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = model.state_dict()
    missing = sorted(set(state).difference(own))
    if missing:
        raise BundleError(
            f"{len(missing)} bundled LoRA tensors have no matching layer in the backbone, e.g. {missing[:3]}. "
            "The bundle was trained against a different backbone or LoRA rank."
        )
    for name, value in state.items():
        if own[name].shape != value.shape:
            raise BundleError(
                f"Bundled LoRA tensor {name} has shape {tuple(value.shape)} but the rebuilt layer expects "
                f"{tuple(own[name].shape)}. Check that lora_rank in the manifest matches the saved weights."
            )
        own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))


def _inverse_scale(bundle: Bundle, predictions: np.ndarray) -> np.ndarray:
    scaler = bundle.label_scaler
    if not scaler:
        warnings.warn(
            "This bundle carries no label scaler, so predictions stay on the z-scored training scale. "
            "Ranking is unaffected; absolute values are not comparable to the assay.",
            stacklevel=3,
        )
        return predictions
    mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, -1)
    if mean.shape[1] != predictions.shape[1]:
        raise BundleError(
            f"The bundled label scaler covers {mean.shape[1]} conditions but the head predicts "
            f"{predictions.shape[1]}. Re-create the bundle from the training run."
        )
    return (predictions * std + mean).astype(np.float32)


def _resolve_device(device: str) -> torch.device:
    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is not available; scoring on CPU instead. This is slow for large backbones.", stacklevel=3)
        return torch.device("cpu")
    return requested

