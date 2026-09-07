"""Fine-tune one LoRA configuration and score the split it was given.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(Python package ``seqdisplay_opt``), which owns this science: every number the
paper reports came out of this loop. Original modules:

* ``seqdisplay_opt/finetuning/lora_reevaluate.py`` — ``load_lora_best_config``,
  ``SourceTrial``, ``train_eval_config`` and the private ``_set_reproducible_seed``,
  ``_configure_model``, ``_lora_state_dict``, ``_load_lora_state_dict``,
  ``_clone_head_state``, ``_load_head_state`` and ``_evaluate_split``.
* ``seqdisplay_opt/finetuning/lora_optuna.py`` — ``LabelScaler`` and the private
  ``_device`` and ``_predict``.

See ``ATTRIBUTION.md``.

Left behind: everything Optuna. Upstream's two modules also build and read study
databases (``run_model_study``, ``load_top_trials``, ``train_one_trial``, trial
pruning and the per-trial artefact writer), take an exclusive ``fcntl`` lock so
two HPC workers cannot share a study, and expose two ``argparse`` command lines.
ColabSeqDisplay runs *one* already-chosen configuration out of ``config/best/``,
so none of that is reachable from a notebook and none of it is here.

Changed while copying, both deliberate and both covered by the side-by-side
equivalence test in ``tests/test_engine_train_config.py``:

1. **``micro_batch_size: auto`` no longer raises in the loader.** Upstream
   defaults ``training.micro_batch_size`` to the string ``"auto"`` and then calls
   ``int()`` on it, so a best-config that omits the field dies with
   ``ValueError: invalid literal for int() with base 10: 'auto'`` -- while
   ``train_eval_config`` resolves the very same ``"auto"`` correctly. The two now
   agree because both go through :func:`resolve_micro_batch_size`. Every entry in
   ``config/best/`` sets the field explicitly, so no shipped configuration parses
   differently than it did upstream.
2. **The names we depend on are public.** ``_predict`` is :func:`predict`,
   ``_configure_model`` is :func:`configure_model`, and the state-dict and
   split-evaluation helpers lost their underscores: they are this package's own
   API now, not another package's internals.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW

from colabsd.engine.adapters import SequenceAdapter
from colabsd.engine.config_space import pooling_positions_1based, validation_metric_name
from colabsd.engine.heads import MLPHead
from colabsd.engine.lora import LORA_TARGET_MODULES, inject_lora, lora_parameters
from colabsd.engine.metrics import evaluate_predictions, summarize_metrics, validation_log_row
from colabsd.engine.training import batch_indices, train_epoch

# Upstream's ceiling on a derived micro-batch: with `micro_batch_size: auto` the
# forward pass runs at most this many sequences at a time, and gradient
# accumulation makes up the configured effective batch.
AUTO_MICRO_BATCH_SIZE = 4

# The three validation objectives a best-config may select on.
SELECTION_OBJECTIVES = ("mean_validation_r2", "mean_validation_pearson", "mean_validation_spearman")


@dataclass(frozen=True)
class SourceTrial:
    """Provenance of a configuration that came out of an Optuna study."""

    number: int
    value: float
    params: dict[str, Any]


@dataclass
class LabelScaler:
    """Per-target z-score fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> LabelScaler:
        return cls(
            mean=values.mean(axis=0, keepdims=True).astype(np.float32),
            std=(values.std(axis=0, keepdims=True) + 1e-6).astype(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)


def resolve_micro_batch_size(micro: Any, effective_batch_size: int) -> int:
    """Resolve ``training.micro_batch_size``, including upstream's ``"auto"``.

    ``"auto"`` means "as many sequences per forward pass as fit comfortably", which
    upstream caps at :data:`AUTO_MICRO_BATCH_SIZE` and never lets exceed the
    effective batch. This one function is what makes the loader and the trainer
    agree; see the module docstring.
    """
    if micro == "auto":
        return min(AUTO_MICRO_BATCH_SIZE, int(effective_batch_size))
    return int(micro)


def load_lora_best_config(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load one fixed LoRA configuration without requiring an Optuna study.

    Returns the model name, the seven LoRA hyperparameters and the runtime config
    that :func:`train_eval_config` expects.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    model_name = raw.get("model")
    params = raw.get("parameters", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})

    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model must be a non-empty string")
    objective = str(evaluation.get("selection_objective", "mean_validation_spearman"))
    if objective not in set(SELECTION_OBJECTIVES):
        raise ValueError(f"Unsupported evaluation.selection_objective: {objective}")

    required_params = {
        "adapter_lr",
        "head_lr",
        "adapter_weight_decay",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "effective_batch_size",
    }
    missing_params = required_params.difference(params)
    if missing_params:
        raise ValueError(f"Missing LoRA parameters: {sorted(missing_params)}")
    split_seeds = evaluation.get("split_seeds")
    model_seeds = evaluation.get("model_seeds")
    for name, values in (("split_seeds", split_seeds), ("model_seeds", model_seeds)):
        if not isinstance(values, list) or not values or any(not isinstance(value, int) for value in values):
            raise ValueError(f"evaluation.{name} must be a non-empty list of integers")

    fixed_defaults = {
        "tuning_method": "lora",
        "head": "mlp",
        "pooling": "cosine_p90_mean",
        "max_epochs": 20,
        "early_stopping_patience": 3,
        "max_grad_norm": 1.0,
        "head_weight_decay": 0.01,
        "loss": "mse",
        "micro_batch_size": "auto",
        "gradient_accumulation": "derive_from_effective_batch_size",
        "label_preprocessing": "train_split_zscore",
        "evaluate_test_during_optimization": False,
    }
    fixed = {**fixed_defaults, **training}
    if fixed["tuning_method"] != "lora":
        raise ValueError("training.tuning_method must be 'lora'")
    if fixed["head"] != "mlp":
        raise ValueError("Only the MLP LoRA head is currently supported")
    if fixed["loss"] != "mse":
        raise ValueError("Only MSE loss is currently supported")
    if fixed["label_preprocessing"] != "train_split_zscore":
        raise ValueError("Only train-split z-score label preprocessing is currently supported")
    if fixed["gradient_accumulation"] != "derive_from_effective_batch_size":
        raise ValueError("gradient_accumulation must derive from effective_batch_size")
    effective_batch_size = int(params["effective_batch_size"])
    # Upstream reads int(fixed["micro_batch_size"]) here, which cannot parse its own
    # "auto" default; the trainer resolves "auto" a few lines further down the file.
    micro_batch_size = resolve_micro_batch_size(fixed["micro_batch_size"], effective_batch_size)
    if micro_batch_size < 1 or effective_batch_size < micro_batch_size:
        raise ValueError("effective_batch_size must be at least micro_batch_size")
    if effective_batch_size % micro_batch_size != 0:
        raise ValueError("effective_batch_size must be exactly divisible by micro_batch_size")

    runtime_config = {
        "study": {
            "direction": "maximize",
            "objective": objective,
            "top_k": 1,
            "re_evaluation_split_seeds": split_seeds,
            "re_evaluation_seeds": model_seeds,
        },
        # Upstream's fallback names its own bundled 5NNK database. It is kept
        # verbatim so a config that omits `protein:` parses identically; colabsd
        # always passes one, written by colabsd.protein_db.write_protein_record.
        "protein": raw.get(
            "protein",
            {"id": "slugcas9", "database": "data/protein/proteins.yaml"},
        ),
        "fixed": fixed,
        "data": raw.get("data", {}),
    }
    return model_name, dict(params), runtime_config


def default_device() -> torch.device:
    """Return CUDA when the runtime has it, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy and torch, and put cuDNN in deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_model(
    *,
    adapter: SequenceAdapter,
    params: dict[str, Any],
    n_outputs: int,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, list[str]]:
    """Load the backbone frozen, inject LoRA into it and build the MLP head."""
    model = adapter.load_model().to(device)
    for param in model.parameters():
        param.requires_grad = False
    matched = inject_lora(
        model,
        rank=int(params["lora_rank"]),
        alpha=int(params["lora_alpha"]),
        dropout=float(params["lora_dropout"]),
    )
    if not matched:
        raise RuntimeError(f"No LoRA query/key/value/output projections matched in {adapter.model_name}.")
    head = MLPHead().build_model(adapter.embed_dim, n_outputs=n_outputs).to(device)
    return model, head, matched


def predict(
    *,
    adapter: SequenceAdapter,
    model: nn.Module,
    head: nn.Module,
    sequences: list[str],
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score *indices* of *sequences* in evaluation mode, on the model's own scale."""
    model.eval()
    head.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batch_indices(indices, batch_size, seed=0, shuffle=False):
            batch_idx = [int(i) for i in batch]
            pooled = adapter.pooled_forward(model, [sequences[i] for i in batch_idx], device).float()
            preds.append(head(pooled).detach().cpu())
    return torch.cat(preds, dim=0).numpy().astype(np.float32)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only the LoRA A/B matrices, detached on the CPU."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "lora_a" in name or "lora_b" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    """Copy LoRA weights back into an injected model, refusing keys it does not have."""
    own_state = model.state_dict()
    missing = sorted(set(state).difference(own_state))
    if missing:
        raise RuntimeError(f"Checkpoint contains LoRA keys missing from model: {missing[:5]}")
    for name, value in state.items():
        own_state[name].copy_(value.to(device=device, dtype=own_state[name].dtype))


def clone_head_state(head: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU copy of the head weights."""
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def load_head_state(head: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    """Restore head weights saved by :func:`clone_head_state`."""
    head.load_state_dict({name: value.to(device) for name, value in state.items()})


def evaluate_split(
    *,
    adapter: SequenceAdapter,
    model: nn.Module,
    head: nn.Module,
    sequences: list[str],
    targets: np.ndarray,
    indices: list[int],
    scaler: LabelScaler,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict one partition, undo the label z-score and score it per target."""
    pred_scaled = predict(
        adapter=adapter,
        model=model,
        head=head,
        sequences=sequences,
        indices=indices,
        batch_size=batch_size,
        device=device,
    )
    pred = scaler.inverse(pred_scaled)
    metrics = evaluate_predictions(targets[indices], pred)
    return pred, metrics


def train_eval_config(
    *,
    adapter: SequenceAdapter,
    sequences: list[str],
    targets: np.ndarray,
    split: dict[str, list[int]],
    params: dict[str, Any],
    config: dict[str, Any],
    split_seed: int,
    seed: int,
    output_dir: Path,
    source_trial: SourceTrial | None,
) -> dict[str, Any]:
    """Train one LoRA configuration on one split and score validation *and* test.

    Early stopping keeps the epoch with the best validation objective, restores it,
    and writes ``best_checkpoint.pt``, ``predictions.npz``, ``training_log.json`` and
    ``metrics.json`` into *output_dir*. The returned dictionary is ``metrics.json``.
    """
    set_reproducible_seed(seed)
    fixed = config["fixed"]
    objective_name = str(config["study"]["objective"])
    objective_metric = validation_metric_name(config)
    mutation_positions_1based = pooling_positions_1based(config)
    device = default_device()
    train_idx = [int(i) for i in split["train_idx"]]
    val_idx = [int(i) for i in split["val_idx"]]
    test_idx = [int(i) for i in split["test_idx"]]
    scaler = LabelScaler.fit(targets[train_idx])
    y_scaled = scaler.transform(targets)

    effective_batch_size = int(params["effective_batch_size"])
    micro_batch_size = resolve_micro_batch_size(fixed.get("micro_batch_size", "auto"), effective_batch_size)
    grad_accum = max(1, math.ceil(effective_batch_size / micro_batch_size))

    model, head, matched = configure_model(
        adapter=adapter,
        params=params,
        n_outputs=targets.shape[1],
        device=device,
    )
    optimizer = AdamW(
        [
            {
                "params": lora_parameters(model),
                "lr": float(params["adapter_lr"]),
                "weight_decay": float(params["adapter_weight_decay"]),
            },
            {
                "params": [p for p in head.parameters() if p.requires_grad],
                "lr": float(params["head_lr"]),
                "weight_decay": float(fixed.get("head_weight_decay", 0.01)),
            },
        ]
    )
    max_epochs = int(fixed["max_epochs"])
    patience_limit = int(fixed["early_stopping_patience"])
    best_score = -float("inf")
    best_epoch = 0
    best_lora_state: dict[str, torch.Tensor] | None = None
    best_head_state: dict[str, torch.Tensor] | None = None
    best_metric_summary: dict[str, float] | None = None
    patience = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(
            adapter=adapter,
            model=model,
            head=head,
            optimizer=optimizer,
            sequences=sequences,
            scaled_targets=y_scaled,
            train_indices=train_idx,
            micro_batch_size=micro_batch_size,
            accumulation_steps=grad_accum,
            seed=seed + epoch,
            device=device,
            max_grad_norm=float(fixed.get("max_grad_norm", 1.0)),
        )

        _, val_metrics = evaluate_split(
            adapter=adapter,
            model=model,
            head=head,
            sequences=sequences,
            targets=targets,
            indices=val_idx,
            scaler=scaler,
            batch_size=micro_batch_size,
            device=device,
        )
        metric_summary = summarize_metrics(val_metrics)
        score = metric_summary[objective_metric]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "objective": score,
            **validation_log_row(metric_summary),
        }
        log_rows.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_lora_state = lora_state_dict(model)
            best_head_state = clone_head_state(head)
            best_metric_summary = dict(metric_summary)
            patience = 0
        else:
            patience += 1
        if patience >= patience_limit:
            break

    if best_lora_state is None or best_head_state is None or best_metric_summary is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    load_lora_state_dict(model, best_lora_state, device)
    load_head_state(head, best_head_state, device)
    val_predictions, val_metrics = evaluate_split(
        adapter=adapter,
        model=model,
        head=head,
        sequences=sequences,
        targets=targets,
        indices=val_idx,
        scaler=scaler,
        batch_size=micro_batch_size,
        device=device,
    )
    test_predictions, test_metrics = evaluate_split(
        adapter=adapter,
        model=model,
        head=head,
        sequences=sequences,
        targets=targets,
        indices=test_idx,
        scaler=scaler,
        batch_size=micro_batch_size,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": adapter.model_name,
        "split_seed": split_seed,
        "seed": seed,
        "params": params,
        "matched_lora_modules": matched,
        "lora_target_modules": LORA_TARGET_MODULES,
        "selection_objective": objective_name,
        "selection_metric": objective_metric,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": best_metric_summary,
        "best_validation_r2": best_metric_summary["R2"],
        "best_validation_spearman": best_metric_summary["Spearman"],
        "lora_state_dict": best_lora_state,
        "head_state_dict": best_head_state,
        "label_scaler": {
            "mean": scaler.mean.tolist(),
            "std": scaler.std.tolist(),
        },
        "pooling": fixed["pooling"],
        "mutation_positions_1based": mutation_positions_1based,
    }
    if source_trial is not None:
        checkpoint["source_trial"] = asdict(source_trial)
    torch.save(checkpoint, output_dir / "best_checkpoint.pt")
    np.savez_compressed(
        output_dir / "predictions.npz",
        validation_indices=np.asarray(val_idx, dtype=np.int64),
        validation_targets=np.asarray(targets[val_idx], dtype=np.float32),
        validation_predictions=np.asarray(val_predictions, dtype=np.float32),
        test_indices=np.asarray(test_idx, dtype=np.int64),
        test_targets=np.asarray(targets[test_idx], dtype=np.float32),
        test_predictions=np.asarray(test_predictions, dtype=np.float32),
    )
    (output_dir / "training_log.json").write_text(json.dumps(log_rows, indent=2))
    if source_trial is not None:
        (output_dir / "source_trial.json").write_text(json.dumps(asdict(source_trial), indent=2))
    validation_summary = summarize_metrics(val_metrics)
    test_summary = summarize_metrics(test_metrics)
    metrics = {
        "model": adapter.model_name,
        "split_seed": split_seed,
        "seed": seed,
        "params": params,
        "lora_target_modules": LORA_TARGET_MODULES,
        "selection_objective": objective_name,
        "selection_metric": objective_metric,
        "pooling": fixed["pooling"],
        "mutation_positions_1based": mutation_positions_1based,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_r2": best_metric_summary["R2"],
        "best_validation_spearman": best_metric_summary["Spearman"],
        "validation": {
            "mean_R2": validation_summary["R2"],
            "mean_Pearson": validation_summary["Pearson"],
            "mean_Spearman": validation_summary["Spearman"],
            "mean": validation_summary,
            "per_target": val_metrics,
        },
        "test": {
            "mean_R2": test_summary["R2"],
            "mean_Pearson": test_summary["Pearson"],
            "mean_Spearman": test_summary["Spearman"],
            "mean": test_summary,
            "per_target": test_metrics,
        },
        "checkpoint": "best_checkpoint.pt",
        "timestamp": datetime.now().isoformat(),
    }
    if source_trial is not None:
        metrics["source_trial"] = asdict(source_trial)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
