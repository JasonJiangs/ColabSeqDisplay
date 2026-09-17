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

Changed while copying, all deliberate and covered by the side-by-side equivalence test
in ``tests/test_engine_train_config.py``:

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
3. **The two defaults that name upstream's own study are gone.** Upstream defaults
   ``training.pooling`` to the strategy its study selected and ``protein:`` to the
   protein and database packaged in its checkout. colabsd pools one way and ships no
   protein database, so the pooling defaults to
   :data:`colabsd.engine.pooling.POOLING_NAME` and the protein block defaults to empty:
   a config that omits it is told which field is missing instead of being resolved
   against a protein nobody loaded. Every shipped ``config/best/`` entry sets the
   pooling explicitly and none sets ``protein:``, so nothing on disk parses differently.
4. **The loop can report progress.** :func:`train_eval_config` takes ``on_epoch`` and
   ``on_batch`` callbacks, both defaulting to ``None``, and passes the batch one down into
   :func:`colabsd.engine.training.train_epoch`. colabsd runs in a notebook cell, where a
   fine-tune that reports nothing for tens of minutes is indistinguishable from a hung one.
   The callbacks only read: what is trained, logged and written is identical whether they are
   given or not, which is why the side-by-side tests pass neither and still compare every number.
5. **A stop press that lands outside the epoch loop raises an ordinary exception.** Upstream has
   no notebook and no stop button: a ``KeyboardInterrupt`` anywhere in this function is its
   caller's problem. colabsd runs inside a widget callback whose guard catches ``Exception``,
   which ``KeyboardInterrupt`` is not, so an interrupt during the final evaluation or the
   artefact writes killed the run with no message at all. The epoch loop still absorbs one
   (that is early stopping by hand); everything after it records the cut-short run on disk and
   re-raises it as :class:`TrainingStopped`, which every guard can see.
6. **The per-target metric names come from the config.** Upstream names the first four target
   columns after its own study's four SlugCas9 PAMs. When ``config["data"]["target_names"]``
   lists the user's condition columns -- :func:`colabsd.train.finetune` always sets it -- the
   ``metrics.json`` this writes carries those names instead. Without the key the upstream
   default is unchanged.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Callable, Mapping
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
from colabsd.engine.pooling import POOLING_NAME
from colabsd.engine.training import BatchCallback, batch_indices, train_epoch

# Upstream's ceiling on a derived micro-batch: with `micro_batch_size: auto` the
# forward pass runs at most this many sequences at a time, and gradient
# accumulation makes up the configured effective batch.
AUTO_MICRO_BATCH_SIZE = 4

# The three validation objectives a best-config may select on.
SELECTION_OBJECTIVES = ("mean_validation_r2", "mean_validation_pearson", "mean_validation_spearman")

#: ``on_epoch(epoch, max_epochs, validation_score, train_loss)`` -- one epoch has closed.
#: ``epoch`` is 0-based, and the score is the objective named by ``study.objective``.
EpochProgress = Callable[[int, int, float, float], None]

#: ``on_batch(epoch, max_epochs, step, n_batches, running_loss)`` -- where inside the epoch the
#: fine-tune has got to, and its training loss so far. Called once per micro-batch, which is
#: thousands of times an epoch: a caller that draws a widget from it has to throttle it itself.
BatchProgress = Callable[[int, int, int, int, float], None]


class TrainingStopped(RuntimeError):
    """A stop press that landed outside the epoch loop, as an exception a guard can catch.

    A `KeyboardInterrupt` inside the epoch loop is early stopping by hand: the loop ends and
    the run finishes normally, marked `stopped_early: "interrupted"`. One that lands after it
    -- during the final validation and test passes, or while the artefacts are being written --
    cannot be absorbed that way, because the numbers those passes were going to produce do not
    exist. The weights of the best epoch are still on disk, so the run is recorded as cut short
    and this is raised. It derives from `RuntimeError`, i.e. from `Exception`, because the panel
    that calls this runs inside a widget callback whose guard catches `Exception`: raising
    `KeyboardInterrupt` past it is what made the run die with a blank log and a frozen progress
    line.
    """


def configured_target_names(config: Mapping[str, Any], n_targets: int) -> list[str] | None:
    """The condition names `config["data"]["target_names"]` lists, or None.

    Upstream names unnamed target columns after its own study's PAMs, which is a surprise in
    anybody else's `metrics.json`. `colabsd.train.finetune` writes the library's own condition
    columns here; a config that does not carry them (or carries the wrong number of them) falls
    back to the upstream default rather than mislabelling anything.
    """
    names = (config.get("data") or {}).get("target_names")
    if not isinstance(names, (list, tuple)) or len(names) != int(n_targets):
        return None
    return [str(name) for name in names]


def relabel_per_target(per_target: Mapping[str, Any], names: list[str] | None) -> dict[str, Any]:
    """Re-key one per-target metric block positionally, leaving the numbers alone."""
    if names is None or len(names) != len(per_target):
        return dict(per_target)
    return {name: dict(block) for name, block in zip(names, per_target.values(), strict=True)}


def epoch_batch_callback(on_batch: BatchProgress | None, epoch: int, max_epochs: int) -> BatchCallback | None:
    """Bind the epoch a micro-batch belongs to onto the caller's batch callback."""
    if on_batch is None:
        return None
    return lambda step, n_batches, loss: on_batch(epoch, max_epochs, step, n_batches, loss)


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
        "pooling": POOLING_NAME,
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
        # No fallback. Upstream defaults this to its own bundled protein and database;
        # colabsd ships neither, so an omitted block stays empty and
        # `colabsd.engine.config_space` says which field is missing rather than silently
        # resolving the pooled residues of somebody else's protein.
        "protein": raw.get("protein", {}),
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


#: The file a run's weights live in, written every time the validation score improves and
#: once more when the run finishes. Named here because `colabsd.train` and `colabsd.bundle`
#: both reach for it, and a second spelling drifting from this one is a bundle that cannot
#: find the weights it is made of.
CHECKPOINT_NAME = "best_checkpoint.pt"


def build_checkpoint(
    *,
    adapter: SequenceAdapter,
    split_seed: int,
    seed: int,
    params: Mapping[str, Any],
    matched: Any,
    objective_name: str,
    objective_metric: str,
    best_epoch: int,
    best_score: float,
    best_metric_summary: Mapping[str, float],
    best_lora_state: Mapping[str, torch.Tensor],
    best_head_state: Mapping[str, torch.Tensor],
    scaler: LabelScaler,
    pooling: str,
    mutation_positions_1based: Any,
    source_trial: Any = None,
    complete: bool = False,
    stopped_early: str | None = None,
    epochs_run: int = 0,
    epochs_budget: int | None = None,
) -> dict[str, Any]:
    """Everything needed to rebuild the trained model, for the epoch holding the record.

    Built in one place because it is written twice: once per improving epoch, so that a
    session that dies leaves something behind, and once at the end with `complete=True`.
    A reader that finds `complete: False` has the weights of a run that was cut short --
    they are real and loadable, and its `metrics.json` and `predictions.npz` are missing.
    `colabsd.train.recover_runs` is that reader, and `epochs_run` and `stopped_early` are
    what let it say how far the run got without any other file to go on.
    """
    payload: dict[str, Any] = {
        "model_name": adapter.model_name,
        "split_seed": split_seed,
        "seed": seed,
        "params": dict(params),
        "matched_lora_modules": matched,
        "lora_target_modules": LORA_TARGET_MODULES,
        "selection_objective": objective_name,
        "selection_metric": objective_metric,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": dict(best_metric_summary),
        "best_validation_r2": best_metric_summary.get("R2"),
        "best_validation_spearman": best_metric_summary.get("Spearman"),
        "lora_state_dict": dict(best_lora_state),
        "head_state_dict": dict(best_head_state),
        "label_scaler": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()},
        "pooling": pooling,
        "mutation_positions_1based": mutation_positions_1based,
        "complete": complete,
        # Why the run ended and how far it got, in the file that survives when nothing else
        # does: `None` while the loop is still running, "patience" for early stopping and
        # "interrupted" for a stop press.
        "stopped_early": stopped_early,
        "epochs_run": int(epochs_run),
        # The budget *this run* was given, not the registry's. A session that vanished
        # leaves only this file, and the reader that rescues it would otherwise have to fall
        # back on the looked-up number -- which is wrong for anyone who moved the box.
        "epochs_budget": None if epochs_budget is None else int(epochs_budget),
    }
    if source_trial is not None:
        payload["source_trial"] = asdict(source_trial)
    return payload


def write_checkpoint_atomically(payload: dict[str, Any], path: Path) -> None:
    """`torch.save` to a sibling temp file, then rename over *path*.

    The point of writing a checkpoint every time the score improves is to survive a session
    that ends without warning. A plain `torch.save` straight onto the destination turns a
    crash *during* the write into a truncated file where the last good one used to be --
    which would make this feature the cause of the loss it exists to prevent. `os.replace`
    is atomic on POSIX, so the file at `path` is always one complete checkpoint or the
    previous one.
    """
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
    on_epoch: EpochProgress | None = None,
    on_batch: BatchProgress | None = None,
) -> dict[str, Any]:
    """Train one LoRA configuration on one split and score validation *and* test.

    Early stopping keeps the epoch with the best validation objective, restores it,
    and writes ``best_checkpoint.pt``, ``predictions.npz``, ``training_log.json`` and
    ``metrics.json`` into *output_dir*. The returned dictionary is ``metrics.json``.

    A stop press inside the epoch loop ends it and finishes the run, which then records
    ``stopped_early: "interrupted"`` and the number of epochs it got through. One that lands
    after the loop cannot be finished: the checkpoint is rewritten with ``complete: False`` so
    the weights say what they are, and :class:`TrainingStopped` is raised.
    """
    set_reproducible_seed(seed)
    fixed = config["fixed"]
    objective_name = str(config["study"]["objective"])
    objective_metric = validation_metric_name(config)
    mutation_positions_1based = pooling_positions_1based(config)
    target_names = configured_target_names(config, int(targets.shape[1]))
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
    stopped_early: str | None = None
    output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot() -> dict[str, Any]:
        """The checkpoint as it stands, for the epoch that currently holds the record."""
        return build_checkpoint(
            adapter=adapter,
            split_seed=split_seed,
            seed=seed,
            params=params,
            matched=matched,
            objective_name=objective_name,
            objective_metric=objective_metric,
            best_epoch=best_epoch,
            best_score=best_score,
            best_metric_summary=best_metric_summary or {},
            best_lora_state=best_lora_state or {},
            best_head_state=best_head_state or {},
            scaler=scaler,
            pooling=str(fixed["pooling"]),
            mutation_positions_1based=mutation_positions_1based,
            source_trial=source_trial,
            epochs_run=len(log_rows),
            epochs_budget=max_epochs,
        )

    try:
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
                on_batch=epoch_batch_callback(on_batch, epoch, max_epochs),
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
            if on_epoch is not None:
                # An epoch closes on the only number that decides anything -- the validation score
                # this epoch is kept or discarded on -- so it is reported even though `on_batch` has
                # been reporting the loss all the way through the epoch that produced it.
                on_epoch(epoch, max_epochs, float(score), float(train_loss))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_lora_state = lora_state_dict(model)
                best_head_state = clone_head_state(head)
                best_metric_summary = dict(metric_summary)
                patience = 0
                # On disk the moment it becomes the best, not at the end of the run. A free
                # Colab session is pre-empted without notice, and until this line the whole
                # run lived in memory: an hour of training ended as nothing. Written only on
                # an improvement, so what survives is the epoch that would have been exported
                # anyway -- never the latest, which may be worse than one already passed.
                write_checkpoint_atomically(snapshot(), output_dir / CHECKPOINT_NAME)
            else:
                patience += 1
            if patience >= patience_limit:
                stopped_early = "patience"
                break

    except KeyboardInterrupt:
        # Colab's own stop button, and the only one that can reach a training loop: the
        # panel runs this synchronously inside a widget callback, so no second widget's
        # handler can fire while it is here. Interrupting is treated exactly like early
        # stopping -- the epoch loop ends, the best epoch so far is what gets restored,
        # re-evaluated and written -- and the run says which of the two ended it.
        stopped_early = "interrupted"
        if best_lora_state is None:
            # `TrainingStopped` rather than a bare RuntimeError so that every stop press this
            # function can see raises one class -- and so that it reaches the panel's guard.
            raise TrainingStopped(
                "Interrupted before the first epoch finished, so there is no checkpoint to keep. "
                "Lower `max_epochs` or let one epoch complete."
            ) from None

    if best_lora_state is None or best_head_state is None or best_metric_summary is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    try:
        return _finalise_run(
            adapter=adapter,
            model=model,
            head=head,
            sequences=sequences,
            targets=targets,
            val_idx=val_idx,
            test_idx=test_idx,
            scaler=scaler,
            micro_batch_size=micro_batch_size,
            device=device,
            params=params,
            fixed=fixed,
            split_seed=split_seed,
            seed=seed,
            matched=matched,
            objective_name=objective_name,
            objective_metric=objective_metric,
            mutation_positions_1based=mutation_positions_1based,
            target_names=target_names,
            best_epoch=best_epoch,
            best_score=best_score,
            best_metric_summary=best_metric_summary,
            best_lora_state=best_lora_state,
            best_head_state=best_head_state,
            log_rows=log_rows,
            stopped_early=stopped_early,
            output_dir=output_dir,
            source_trial=source_trial,
        )
    except KeyboardInterrupt:
        # A second stop press, landing in the final evaluation or the artefact writes. The
        # epoch loop is over and its numbers are gone, so this run cannot be finished; what it
        # can do is leave the best epoch's weights saying exactly that, and raise something the
        # notebook's guard can catch. `colabsd.train.finetune` writes no `colabsd_run.json` for
        # a run that raised, so the next Train press retrains it rather than reusing it.
        write_checkpoint_atomically(
            build_checkpoint(
                adapter=adapter,
                split_seed=split_seed,
                seed=seed,
                params=params,
                matched=matched,
                objective_name=objective_name,
                objective_metric=objective_metric,
                best_epoch=best_epoch,
                best_score=best_score,
                best_metric_summary=best_metric_summary,
                best_lora_state=best_lora_state,
                best_head_state=best_head_state,
                scaler=scaler,
                pooling=str(fixed["pooling"]),
                mutation_positions_1based=mutation_positions_1based,
                source_trial=source_trial,
                complete=False,
                stopped_early="interrupted",
                epochs_run=len(log_rows),
                epochs_budget=max_epochs,
            ),
            output_dir / CHECKPOINT_NAME,
        )
        raise TrainingStopped(
            f"Stopped after the epoch loop of split seed {split_seed}, model seed {seed}, so this run has no "
            f"validation or test numbers. The {len(log_rows)} epoch(s) it did train are not lost: the weights of "
            f"its best epoch ({best_epoch + 1}) are in {output_dir / CHECKPOINT_NAME}, marked as a run that was "
            "cut short. Press Train again to run it properly -- that moves these weights aside rather than "
            "overwriting them -- or keep them with colabsd.bundle.save_bundle_from_checkpoint()."
        ) from None


def _finalise_run(
    *,
    adapter: SequenceAdapter,
    model: nn.Module,
    head: nn.Module,
    sequences: list[str],
    targets: np.ndarray,
    val_idx: list[int],
    test_idx: list[int],
    scaler: LabelScaler,
    micro_batch_size: int,
    device: torch.device,
    params: dict[str, Any],
    fixed: dict[str, Any],
    split_seed: int,
    seed: int,
    matched: Any,
    objective_name: str,
    objective_metric: str,
    mutation_positions_1based: Any,
    target_names: list[str] | None,
    best_epoch: int,
    best_score: float,
    best_metric_summary: dict[str, float],
    best_lora_state: dict[str, torch.Tensor],
    best_head_state: dict[str, torch.Tensor],
    log_rows: list[dict[str, Any]],
    stopped_early: str | None,
    output_dir: Path,
    source_trial: SourceTrial | None,
) -> dict[str, Any]:
    """Restore the best epoch, score both partitions and write the run's artefacts.

    Split out of :func:`train_eval_config` so that a stop press landing in here is caught in
    one place: everything this function does is after the point where a run can still be
    finished honestly.
    """
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
    # The numbers are positional; only the labels change. Without this the per-target blocks
    # in metrics.json would be named after upstream's own study's four PAMs.
    val_metrics = relabel_per_target(val_metrics, target_names)
    test_metrics = relabel_per_target(test_metrics, target_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = build_checkpoint(
        adapter=adapter,
        split_seed=split_seed,
        seed=seed,
        params=params,
        matched=matched,
        objective_name=objective_name,
        objective_metric=objective_metric,
        best_epoch=best_epoch,
        best_score=best_score,
        best_metric_summary=best_metric_summary,
        best_lora_state=best_lora_state,
        best_head_state=best_head_state,
        scaler=scaler,
        pooling=str(fixed["pooling"]),
        mutation_positions_1based=mutation_positions_1based,
        source_trial=source_trial,
        complete=True,
        stopped_early=stopped_early,
        epochs_run=len(log_rows),
        epochs_budget=int(fixed["max_epochs"]),
    )
    write_checkpoint_atomically(checkpoint, output_dir / CHECKPOINT_NAME)
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
        # How many rows each number was scored on. Every ranking metric here is a cut of the
        # top k -- `precision_k` and `ndcg_k` both score min(k, n) -- so a partition shorter
        # than k turns `P@50` into "all of them", which reads as a perfect score. Nothing
        # downstream could say that, because nothing recorded n.
        # The split is a partition of the library, so the training count is the remainder --
        # `_finalise_run` is handed the two partitions it scores and the whole sequence list.
        "partition_rows": {
            "train": len(sequences) - len(val_idx) - len(test_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_r2": best_metric_summary["R2"],
        "best_validation_spearman": best_metric_summary["Spearman"],
        # Why the epoch loop ended: `patience` for early stopping, `interrupted` for a
        # KeyboardInterrupt, absent when it simply ran out of epochs. A reader comparing two
        # runs needs to know which of them was cut short by a person.
        "stopped_early": stopped_early,
        "epochs_run": len(log_rows),
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
