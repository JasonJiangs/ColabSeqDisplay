"""The one-hot floor: ridge and MLP on the k mutated residues, on the pLM's own splits.

A pLM is only worth its GPU minutes if it beats a model that sees nothing but which
amino acid sits at each randomized site. This module builds that model on exactly the
splits the fine-tuning run used, so the two numbers in the report are comparable.

Encoding delegates to ``colabsd.engine.one_hot.encode_site_one_hot``; the predictors are
``colabsd.engine.heads.HEAD_REGISTRY`` entries; metrics are
``colabsd.engine.metrics.evaluate_predictions``. Nothing here is reimplemented science.

Runs are aggregated with ``colabsd.train.summarize_across_runs``, so the floor and the pLM
report a single run's standard deviation the same way: undefined, never zero.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .data import require_unique_columns
from .errors import DataError
from .train import summarize_across_runs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .spec import LibrarySpec

DEFAULT_HEADS: tuple[str, ...] = ("ridge", "mlp")
DEFAULT_MODEL_SEEDS: tuple[int, ...] = (11, 22, 33)
PARTITIONS: tuple[str, ...] = ("validation", "test")
SPLIT_KEYS: tuple[str, ...] = ("train_idx", "val_idx", "test_idx")

# Notebook-sized overrides of upstream's TrainingConfig defaults: 20*k features fit in
# seconds with a wide batch and a short schedule, on CPU.
TRAINING_DEFAULTS: dict[str, Any] = {
    "lr": 1.0e-3,
    "weight_decay": 0.01,
    "batch_size": 512,
    "max_epochs": 60,
    "patience": 8,
}


def one_to_three_letter() -> dict[str, str]:
    """Invert `colabsd.data.THREE_TO_ONE`, the one residue table this package owns.

    Cross-checked against upstream's `AA3` vocabulary so a divergence fails loudly here
    rather than silently encoding the wrong amino acid.
    """
    from colabsd.engine.one_hot import AA3

    from .data import THREE_TO_ONE

    mapping = {one: three for three, one in THREE_TO_ONE.items()}
    if sorted(THREE_TO_ONE) != sorted(AA3) or len(mapping) != len(THREE_TO_ONE):
        raise DataError(
            "colabsd.data.THREE_TO_ONE and colabsd.engine.one_hot's AA3 vocabulary disagree "
            f"({sorted(set(THREE_TO_ONE).symmetric_difference(AA3))}). One of the two residue tables "
            "changed; reconcile them before encoding a one-hot baseline."
        )
    return mapping


def encode_one_hot(df: pd.DataFrame, spec: LibrarySpec) -> np.ndarray:
    """One-hot encode the mutated residues into a flat ``(N, 20 * k)`` array."""
    from colabsd.engine.one_hot import encode_site_one_hot

    columns = [str(column) for column in spec.mutation_columns]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DataError(
            f"The variant table has no column(s) {missing}. Set LibrarySpec.mutation_columns to the "
            f"columns holding the mutated residues; the table has {list(df.columns)}."
        )
    require_unique_columns(df, columns)
    frame = _three_letter_frame(df, columns, bool(getattr(spec, "three_letter", True)))
    try:
        encoded = encode_site_one_hot(frame, columns)
    except ValueError as exc:  # pragma: no cover - guarded by _three_letter_frame
        raise DataError(f"{exc}. Every mutated residue must be one of the 20 standard amino acids.") from exc
    return encoded.reshape(len(frame), -1).astype(np.float32)


def one_hot_baseline(
    df: pd.DataFrame,
    spec: LibrarySpec,
    targets: np.ndarray,
    splits: dict[int, dict] | Iterable[dict],
    *,
    model_seeds: Sequence[int] | None = None,
    heads: Sequence[str] = DEFAULT_HEADS,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Fit the one-hot floor across ``splits x model_seeds`` and return its metrics.

    Every run reports both partitions. Validation numbers come from the same fitted model
    that selected on validation — exactly as the pLM run's validation numbers do — so the
    floor stays comparable whether or not the test set has been unlocked.
    """
    from colabsd.engine.heads import HEAD_REGISTRY, create_head
    from colabsd.engine.metrics import evaluate_predictions, summarize_metrics
    from colabsd.engine.schema import TrainingConfig

    head_names = [str(name) for name in heads]
    unknown = [name for name in head_names if name not in HEAD_REGISTRY]
    if unknown:
        raise DataError(f"Unknown one-hot head(s) {unknown}. Available heads: {sorted(HEAD_REGISTRY)}.")
    if not head_names:
        raise DataError("one_hot_baseline needs at least one head, for example heads=('ridge', 'mlp').")

    features = encode_one_hot(df, spec)
    target_columns = [str(column) for column in spec.condition_columns]
    y = _as_target_matrix(targets, len(df), target_columns)
    seeds = [int(seed) for seed in (model_seeds if model_seeds is not None else DEFAULT_MODEL_SEEDS)]
    if not seeds:
        raise DataError("model_seeds is empty; pass at least one seed, for example model_seeds=[11].")
    split_items = _normalized_splits(splits, len(df))

    cfg = TrainingConfig(
        split_seeds=[seed for seed, _ in split_items],
        model_seeds=seeds,
        **TRAINING_DEFAULTS,
    )

    total = len(head_names) * len(split_items) * len(seeds)
    runs: list[dict[str, Any]] = []
    done = 0
    for head_name in head_names:
        head = create_head(head_name)
        for split_seed, split in split_items:
            train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
            x_train, x_val, x_test = features[train_idx], features[val_idx], features[test_idx]
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
            x_eval = np.concatenate([x_val, x_test], axis=0)
            for model_seed in seeds:
                label = f"one_hot_{head_name} split={split_seed} seed={model_seed}"
                started = time.time()
                predictions, _state, _log = head.run(x_train, y_train, x_val, y_val, x_eval, model_seed, cfg)
                predictions = np.asarray(predictions, dtype=np.float32).reshape(len(x_eval), -1)
                val_metrics = evaluate_predictions(y_val, predictions[: len(x_val)], target_columns)
                test_metrics = evaluate_predictions(y_test, predictions[len(x_val) :], target_columns)
                runs.append(
                    {
                        "source": f"one_hot_{head_name}",
                        "head": head_name,
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "seed": model_seed,
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "seconds": float(time.time() - started),
                        "validation": {"per_target": val_metrics, "mean": summarize_metrics(val_metrics)},
                        "test": {"per_target": test_metrics, "mean": summarize_metrics(test_metrics)},
                    }
                )
                done += 1
                if progress is not None:
                    progress(done, total, label)

    result = {
        "kind": "one_hot_baseline",
        "heads": head_names,
        "split_seeds": [seed for seed, _ in split_items],
        "model_seeds": seeds,
        "target_columns": target_columns,
        "n_runs": len(runs),
        "n_features": int(features.shape[1]),
        "n_sites": int(features.shape[1] // 20),
        "training": {"batch_size": cfg.batch_size, "max_epochs": cfg.max_epochs, "patience": cfg.patience},
        "runs": runs,
    }
    result["summary"] = _summarize(runs, head_names, target_columns)
    result["floor"] = {
        partition: baseline_floor(result, metric="Spearman", partition=partition) for partition in PARTITIONS
    }
    return result


def baseline_floor(baseline: dict, *, metric: str = "Spearman", partition: str = "validation") -> dict | None:
    """Return the strongest one-hot head for *metric*: the line a pLM has to clear."""
    summary = (baseline or {}).get("summary", {}).get(partition, {})
    best: dict | None = None
    for head_name, per_condition in summary.items():
        entry = per_condition.get("mean", {}).get(metric)
        if entry is None or not np.isfinite(entry["mean"]):
            continue
        if best is None or entry["mean"] > best["mean"]:
            best = {
                "head": head_name,
                "metric": metric,
                "partition": partition,
                "mean": float(entry["mean"]),
                "sd": float(entry["sd"]),
                "n": int(entry["n"]),
            }
    return best


def _three_letter_frame(df: pd.DataFrame, columns: list[str], three_letter: bool) -> pd.DataFrame:
    from colabsd.engine.one_hot import AA3_TO_INDEX

    mapping = None if three_letter else one_to_three_letter()
    normalized = {}
    for column in columns:
        values = df[column].astype(str).str.strip()
        values = values.str.upper() if mapping is not None else values.str.capitalize()
        unknown = sorted(set(values).difference(mapping if mapping is not None else AA3_TO_INDEX))
        if unknown:
            style = "one-letter codes such as 'K'" if mapping is not None else "three-letter names such as 'Lys'"
            raise DataError(
                f"Column '{column}' holds residues this encoder does not know: {unknown[:10]}. "
                f"LibrarySpec.three_letter={three_letter} expects {style}; fix the column or flip the flag."
            )
        normalized[column] = values.map(mapping) if mapping is not None else values
    return pd.DataFrame(normalized, index=df.index)


def _as_target_matrix(targets: np.ndarray, n_rows: int, target_columns: list[str]) -> np.ndarray:
    y = np.asarray(targets, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != n_rows:
        raise DataError(
            f"targets has {y.shape[0]} rows but the variant table has {n_rows}. "
            "Pass the targets returned by colabsd.data.load_library for this same table."
        )
    if y.shape[1] != len(target_columns):
        raise DataError(
            f"targets has {y.shape[1]} columns but LibrarySpec.condition_columns names "
            f"{len(target_columns)} ({target_columns}). They must agree column for column."
        )
    if not np.all(np.isfinite(y)):
        raise DataError("targets contains NaN or inf. Drop or impute those rows before fitting the baseline.")
    return y


def _split_seed(raw_seed: Any) -> int:
    try:
        return int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"Split key {raw_seed!r} is not a seed number. Pass the mapping returned by "
            "colabsd.data.make_splits(n, seeds, out_dir), which is keyed by integer split seed."
        ) from exc


def _normalized_splits(splits: dict[int, dict] | Iterable[dict], n_rows: int) -> list[tuple[int, dict]]:
    if isinstance(splits, dict):
        items = [(_split_seed(key), value) for key, value in splits.items()]
    else:
        items = [(_split_seed(split.get("split_seed", index)), split) for index, split in enumerate(splits)]
    if not items:
        raise DataError("splits is empty. Build it with colabsd.data.make_splits(n, seeds, out_dir) first.")

    normalized: list[tuple[int, dict]] = []
    for seed, split in sorted(items, key=lambda item: item[0]):
        missing = [key for key in SPLIT_KEYS if key not in split]
        if missing:
            raise DataError(
                f"Split {seed} is missing {missing}. Splits must come from "
                "colabsd.data.make_splits, which stores train_idx / val_idx / test_idx."
            )
        indices = {}
        for key in SPLIT_KEYS:
            idx = np.asarray(split[key], dtype=np.int64).ravel()
            if idx.size == 0:
                raise DataError(f"Split {seed} has an empty {key}; it cannot be used to fit or score a model.")
            if idx.min() < 0 or idx.max() >= n_rows:
                offender = int(idx.min()) if idx.min() < 0 else int(idx.max())
                raise DataError(
                    f"Split {seed} indexes row {offender} of a table with {n_rows} rows, so {key} is "
                    "out of range. The splits were built for a different (probably unfiltered) library."
                )
            if np.unique(idx).size != idx.size:
                raise DataError(
                    f"Split {seed} lists {idx.size - int(np.unique(idx).size)} duplicated row(s) in {key}. "
                    "Duplicated rows are scored more than once and inflate the floor; rebuild the split."
                )
            indices[key] = idx
        _require_disjoint(seed, indices)
        normalized.append((seed, indices))
    return normalized


def _require_disjoint(seed: int, indices: dict[str, np.ndarray]) -> None:
    """A floor fitted on rows it is scored on is not a floor. Refuse overlapping partitions."""
    for left, right in (("train_idx", "val_idx"), ("train_idx", "test_idx"), ("val_idx", "test_idx")):
        shared = np.intersect1d(indices[left], indices[right])
        if shared.size:
            raise DataError(
                f"Split {seed} puts {shared.size} row(s) in both {left} and {right} "
                f"(for example row {int(shared[0])}). The one-hot floor would be fitted on rows it is "
                "scored on, so it would look better than it is; rebuild the split with "
                "colabsd.data.make_splits."
            )


def _summarize(runs: list[dict], head_names: list[str], target_columns: list[str]) -> dict:
    from colabsd.engine.metrics import TRACKED_METRICS

    summary: dict[str, dict] = {partition: {} for partition in PARTITIONS}
    for partition in PARTITIONS:
        for head_name in head_names:
            selected = [run for run in runs if run["head"] == head_name]
            per_condition: dict[str, dict] = {}
            for condition in target_columns:
                per_condition[condition] = {
                    metric: summarize_across_runs([run[partition]["per_target"][condition][metric] for run in selected])
                    for metric in TRACKED_METRICS
                }
            per_condition["mean"] = {
                metric: summarize_across_runs([run[partition]["mean"][metric] for run in selected])
                for metric in TRACKED_METRICS
            }
            summary[partition][head_name] = per_condition
    return summary
