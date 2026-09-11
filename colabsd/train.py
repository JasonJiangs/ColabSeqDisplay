"""Training orchestration across splits x model seeds, with a locked test set.

The vendored `colabsd.engine.train_config.train_eval_config` owns the
training loop, LoRA injection, early stopping and checkpointing. This module only
decides *what* to run, keeps the test partition locked, and turns the per-run
metric dictionaries into a table the report can render.

Locked-test discipline
----------------------
`train_eval_config` always evaluates the test partition at the end of a run and
writes it next to the validation numbers. `finetune` moves every test artefact
into `output_dir/locked_test/` the moment a run finishes, so nothing a notebook
touches -- neither `RunResult` nor `runs/<run>/metrics.json` nor
`runs/<run>/predictions.npz` -- carries test information. Reading it back is
`unlock_test`, which increments and persists `output_dir/unlock.json`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from colabsd.errors import ColabSDError, ConfigError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from colabsd.bestconfig import BestConfig
    from colabsd.spec import LibrarySpec

PROTEIN_ID = "user_protein"
LOCKED_DIRNAME = "locked_test"
UNLOCK_FILENAME = "unlock.json"
RUNS_DIRNAME = "runs"
METRICS = ("R2", "Pearson", "Spearman", "P@10", "P@50", "NDCG@10", "NDCG@50")
SKIPPED_FINGERPRINT = "interrupted"
LOCKED_WARNING = "Locked test partition. Read it through colabsd.train.unlock_test, which records every unlock."

ProgressFn = Callable[[int, int, str], None]


@dataclass
class RunResult:
    """Everything a report needs about one `finetune` call.

    Attributes:
        model_name: upstream model key the best-config registry was keyed on.
        adapter_name: backbone registry name the adapter was built from.
        condition_columns: measured conditions, in target-column order.
        params: the seven LoRA hyperparameters actually trained with.
        fixed: the training block (epochs, patience, batch sizes, ...).
        is_provisional: True when the hyperparameters are a placeholder.
        runs: one dict per (split seed, model seed), validation metrics only.
        aggregate: tidy frame with columns
            partition / condition / metric / mean / sd / n_runs.
        validation_summary: macro mean over conditions, mean +- sd across runs.
        test_runs, test_aggregate: None until `unlock_test` fills them in.
        unlock_count: how often the test partition has been read.
        paths: files written by `finetune`.
    """

    model_name: str
    adapter_name: str
    condition_columns: list[str]
    params: dict[str, Any]
    fixed: dict[str, Any]
    is_provisional: bool
    selection_metric: str
    split_seeds: list[int]
    model_seeds: list[int]
    runs: list[dict[str, Any]]
    aggregate: pd.DataFrame
    validation_summary: dict[str, dict[str, float]]
    output_dir: Path
    run_dirs: list[Path]
    locked_dir: Path
    paths: dict[str, Path]
    n_sequences: int
    created_utc: str
    unlock_count: int = 0
    test_runs: list[dict[str, Any]] | None = None
    test_aggregate: pd.DataFrame | None = None
    minutes: float = 0.0
    warnings: list[str] = field(default_factory=list)
    adapter_dtype: str | None = None
    adapter_hf_id: str | None = None

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def test_unlocked(self) -> bool:
        return self.test_runs is not None

    def best_run(self) -> dict[str, Any]:
        """Return the run with the highest validation selection score."""
        if not self.runs:
            raise ColabSDError("This RunResult has no runs; call finetune() before asking for the best one.")
        return max(self.runs, key=lambda row: row["best_validation_score"])

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "adapter_name": self.adapter_name,
            "condition_columns": list(self.condition_columns),
            "params": self.params,
            "fixed": self.fixed,
            "is_provisional": self.is_provisional,
            "selection_metric": self.selection_metric,
            "split_seeds": list(self.split_seeds),
            "model_seeds": list(self.model_seeds),
            "n_sequences": self.n_sequences,
            "n_runs": self.n_runs,
            "created_utc": self.created_utc,
            "minutes": self.minutes,
            "unlock_count": self.unlock_count,
            "test_unlocked": self.test_unlocked,
            "runs": self.runs,
            "validation_summary": self.validation_summary,
            "aggregate": self.aggregate.to_dict(orient="records"),
            "output_dir": str(self.output_dir),
            "run_dirs": [str(path) for path in self.run_dirs],
            "warnings": list(self.warnings),
            "adapter_dtype": self.adapter_dtype,
            "adapter_hf_id": self.adapter_hf_id,
        }
        if self.test_runs is not None:
            payload["test_runs"] = self.test_runs
        if self.test_aggregate is not None:
            payload["test_aggregate"] = self.test_aggregate.to_dict(orient="records")
        return payload

    def save(self) -> dict[str, Path]:
        """Write `run_result.json` plus the two CSV views; returns the paths."""
        import pandas as pd

        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / "run_result.json"
        result_path.write_text(json.dumps(self.to_dict(), indent=2))
        runs_path = self.output_dir / "validation_runs.csv"
        pd.DataFrame(_flatten_runs(self.runs, self.condition_columns)).to_csv(runs_path, index=False)
        summary_path = self.output_dir / "validation_summary.csv"
        self.aggregate.to_csv(summary_path, index=False)
        self.paths.update({"run_result": result_path, "runs_csv": runs_path, "summary_csv": summary_path})
        if self.test_aggregate is not None:
            test_path = self.output_dir / "test_summary.csv"
            self.test_aggregate.to_csv(test_path, index=False)
            self.paths["test_summary_csv"] = test_path
        return dict(self.paths)


def finetune(
    *,
    spec: LibrarySpec,
    sequences: Sequence[str],
    targets: np.ndarray,
    adapter: Any,
    best: BestConfig,
    splits: dict[int, dict[str, list[int]]],
    output_dir: str | Path,
    model_seeds: Sequence[int] | None = None,
    progress: ProgressFn | None = None,
    config: dict[str, Any] | None = None,
    resume: bool = True,
) -> RunResult:
    """Train `splits x model_seeds` LoRA runs and report validation only.

    `config` overrides the runtime config that would otherwise be built through
    `colabsd.protein_db`; `resume` reuses a finished run only when its fingerprint --
    the model, the hyperparameters, the mutated sites, the split and a hash of the
    sequences and targets themselves -- is identical, so a changed library always
    retrains. Test metrics are never returned: call `unlock_test` for those.
    """
    from colabsd.engine.train_config import train_eval_config

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = [str(item) for item in sequences]
    targets = np.asarray(targets, dtype=np.float32)
    condition_columns = list(spec.condition_columns)
    _validate_inputs(sequences, targets, condition_columns, splits, spec)

    seeds = _unique([int(seed) for seed in (model_seeds if model_seeds else _default_model_seeds(best))])
    try:
        splits = {int(seed): split for seed, split in splits.items()}
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"splits must be keyed by integer split seeds ({exc}). Pass the mapping returned by "
            "colabsd.data.make_splits(n, seeds, out_dir) unchanged."
        ) from exc
    split_seeds = sorted(splits)
    runtime = _runtime_config(
        spec=spec,
        best=best,
        output_dir=output_dir,
        split_seeds=split_seeds,
        model_seeds=seeds,
        override=config,
    )
    params = dict(best.params)
    fixed = dict(runtime["fixed"])
    selection_metric = _selection_metric(runtime)
    pooled_positions = _pooled_positions(adapter, spec, runtime)
    data_digest = _data_digest(sequences, targets)

    locked_dir = output_dir / LOCKED_DIRNAME
    runs_root = output_dir / RUNS_DIRNAME
    total = len(split_seeds) * len(seeds)
    _notify(progress, 0, total, "starting")

    rows: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    started = time.time()
    for split_seed in split_seeds:
        split = _clean_split(splits[split_seed], len(sequences), split_seed)
        for model_seed in seeds:
            name = f"split{split_seed}_seed{model_seed}"
            run_dir = runs_root / name
            fingerprint = _fingerprint(
                model=str(best.model),
                params=params,
                fixed=fixed,
                split=split,
                data=data_digest,
                pooled_positions=pooled_positions,
                n_sequences=len(sequences),
                n_targets=int(targets.shape[1]),
                split_seed=split_seed,
                model_seed=model_seed,
            )
            cached = _cached_row(run_dir, fingerprint) if resume else None
            if cached is None:
                run_dir.mkdir(parents=True, exist_ok=True)
                _lock_interrupted_artifacts(run_dir, locked_dir / name, condition_columns)
                t0 = time.time()
                metrics = train_eval_config(
                    adapter=adapter,
                    sequences=sequences,
                    targets=targets,
                    split=split,
                    params=params,
                    config=runtime,
                    split_seed=split_seed,
                    seed=model_seed,
                    output_dir=run_dir,
                    source_trial=None,
                    on_epoch=_epoch_reporter(progress, len(rows), total, name, t0),
                )
                minutes = (time.time() - t0) / 60.0
                _lock_test_artifacts(
                    run_dir=run_dir,
                    locked_run_dir=locked_dir / name,
                    metrics=metrics,
                    fingerprint=fingerprint,
                    condition_columns=condition_columns,
                )
                row = _visible_row(
                    metrics=metrics,
                    condition_columns=condition_columns,
                    split_seed=split_seed,
                    model_seed=model_seed,
                    run_dir=run_dir,
                    minutes=minutes,
                    fingerprint=fingerprint,
                )
                _write_visible_metrics(run_dir, row)
            else:
                row = cached
            rows.append(row)
            run_dirs.append(run_dir)
            _notify(progress, len(rows), total, name)

    aggregate = _aggregate(rows, condition_columns, partition="validation")
    result = RunResult(
        model_name=str(best.model),
        adapter_name=_adapter_name(adapter, best),
        condition_columns=condition_columns,
        params=params,
        fixed=fixed,
        is_provisional=bool(getattr(best, "is_provisional", False)),
        selection_metric=selection_metric,
        split_seeds=split_seeds,
        model_seeds=seeds,
        runs=rows,
        aggregate=aggregate,
        validation_summary=_macro_summary(rows, "validation"),
        output_dir=output_dir,
        run_dirs=run_dirs,
        locked_dir=locked_dir,
        paths={},
        n_sequences=len(sequences),
        created_utc=_utc_now(),
        unlock_count=read_unlock_count(output_dir),
        minutes=(time.time() - started) / 60.0,
        warnings=_provisional_warning(best),
        adapter_dtype=_text_attr(adapter, "dtype"),
        adapter_hf_id=_text_attr(adapter, "hf_id"),
    )
    result.save()
    return result


def unlock_test(run_result: RunResult, *, output_dir: str | Path) -> dict[str, Any]:
    """Read the locked test partition once, incrementing the persisted unlock count.

    Every call is recorded in `output_dir/unlock.json` and echoed into the returned
    payload, `output_dir/test_metrics.json` and the updated `RunResult`.
    """
    output_dir = Path(output_dir)
    locked_dir = output_dir / LOCKED_DIRNAME
    payloads = sorted(locked_dir.glob("*/test_metrics.json")) if locked_dir.is_dir() else []
    if not payloads:
        raise ColabSDError(
            f"No locked test results under {locked_dir}. Run finetune(..., output_dir={output_dir!s}) first, "
            "and pass that same output_dir here."
        )

    condition_columns = list(run_result.condition_columns)
    if not run_result.runs:
        raise ColabSDError("This RunResult has no runs, so there is nothing to unlock. Call finetune() first.")
    wanted = {(int(run["split_seed"]), int(run["model_seed"])): run.get("fingerprint") for run in run_result.runs}
    rows: list[dict[str, Any]] = []
    for path in payloads:
        locked = json.loads(path.read_text())
        key = (int(locked["split_seed"]), int(locked["model_seed"]))
        if key not in wanted:
            continue
        _require_matching_fingerprint(path, locked.get("fingerprint"), wanted[key], key, output_dir)
        per_condition = _rename_targets(locked["test"]["per_target"], condition_columns)
        rows.append(
            {
                "split_seed": int(locked["split_seed"]),
                "model_seed": int(locked["model_seed"]),
                "run_dir": locked.get("run_dir", ""),
                "test": {"mean": dict(locked["test"]["mean"]), "per_condition": per_condition},
            }
        )
    rows.sort(key=lambda row: (row["split_seed"], row["model_seed"]))
    if not rows:
        raise ColabSDError(
            f"{locked_dir} holds locked results, but none of them belong to this RunResult "
            f"(split seeds {run_result.split_seeds}, model seeds {run_result.model_seeds}). "
            "Pass the output_dir that finetune() wrote this result to."
        )

    count = _record_unlock(output_dir, n_runs=len(rows), model_name=run_result.model_name)
    aggregate = _aggregate(rows, condition_columns, partition="test")
    payload = {
        "unlock_count": count,
        "unlocked_utc": _utc_now(),
        "model_name": run_result.model_name,
        "is_provisional": run_result.is_provisional,
        "condition_columns": condition_columns,
        "n_runs": len(rows),
        "runs": rows,
        "summary": _macro_summary(rows, "test"),
        "aggregate": aggregate.to_dict(orient="records"),
        "output_dir": str(output_dir),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(payload, indent=2))
    run_result.test_runs = rows
    run_result.test_aggregate = aggregate
    run_result.unlock_count = count
    run_result.save()
    return payload


def _require_matching_fingerprint(
    path: Path,
    found: str | None,
    expected: str | None,
    key: tuple[int, int],
    output_dir: Path,
) -> None:
    """Refuse to hand back test numbers produced by a different configuration.

    Run directories are keyed by (split seed, model seed) alone, so a second `finetune` into the same
    `output_dir` with different hyperparameters overwrites the locked payload. Without this check the
    older `RunResult` would silently report the newer model's test metrics.
    """
    if not found or not expected or found == SKIPPED_FINGERPRINT:
        return
    if found == expected:
        return
    raise ColabSDError(
        f"{path} holds the test metrics of a different training configuration for split seed {key[0]}, model "
        f"seed {key[1]} (fingerprint {found} rather than {expected}): {output_dir} was reused by a later "
        "finetune() call with different hyperparameters, data or splits. Re-run finetune() for this "
        "configuration, or give each configuration its own output_dir."
    )


def read_unlock_count(output_dir: str | Path) -> int:
    """Return how often the test partition under *output_dir* has been unlocked."""
    return _read_unlock_payload(Path(output_dir))[0]


def _read_unlock_payload(output_dir: Path) -> tuple[int, list[dict[str, Any]]]:
    path = output_dir / UNLOCK_FILENAME
    if not path.is_file():
        return 0, []
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"expected a JSON object, got {type(payload).__name__}")
        count = int(payload.get("unlock_count", 0))
        history = payload.get("history", [])
        if not isinstance(history, list):
            raise TypeError(f"'history' must be a list, got {type(history).__name__}")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ColabSDError(
            f"{path} is not readable as an unlock record ({exc}). Delete it to reset the unlock counter to zero, "
            "and say so in the report."
        ) from exc
    return count, list(history)


def _record_unlock(output_dir: Path, *, n_runs: int, model_name: str) -> int:
    path = output_dir / UNLOCK_FILENAME
    previous, history = _read_unlock_payload(output_dir)
    count = previous + 1
    history.append({"unlock_count": count, "utc": _utc_now(), "model": model_name, "n_runs": n_runs})
    path.write_text(json.dumps({"unlock_count": count, "history": history}, indent=2))
    return count


def _lock_test_artifacts(
    *,
    run_dir: Path,
    locked_run_dir: Path,
    metrics: dict[str, Any],
    fingerprint: str,
    condition_columns: list[str],
) -> None:
    locked_run_dir.mkdir(parents=True, exist_ok=True)
    locked_run_dir.joinpath("test_metrics.json").write_text(
        json.dumps(
            {
                "_warning": LOCKED_WARNING,
                "model": metrics["model"],
                "split_seed": int(metrics["split_seed"]),
                "model_seed": int(metrics["seed"]),
                "run_dir": str(run_dir),
                "fingerprint": fingerprint,
                "condition_columns": condition_columns,
                "test": metrics["test"],
            },
            indent=2,
        )
    )
    _strip_test_from_metrics_json(run_dir)
    predictions = run_dir / "predictions.npz"
    if predictions.is_file():
        with np.load(predictions) as blob:
            arrays = {key: blob[key] for key in blob.files}
        np.savez_compressed(
            locked_run_dir / "test_predictions.npz",
            **{key: value for key, value in arrays.items() if key.startswith("test_")},
        )
        np.savez_compressed(
            predictions,
            **{key: value for key, value in arrays.items() if not key.startswith("test_")},
        )
    metrics.pop("test", None)


def _strip_test_from_metrics_json(run_dir: Path) -> None:
    """Replace the test block upstream just wrote with the locked-partition notice."""
    path = run_dir / "metrics.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return
    if not isinstance(payload, dict):
        return
    payload["test"] = LOCKED_WARNING
    payload["test_locked"] = True
    path.write_text(json.dumps(payload, indent=2))


def _lock_interrupted_artifacts(run_dir: Path, locked_run_dir: Path, condition_columns: list[str]) -> None:
    """Quarantine test artefacts left behind by a run that died before they were locked."""
    path = run_dir / "metrics.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return
    if not isinstance(payload.get("test"), dict):
        return
    _lock_test_artifacts(
        run_dir=run_dir,
        locked_run_dir=locked_run_dir,
        metrics=payload,
        fingerprint=SKIPPED_FINGERPRINT,
        condition_columns=condition_columns,
    )


def _visible_row(
    *,
    metrics: dict[str, Any],
    condition_columns: list[str],
    split_seed: int,
    model_seed: int,
    run_dir: Path,
    minutes: float,
    fingerprint: str,
) -> dict[str, Any]:
    validation = metrics["validation"]
    return {
        "split_seed": int(split_seed),
        "model_seed": int(model_seed),
        "model": metrics["model"],
        "selection_metric": metrics["selection_metric"],
        "best_epoch": int(metrics["best_epoch"]),
        "best_validation_score": float(metrics["best_validation_score"]),
        "minutes": float(minutes),
        "run_dir": str(run_dir),
        "checkpoint": str(run_dir / "best_checkpoint.pt"),
        "fingerprint": fingerprint,
        "test_locked": True,
        "validation": {
            "mean": dict(validation["mean"]),
            "per_condition": _rename_targets(validation["per_target"], condition_columns),
        },
    }


def _write_visible_metrics(run_dir: Path, row: dict[str, Any]) -> None:
    path = run_dir / "metrics.json"
    payload = json.loads(path.read_text()) if path.is_file() else {}
    payload["test"] = LOCKED_WARNING
    payload["test_locked"] = True
    payload["fingerprint"] = row["fingerprint"]
    payload["validation_per_condition"] = row["validation"]["per_condition"]
    path.write_text(json.dumps(payload, indent=2))
    (run_dir / "colabsd_run.json").write_text(json.dumps(row, indent=2))


def _cached_row(run_dir: Path, fingerprint: str) -> dict[str, Any] | None:
    marker = run_dir / "colabsd_run.json"
    if not marker.is_file():
        return None
    try:
        row = json.loads(marker.read_text())
    except ValueError:
        return None
    if not isinstance(row, dict) or row.get("fingerprint") != fingerprint:
        return None
    checkpoint = run_dir / "best_checkpoint.pt"
    if not checkpoint.is_file():
        return None
    row["run_dir"] = str(run_dir)
    row["checkpoint"] = str(checkpoint)
    return row


def _validate_inputs(
    sequences: list[str],
    targets: np.ndarray,
    condition_columns: list[str],
    splits: dict[int, dict[str, list[int]]],
    spec: LibrarySpec | None = None,
) -> None:
    if targets.ndim != 2:
        raise DataError(f"targets must be 2-D (N, T); got shape {targets.shape}.")
    if len(sequences) != targets.shape[0]:
        raise DataError(
            f"{len(sequences)} sequences but {targets.shape[0]} target rows. "
            "Both come from load_library(); pass its outputs unchanged."
        )
    if targets.shape[1] != len(condition_columns):
        raise DataError(
            f"targets has {targets.shape[1]} columns but the spec lists {len(condition_columns)} "
            f"condition_columns ({condition_columns}). Rebuild the targets with load_library()."
        )
    if not isinstance(splits, dict) or not splits:
        raise DataError(
            "splits must be the {split_seed: split} mapping returned by colabsd.data.make_splits(); got "
            f"{type(splits).__name__}."
        )
    lengths = {len(seq) for seq in sequences}
    if len(lengths) > 1:
        raise DataError(
            f"Sequences have {len(lengths)} different lengths; every variant must be the full-length WT with "
            "substitutions. Rebuild them with load_library()."
        )
    wt = str(getattr(spec, "wt_sequence", "") or "")
    if wt and lengths and lengths != {len(wt)}:
        raise DataError(
            f"The variant sequences are {sorted(lengths)[0]} residues but spec.wt_sequence is {len(wt)}. "
            "The mutated positions are wild-type coordinates, so a length mismatch averages the wrong "
            "residues: rebuild the sequences with load_library(csv, spec)."
        )


def _clean_split(split: dict[str, list[int]], n: int, split_seed: int) -> dict[str, list[int]]:
    missing = [key for key in ("train_idx", "val_idx", "test_idx") if key not in split]
    if missing:
        raise DataError(f"Split seed {split_seed} is missing {missing}; build splits with colabsd.data.make_splits().")
    cleaned = {key: [int(index) for index in split[key]] for key in ("train_idx", "val_idx", "test_idx")}
    for key, indices in cleaned.items():
        if not indices:
            raise DataError(f"Split seed {split_seed} has an empty {key}; the library is too small to split 8:1:1.")
        if min(indices) < 0 or max(indices) >= n:
            raise DataError(
                f"Split seed {split_seed} indexes row {max(indices)} of a {n}-row library. "
                "Rebuild the splits after filtering the library."
            )
        if len(set(indices)) != len(indices):
            raise DataError(
                f"Split seed {split_seed} lists the same row twice in {key}. Rebuild the splits with "
                "colabsd.data.make_splits(); a repeated row is counted twice in every metric."
            )
    for left, right in (("train_idx", "val_idx"), ("train_idx", "test_idx"), ("val_idx", "test_idx")):
        shared = sorted(set(cleaned[left]) & set(cleaned[right]))
        if shared:
            raise DataError(
                f"Split seed {split_seed} puts {len(shared)} rows (for example {shared[:5]}) in both {left} and "
                f"{right}. That leaks the held-out data into training and makes every number optimistic: "
                "rebuild the splits with colabsd.data.make_splits()."
            )
    return cleaned


def _unique(values: list[int]) -> list[int]:
    """Keep the first occurrence of each seed: a repeated seed would train twice into one run directory."""
    seen: set[int] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _default_model_seeds(best: BestConfig) -> list[int]:
    evaluation = dict(getattr(best, "evaluation", {}) or {})
    seeds = evaluation.get("model_seeds") or [0]
    return [int(seed) for seed in seeds]


def _adapter_name(adapter: Any, best: BestConfig) -> str:
    return str(getattr(adapter, "model_name", None) or best.model)


def _text_attr(adapter: Any, name: str) -> str | None:
    value = getattr(adapter, name, None)
    return None if value is None else str(value)


def _provisional_warning(best: BestConfig) -> list[str]:
    if getattr(best, "is_provisional", False):
        return [
            f"{best.model} hyperparameters are PROVISIONAL: they are a placeholder, not a tuned configuration. "
            "Treat the numbers as a lower bound."
        ]
    return []


def _runtime_config(
    *,
    spec: LibrarySpec,
    best: BestConfig,
    output_dir: Path,
    split_seeds: list[int],
    model_seeds: list[int],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    fixed_best = dict(getattr(best, "fixed", {}) or {})
    if override is not None:
        config = copy.deepcopy(override)
    else:
        config = _config_from_protein_db(
            spec=spec,
            best=best,
            output_dir=output_dir,
            split_seeds=split_seeds,
            model_seeds=model_seeds,
        )
    config.setdefault("data", {})
    config.setdefault("protein", {})
    fixed = {**dict(config.get("fixed", {})), **fixed_best}
    if fixed.get("evaluate_test_during_optimization"):
        raise ConfigError(
            "training.evaluate_test_during_optimization must stay false. Remove it from the best-config YAML; "
            "colabsd reads the test partition only through unlock_test()."
        )
    fixed["evaluate_test_during_optimization"] = False
    config["fixed"] = fixed
    evaluation = dict(getattr(best, "evaluation", {}) or {})
    study = dict(config.get("study", {}))
    study.update(
        {
            "direction": "maximize",
            "objective": str(evaluation.get("selection_objective", study.get("objective", "mean_validation_spearman"))),
            "top_k": 1,
            "re_evaluation_split_seeds": list(split_seeds),
            "re_evaluation_seeds": list(model_seeds),
        }
    )
    config["study"] = study
    return config


def _config_from_protein_db(
    *,
    spec: LibrarySpec,
    best: BestConfig,
    output_dir: Path,
    split_seeds: list[int],
    model_seeds: list[int],
) -> dict[str, Any]:
    """Write the one-protein record the engine resolves coordinates from, and configure a run against it.

    The record's coordinates are `spec.positions_1based` and nothing else: the model averages the
    embeddings at the sites the library varies.
    """
    try:
        from colabsd import protein_db
    except ImportError as exc:  # pragma: no cover - only before protein_db lands
        raise ConfigError(
            "colabsd.protein_db is unavailable, so the mutated sites cannot be written where the engine "
            f"reads them. Pass finetune(config=...) with a prepared runtime config instead ({exc})."
        ) from exc
    database_path = protein_db.write_protein_record(spec, out_dir=output_dir, protein_id=PROTEIN_ID)
    return protein_db.runtime_config(
        spec,
        params=best,
        protein_id=PROTEIN_ID,
        database_path=Path(database_path).resolve(),
        split_seeds=list(split_seeds),
        model_seeds=list(model_seeds),
    )


def _selection_metric(config: dict[str, Any]) -> str:
    from colabsd.engine.config_space import validation_metric_name

    try:
        return validation_metric_name(config)
    except ValueError as exc:
        raise ConfigError(
            f"{exc} Fix evaluation.selection_objective in the best-config YAML for this model."
        ) from exc


def _rename_targets(per_target: dict[str, Any], condition_columns: list[str]) -> dict[str, dict[str, float]]:
    """Relabel the engine's positional target names with the user's condition columns."""
    names = list(per_target)
    if len(names) != len(condition_columns):
        raise DataError(
            f"The engine reported {len(names)} targets but the spec lists {len(condition_columns)} condition_columns. "
            "The targets array and spec.condition_columns must agree."
        )
    return {column: dict(per_target[name]) for column, name in zip(condition_columns, names, strict=True)}


def summarize_across_runs(values: Iterable[float]) -> dict[str, float]:
    """Mean, sample sd and count over the finite values of one metric across repeated runs.

    The sd of a single run is **undefined**, not zero: reporting 0.0 would claim a
    reproducibility that one run cannot show. `colabsd.train`, `colabsd.baseline` and
    `colabsd.report` all aggregate through this one function so their tables agree.
    """
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    return {"mean": float(np.mean(finite)), "sd": sd, "n": len(finite)}


def _aggregate(rows: list[dict[str, Any]], condition_columns: list[str], *, partition: str) -> pd.DataFrame:
    import pandas as pd

    records = []
    for condition in condition_columns:
        for metric in METRICS:
            values = [float(row[partition]["per_condition"][condition][metric]) for row in rows]
            stats = summarize_across_runs(values)
            records.append(
                {
                    "partition": partition,
                    "condition": condition,
                    "metric": metric,
                    "mean": stats["mean"],
                    "sd": stats["sd"],
                    "n_runs": int(stats["n"]),
                }
            )
    return pd.DataFrame.from_records(records)


def _macro_summary(rows: list[dict[str, Any]], partition: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        stats = summarize_across_runs([float(row[partition]["mean"][metric]) for row in rows])
        summary[metric] = {"mean": stats["mean"], "sd": stats["sd"], "n_runs": int(stats["n"])}
    return summary


def _flatten_runs(rows: list[dict[str, Any]], condition_columns: list[str]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        partition = "test" if "test" in row else "validation"
        record = {
            "split_seed": row["split_seed"],
            "model_seed": row["model_seed"],
            "best_epoch": row.get("best_epoch"),
            "best_validation_score": row.get("best_validation_score"),
            "minutes": row.get("minutes"),
        }
        for metric, value in row[partition]["mean"].items():
            record[f"{partition}_mean_{metric}"] = value
        for condition in condition_columns:
            for metric, value in row[partition]["per_condition"][condition].items():
                record[f"{partition}_{condition}_{metric}"] = value
        flat.append(record)
    return flat


def _fingerprint(**payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _data_digest(sequences: list[str], targets: np.ndarray) -> str:
    """Hash the training data so a resumed run cannot reuse a checkpoint fitted to other numbers."""
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(sequence.encode())
        digest.update(b"\x00")
    digest.update(np.ascontiguousarray(targets, dtype=np.float32).tobytes())
    return digest.hexdigest()[:32]


def _pooled_positions(adapter: Any, spec: LibrarySpec, config: dict[str, Any]) -> list[int]:
    """The residues this run averages: the library's mutated sites, agreed on by all three parties.

    The engine averages the embeddings the *adapter* selects but records the coordinates it resolves
    from the *protein record*, and both must be the sites the *spec* says the library varies. A
    disagreement is invisible otherwise: the metrics, the RunResult and the bundle would all describe
    residues the model never averaged.
    """
    from colabsd.engine.config_space import pooling_positions_0based
    from colabsd.engine.protein_db import clear_database_cache

    expected = sorted(int(position) for position in spec.positions_0based())
    positions = getattr(adapter, "pooling_positions_0based", None)
    if not positions:
        raise ConfigError(
            "The adapter was given no residue positions, so there is nothing for it to average. Build it "
            "with create_adapter(name, pooling_positions_0based=spec.positions_0based(), ...)."
        )
    resolved = sorted(int(position) for position in positions)
    if resolved != expected:
        raise ConfigError(
            f"The adapter averages {len(resolved)} residues (0-based {resolved[:5]}) but the library varies "
            f"{len(expected)} (0-based {expected[:5]}). Build the adapter with "
            "pooling_positions_0based=spec.positions_0based(), from the same spec you are training on."
        )
    clear_database_cache()
    try:
        recorded = sorted(int(position) for position in pooling_positions_0based(config))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ConfigError(
            f"The mutated sites could not be resolved from the protein record ({exc}). Write it with "
            "colabsd.protein_db.write_protein_record(spec, out_dir) before calling finetune(config=...)."
        ) from exc
    if recorded != expected:
        raise ConfigError(
            f"The protein record lists {len(recorded)} mutated residues (0-based {recorded[:5]}) but this "
            f"library varies {len(expected)} (0-based {expected[:5]}). Every artefact would record "
            "coordinates the model never averaged: rewrite the record with "
            "colabsd.protein_db.write_protein_record(spec, out_dir) for this spec."
        )
    return expected


def _elapsed(seconds: float) -> str:
    """`6m12s`, so a reader can tell a slow run from a stopped one."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _epoch_reporter(
    progress: ProgressFn | None,
    done: int,
    total: int,
    name: str,
    started: float,
) -> Callable[[int, int, float], None] | None:
    """Forward each finished epoch to `progress` as a line the panel can display.

    `done` stays the number of *completed runs*, so the run counter does not advance until
    the run really finishes; the epoch and the validation score go in the label.
    """
    if progress is None:
        return None

    def report(epoch: int, max_epochs: int, score: float) -> None:
        _notify(
            progress,
            done,
            total,
            f"{name} · epoch {epoch + 1}/{max_epochs} · val {score:.4f} · {_elapsed(time.time() - started)}",
        )

    return report

def _notify(progress: ProgressFn | None, done: int, total: int, label: str) -> None:
    if progress is not None:
        progress(done, total, label)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 -- datetime.UTC needs 3.11
