"""Prediction metrics: per-target correlation, precision@k and NDCG@k, and their macro means.

Vendored from the *SequenceDisplay Workflow Optimization* research package
(``seqdisplay-opt``), which owns this science. Original modules:

* ``seqdisplay_opt/metrics/evaluation.py`` — ``precision_k``, ``ndcg_k``,
  ``evaluate_predictions``, ``mean_metric`` and the safe correlation / R2 helpers.
* ``seqdisplay_opt/finetuning/validation.py`` — ``TRACKED_METRICS``, ``summarize_metrics``,
  ``validation_log_row``.

Copied verbatim except where noted, so that ColabSeqDisplay installs without the research
checkout. Every metric was checked against the original, value for value.

Changed while copying: upstream names the first four unnamed targets after the four 5NNK
PAMs through a module-private ``INDEX2PAM`` table, which surprises anyone whose conditions
are not SlugCas9 PAMs. The behaviour is unchanged, but the table is now the public,
documented ``DEFAULT_TARGET_NAMES`` and the naming rule is the public
``default_target_names``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import ndcg_score

# Upstream's default target names: the four SlugCas9 5NNK PAMs, in output order. They are
# applied only when a caller passes no target names at all; pass `target_names` explicitly
# whenever the columns are not these PAMs.
DEFAULT_TARGET_NAMES: tuple[str, ...] = ("NNGA", "NNGT", "NNGC", "NNGG")

# Every metric `evaluate_predictions` records, in report order.
TRACKED_METRICS: tuple[str, ...] = (
    "R2",
    "Pearson",
    "Spearman",
    "P@10",
    "P@50",
    "NDCG@10",
    "NDCG@50",
)


def default_target_names(n_targets: int) -> list[str]:
    """Name *n_targets* output columns: the 5NNK PAMs first, then ``target_i``."""
    return [
        DEFAULT_TARGET_NAMES[index] if index < len(DEFAULT_TARGET_NAMES) else f"target_{index}"
        for index in range(int(n_targets))
    ]


def precision_k(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Fraction of the true top-*k* that the predicted top-*k* recovers."""
    if np.ptp(pred) < 1e-8:
        return float("nan")
    kk = min(k, len(true))
    idx_pred = np.argsort(-pred)[:kk]
    idx_true = np.argsort(-true)[:kk]
    return len(set(idx_pred) & set(idx_true)) / kk if kk else float("nan")


def ndcg_k(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Compute standard NDCG@*k* from continuous assay relevance values."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if true.size == 0 or true.shape != pred.shape:
        return float("nan")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        return float("nan")
    relevance = true.copy()
    if relevance.min() < 0:
        relevance -= relevance.min()
    if np.max(relevance) <= 0:
        return float("nan")
    kk = min(int(k), len(relevance))
    if kk <= 0:
        return float("nan")
    return float(ndcg_score(relevance[None, :], pred[None, :], k=kk, ignore_ties=False))


def _safe_corr(fn: Any, t: np.ndarray, p: np.ndarray) -> float:
    """Correlation that reports 0.0 rather than NaN for a constant or degenerate prediction."""
    if np.ptp(p) < 1e-8:
        return 0.0
    val = fn(t, p)[0]
    return float(val) if np.isfinite(val) else 0.0


def _safe_r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R2 that reports 0.0 rather than dividing by a zero target variance."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_predictions(
    trues: np.ndarray,
    preds: np.ndarray,
    target_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-target metrics for each output column.

    ``target_names`` defaults to :func:`default_target_names`, i.e. the four 5NNK PAMs
    followed by ``target_i``. Pass the real condition names for any other library.
    """
    trues = np.asarray(trues)
    preds = np.asarray(preds)
    if trues.ndim == 1:
        trues = trues[:, None]
    if preds.ndim == 1:
        preds = preds[:, None]
    if target_names is None:
        target_names = default_target_names(trues.shape[1])
    metrics: dict[str, dict[str, float]] = {}
    for i, name in enumerate(target_names):
        t, p = trues[:, i], preds[:, i]
        metrics[name] = {
            "R2": _safe_r2(t, p),
            "Spearman": _safe_corr(spearmanr, t, p),
            "Pearson": _safe_corr(pearsonr, t, p),
            "P@10": float(precision_k(t, p, 10)),
            "P@50": float(precision_k(t, p, 50)),
            "NDCG@10": ndcg_k(t, p, 10),
            "NDCG@50": ndcg_k(t, p, 50),
        }
    return metrics


def mean_metric(metrics: dict[str, Any], metric_name: str) -> float:
    """Average *metric_name* across all targets, ignoring non-finite values."""
    vals = [metrics[name][metric_name] for name in metrics]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else 0.0


def summarize_metrics(per_target: dict[str, Any]) -> dict[str, float]:
    """Return macro means for every metric recorded by the evaluation layer."""
    return {metric: mean_metric(per_target, metric) for metric in TRACKED_METRICS}


def validation_log_row(summary: dict[str, float]) -> dict[str, float]:
    """Format macro validation metrics for an epoch-level training log."""
    return {f"val_mean_{metric}": value for metric, value in summary.items()}
