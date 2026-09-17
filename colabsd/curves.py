"""The training curves: what the loss and the validation score did, epoch by epoch.

`colabsd.report` answers "how good is the finished model". This answers "did training behave",
which is a different question and the one somebody asks when a run disappoints: did the loss
fall at all, did it stop falling, did validation turn over while training loss kept dropping,
did the two seeds diverge.

The numbers are not new. `train_eval_config` has always written one row per epoch into each
run's `training_log.json` — the training loss, which is MSE because `loss: mse` is fixed for
every entry in the registry, and the macro validation score for every tracked metric. What was
missing was a way to see them: the file sat in the run directory and never reached the archive
the user takes away.

Both are exported, and that is deliberate. A figure nobody can re-plot is a picture; the CSV
beside it is the same numbers in the form a reader can check, re-draw or paste into their own
tool. `curve_frame` is what both are made of, so they cannot disagree.

One line per run. A default run is one line; a 3 x 3 evaluation is nine, and the spread between
them at the same epoch is the thing the aggregate report can only summarise afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cost, not behaviour
    import pandas as pd
    from matplotlib.figure import Figure

#: What each run writes, one row per epoch. Named here because this module is the only reader.
TRAINING_LOG_NAME = "training_log.json"

#: The two files this module writes into the report directory.
CURVE_CSV_NAME = "training_curve.csv"
CURVE_PNG_NAME = "training_curve.png"

#: The validation metric drawn beside the loss. Spearman is what the registry selects on
#: (`selection_objective: mean_validation_spearman`), so it is the curve that decided which
#: epoch was kept -- drawing anything else would show a run being judged on one number while
#: the reader watches another.
CURVE_METRIC = "Spearman"

#: The column `validation_log_row` writes that metric under.
_METRIC_COLUMN = f"val_mean_{CURVE_METRIC}"


@dataclass(frozen=True)
class CurvePaths:
    """Where the two files landed. Either may be None when there was nothing to draw."""

    csv: Path | None
    png: Path | None
    n_runs: int
    n_epochs: int


def _run_records(run_result: Any) -> list[dict[str, Any]]:
    """The per-run dicts, whether they arrive on a `RunResult` or as a plain list."""
    runs = getattr(run_result, "runs", None)
    if runs is None and isinstance(run_result, dict):
        runs = run_result.get("runs")
    if runs is None and isinstance(run_result, list):
        runs = run_result
    return [record for record in (runs or []) if isinstance(record, dict)]


def read_training_log(run_dir: str | Path) -> list[dict[str, Any]]:
    """One row per epoch for a single run, or an empty list when the file is not there.

    Missing is normal rather than exceptional: a run recovered from an interrupted session may
    have a checkpoint and no log, and a hand-built `RunResult` in a test has neither. The curve
    is a convenience, so a missing log costs that run its line and nothing else.
    """
    path = Path(run_dir) / TRAINING_LOG_NAME
    if not path.is_file():
        return []
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def curve_frame(run_result: Any) -> pd.DataFrame:
    """Every epoch of every run, tidy: one row per (run, epoch).

    `is_best` marks the epoch each run kept — the one whose weights were exported. Without it a
    reader of the CSV cannot tell which point on the curve became the model.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for record in _run_records(run_result):
        log = read_training_log(record.get("run_dir", ""))
        if not log:
            continue
        best_epoch = record.get("best_epoch")
        split_seed = record.get("split_seed")
        model_seed = record.get("model_seed")
        for entry in log:
            epoch = entry.get("epoch")
            rows.append(
                {
                    "split_seed": split_seed,
                    "model_seed": model_seed,
                    "run": f"split {split_seed} · seed {model_seed}",
                    "epoch": epoch,
                    "train_loss_mse": entry.get("train_loss"),
                    f"val_{CURVE_METRIC.lower()}": entry.get(_METRIC_COLUMN),
                    "objective": entry.get("objective"),
                    "is_best": epoch is not None and epoch == best_epoch,
                }
            )
    return pd.DataFrame(rows)


def build_curve_figure(frame: pd.DataFrame, *, model_label: str = "") -> Figure:
    """Two stacked panels sharing an epoch axis: MSE on top, validation below."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.0, 7.0))
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.18})

    metric_column = f"val_{CURVE_METRIC.lower()}"
    runs = list(dict.fromkeys(frame["run"])) if len(frame) else []
    for name in runs:
        part = frame[frame["run"] == name].sort_values("epoch")
        # An epoch axis a reader counts from 1, matching the panel's "epoch 3/20" line. The
        # log counts from 0 because that is the loop variable.
        epochs = part["epoch"] + 1
        top.plot(epochs, part["train_loss_mse"], marker="o", markersize=3, linewidth=1.4, label=name)
        bottom.plot(epochs, part[metric_column], marker="o", markersize=3, linewidth=1.4, label=name)
        kept = part[part["is_best"]]
        if len(kept):
            bottom.scatter(
                kept["epoch"] + 1,
                kept[metric_column],
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                zorder=5,
            )

    top.set_ylabel("training loss (MSE)", fontsize=11)
    title = "Training curves" + (f" — {model_label}" if model_label else "")
    top.set_title(title, fontsize=12)
    bottom.set_ylabel(f"validation {CURVE_METRIC} (mean over conditions)", fontsize=11)
    bottom.set_xlabel("epoch", fontsize=11)
    for axis in (top, bottom):
        axis.grid(True, alpha=0.25, linewidth=0.6)
    if len(runs) > 1:
        bottom.legend(fontsize=8, ncol=min(3, len(runs)), frameon=False)
    if len(frame):
        bottom.annotate(
            "○ the epoch each run kept",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            alpha=0.75,
        )
    # No `tight_layout`: the two shared-axis panels make it warn that it may get the result
    # wrong, and `savefig(bbox_inches="tight")` below already trims the margins.
    return figure


def write_training_curves(run_result: Any, out_dir: str | Path, *, model_label: str = "") -> CurvePaths:
    """Write `training_curve.csv` and `training_curve.png`; return what was written.

    Both files or neither: a figure with no CSV beside it is the situation this module exists
    to avoid.
    """
    frame = curve_frame(run_result)
    directory = Path(out_dir)
    if not len(frame):
        return CurvePaths(csv=None, png=None, n_runs=0, n_epochs=0)

    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / CURVE_CSV_NAME
    frame.to_csv(csv_path, index=False)

    png_path = directory / CURVE_PNG_NAME
    figure = build_curve_figure(frame, model_label=model_label)
    figure.savefig(png_path, dpi=150, bbox_inches="tight")

    return CurvePaths(
        csv=csv_path,
        png=png_path,
        n_runs=int(frame["run"].nunique()),
        n_epochs=int(frame["epoch"].nunique()),
    )
