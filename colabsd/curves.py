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


#: One hue per split seed, lightness per model seed inside it. A 3 x 3 evaluation is nine
#: curves that lie almost on top of each other, and a nine-colour cycle says only "these are
#: different"; grouping the hue says *which* of the two seeds a spread belongs to, which is
#: the question somebody plotting nine runs actually has.
_SPLIT_HUES: tuple[str, ...] = ("#0d5c63", "#9b4a1f", "#3f3d8f", "#2f6b40", "#7a2d52")

#: The single-run colour. Most runs are one run, and one line needs no palette at all.
_SOLO = "#0d5c63"


def _lighten(hex_colour: str, amount: float) -> tuple[float, float, float]:
    """Mix *hex_colour* toward white by *amount* in [0, 1)."""
    value = hex_colour.lstrip("#")
    rgb = tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))
    return tuple(channel + (1.0 - channel) * amount for channel in rgb)


def curve_colours(frame: pd.DataFrame) -> dict[str, tuple[float, float, float] | str]:
    """A colour per run label: hue from the split seed, lightness from the model seed."""
    runs = list(dict.fromkeys(frame["run"]))
    if len(runs) == 1:
        return {runs[0]: _SOLO}

    splits = list(dict.fromkeys(frame["split_seed"]))
    colours: dict[str, tuple[float, float, float] | str] = {}
    for name in runs:
        part = frame[frame["run"] == name]
        split = part["split_seed"].iloc[0]
        seeds = list(dict.fromkeys(frame[frame["split_seed"] == split]["model_seed"]))
        hue = _SPLIT_HUES[splits.index(split) % len(_SPLIT_HUES)]
        step = seeds.index(part["model_seed"].iloc[0])
        colours[name] = _lighten(hue, 0.0 if len(seeds) < 2 else 0.42 * step / (len(seeds) - 1))
    return colours


def build_curve_figure(frame: pd.DataFrame, *, model_label: str = "") -> Figure:
    """Two stacked panels sharing an epoch axis: MSE on top, validation below."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    figure = Figure(figsize=(10.0, 6.6))
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.13})

    metric_column = f"val_{CURVE_METRIC.lower()}"
    runs = list(dict.fromkeys(frame["run"])) if len(frame) else []
    colours = curve_colours(frame) if runs else {}
    for name in runs:
        part = frame[frame["run"] == name].sort_values("epoch")
        colour = colours[name]
        # An epoch axis a reader counts from 1, matching the panel's "epoch 3/20" line. The
        # log counts from 0 because that is the loop variable.
        epochs = part["epoch"] + 1
        style = {"color": colour, "marker": "o", "markersize": 2.6, "linewidth": 1.3}
        top.plot(epochs, part["train_loss_mse"], **style)
        bottom.plot(epochs, part[metric_column], label=name, **style)
        kept = part[part["is_best"]]
        if len(kept):
            # The exported epoch, in the line's own colour so it reads as part of that curve
            # rather than as a separate series. A heavy black ring did the opposite.
            bottom.scatter(
                kept["epoch"] + 1,
                kept[metric_column],
                s=58,
                facecolors="white",
                edgecolors=[colour],
                linewidths=1.8,
                zorder=5,
            )

    top.set_ylabel("training loss (MSE)", fontsize=10.5)
    title = "Training curves" + (f" — {model_label}" if model_label else "")
    top.set_title(title, fontsize=12.5, pad=10)
    bottom.set_ylabel(f"validation {CURVE_METRIC}", fontsize=10.5)
    bottom.set_xlabel("epoch", fontsize=10.5)
    for axis in (top, bottom):
        axis.grid(True, alpha=0.22, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(labelsize=9.5)
    # Epochs are whole numbers; the default locator was offering 2.5 and 7.5.
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

    if runs:
        # Under the axes, never over the data: nine curves that converge leave no corner free,
        # and a legend that covers the interesting part of a plot is worse than no legend.
        kept_note = "○ marks the epoch each run kept — the weights that were exported"
        if len(runs) == 1:
            figure.text(0.5, 0.02, kept_note, ha="center", fontsize=8.5, alpha=0.7)
        else:
            # The note is the legend's title rather than a second floating label: laid out as
            # part of the legend it cannot land on top of it, which is what happened when the
            # two were positioned independently and the entries wrapped to a second row.
            handles, labels = bottom.get_legend_handles_labels()
            legend = figure.legend(
                handles,
                labels,
                title=kept_note,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.055),
                ncol=min(5, len(runs)),
                fontsize=8.5,
                frameon=False,
                handlelength=1.6,
                columnspacing=1.4,
            )
            legend.get_title().set_fontsize(8.5)
            legend.get_title().set_alpha(0.7)
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
