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
them at the same epoch is the thing the aggregate report can only summarize afterwards.

The same three panels are also drawn *during* a run, into the panel's image widget, from events
instead of from files -- `LiveCurveState` accumulates, `live_curve_png` draws, and
`should_redraw` decides how rarely. It is the same `build_curve_figure` either way, with the
batch-level loss of the epoch under way overlaid on the loss panel, because a run that is not
converging should say so before its first epoch closes and because a screenshot of the panel
and `training_curve.png` must not be able to tell two stories about one run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cost, not behavior
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


def run_label(split_seed: Any, model_seed: Any) -> str:
    """The name a run is drawn and legended under, from the two seeds that identify it.

    One function because the two runs of a training press reach the figure by two different
    roads -- `curve_frame` off the finished logs, `LiveCurveState` off the events while they
    are still arriving -- and a label that differed between them would color and legend the
    same run twice.
    """
    return f"split {split_seed} · seed {model_seed}"


def _as_float(value: Any) -> float:
    """A number, or NaN for the absent key of a log written before it was recorded."""
    try:
        return float("nan") if value is None else float(value)
    except (TypeError, ValueError):
        return float("nan")


def exported_run(run_result: Any) -> tuple[int, int] | None:
    """The `(split_seed, model_seed)` of the run whose weights a bundle would carry.

    `colabsd.bundle.save_bundle_from_run` exports exactly one run -- `run_result.best_run()`,
    the highest validation selection score -- and records it in the manifest as `source_run`.
    The figure draws every run and rings the kept epoch of each, so without this it could only
    say something true of all of them, and what it said instead ("the weights that were
    exported") was true of one.

    None when the runs do not say: a hand-built result, or one whose rows carry no score.
    """
    records = [
        record
        for record in _run_records(run_result)
        if record.get("split_seed") is not None and record.get("model_seed") is not None
    ]
    if not records:
        return None
    scored = [record for record in records if record.get("best_validation_score") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda record: record["best_validation_score"])
    return int(best["split_seed"]), int(best["model_seed"])


def curve_frame(run_result: Any) -> pd.DataFrame:
    """Every epoch of every run, tidy: one row per (run, epoch).

    `is_best` marks the epoch each run kept — the checkpoint that run wrote. Without it a reader
    of the CSV cannot tell which point on the curve that run settled on. Which run's checkpoint
    a bundle carries is a separate question, and `exported_run` is what answers it: with several
    runs there are several rings and only one of them left the working directory.
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
                    "run": run_label(split_seed, model_seed),
                    "epoch": epoch,
                    "train_loss_mse": entry.get("train_loss"),
                    # The same MSE on the partition the model did not train on. A log written
                    # before this existed has no such key, and the column is then all-NaN,
                    # which the figure reads as "that panel has nothing to draw".
                    # NaN rather than None for a log written before this key existed, because
                    # `LiveCurveState.frame()` produces NaN and the two frames are held to the
                    # same columns *and* the same dtypes.
                    "val_loss_mse": _as_float(entry.get("val_loss")),
                    f"val_{CURVE_METRIC.lower()}": entry.get(_METRIC_COLUMN),
                    "objective": entry.get("objective"),
                    "is_best": epoch is not None and epoch == best_epoch,
                }
            )
    return pd.DataFrame(rows)


#: One hue per split seed, lightness per model seed inside it. A 3 x 3 evaluation is nine
#: curves that lie almost on top of each other, and a nine-color cycle says only "these are
#: different"; grouping the hue says *which* of the two seeds a spread belongs to, which is
#: the question somebody plotting nine runs actually has.
_SPLIT_HUES: tuple[str, ...] = ("#0d5c63", "#9b4a1f", "#3f3d8f", "#2f6b40", "#7a2d52")

#: The single-run color. Most runs are one run, and one line needs no palette at all.
_SOLO = "#0d5c63"


def _lighten(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Mix *hex_color* toward white by *amount* in [0, 1)."""
    value = hex_color.lstrip("#")
    rgb = tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))
    return tuple(channel + (1.0 - channel) * amount for channel in rgb)


def curve_colors(frame: pd.DataFrame) -> dict[str, tuple[float, float, float] | str]:
    """A color per run label: hue from the split seed, lightness from the model seed."""
    runs = list(dict.fromkeys(frame["run"]))
    if len(runs) == 1:
        return {runs[0]: _SOLO}

    splits = list(dict.fromkeys(frame["split_seed"]))
    colors: dict[str, tuple[float, float, float] | str] = {}
    for name in runs:
        part = frame[frame["run"] == name]
        split = part["split_seed"].iloc[0]
        seeds = list(dict.fromkeys(frame[frame["split_seed"] == split]["model_seed"]))
        hue = _SPLIT_HUES[splits.index(split) % len(_SPLIT_HUES)]
        step = seeds.index(part["model_seed"].iloc[0])
        colors[name] = _lighten(hue, 0.0 if len(seeds) < 2 else 0.42 * step / (len(seeds) - 1))
    return colors


def _figure_colors(
    frame: pd.DataFrame, batch_points: pd.DataFrame | None
) -> dict[str, tuple[float, float, float] | str]:
    """A color per run label covering both frames.

    Mid-evaluation the two disagree about which runs exist: run 4 has batch points and no
    closed epoch yet, so asking `curve_colors` for the epoch frame alone would leave its cloud
    gray. Built from the union, a run keeps one color from its first batch point to its last
    epoch, which is what lets the eye follow it across a redraw.
    """
    if batch_points is None or not len(batch_points):
        return curve_colors(frame) if len(frame) else {}

    import pandas as pd

    keys = ["run", "split_seed", "model_seed"]
    parts = [part[keys] for part in (frame, batch_points) if part is not None and len(part)]
    if not parts:
        return {}
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset="run", keep="first")
    return curve_colors(combined)


#: Below this many points a run's batch cloud is drawn larger and darker: at twenty points
#: the cloud carries the whole figure, at two hundred it is a backdrop for the epoch curve.
_CLOUD_DENSE_POINTS = 120

#: A batch loss this many times the bulk of the data is an outlier rather than a reading.
_LOSS_OUTLIER_FACTOR = 4.0

#: What counts as "the bulk" when deciding that. High enough that an ordinary run -- whose
#: largest batch losses are its first few, and are the story -- is never clipped.
_LOSS_BULK_QUANTILE = 0.95


def _loss_limits(frame: pd.DataFrame, batch_points: pd.DataFrame) -> tuple[float, float, int] | None:
    """`(low, high, hidden)` for the loss panel, or None to leave the autoscale alone.

    The first micro-batches of a run are a running mean over a handful of examples and can sit
    far above everything that follows. Drawn to scale, one of them flattens the rest of the
    panel into a line along the bottom -- which is precisely the shape somebody watching for
    convergence needs to read. So a limit is applied, but only when it is earned: when the
    largest batch loss is more than `_LOSS_OUTLIER_FACTOR` times the bulk of the data. An
    ordinary run is drawn untouched, and `hidden` says how many points fell outside so the
    figure can admit to it rather than quietly dropping them.

    Every epoch-level loss is inside the limit by construction. The epoch curve is the
    authoritative series; it is never the thing that gets clipped.
    """
    import numpy as np

    batch = np.asarray(batch_points["train_loss_mse"], dtype="float64")
    batch = batch[np.isfinite(batch)]
    if not batch.size:
        return None
    epochs = np.asarray(frame["train_loss_mse"], dtype="float64") if len(frame) else np.empty(0)
    epochs = epochs[np.isfinite(epochs)]

    bulk = float(np.quantile(batch, _LOSS_BULK_QUANTILE))
    if epochs.size:
        bulk = max(bulk, float(epochs.max()))
    if bulk <= 0.0 or float(batch.max()) <= _LOSS_OUTLIER_FACTOR * bulk:
        return None

    high = bulk * 1.25
    low = float(min(batch.min(), epochs.min())) if epochs.size else float(batch.min())
    return low - 0.05 * (high - low), high, int((batch > high).sum())


def _attr(frame: pd.DataFrame, batch_points: pd.DataFrame | None, key: str) -> Any:
    """A value a live caller attached to either frame, or None for the archive's frames."""
    for part in (frame, batch_points):
        value = getattr(part, "attrs", {}).get(key) if part is not None else None
        if value is not None:
            return value
    return None


def _progress_suffix(frame: pd.DataFrame, batch_points: pd.DataFrame | None) -> str:
    """`run 2/9 · epoch 3/20` -- where the run had got to when this figure was drawn.

    A partial run and a finished short one plot identically: three epochs of curve either way.
    The axis cannot say which it is without inventing a scale the data does not support, so the
    title says it in words, in the same dialect as the progress line above the image -- and the
    archive's figure, whose frames carry none of these, keeps exactly the title it always had.
    """
    epoch = _attr(frame, batch_points, "epoch")
    max_epochs = _attr(frame, batch_points, "max_epochs")
    run_index = _attr(frame, batch_points, "run_index")
    n_runs = _attr(frame, batch_points, "n_runs")

    parts = []
    if n_runs is not None and int(n_runs) > 1 and run_index is not None:
        parts.append(f"run {int(run_index) + 1}/{int(n_runs)}")
    if epoch is not None and max_epochs:
        parts.append(f"epoch {int(epoch) + 1}/{int(max_epochs)}")
    return " · ".join(parts)


def build_curve_figure(
    frame: pd.DataFrame,
    *,
    model_label: str = "",
    batch_points: pd.DataFrame | None = None,
    metric_label: str = CURVE_METRIC,
    source_run: tuple[int, int] | None = None,
) -> Figure:
    """Three panels side by side on a shared epoch axis: training MSE, validation MSE, validation score.

    One function draws both `training_curve.png` and the live figure in the panel, because a
    second drawing function is how a screenshot and the archived file start disagreeing about
    the same run. `batch_points` is everything a live call adds: the per-batch loss inside the
    epoch under way, overlaid faintly on the loss panel so a run that is not converging says so
    before the first epoch closes. Left at None -- every call the archive makes -- not one
    artist on the figure changes.

    `metric_label` names the validation series, both the column read (`val_<lower>`) and the
    label written. It follows the run's own selection metric rather than assuming Spearman: a
    config selecting on R2 would otherwise be drawn under a Spearman heading.

    A live caller may attach `epoch`, `max_epochs`, `run_index` and `n_runs` to either frame's
    `.attrs`, counted the way the events count them, and the title then says how far along the
    run was. Nothing else reads them and the archive's frames set none.

    `source_run` is the `(split_seed, model_seed)` of the run a bundle built from these numbers
    carries -- `colabsd.curves.exported_run`. Every run gets a ring, because every run did keep
    that epoch, but only one of them was exported, and the note used to tell a nine-run reader
    that all nine sets of weights had been. Left at None the note claims nothing about export.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    # Side by side, not stacked: the two losses are read against each other -- that comparison
    # is the whole reason the validation one exists -- and a reader cannot compare two curves
    # they have to scroll between. Sharing the epoch axis keeps the three at one scale.
    figure = Figure(figsize=(15.0, 4.4))
    FigureCanvasAgg(figure)
    # Named for what each one draws rather than for where it sits. They were `top`, `middle`
    # and `bottom` when the figure was a stack, and those names outlived the layout: the
    # docstrings and comments that explain the figure went on putting the score underneath the
    # loss, so a reader following the prose looked below for a panel that is to the right of it.
    loss_panel, val_loss_panel, score_panel = figure.subplots(
        1, 3, sharex=True, gridspec_kw={"wspace": 0.22}
    )

    live = batch_points is not None and len(batch_points) > 0
    metric_column = f"val_{metric_label.lower()}"
    runs = list(dict.fromkeys(frame["run"])) if len(frame) else []
    # A live frame that has not seen an epoch close yet has no metric column to read, and
    # neither has one whose run selects on something this call was not told about.
    has_metric = bool(runs) and metric_column in frame.columns
    # An all-NaN column is a log written before the validation loss was recorded; pandas keeps
    # the column, so presence alone is not the question.
    has_val_loss = bool(runs) and "val_loss_mse" in frame.columns and frame["val_loss_mse"].notna().any()
    colors = _figure_colors(frame, batch_points)
    style = {"marker": "o", "markersize": 2.6, "linewidth": 1.3}

    for name in runs:
        part = frame[frame["run"] == name].sort_values("epoch")
        color = colors[name]
        # An epoch axis a reader counts from 1, matching the panel's "epoch 3/20" line. The
        # log counts from 0 because that is the loop variable.
        epochs = part["epoch"] + 1
        loss_panel.plot(epochs, part["train_loss_mse"], color=color, **style)
        if has_val_loss:
            val_loss_panel.plot(epochs, part["val_loss_mse"], color=color, **style)
        if not has_metric:
            continue
        score_panel.plot(epochs, part[metric_column], label=name, color=color, **style)
        kept = part[part["is_best"]]
        if len(kept):
            # The exported epoch, in the line's own color so it reads as part of that curve
            # rather than as a separate series. A heavy black ring did the opposite.
            score_panel.scatter(
                kept["epoch"] + 1,
                kept[metric_column],
                s=58,
                facecolors="white",
                edgecolors=[color],
                linewidths=1.8,
                zorder=5,
            )

    hidden = 0
    if live:
        for name in dict.fromkeys(batch_points["run"]):
            part = batch_points[batch_points["run"] == name]
            color = colors.get(name, _SOLO)
            # Small, faint and under the epoch line: at a couple of hundred points a run the
            # cloud reads as texture with a direction, not as a series to follow point by
            # point, and solid markers at that density smear over the curve they surround.
            # Sparse is the opposite problem -- in the first minute the cloud *is* the figure,
            # twenty points of it, and drawn for the crowded case it was barely there.
            sparse = len(part) < _CLOUD_DENSE_POINTS
            loss_panel.scatter(
                part["epoch"] + 1,
                part["train_loss_mse"],
                s=7.0 if sparse else 3.4,
                color=[color],
                alpha=0.55 if sparse else 0.3,
                linewidths=0,
                zorder=1,
            )
            head = part.iloc[-1]
            # Where the run had got to at this redraw: the one mark that says the figure is of
            # something still moving rather than of something finished.
            loss_panel.scatter(
                [head["epoch"] + 1],
                [head["train_loss_mse"]],
                s=22,
                color=[color],
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
            if name not in runs:
                # A run whose first epoch has not closed has no line to put in the legend, and
                # an unlabeled cloud in a nine-run evaluation belongs to nobody.
                score_panel.plot([], [], label=name, color=color, **style)

        limits = _loss_limits(frame, batch_points)
        if limits is not None:
            low, high, hidden = limits
            loss_panel.set_ylim(low, high)
        if not has_metric:
            # Before the first epoch closes there is no validation number in existence. Saying
            # so is the difference between "not yet" and "this run scores zero".
            # ...and the tick labels are dropped with it: an axis ranging -0.04 to 0.04 is a
            # scale invented by the autoscaler for an empty panel, and it reads as data.
            score_panel.set_yticks([])
            score_panel.text(
                0.5,
                0.5,
                f"validation {metric_label} appears when the first epoch closes",
                transform=score_panel.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                alpha=0.55,
            )

    if hidden:
        loss_panel.text(
            0.995,
            0.96,
            f"{hidden} batch point{'s' if hidden != 1 else ''} above the axis",
            transform=loss_panel.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            alpha=0.6,
        )

    loss_panel.set_ylabel("training loss (MSE)", fontsize=10.5)
    val_loss_panel.set_ylabel("validation loss (MSE)", fontsize=10.5)
    if not has_val_loss:
        # A run from before the validation loss was recorded. The panel stays, so the three
        # are always in the same places, and says why it is empty rather than looking broken.
        val_loss_panel.text(
            0.5, 0.5, "not recorded\nfor this run", transform=val_loss_panel.transAxes,
            ha="center", va="center", fontsize=9.5, alpha=0.55,
        )
    title = "Training curves" + (f" — {model_label}" if model_label else "")
    # Read off `.attrs` rather than off `live`, so a live figure whose batch reports never
    # arrived still says which epoch of how many it is drawn at. The archive's frames carry
    # none of them and its title is unchanged.
    progress = _progress_suffix(frame, batch_points)
    figure.suptitle(f"{title} · {progress}" if progress else title, fontsize=12.5, y=0.99)
    # Which panel decides. Side by side, the validation loss sits between the training loss
    # and the score, and the obvious reading is that the panel in the center is what early
    # stopping watches. It is not: the run keeps the epoch with the best validation *score*,
    # and the loss panels are diagnosis. Saying so on the axis is cheaper than a note nobody
    # reads.
    score_panel.set_ylabel(f"validation {metric_label}  (selects the epoch)", fontsize=10.5)
    for axis in (loss_panel, val_loss_panel, score_panel):
        # Every panel carries the label: side by side they are three plots a reader scans
        # across, not one stack with a shared foot.
        axis.set_xlabel("epoch", fontsize=10.5)
        axis.grid(True, alpha=0.22, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(labelsize=9.5)
        # Epochs are whole numbers; the default locator was offering 2.5 and 7.5.
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if live:
        # The first epoch is a whole unit of the axis rather than the left edge of it, so a run
        # 70% of the way through epoch 1 is drawn 70% of the way to the first tick. Autoscale
        # would have put those same points across the full width and called it an epoch.
        right = max(1.0, float(batch_points["epoch"].max()) + 1.0)
        if len(frame):
            right = max(right, float(frame["epoch"].max()) + 1.0)
        # The pad is what keeps the ring around the newest kept epoch -- which is usually the
        # rightmost point there is -- from being drawn half outside the axes. Set on one panel
        # and true of all three, because `sharex` ties them; the score panel is the one asked
        # because the ring that needs the room is drawn on it.
        score_panel.set_xlim(0.0, right + max(0.12, 0.03 * right))

    if runs or live:
        # Under the axes, never over the data: nine curves that converge leave no corner free,
        # and a legend that covers the interesting part of a plot is worse than no legend.
        # What is true of every ring on the figure: each run kept that epoch and wrote that
        # checkpoint. Which of those checkpoints left the working directory is a different
        # claim, true of one run, and it is only made when the caller has said which.
        kept_note = "○ marks the epoch each run kept — the checkpoint that run wrote"
        if source_run is not None:
            kept_note += f" · the bundle carries split seed {source_run[0]} / model seed {source_run[1]}"
        if live:
            # A different claim from the archive's, and the difference matters: this ring is
            # the checkpoint on disk at this instant, which the next epoch may replace.
            kept_note = (
                "live · faint points are batch loss, ● the latest"
                " · ○ the best epoch so far — the checkpoint on disk now"
            )
        elif progress:
            kept_note = "live · ○ the best epoch so far — the checkpoint on disk now"
        handles, labels = score_panel.get_legend_handles_labels()
        # One anchor for both branches. They were positioned independently and drifted: the
        # legend was moved below the x-labels when the figure became three short panels and
        # this one was not, so a single-run figure -- the common case -- printed its note
        # across the center panel's "epoch" label.
        note_y = -0.02
        if len(handles) < 2:
            figure.text(0.5, note_y, kept_note, ha="center", fontsize=8.5, alpha=0.7)
        else:
            # The note is the legend's title rather than a second floating label: laid out as
            # part of the legend it cannot land on top of it, which is what happened when the
            # two were positioned independently and the entries wrapped to a second row.
            legend = figure.legend(
                handles,
                labels,
                title=kept_note,
                loc="upper center",
                bbox_to_anchor=(0.5, note_y),
                ncol=min(5, len(handles)),
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
    figure = build_curve_figure(frame, model_label=model_label, source_run=exported_run(run_result))
    figure.savefig(png_path, dpi=150, bbox_inches="tight")

    return CurvePaths(
        csv=csv_path,
        png=png_path,
        n_runs=int(frame["run"].nunique()),
        n_epochs=int(frame["epoch"].nunique()),
    )


# --- the same curves, while the run is still going -------------------------------------
#
# The archive's figure is drawn once, from files, when there is nothing left to decide. The
# panel's is drawn repeatedly, from events, while the decision that matters -- let it run or
# stop it -- is still open. Only the input differs: `build_curve_figure` draws both, so a
# screenshot of the panel and `training_curve.png` cannot tell two stories about one run.
#
# Everything below exists because training holds the kernel. A redraw is 79 ms of matplotlib
# for the first figure of a run, 114 ms for one run mid-flight and 207 ms for nine (measured as the
# minimum of 15 interleaved rounds, so a busy machine cannot flatter the result), and every one of those
# milliseconds is a millisecond the GPU spends waiting. So the question is not how to make a
# redraw cheap -- it is not going to be cheap -- but how few of them will do.
#
# Measured over a simulated forty-minute nine-run press with every figure drawn for
# real, on the event stream the reporters actually emit: the policy below draws 70 pictures and spends 0.42% of
# the run drawing them. Redrawing on the text line's own 0.25 s gate would draw ~9,600 and
# spend 64% of the run inside matplotlib -- twenty-five minutes of a forty-minute run. A flat
# two seconds is 8.1%, a flat twelve seconds 1.28%. The budget is 2%.

#: The live figure's resolution, and the first thing that looks like a speed lever and is not:
#: between dpi 80 and 150 a nine-run redraw moves 202 -> 230 ms, because the cost is building
#: the figure and laying it out, not rasterizing it. 96 is chosen for the bytes that cross the
#: notebook's comm channel on every redraw -- 60 KB against 72 KB at 110 and 104 KB at 150.
#: The archive keeps dpi 150: nothing about the exported file changes.
LIVE_DPI = 96

#: Batch points kept per run, and the second thing that is not a speed lever: 240 against 2000
#: points a run is 207 ms against 218 ms at nine runs, and the PNG is 60 KB either way. What
#: the cap buys is a redraw whose cost does not grow with the length of the run: nine runs over
#: 20 epochs and over 60 epochs both draw in 207 ms, where one uncapped run of three hours
#: reaches ~40,000 points and 161 ms against a capped run's 114 ms. And a cloud that stays a
#: trend rather than a smear -- two hundred points is denser than this figure resolves.
LIVE_MAX_BATCH_POINTS = 240

#: The first figure appears this many seconds into a run. Early, because the first minute is
#: when somebody is deciding whether to sit and watch or come back later, and because a panel
#: that shows nothing for a minute reads as a panel that is broken.
LIVE_FIRST_DRAW_S = 3.0

#: Never redraw more often than this. Nearly sixty times the slowest measured redraw (207 ms,
#: nine runs): even drawing at the floor for a whole run, matplotlib gets 1.7% of the wall
#: clock. That ratio is where the number comes from -- it is the 2% budget, with room.
LIVE_MIN_INTERVAL_S = 12.0

#: ...and never less often than this, however long the run goes on. Beyond a minute between
#: frames the figure stops being live and becomes a picture that happens to update.
LIVE_MAX_INTERVAL_S = 60.0

#: Between those two the interval is this fraction of the run so far. A long run earns a longer
#: interval -- there is proportionally less news in each additional minute -- and that is what
#: turns 200 redraws into 71 over forty minutes while the first minute is drawn exactly as
#: often either way. A flat twelve seconds also fits the budget; it just spends three times as
#: much of it on pictures of a curve that has stopped moving.
LIVE_INTERVAL_FRACTION = 0.05

#: What `LiveCurveState.batch_frame` produces. `epoch` is fractional there and only there:
#: `epoch - 1 + step/n_batches`, so the last batch of an epoch lands exactly on that epoch's
#: marker and the first epoch's points span the first unit of the axis.
BATCH_COLUMNS = ("split_seed", "model_seed", "run", "epoch", "train_loss_mse")


def redraw_interval_s(elapsed_s: float) -> float:
    """Seconds to wait between redraws, for a run that has been going `elapsed_s`."""
    if elapsed_s <= 0.0:
        return LIVE_MIN_INTERVAL_S
    return min(LIVE_MAX_INTERVAL_S, max(LIVE_MIN_INTERVAL_S, LIVE_INTERVAL_FRACTION * elapsed_s))


def should_redraw(*, now: float, started_at: float, last_drawn_at: float, changed: bool) -> bool:
    """Whether the figure is worth redrawing at `now`. Reads no clock and touches no widget.

    The timing policy lives here, as arithmetic over four floats, rather than inside the
    panel's event handler: this way it is tested against a table of times instead of against a
    real run, and the handler that calls it has no arithmetic in it to get wrong. Call it with
    `last_drawn_at` at 0.0 to mean "nothing drawn yet"; `now` and `started_at` must come from
    the same clock.
    """
    if not changed:
        return False
    if last_drawn_at <= 0.0:
        return now - started_at >= LIVE_FIRST_DRAW_S
    return now - last_drawn_at >= redraw_interval_s(now - started_at)


@dataclass
class _LiveRun:
    """One run's points, in the order they arrived, which is the order they are drawn."""

    split_seed: Any
    model_seed: Any
    label: str
    #: (epoch, train_loss_mse, score, val_loss_mse) per closed epoch -- the three series the
    #: figure draws, plus the epoch they sit at. `score` is None until an epoch closes with one,
    #: which a well-formed epoch event always does; `val_loss_mse` is NaN when the event carried
    #: none, so the validation-loss panel simply has no point there.
    epochs: list[tuple[int, float, float | None, float]] = field(default_factory=list)
    #: (fractional epoch, train_loss_mse) per surviving batch report.
    batches: list[tuple[float, float]] = field(default_factory=list)
    #: One batch report in `stride` is kept. Doubles each time the cap is reached, so the
    #: points that survive stay evenly spread over the whole run rather than over its start.
    stride: int = 1
    seen: int = 0


class LiveCurveState:
    """What has happened so far in one training press, in the shape the figure wants.

    Fed `TrainingEvent`s by the panel, it hands back the two frames `build_curve_figure`
    draws. It holds points, not artists: no matplotlib object survives a redraw, and nothing
    here touches a widget or reads a clock.

    Bounded by construction. The epoch rows are a run's epoch budget -- tens, at most. The
    batch points are capped per run at `max_batch_points` by halving the list and doubling the
    stride whenever the cap is passed, so a forty-minute run and a forty-second one both end up
    drawing about the same number of points, spread evenly over whatever actually happened.
    """

    def __init__(self, *, max_batch_points: int = LIVE_MAX_BATCH_POINTS) -> None:
        #: Two is the floor: one point to halve toward and one to keep.
        self._cap = max(2, int(max_batch_points))
        #: The validation metric the events say this run is being selected on. It names the
        #: frame's metric column and the label on the figure's score panel, so a config
        #: selecting on R2 is not drawn under a Spearman heading. Every shipped registry entry
        #: selects on Spearman, which is why the default is the archive's own.
        self.metric: str = CURVE_METRIC
        self._runs: dict[tuple[Any, Any], _LiveRun] = {}
        self._where: dict[str, int] = {}

    @property
    def has_points(self) -> bool:
        """Whether there is anything to draw yet."""
        return any(run.epochs or run.batches for run in self._runs.values())

    @property
    def n_runs(self) -> int:
        """How many runs have reported anything. Rises as an evaluation works through them."""
        return len(self._runs)

    def reset(self) -> None:
        """Forget the last training press. The panel calls this when a new one starts."""
        self._runs.clear()
        self._where.clear()
        self.metric = CURVE_METRIC

    def record(self, event: Any) -> bool:
        """Take one event in; answer whether the drawing would change because of it.

        The answer is the throttle's other half: a redraw that would produce the same picture
        is 114 ms spent on nothing, and once the cap is full most batch events are decimated
        away before they reach the figure -- fewer than a fifth of a long run's ticks change
        anything. Events of a kind this module does not draw answer False too, rather than
        raising: a new kind added on the training side must not be able to stop a run.

        Deliberately not defensive about anything else. A malformed event raises, `finetune`
        catches it once, and the run loses its picture and keeps its model; swallowing it here
        would turn the same failure into a figure that silently stopped updating.
        """
        kind = getattr(event, "kind", "")
        if kind not in ("epoch", "batch"):
            return False

        metric = getattr(event, "metric", "")
        if metric:
            self.metric = metric
        for name in ("epoch", "max_epochs", "run_index", "n_runs"):
            value = getattr(event, name, None)
            if value is not None:
                self._where[name] = int(value)

        split_seed = getattr(event, "split_seed", 0)
        model_seed = getattr(event, "model_seed", 0)
        key = (split_seed, model_seed)
        run = self._runs.get(key)
        if run is None:
            run = self._runs[key] = _LiveRun(split_seed, model_seed, run_label(split_seed, model_seed))

        epoch = int(getattr(event, "epoch", 0))
        loss = float(getattr(event, "loss", float("nan")))
        if kind == "epoch":
            score = getattr(event, "score", None)
            val_loss = getattr(event, "val_loss", None)
            run.epochs.append(
                (
                    epoch,
                    loss,
                    None if score is None else float(score),
                    float("nan") if val_loss is None else float(val_loss),
                )
            )
            return True

        step, n_batches = getattr(event, "step", None), getattr(event, "n_batches", None)
        if step is None or not n_batches:
            # A batch event that cannot say where in the epoch it is has no x to be drawn at.
            return False
        run.seen += 1
        if run.seen % run.stride:
            return False
        run.batches.append((epoch - 1.0 + step / n_batches, loss))
        if len(run.batches) > self._cap:
            run.batches = run.batches[::2]
            run.stride *= 2
        return True

    def frame(self) -> pd.DataFrame:
        """The closed epochs, in `curve_frame`'s columns, so one function draws both.

        `is_best` is the first epoch holding the running maximum, which is not argmax dressed
        up: the engine replaces its checkpoint only on `score > best_score`, so on a tie the
        earlier epoch is the one whose weights are on disk, and this frame is a statement about
        what is on disk right now.
        """
        import pandas as pd

        column = f"val_{self.metric.lower()}"
        rows: list[dict[str, Any]] = []
        for run in self._runs.values():
            best_index, best_score = None, float("-inf")
            for index, (_, _, score, _) in enumerate(run.epochs):
                if score is not None and score > best_score:
                    best_index, best_score = index, score
            for index, (epoch, loss, score, val_loss) in enumerate(run.epochs):
                value = float("nan") if score is None else score
                rows.append(
                    {
                        "split_seed": run.split_seed,
                        "model_seed": run.model_seed,
                        "run": run.label,
                        "epoch": epoch,
                        "train_loss_mse": loss,
                        "val_loss_mse": val_loss,
                        column: value,
                        "objective": value,
                        "is_best": index == best_index,
                    }
                )
        # The archive's `curve_frame` emits these in this order; the two must match column for
        # column or the live picture and the exported one are two different drawings.
        columns = [
            "split_seed", "model_seed", "run", "epoch",
            "train_loss_mse", "val_loss_mse", column, "objective", "is_best",
        ]
        return self._stamped(pd.DataFrame(rows, columns=columns))

    def batch_frame(self) -> pd.DataFrame:
        """The surviving batch points, `BATCH_COLUMNS`, one row per point per run."""
        import pandas as pd

        rows = [
            {
                "split_seed": run.split_seed,
                "model_seed": run.model_seed,
                "run": run.label,
                "epoch": epoch,
                "train_loss_mse": loss,
            }
            for run in self._runs.values()
            for epoch, loss in run.batches
        ]
        return self._stamped(pd.DataFrame(rows, columns=list(BATCH_COLUMNS)))

    def _stamped(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Where the run had got to, carried beside the points rather than inside them.

        The figure's title says "run 2/9 · epoch 3/20" because a partial run and a finished
        short one plot identically. That fact is not a point and does not belong in a column
        the CSV would have to grow a meaning for, so it travels in `.attrs` -- set here, read
        only by `_progress_suffix`, and absent from every frame `curve_frame` builds.
        """
        frame.attrs.update(self._where)
        return frame


def figure_png(figure: Figure, *, dpi: int = LIVE_DPI) -> bytes:
    """A figure as PNG bytes, cleared afterwards. The one encoder both curve paths use.

    `figure.clear()` is not decoration: without it a long session accumulates a few megabytes
    over a few hundred redraws. Measured on a nine-run state, 50 redraws: resident memory
    settles rather than growing.
    """
    import io

    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        figure.clear()
    return buffer.getvalue()


def live_curve_png(state: LiveCurveState, *, model_label: str = "", dpi: int = LIVE_DPI) -> bytes:
    """One redraw: the current state as PNG bytes, ready to become `Image.value`.

    Bytes rather than a `Figure` so that the panel never imports matplotlib, never owns a
    figure and cannot leak one. The encoding, and the `figure.clear()` that keeps a long
    session from accumulating figures, live in `figure_png`.

    Raises rather than returning something blank on a drawing failure. The caller that catches
    it is `finetune`, which stops sending events, warns, and lets the run finish: one guard,
    in the place that can actually report it.
    """
    return figure_png(
        build_curve_figure(
            state.frame(),
            model_label=model_label,
            batch_points=state.batch_frame(),
            metric_label=state.metric,
        ),
        dpi=dpi,
    )
