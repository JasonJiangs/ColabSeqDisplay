"""The report: one CSV, one figure, and the three things a reader would otherwise assume.

`report.png` always states (1) the spread across the repeated runs, (2) where the one-hot
floor sits, and (3) how many times the locked test set has been unlocked. `report.csv`
carries every tracked metric for every condition, mean +/- sd across runs; it holds the test
partition only once `colabsd.train.unlock_test` has recorded an unlock, so a locked report
cannot be misread as a validation-versus-test comparison.

Metrics are upstream's: an already-scored run is read as-is, and a run that only carries
predictions is scored with `colabsd.engine.metrics.evaluate_predictions`.

The run under report is a `colabsd.train.RunResult`, the unlock counter is read with
`colabsd.train.read_unlock_count` and runs are averaged with
`colabsd.train.summarize_across_runs`: this module owns no second copy of any of the three.
Mapping-shaped runs are still accepted, so a notebook can report a `run_result.json` it
loaded back from disk. Nothing here needs a GPU or a network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .baseline import PARTITIONS, baseline_floor
from .errors import ColabSDError
from .train import UNLOCK_FILENAME, RunResult, read_unlock_count, summarize_across_runs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

    from .spec import LibrarySpec

PARTITION_ALIASES: dict[str, str] = {
    "validation": "validation",
    "val": "validation",
    "valid": "validation",
    "test": "test",
    "locked_test": "test",
}

# R2 is the one tracked metric with no lower bound; below this the panel stops following it
# down and prints the true value on the clipped bar instead.
UNBOUNDED_METRIC_FLOOR = -1.0

BAR_STYLES: tuple[tuple[str, str], ...] = (
    ("#2f2f2f", ""),
    ("#9a9a9a", "///"),
    ("#d9d9d9", "xxx"),
    ("#6e6e6e", "..."),
    ("#efefef", "\\\\\\"),
)


class ReportError(ColabSDError, ValueError):
    """There is nothing to report, or the report was asked for something it does not track.

    Subclasses both `ColabSDError` (so notebooks show a friendly message) and `ValueError`
    (so callers that expect the plain Python error still catch it).
    """


@dataclass(frozen=True)
class ReportPaths:
    """Where `build_report` put its output."""

    csv: Path
    png: Path
    json: Path


def build_report(
    run_result: RunResult | Any,
    baseline: dict | None,
    spec: LibrarySpec | None,
    *,
    out_dir: Path | str,
    metric: str = "Spearman",
    partition: str | None = None,
) -> ReportPaths:
    """Write `report.csv`, `report.png` and `report.json` into *out_dir*.

    The figure shows the test partition only once the test set has actually been
    unlocked by `colabsd.train.unlock_test`; otherwise it shows validation and says so.
    The CSV carries the same partitions the figure is allowed to show, for every source.
    """
    from colabsd.engine.metrics import TRACKED_METRICS

    if metric not in TRACKED_METRICS:
        raise ReportError(f"Unknown metric {metric!r}. Tracked metrics are {list(TRACKED_METRICS)}.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = _condition_names(spec, run_result, baseline)
    model_label = _model_label(run_result)
    sources: dict[str, list[dict]] = {}
    plm_records = _collect_records(run_result, conditions)
    if plm_records:
        sources[model_label] = plm_records
    for head_name in (baseline or {}).get("heads", []):
        head_records = _collect_records(
            [run for run in (baseline or {}).get("runs", []) if run.get("head") == head_name], conditions
        )
        if head_records:
            sources[f"one-hot {head_name}"] = head_records
    if not sources:
        raise ReportError(
            "build_report received neither pLM runs nor one-hot baseline runs. Pass the RunResult "
            "from colabsd.train.finetune and/or the dict from colabsd.baseline.one_hot_baseline."
        )

    aggregated = {name: _aggregate_source(records, conditions, TRACKED_METRICS) for name, records in sources.items()}
    unlock = _unlock_info(run_result, out_dir)
    primary = model_label if model_label in aggregated else None
    shown = _shown_partition(partition, aggregated, unlock["count"], primary)
    headline = aggregated[primary] if primary is not None else _any_source(aggregated)
    visible = _visible_partitions(unlock["count"], shown, "test" in headline)

    frame = _report_frame(aggregated, conditions, TRACKED_METRICS, visible)
    csv_path = out_dir / "report.csv"
    frame.to_csv(csv_path, index=False)

    floor = baseline_floor(baseline or {}, metric=metric, partition=shown)
    figure = _build_figure(
        aggregated=aggregated,
        conditions=conditions,
        metrics=list(TRACKED_METRICS),
        metric=metric,
        partition=shown,
        unlock=unlock,
        floor=floor,
        model_label=primary,
        spec=spec,
        n_runs={name: len(records) for name, records in sources.items()},
    )
    png_path = out_dir / "report.png"
    figure.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "headline_metric": metric,
        "shown_partition": shown,
        "unlock_count": unlock["count"],
        "unlock_source": unlock["source"],
        "n_runs": {name: len(records) for name, records in sources.items()},
        "one_hot_floor": floor,
        "reported_partitions": list(visible),
        "sources": {
            name: {
                partition_name: table.get("mean", {})
                for partition_name, table in per_partition.items()
                if partition_name in visible
            }
            for name, per_partition in aggregated.items()
        },
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(_json_safe(summary), indent=2))
    return ReportPaths(csv=csv_path, png=png_path, json=json_path)


def _visible_partitions(unlock_count: int, shown: str, headline_has_test: bool) -> tuple[str, ...]:
    """While the test set is locked the report carries validation only, for every source.

    A `report.csv` that listed the one-hot floor's *test* numbers next to the pLM's
    validation-only rows would invite exactly the validation-versus-test comparison the
    locked-test protocol exists to prevent, so the test partition is published only when the
    source the figure is named after has test metrics of its own.
    """
    if shown == "test" or (unlock_count > 0 and headline_has_test):
        return PARTITIONS
    return ("validation",)


def _any_source(aggregated: dict[str, dict]) -> dict:
    return next(iter(aggregated.values()), {})


def _json_safe(value: Any) -> Any:
    """NaN and inf are not JSON; write them as null so `report.json` parses anywhere."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    return value


def _condition_names(spec: Any, run_result: RunResult | Any, baseline: dict | None) -> list[str]:
    names = _lookup(spec, "condition_columns")
    if not names and isinstance(run_result, RunResult):
        names = run_result.condition_columns
    if names:
        return [str(name) for name in names]
    names = (baseline or {}).get("target_columns")
    if names:
        return [str(name) for name in names]
    for record in _raw_records(run_result):
        for key in PARTITIONS:
            block = _lookup(record, key)
            for inner in ("per_condition", "per_target"):
                per_target = _lookup(block, inner) if isinstance(block, dict) else None
                if isinstance(per_target, dict) and per_target:
                    return [str(name) for name in per_target]
    raise ReportError(
        "Could not work out the condition names. Pass the LibrarySpec used for the run so the "
        "report can label its columns."
    )


def _model_label(run_result: RunResult | Any) -> str:
    if isinstance(run_result, RunResult) and run_result.model_name:
        return run_result.model_name
    for key in ("model_name", "adapter_name", "model", "backbone"):
        value = _lookup(run_result, key)
        if isinstance(value, str) and value:
            return value
    for record in _raw_records(run_result):
        value = _lookup(record, "model")
        if isinstance(value, str) and value:
            return value
    return "pLM"


def _raw_records(run_result: RunResult | Any) -> list[Any]:
    if isinstance(run_result, RunResult):
        return list(run_result.runs)
    if run_result is None:
        return []
    if isinstance(run_result, (list, tuple)):
        return list(run_result)
    for key in ("runs", "per_run", "run_metrics", "records", "results"):
        value = _lookup(run_result, key)
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    if isinstance(run_result, dict) and any(key in run_result for key in ("validation", "test", "metrics")):
        return [run_result]
    return []


def _collect_records(run_result: RunResult | Any, conditions: list[str]) -> list[dict]:
    """Normalize whatever a RunResult carries into one record per run, per partition.

    `colabsd.train.finetune` keeps the test partition out of `runs` entirely; `unlock_test`
    fills `test_runs` in later, so the two lists are merged on (split seed, model seed).
    """
    records = _records_from(_raw_records(run_result), conditions)
    unlocked = run_result.test_runs if isinstance(run_result, RunResult) else _lookup(run_result, "test_runs")
    if isinstance(unlocked, (list, tuple)) and unlocked:
        by_key = {_run_key(record): record for record in records}
        for extra in _records_from(unlocked, conditions):
            target = by_key.get(_run_key(extra))
            if target is None:
                records.append(extra)
                continue
            for partition in PARTITIONS:
                if partition in extra and partition not in target:
                    target[partition] = extra[partition]
    return records


def _records_from(raw_records: Any, conditions: list[str]) -> list[dict]:
    records = []
    for index, raw in enumerate(raw_records):
        blocks = {}
        for partition in PARTITIONS:
            block = _metric_block(_partition_payload(raw, partition), conditions)
            if block is not None:
                blocks[partition] = block
        if not blocks:
            continue
        model_seed = _lookup(raw, "model_seed")
        records.append(
            {
                "split_seed": _lookup(raw, "split_seed"),
                "model_seed": model_seed if model_seed is not None else _lookup(raw, "seed"),
                "index": index,
                **blocks,
            }
        )
    return records


def _run_key(record: dict) -> tuple:
    if record["split_seed"] is None and record["model_seed"] is None:
        return ("index", record["index"])
    return (record["split_seed"], record["model_seed"])


def _partition_payload(record: Any, partition: str) -> Any:
    aliases = [name for name, canonical in PARTITION_ALIASES.items() if canonical == partition]
    for name in aliases:
        for key in (name, f"{name}_metrics", f"{name}_results"):
            value = _lookup(record, key)
            if value is not None:
                return value
    for name in aliases:
        trues = _lookup(record, f"{name}_targets")
        preds = _lookup(record, f"{name}_predictions")
        if trues is not None and preds is not None:
            return {"y_true": trues, "y_pred": preds}
    if partition == "validation":
        for key in ("metrics", "final_metrics"):
            value = _lookup(record, key)
            if value is not None and _lookup(record, "validation") is None:
                declared = str(_lookup(record, "partition") or "validation")
                if PARTITION_ALIASES.get(declared, "validation") == "validation":
                    return value
    return None


def _metric_block(payload: Any, conditions: list[str]) -> dict | None:
    from colabsd.engine.metrics import TRACKED_METRICS, evaluate_predictions, summarize_metrics

    if payload is None or not isinstance(payload, dict) or not payload:
        return None
    if "y_true" in payload and "y_pred" in payload:
        per_target = evaluate_predictions(np.asarray(payload["y_true"]), np.asarray(payload["y_pred"]), conditions)
        return {"per_target": per_target, "mean": summarize_metrics(per_target)}

    per_target = None
    for key in ("per_condition", "per_target"):
        if isinstance(payload.get(key), dict):
            per_target = payload[key]
            break
    if per_target is None and conditions and set(conditions).issubset(payload):
        per_target = {name: payload[name] for name in conditions}
    if per_target is None and all(isinstance(value, dict) for value in payload.values()):
        per_target = payload
    if per_target:
        mean = payload.get("mean") if isinstance(payload.get("mean"), dict) else None
        return {"per_target": per_target, "mean": mean or summarize_metrics(per_target)}

    flat = {}
    for key, value in payload.items():
        name = str(key).removeprefix("mean_")
        if name in TRACKED_METRICS and isinstance(value, (int, float, np.floating)):
            flat[name] = float(value)
    return {"per_target": {}, "mean": flat} if flat else None


def _aggregate_source(records: list[dict], conditions: list[str], metrics: tuple[str, ...]) -> dict:
    aggregated: dict[str, dict] = {}
    for partition in PARTITIONS:
        present = [record[partition] for record in records if partition in record]
        if not present:
            continue
        table: dict[str, dict] = {}
        for condition in conditions:
            values = [block["per_target"].get(condition, {}) for block in present]
            if not any(values):
                continue
            table[condition] = {
                metric: summarize_across_runs([entry.get(metric, float("nan")) for entry in values])
                for metric in metrics
            }
        table["mean"] = {
            metric: summarize_across_runs([block["mean"].get(metric, float("nan")) for block in present])
            for metric in metrics
        }
        aggregated[partition] = table
    return aggregated


def _report_frame(
    aggregated: dict[str, dict],
    conditions: list[str],
    metrics: tuple[str, ...],
    partitions: tuple[str, ...] = PARTITIONS,
) -> pd.DataFrame:
    rows = []
    for source, per_partition in aggregated.items():
        for partition in partitions:
            table = per_partition.get(partition)
            if not table:
                continue
            for condition in [*conditions, "mean"]:
                entry = table.get(condition)
                if not entry:
                    continue
                row: dict[str, Any] = {"source": source, "partition": partition, "condition": condition}
                row["n_runs"] = int(max(item["n"] for item in entry.values()))
                for metric in metrics:
                    row[f"{metric}_mean"] = entry[metric]["mean"]
                    row[f"{metric}_sd"] = entry[metric]["sd"]
                rows.append(row)
    return pd.DataFrame(rows)


def _shown_partition(
    requested: str | None,
    aggregated: dict[str, dict],
    unlock_count: int,
    primary: str | None = None,
) -> str:
    """Pick the partition the figure headlines.

    Switching to test needs both a recorded unlock *and* test metrics on the source the
    figure is named after: a persisted unlock counter with a `RunResult` that predates the
    unlock would otherwise drop the pLM out of its own figure and leave the floor alone.
    """
    available = {partition for per_partition in aggregated.values() for partition in per_partition}
    if requested is not None:
        partition = PARTITION_ALIASES.get(str(requested), str(requested))
        if partition not in available:
            raise ReportError(f"No {partition} metrics in these runs; available partitions: {sorted(available)}.")
        if primary is not None and partition not in aggregated[primary]:
            raise ReportError(
                f"No {partition} metrics for {primary}; it only carries {sorted(aggregated[primary])}. "
                "Call colabsd.train.unlock_test(run_result, output_dir=...) first, and pass the "
                "RunResult it updated to build_report."
            )
        return partition
    headline = aggregated[primary] if primary is not None else _any_source(aggregated)
    if unlock_count > 0 and "test" in available and "test" in headline:
        return "test"
    return "validation" if "validation" in available else sorted(available)[0]


def _unlock_info(run_result: RunResult | Any, out_dir: Path) -> dict:
    """Highest unlock count on record. A RunResult built before `unlock_test` ran is stale, so the
    persisted counter can only ever be larger — reporting the larger number is the honest read."""
    found: list[tuple[int, str]] = []
    value = run_result.unlock_count if isinstance(run_result, RunResult) else _lookup(run_result, "unlock_count")
    if isinstance(value, (int, float)):
        found.append((int(value), "run_result.unlock_count"))
    unlock = _lookup(run_result, "unlock")
    if isinstance(unlock, dict):
        for key in ("unlock_count", "count"):
            if key in unlock:
                found.append((int(unlock[key]), "run_result.unlock"))
                break

    candidates = [out_dir]
    for key in ("output_dir", "out_dir", "run_dir"):
        candidate = _lookup(run_result, key)
        if candidate:
            candidates.insert(0, Path(candidate))
    paths = _lookup(run_result, "paths")
    if paths is not None:
        for key in ("output_dir", "out_dir", "unlock"):
            candidate = _lookup(paths, key)
            if candidate:
                candidates.insert(0, Path(candidate))
    for candidate in candidates:
        directory = candidate.parent if candidate.suffix == ".json" else candidate
        if (directory / UNLOCK_FILENAME).is_file():
            found.append((_unlock_count_in(directory), str(directory / UNLOCK_FILENAME)))
    if not found:
        return {"count": 0, "source": "not recorded"}
    count, source = max(found, key=lambda item: item[0])
    return {"count": count, "source": source}


def _unlock_count_in(directory: Path) -> int:
    """Read the counter with `colabsd.train.read_unlock_count`, the reader `unlock_test` writes for.

    An unreadable counter is not zero: reporting "0 unlocks" over a test set that has in
    fact been read is the one lie this figure exists to prevent, so it raises instead.
    """
    try:
        return read_unlock_count(directory)
    except ColabSDError as exc:
        raise ReportError(
            f"{exc} Until then the report cannot state how often the test set has been unlocked."
        ) from exc


def _lookup(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _bar(ax, x, values, sds, *, width: float, color: str, hatch: str, label: str) -> None:
    ax.bar(
        x,
        np.nan_to_num(np.asarray(values, dtype=float), nan=0.0),
        width=width,
        yerr=np.nan_to_num(np.asarray(sds, dtype=float), nan=0.0),
        color=color,
        hatch=hatch,
        edgecolor="black",
        linewidth=0.7,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "black"},
        label=label,
        zorder=3,
    )


def _extremes(means: list[float], sds: list[float]) -> list[float]:
    values = []
    for mean, sd in zip(means, sds, strict=True):
        if not np.isfinite(mean):
            continue
        spread = float(sd) if np.isfinite(sd) else 0.0
        values.extend([float(mean) - spread, float(mean) + spread])
    return values


def _set_ylim(ax, values: list[float], *, headroom: float, low_clip: float | None = None) -> None:
    """Scale an axis to its data, optionally refusing to follow an unbounded metric down.

    R2 has no lower bound, so one badly-scaled run flattens every other metric in the panel
    into an unreadable strip. Below `low_clip` the exact value stops carrying information —
    the bar is cut off and `_annotate_clipped` prints the number instead.
    """
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return
    low, high = min(0.0, min(finite)), max(finite)
    if low_clip is not None:
        low = max(low, low_clip)
    span = max(high - low, 1.0e-6)
    ax.set_ylim(low - 0.05 * span, high + headroom * span)


def _annotate_clipped(ax, bars: list[tuple[float, float]]) -> None:
    """Print the true value of every bar that runs off the bottom of the axis."""
    low = ax.get_ylim()[0]
    for x, value in bars:
        if np.isfinite(value) and value < low:
            ax.annotate(
                f"{value:.2f}",
                (x, low),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="black",
                bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                zorder=6,
            )


def _group_top(aggregated: dict, sources: list[str], partition: str, metric: str, group: str) -> float:
    """Highest bar top (mean + sd) in one x-group, used to keep labels off the bars."""
    tops = [
        entry["mean"] + (entry["sd"] if np.isfinite(entry["sd"]) else 0.0)
        for source in sources
        for entry in [aggregated[source][partition].get(group, {}).get(metric)]
        if isinstance(entry, dict) and np.isfinite(entry["mean"])
    ]
    return max(tops) if tops else float("-inf")


def _build_figure(
    *,
    aggregated: dict[str, dict],
    conditions: list[str],
    metrics: list[str],
    metric: str,
    partition: str,
    unlock: dict,
    floor: dict | None,
    model_label: str | None,
    spec: Any,
    n_runs: dict[str, int],
) -> Figure:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sources = [name for name in aggregated if partition in aggregated[name]]
    figure = Figure(figsize=(11.0, 8.0))
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.45})

    groups = [*conditions, "mean"]
    positions = np.arange(len(groups), dtype=float)
    width = 0.8 / max(len(sources), 1)
    extremes: list[float] = []
    for index, source in enumerate(sources):
        color, hatch = BAR_STYLES[index % len(BAR_STYLES)]
        table = aggregated[source][partition]
        means = [table.get(group, {}).get(metric, {}).get("mean", float("nan")) for group in groups]
        sds = [table.get(group, {}).get(metric, {}).get("sd", float("nan")) for group in groups]
        offset = (index - (len(sources) - 1) / 2.0) * width
        _bar(top, positions + offset, means, sds, width=width, color=color, hatch=hatch, label=source)
        extremes.extend(_extremes(means, sds))

    if floor is not None:
        extremes.extend(_extremes([floor["mean"]], [floor["sd"]]))
        top.axhline(floor["mean"], color="black", linestyle="--", linewidth=1.3, zorder=4)
        if np.isfinite(floor["sd"]):
            top.axhspan(floor["mean"] - floor["sd"], floor["mean"] + floor["sd"], color="0.85", alpha=0.6, zorder=0)
        left_top = _group_top(aggregated, sources, partition, metric, groups[0])
        right_top = _group_top(aggregated, sources, partition, metric, groups[-1])
        on_left = left_top <= right_top
        top.text(
            0.012 if on_left else 0.988,
            floor["mean"],
            f"one-hot floor ({floor['head']}): {floor['mean']:.3f}",
            transform=top.get_yaxis_transform(),
            ha="left" if on_left else "right",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=5,
        )

    _set_ylim(top, extremes, headroom=0.22)
    top.set_xticks(positions)
    top.set_xticklabels([*groups[:-1], "mean"], fontsize=10)
    top.set_ylabel(metric, fontsize=11)
    top.set_title(f"{metric} per condition — {partition} partition (error bars: ±1 sd across runs)", fontsize=12)
    top.axhline(0.0, color="black", linewidth=0.8)
    top.grid(axis="y", linestyle=":", linewidth=0.6, color="0.6")
    top.set_axisbelow(True)
    top.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=min(len(sources), 3), frameon=False, fontsize=9)

    if not unlock["count"]:
        unlock_text = f"test set LOCKED (0 unlocks) — {partition} shown"
    elif partition == "test":
        unlock_text = f"test set unlocked {unlock['count']}×"
    else:
        unlock_text = f"test set unlocked {unlock['count']}× — {partition} shown, no test metrics in this run"
    top.text(
        0.99,
        0.97,
        unlock_text,
        transform=top.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
    )

    metric_positions = np.arange(len(metrics), dtype=float)
    bottom_extremes: list[float] = []
    bottom_bars: list[tuple[float, float]] = []
    for index, source in enumerate(sources):
        color, hatch = BAR_STYLES[index % len(BAR_STYLES)]
        table = aggregated[source][partition]["mean"]
        means = [table.get(name, {}).get("mean", float("nan")) for name in metrics]
        sds = [table.get(name, {}).get("sd", float("nan")) for name in metrics]
        offset = (index - (len(sources) - 1) / 2.0) * width
        _bar(bottom, metric_positions + offset, means, sds, width=width, color=color, hatch=hatch, label=source)
        bottom_extremes.extend(_extremes(means, sds))
        bottom_bars.extend(zip(metric_positions + offset, means, strict=True))
    _set_ylim(bottom, bottom_extremes, headroom=0.10, low_clip=UNBOUNDED_METRIC_FLOOR)
    _annotate_clipped(bottom, bottom_bars)
    bottom.set_xticks(metric_positions)
    bottom.set_xticklabels(metrics, fontsize=9.5)
    bottom.set_ylabel("value (mean over conditions)", fontsize=10)
    bottom.set_title("All tracked metrics, averaged over conditions", fontsize=11)
    bottom.axhline(0.0, color="black", linewidth=0.8)
    bottom.grid(axis="y", linestyle=":", linewidth=0.6, color="0.6")
    bottom.set_axisbelow(True)

    headline = model_label or "one-hot baseline only"
    verdict = _verdict(aggregated, model_label, partition, metric, floor)
    figure.suptitle(f"{headline} vs the one-hot floor — {metric}", fontsize=13.5, y=0.98)
    figure.text(0.5, 0.945, verdict, ha="center", va="top", fontsize=10.5)
    figure.text(0.5, 0.015, _footer(n_runs, unlock, spec, conditions), ha="center", va="bottom", fontsize=8.5)
    return figure


def _verdict(aggregated: dict, model_label: str | None, partition: str, metric: str, floor: dict | None) -> str:
    if model_label is None or floor is None or model_label not in aggregated:
        return "no pLM/one-hot comparison available"
    entry = aggregated[model_label].get(partition, {}).get("mean", {}).get(metric)
    if entry is None or not np.isfinite(entry["mean"]):
        return "no pLM/one-hot comparison available"
    delta = entry["mean"] - floor["mean"]
    sign = "+" if delta >= 0 else "−"
    beats = "beats" if delta > 0 else "does NOT beat"
    return f"{model_label} {beats} the one-hot floor: {entry['mean']:.3f} vs {floor['mean']:.3f} ({sign}{abs(delta):.3f})"


def _footer(n_runs: dict[str, int], unlock: dict, spec: Any, conditions: list[str]) -> str:
    runs = ", ".join(f"{name}: {count} runs" for name, count in n_runs.items())
    sites = _lookup(spec, "k")
    site_text = f"{sites} mutated sites · " if isinstance(sites, int) else ""
    source = unlock["source"]
    source = Path(source).name if source.endswith(".json") else source
    return (
        f"{runs} · error bars are ±1 sd across runs · {site_text}{len(conditions)} conditions · "
        f"test-set unlock count: {unlock['count']} (from {source}) · colabsd"
    )
