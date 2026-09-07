"""The deliberate test-set step: the fourth cell of the main notebook.

This is the one place where ColabSeqDisplay departs from the ColabPLM layout on purpose.
The wizard in cell 2 is the whole training interface, but unlocking the test set is not part
of it: it is a separate cell, with its own confirmation, that you run once when you have
stopped changing things.

Why separate. Every choice made in the wizard — the backbone, the pooling, the
hyperparameters, how many seeds to average — is made on validation data. The test
partition is written to disk and left alone while that happens. Reading it is therefore not
a step in a workflow; it is a decision, taken once, at the end. Every read increments a
counter in `unlock.json`, and that counter is printed in the report, because a test set read
repeatedly while things are still being tuned is no longer a test set, and a reader of the
report deserves to know which one they are looking at.

Nothing here prevents a second unlock. It makes it visible: the panel says how many times
this run directory has already been unlocked before you can unlock it again, the
confirmation resets after every unlock so the next one has to be given deliberately, and the
final report carries the count beside the numbers it qualifies.

What it writes. The same `performance_report.zip` the export cell above writes — the report
CSV, the figure, and the JSON that records what they show — rewritten through
`colabsd.ui.exports` so that it now carries the test numbers and the unlock count. The
validation archive is obtainable without ever running this cell; that is the point of it.
This cell only changes what is inside the archive, and the archive says which it is.

The decision layer is pure — `status_notices`, `blocking`, `condition_rows`, `report_html`,
`floor_line` — and `UnlockPanel` only wires it to two widgets. Presentation is
`colabsd.ui.theme` and `colabsd.ui.core.Message`, the vocabulary the other three interfaces
use, so this cell does not look like a different program. The sentence that says what a test
number is worth after N reads lives in `colabsd.ui.exports`, so the panel, the archive's JSON
and its README all say it the same way.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from colabsd.ui import exports, theme
from colabsd.ui.core import Message, render_messages
from colabsd.ui.exports import number, unlock_verdict

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

#: Only used when `colabsd.train` cannot be imported at all. It is a fallback, not a
#: mirror: the dropdown is filled from `colabsd.train.METRICS` itself whenever that import
#: works, so what the form offers is what a run records, by construction rather than by
#: someone remembering to edit two lists. `tests/test_ui_side.py` holds them equal.
FALLBACK_METRICS: tuple[str, ...] = ("R2", "Pearson", "Spearman", "P@10", "P@50", "NDCG@10", "NDCG@50")


def _recorded_metrics() -> tuple[str, ...]:
    try:
        from colabsd.train import METRICS as recorded
    except ImportError:  # pragma: no cover - colabsd.train is a hard dependency in Colab
        return FALLBACK_METRICS
    return tuple(str(metric) for metric in recorded) or FALLBACK_METRICS


METRICS: tuple[str, ...] = _recorded_metrics()
#: One value, so the dropdown, the archive and the report cannot default to different metrics.
DEFAULT_METRIC = exports.DEFAULT_METRIC


@dataclass(frozen=True)
class UnlockInputs:
    """What the unlock needs, gathered from wherever the notebook keeps it."""

    run_result: Any = None
    output_dir: Path | None = None
    baseline: dict | None = None
    spec: Any = None
    work_dir: Path | None = None

    @property
    def ready(self) -> bool:
        return not missing_inputs(self)


@dataclass(frozen=True)
class ConditionRow:
    """One condition's test-partition number for the metric on show.

    Named for the condition rather than for the partition so that pytest never mistakes it
    for a test case of its own.
    """

    condition: str
    mean: float
    sd: float
    n_runs: int


@dataclass(frozen=True)
class FinalReport:
    """Everything the last cell of the notebook produced."""

    unlock_count: int
    metric: str
    rows: tuple[ConditionRow, ...]
    macro_mean: float
    macro_sd: float
    floor: dict | None
    is_provisional: bool
    model_label: str
    paths: dict[str, Path]
    html: str
    #: The performance archive rewritten by this unlock, the one file worth keeping.
    archive: Path | None = None


#: What a training wizard might have called each thing it hands over. Aliases, so this cell
#: keeps working if the wizard names its result `run` rather than `run_result`.
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "run_result": ("run_result", "run", "result"),
    "output_dir": ("output_dir", "run_dir", "out_dir"),
    "baseline": ("baseline", "one_hot_baseline"),
    "spec": ("spec", "library_spec"),
    "work_dir": ("work_dir", "working_dir"),
}


def resolve_inputs(
    source: Any = None,
    *,
    run_result: Any = None,
    output_dir: Path | str | None = None,
    baseline: dict | None = None,
    spec: Any = None,
    work_dir: Path | str | None = None,
) -> UnlockInputs:
    """Collect the inputs from explicit arguments, falling back to attributes of *source*.

    `source` is duck-typed on purpose: anything carrying a `run_result` (or `run`, or
    `result`) and an `output_dir` (or `run_dir`) works, which is how the training wizard
    hands its result to this cell without either side importing the other.
    """

    def pick(explicit: Any, name: str) -> Any:
        if explicit is not None:
            return explicit
        if source is None:
            return None
        for alias in SOURCE_ALIASES[name]:
            found = getattr(source, alias, None)
            if found is not None:
                return found
        return None

    directory = pick(output_dir, "output_dir")
    working = pick(work_dir, "work_dir")
    return UnlockInputs(
        run_result=pick(run_result, "run_result"),
        output_dir=None if directory is None else Path(directory),
        baseline=pick(baseline, "baseline"),
        spec=pick(spec, "spec"),
        work_dir=None if working is None else Path(working),
    )


def missing_inputs(inputs: UnlockInputs) -> list[str]:
    """Which of the two things the unlock cannot proceed without are absent."""
    missing = []
    if inputs.run_result is None:
        missing.append("run_result")
    if inputs.output_dir is None:
        missing.append("output_dir")
    return missing


def status_notices(*, inputs: UnlockInputs, unlock_count: int, confirmed: bool) -> list[Message]:
    """The messages this panel shows for the current state, most consequential first.

    The "already unlocked" warning comes before anything else, so it is read before the
    second unlock happens rather than after.
    """
    out: list[Message] = []
    if unlock_count == 1:
        out.append(
            Message(
                "already_unlocked_once",
                "warning",
                "**This run directory has already been unlocked once.** The number you got then is the "
                "held-out one. If you have changed the pooling, the backbone or the hyperparameters since, a "
                "second read is a validation number wearing a test label — and the report will say so.",
            )
        )
    elif unlock_count > 1:
        out.append(
            Message(
                "already_unlocked_repeatedly",
                "warning",
                f"**This run directory has already been unlocked {unlock_count} times.** Each read after the "
                "first informs the choices you make next, so what comes back is no longer held out. Unlocking "
                "again is allowed and will be counted; the count travels in the report and in any bundle you "
                "export afterwards.",
            )
        )
    else:
        out.append(
            Message(
                "never_unlocked",
                "info",
                "This run directory has never been unlocked. The test partition has been on disk, untouched, "
                "since training finished.",
            )
        )

    missing = missing_inputs(inputs)
    if missing:
        out.append(
            Message(
                "missing_inputs",
                "stop",
                f"Nothing to unlock yet: no {' and no '.join(missing)} in this session. Train a model in the "
                "wizard above first — the test partition only exists once something has been trained against it.",
            )
        )
    elif not confirmed:
        out.append(
            Message(
                "not_confirmed",
                "stop",
                "Tick the confirmation below to unlock. It is deliberately not a default: everything above this "
                "cell is validation, and the test partition is read once, on purpose, so the reported number "
                "means what it says.",
            )
        )
    return out


def blocking(found: Sequence[Message]) -> list[Message]:
    """The messages that refuse to let the unlock happen."""
    return [message for message in found if message.severity == "stop"]


def can_unlock(*, inputs: UnlockInputs, unlock_count: int, confirmed: bool) -> bool:
    """True when a click would actually read the test set."""
    return not blocking(status_notices(inputs=inputs, unlock_count=unlock_count, confirmed=confirmed))


def condition_rows(payload: dict, metric: str) -> list[ConditionRow]:
    """The per-condition test numbers for *metric*, out of an `unlock_test` payload."""
    rows = []
    for record in payload.get("aggregate", []) or []:
        if str(record.get("metric")) != metric:
            continue
        rows.append(
            ConditionRow(
                condition=str(record.get("condition", "?")),
                mean=float(record.get("mean", float("nan"))),
                sd=float(record.get("sd", float("nan"))),
                n_runs=int(record.get("n_runs", 0) or 0),
            )
        )
    return rows


def macro_summary(payload: dict, metric: str) -> tuple[float, float]:
    """The test metric averaged over conditions, mean and sd across runs."""
    entry = (payload.get("summary", {}) or {}).get(metric, {}) or {}
    return float(entry.get("mean", float("nan"))), float(entry.get("sd", float("nan")))


def floor_line(floor: dict | None, macro_mean: float, metric: str) -> str:
    """The one-hot floor, and how far above it the language model actually is."""
    if not floor:
        return (
            f"<b>One-hot floor:</b> not available for {metric} — either the one-hot baseline has not been run, "
            "or it recorded no such metric. Without it there is nothing here to say whether a language model "
            "was worth it."
        )
    head = (
        f"<b>One-hot floor:</b> {floor['head']} on the {floor['partition']} partition scores "
        f"{float(floor['mean']):.4f} ± {float(floor['sd']):.4f} {metric}."
    )
    if macro_mean != macro_mean:
        return f"{head} There is no {metric} for the language model to compare against it."
    margin = macro_mean - float(floor["mean"])
    below = f"<b style='color:{theme.SEVERITY_COLOR['stop']}'>below</b>"
    verdict = "above" if margin >= 0 else below
    return f"{head} The language model is {abs(margin):.4f} {verdict} it."


# ----------------------------------------------------------------------------------------
# HTML rendering.
# ----------------------------------------------------------------------------------------

def why_separate_html() -> str:
    """Why this is its own cell, said plainly, above the confirmation."""
    return theme.heading_html("Unlock the test set") + theme.note_html(
        "Everything above this cell is **validation**. The backbone, the pooling, the hyperparameters and the "
        "number of seeds were all chosen by looking at validation numbers, and the test partition was written "
        "to disk and left alone while that happened.\n\n"
        "That is why unlocking is a separate step you take on purpose, and not part of the wizard: the test "
        "partition is touched once, deliberately, at the end, so the number you report means what it says. "
        "Every read is counted in `unlock.json` and printed in the report — including the second one."
    )


def report_html(
    *,
    unlock_count: int,
    metric: str,
    rows: Sequence[ConditionRow],
    macro_mean: float,
    macro_sd: float,
    floor: dict | None,
    is_provisional: bool,
    model_label: str,
    paths: dict[str, Path] | None = None,
) -> str:
    """The final report: test numbers, the one-hot floor and the unlock count, together.

    The three belong on one screen. A test number without the floor does not say whether a
    language model earned its keep, and either without the unlock count does not say how
    much the number should be believed.
    """
    # One unlock is the number this design is for; any other count is coloured like a warning.
    count_color = theme.SEVERITY_COLOR["stop"] if unlock_count != 1 else "inherit"
    body = "".join(
        "<tr>"
        f"<td style='padding:3px 14px 3px 0'>{row.condition}</td>"
        f"<td style='padding:3px 14px 3px 0;font-family:monospace'>{number(row.mean)} ± {number(row.sd)}</td>"
        f"<td style='padding:3px 0;color:#555'>{row.n_runs} run(s)</td>"
        "</tr>"
        for row in rows
    )
    if not rows:
        body = (
            "<tr><td colspan='3' style='padding:3px 0'>The unlocked run recorded no per-condition "
            f"<b>{metric}</b> numbers. Pick another metric above and unlock again, or read "
            "<code>report.csv</code>.</td></tr>"
        )
    files = ""
    if paths:
        files = (
            "<div style='margin-top:8px;color:#555'>written: "
            + ", ".join(f"<code>{Path(path).name}</code>" for path in paths.values())
            + "</div>"
        )
    provisional = ""
    if is_provisional:
        provisional = theme.message_html(
            "These numbers came from **PROVISIONAL** hyperparameters: a lower bound on what this pair can do, "
            "not a benchmark result.",
            "warning",
        )
    return (
        "<div style='border:2px solid rgba(128,128,128,0.6);padding:12px 14px;margin:8px 0;line-height:1.55'>"
        f"<div style='font-size:15px'><b>Final report — test partition · {model_label}</b></div>"
        f"<div style='color:{count_color};margin-top:4px'><b>Test set unlocked {unlock_count}x</b> "
        f"in this run directory. {unlock_verdict(unlock_count)}</div>"
        f"<table style='border-collapse:collapse;margin-top:8px'>"
        f"<tr style='text-align:left;border-bottom:1px solid rgba(128,128,128,0.35)'>"
        f"<th style='padding-right:14px'>condition</th>"
        f"<th style='padding-right:14px'>test {metric}</th><th>runs</th></tr>{body}</table>"
        f"<div style='margin-top:8px'><b>Test {metric}, averaged over conditions:</b> "
        f"{number(macro_mean)} ± {number(macro_sd)}</div>"
        f"<div style='margin-top:4px'>{floor_line(floor, macro_mean, metric)}</div>"
        f"{provisional}{files}</div>"
    )


def render_report(report: FinalReport) -> str:
    """`report_html` for an already-assembled `FinalReport`."""
    return report_html(
        unlock_count=report.unlock_count,
        metric=report.metric,
        rows=report.rows,
        macro_mean=report.macro_mean,
        macro_sd=report.macro_sd,
        floor=report.floor,
        is_provisional=report.is_provisional,
        model_label=report.model_label,
        paths=report.paths,
    )


# ----------------------------------------------------------------------------------------
# The widget layer.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnlockRunners:
    """The side effects, injected so the panel can be driven headlessly in tests."""

    unlock_test: Callable[..., dict]
    read_unlock_count: Callable[[Any], int]
    build_report: Callable[..., Any]
    baseline_floor: Callable[..., dict | None]
    download: Callable[[Path], None]
    show_table: Callable[[Any], None]


def default_runners() -> UnlockRunners:
    """The real implementations, imported on call so importing this module stays cheap."""
    from colabsd.baseline import baseline_floor
    from colabsd.report import build_report
    from colabsd.train import read_unlock_count, unlock_test

    def download(path: Path) -> None:
        try:
            from google.colab import files  # noqa: PLC0415
        except ImportError:
            return
        files.download(str(path))

    def show_table(frame: Any) -> None:
        from IPython.display import display

        display(frame)

    return UnlockRunners(
        unlock_test=unlock_test,
        read_unlock_count=read_unlock_count,
        build_report=build_report,
        baseline_floor=baseline_floor,
        download=download,
        show_table=show_table,
    )


class UnlockPanel:
    """Two widgets and a report. The confirmation is the only way through."""

    def __init__(
        self,
        *,
        inputs: UnlockInputs,
        metric: str = DEFAULT_METRIC,
        runners: UnlockRunners | None = None,
    ) -> None:
        import ipywidgets

        self.inputs = inputs
        self.runners = runners or default_runners()
        self.report: FinalReport | None = None
        self.unlock_calls = 0

        self.intro = theme.html(why_separate_html())
        self.metric = ipywidgets.Dropdown(
            options=list(METRICS),
            value=metric if metric in METRICS else DEFAULT_METRIC,
            description="Report metric:",
            style={"description_width": "initial"},
            layout=ipywidgets.Layout(width="320px"),
        )
        self.confirm = ipywidgets.Checkbox(
            value=False,
            description="I have stopped changing things. Read the test set once and count it.",
            indent=False,
            layout=ipywidgets.Layout(width="640px"),
        )
        self.unlock_button = ipywidgets.Button(
            description="Unlock the test set",
            button_style="danger",
            layout=ipywidgets.Layout(width="240px"),
        )
        self.notice_box = theme.html("")
        self.report_box = theme.html("")
        self.log = ipywidgets.Output()

        self.confirm.observe(self._on_change, names="value")
        self.metric.observe(self._on_change, names="value")
        self.unlock_button.on_click(self.on_unlock)
        self.refresh()

    # -- state ---------------------------------------------------------------------------

    @property
    def unlock_count(self) -> int:
        """How often this run directory has been unlocked, straight off disk."""
        if self.inputs.output_dir is None:
            return 0
        try:
            return int(self.runners.read_unlock_count(self.inputs.output_dir))
        except Exception as exc:
            self._say(f"could not read the unlock counter: {exc}")
            return 0

    def notices(self) -> list[Message]:
        """The messages for the current state."""
        return status_notices(
            inputs=self.inputs, unlock_count=self.unlock_count, confirmed=bool(self.confirm.value)
        )

    def log_text(self) -> str:
        """Everything printed into the log area, as one string."""
        parts = []
        for entry in self.log.outputs:
            parts.append(entry.get("text", "") if isinstance(entry, dict) else getattr(entry, "text", ""))
        return "".join(parts)

    def _say(self, message: str) -> None:
        self.log.append_stdout(message + "\n")

    def _on_change(self, _change: Any = None) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Re-render the notices; the count is read from disk every time."""
        self.notice_box.value = render_messages(self.notices())

    def display_box(self) -> Any:
        """The whole panel as one `VBox`, built but not shown."""
        import ipywidgets

        return ipywidgets.VBox(
            [
                self.intro,
                self.notice_box,
                self.metric,
                self.confirm,
                self.unlock_button,
                self.report_box,
                self.log,
            ]
        )

    def display(self) -> UnlockPanel:
        """Show the panel. Returns self so a notebook cell can keep the handle."""
        from IPython.display import display

        display(self.display_box())
        return self

    # -- action --------------------------------------------------------------------------

    def on_unlock(self, _button: Any = None) -> FinalReport | None:
        """Read the test set once, then write and render the final report.

        One click reads the test set at most once: `colabsd.train.unlock_test` is called a
        single time and the confirmation is reset afterwards, so the next unlock needs a
        fresh, deliberate tick.
        """
        refusals = blocking(self.notices())
        if refusals:
            for message in refusals:
                self._say("not unlocked: " + plain_text(message.text))
            self.refresh()
            return None

        metric = str(self.metric.value)
        try:
            payload = self.runners.unlock_test(self.inputs.run_result, output_dir=self.inputs.output_dir)
            self.unlock_calls += 1
        except Exception as exc:
            self._say(f"unlock failed: {exc}")
            self.refresh()
            return None

        macro_mean, macro_sd = macro_summary(payload, metric)
        export = self._export_performance(metric)
        paths = {} if export is None else export.written()
        shown_partition = "test" if export is None else export.facts.partition
        assembled = FinalReport(
            unlock_count=int(payload.get("unlock_count", self.unlock_count)),
            metric=metric,
            rows=tuple(condition_rows(payload, metric)),
            macro_mean=macro_mean,
            macro_sd=macro_sd,
            floor=self._floor(metric, shown_partition),
            is_provisional=bool(payload.get("is_provisional", False)),
            model_label=str(payload.get("model_name", "this model")),
            paths=paths,
            html="",
            archive=None if export is None else export.path,
        )
        self.report = replace(assembled, html=render_report(assembled))
        count = self.report.unlock_count
        self.report_box.value = self.report.html
        self._say(f"unlock #{count} at {payload.get('unlocked_utc', 'unknown time')}")
        self._say(f"test {metric}, averaged over conditions: {macro_mean:.4f} ± {macro_sd:.4f}")
        if export is not None:
            # Two downloads and no more: the archive is what to keep — it says which
            # partition and how many reads — and the figure is what to look at now.
            self._say(f"performance archive: {export.path} — {export.facts.describe()}")
            self.runners.download(export.path)
            figure = export.report_paths.get("report.png")
            if figure is not None:
                self.runners.download(Path(figure))
        # A second unlock stays possible, but it has to be confirmed again on purpose.
        self.confirm.value = False
        self.refresh()
        return self.report

    def _export_runners(self) -> exports.ExportRunners:
        """This panel's injected side effects, handed to `colabsd.ui.exports` unchanged.

        The export module writes the archive, but it writes it through whatever this panel
        was given — so a test that fakes the report writer fakes it here too.
        """
        return exports.ExportRunners(
            build_report=self.runners.build_report,
            read_unlock_count=self.runners.read_unlock_count,
            download=self.runners.download,
        )

    def _export_performance(self, metric: str) -> exports.PerformanceExport | None:
        """Rewrite the performance archive, now that there are test numbers to put in it.

        The same file name the export cell above writes with validation numbers. Which
        partition it describes is `colabsd.report`'s decision, recorded in `report.json` and
        copied into the archive's own manifest — this panel does not get a second opinion.
        """
        if self.inputs.output_dir is None:
            return None
        work_dir = Path(self.inputs.work_dir or self.inputs.output_dir)
        try:
            return exports.export_performance(
                run_result=self.inputs.run_result,
                baseline=self.inputs.baseline,
                spec=self.inputs.spec,
                best=None,
                work_dir=work_dir,
                report_dir=work_dir / "report",
                metric=metric,
                runners=self._export_runners(),
                download=False,
            )
        except Exception as exc:
            self._say(f"the report could not be written: {exc}")
            return None

    def _floor(self, metric: str, partition: str) -> dict | None:
        """The one-hot floor on the partition the report is showing."""
        if not self.inputs.baseline:
            return None
        try:
            floor = self.runners.baseline_floor(self.inputs.baseline, metric=metric, partition=partition)
            if floor is None and partition != "validation":
                floor = self.runners.baseline_floor(self.inputs.baseline, metric=metric, partition="validation")
            return floor
        except Exception as exc:
            self._say(f"the one-hot floor could not be read: {exc}")
            return None


def plain_text(text: str) -> str:
    """A message's markdown, flattened for a plain-text log line."""
    flat = _MARKDOWN_LINK.sub(r"\1 (\2)", text)
    return flat.replace("**", "").replace("`", "").replace("*", "")


def launch(
    source: Any = None,
    *,
    run_result: Any = None,
    output_dir: Path | str | None = None,
    baseline: dict | None = None,
    spec: Any = None,
    work_dir: Path | str | None = None,
    metric: str = DEFAULT_METRIC,
    runners: UnlockRunners | None = None,
) -> UnlockPanel:
    """Build the panel and show it. This is the whole fourth cell of the main notebook."""
    inputs = resolve_inputs(
        source,
        run_result=run_result,
        output_dir=output_dir,
        baseline=baseline,
        spec=spec,
        work_dir=work_dir,
    )
    return UnlockPanel(inputs=inputs, metric=metric, runners=runners).display()
