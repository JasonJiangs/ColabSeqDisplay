"""The deliberate test-set step: the fourth cell of the main notebook.

This is the one place where ColabSeqDisplay departs from the ColabPLM layout on purpose.
The wizard in cell 2 is the whole training interface; unlocking the test set is a separate
cell, with its own confirmation, run once when you have stopped changing things.

Why separate. Every choice made in the wizard — the backbone, the hyperparameters, how many
seeds to average — is made on validation data, and the test partition is written to disk and
left alone while that happens. Reading it is therefore not a step in a workflow; it is a
decision, taken once, at the end. Every read increments a counter in `unlock.json` and that
counter is printed in the report, because a test set read repeatedly while things are still
being tuned is no longer a test set, and a reader of the report deserves to know which one
they are looking at.

Nothing here prevents a second unlock; it makes one visible. The panel says how many times
this run directory has already been unlocked before you can unlock it again, the confirmation
resets after every unlock, and the report carries the count beside the numbers it qualifies.

What it writes. The same `performance_report.zip` the export cell above writes, rewritten
through `colabsd.ui.exports` so that it now carries the test numbers and the unlock count.
The validation archive is obtainable without ever running this cell.

The final report also prints the budget the run was given — its epochs, its patience and both
batch sizes — with the fields the user set marked as theirs, because a test number nobody can
reproduce is half a result. It is the archive's own record, in `colabsd.bundle`'s words.

The decision layer is pure — `status_notices`, `blocking`, `condition_rows`, `macro_summary`,
`report_html` — and `UnlockPanel` only wires it to two widgets. The sentence that says what a
test number is worth after N reads lives in `colabsd.ui.exports`, so the panel, the archive's
JSON and its README all say it the same way.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from colabsd.ui import exports, theme
from colabsd.ui.core import Message, render_messages
from colabsd.ui.exports import number, unlock_verdict

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

#: Used only when `colabsd.train` cannot be imported at all. The dropdown is otherwise filled
#: from `colabsd.train.METRICS` itself, so the form offers what a run records;
#: `tests/test_ui_side.py` holds the two equal.
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
    spec: Any = None
    work_dir: Path | None = None
    #: The registry entry the run was configured from. Not needed to unlock anything: it is
    #: carried so the archive this cell rewrites can say which of the budget the user set, and
    #: an unlock without it still writes every number, minus that one piece of provenance.
    best: Any = None

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
    is_provisional: bool
    model_label: str
    paths: dict[str, Path]
    html: str
    #: The performance archive rewritten by this unlock, the one file worth keeping.
    archive: Path | None = None
    #: What the run was given: epochs, patience and both batch sizes, with the fields the user
    #: set marked as theirs. Empty when the run recorded none.
    budget: dict[str, Any] = field(default_factory=dict)


#: What a training wizard might have called each thing it hands over.
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "run_result": ("run_result", "run", "result"),
    "output_dir": ("output_dir", "run_dir", "out_dir"),
    "spec": ("spec", "library_spec"),
    "work_dir": ("work_dir", "working_dir"),
    # `trained_best` first: a wizard freezes the entry the run was actually configured from
    # under that name, while `best` follows its dropdown and may already name another backbone.
    "best": ("trained_best", "best", "best_config"),
}


def resolve_inputs(
    source: Any = None,
    *,
    run_result: Any = None,
    output_dir: Path | str | None = None,
    spec: Any = None,
    work_dir: Path | str | None = None,
    best: Any = None,
) -> UnlockInputs:
    """Collect the inputs from explicit arguments, falling back to attributes of *source*.

    `source` is duck-typed: anything carrying a `run_result` (or `run`, or `result`) and an
    `output_dir` (or `run_dir`) works, which is how the training wizard hands its result to
    this cell without either side importing the other.
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
        spec=pick(spec, "spec"),
        work_dir=None if working is None else Path(working),
        best=pick(best, "best"),
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
    """The messages this panel shows, most consequential first.

    The "already unlocked" warning comes first, so it is read before the second unlock rather
    than after.
    """
    out: list[Message] = []
    if unlock_count == 1:
        out.append(
            Message(
                "already_unlocked_once",
                "warning",
                "**This run directory has already been unlocked once.** The number you got then is the "
                "held-out one. If you have changed anything since, a second read is a validation number "
                "wearing a test label, and the report will say so.",
            )
        )
    elif unlock_count > 1:
        out.append(
            Message(
                "already_unlocked_repeatedly",
                "warning",
                f"**This run directory has already been unlocked {unlock_count} times.** What comes back is no "
                "longer held out. Unlocking again is allowed and counted, in the report and in any bundle you "
                "export afterwards.",
            )
        )
    else:
        out.append(
            Message(
                "never_unlocked",
                "info",
                "This run directory has never been unlocked: the test partition has been on disk, untouched, "
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
                "wizard above first.",
            )
        )
    elif not confirmed:
        out.append(
            Message(
                "not_confirmed",
                "stop",
                "Tick the confirmation below to unlock. It is deliberately not a default.",
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


# ----------------------------------------------------------------------------------------
# HTML rendering.
# ----------------------------------------------------------------------------------------

def why_separate_html() -> str:
    """Why this is its own cell, said plainly, above the confirmation."""
    return theme.heading_html("Unlock the test set") + theme.note_html(
        "Everything above this cell is **validation**: the backbone, the hyperparameters and the number of "
        "seeds were all chosen by looking at validation numbers, and the test partition was left on disk "
        "while that happened.\n\n"
        "So it is touched once, deliberately, here rather than inside the wizard, and the number you report "
        "means what it says. Every read is counted in `unlock.json` and printed in the report — including "
        "the second one."
    )


def report_html(
    *,
    unlock_count: int,
    metric: str,
    rows: Sequence[ConditionRow],
    macro_mean: float,
    macro_sd: float,
    is_provisional: bool,
    model_label: str,
    paths: dict[str, Path] | None = None,
    budget: Any = None,
) -> str:
    """The final report: the test numbers and the unlock count, together.

    A test number without the count does not say how much to believe it, so the two are one
    block and never two. The budget the run was given goes in the same block for the same
    reason: it is what somebody would need in order to get this number again.
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
            f"<b>{metric}</b> numbers. <code>report.csv</code> in the archive carries every metric that "
            "was recorded.</td></tr>"
        )
    files = ""
    if paths:
        files = (
            "<div style='margin-top:8px;color:#555'>written: "
            + ", ".join(f"<code>{Path(path).name}</code>" for path in paths.values())
            + "</div>"
        )
    settings = (
        "<div style='margin-top:8px;color:#555'>trained with: "
        f"{exports.budget_line(budget)}</div>"
    )
    provisional = ""
    if is_provisional:
        provisional = theme.message_html(
            "These numbers came from **PROVISIONAL** hyperparameters: a lower bound on what this backbone "
            "can do, not a benchmark result.",
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
        f"{settings}{provisional}{files}</div>"
    )


def render_report(report: FinalReport) -> str:
    """`report_html` for an already-assembled `FinalReport`."""
    return report_html(
        unlock_count=report.unlock_count,
        metric=report.metric,
        rows=report.rows,
        macro_mean=report.macro_mean,
        macro_sd=report.macro_sd,
        is_provisional=report.is_provisional,
        model_label=report.model_label,
        paths=report.paths,
        budget=report.budget,
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
    download: Callable[[Path], None]
    show_table: Callable[[Any], None]


def default_runners() -> UnlockRunners:
    """The real implementations, imported on call so importing this module stays cheap."""
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

        One click reads the test set at most once: `colabsd.train.unlock_test` is called once
        and the confirmation is reset afterwards, so the next unlock needs a fresh tick.
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
        assembled = FinalReport(
            unlock_count=int(payload.get("unlock_count", self.unlock_count)),
            metric=metric,
            rows=tuple(condition_rows(payload, metric)),
            macro_mean=macro_mean,
            macro_sd=macro_sd,
            is_provisional=bool(payload.get("is_provisional", False)),
            model_label=str(payload.get("model_name", "this model")),
            paths=paths,
            html="",
            archive=None if export is None else export.path,
            # The archive's own reading of the budget, so the panel and the file agree; nothing
            # found is nothing shown, rather than the entry the run was looked up from.
            budget=exports.budget_of(
                None if export is None else export.facts.budget, self.inputs.run_result, self.inputs.best
            ),
        )
        self.report = replace(assembled, html=render_report(assembled))
        count = self.report.unlock_count
        self.report_box.value = self.report.html
        self._say(f"unlock #{count} at {payload.get('unlocked_utc', 'unknown time')}")
        self._say(f"test {metric}, averaged over conditions: {macro_mean:.4f} ± {macro_sd:.4f}")
        if export is not None:
            # Two downloads and no more: the archive to keep, the figure to look at now.
            self._say(f"performance archive: {export.path} — {export.facts.describe()}")
            self.runners.download(export.path)
            figure = export.report_paths.get("report.png")
            if figure is not None:
                self.runners.download(Path(figure))
        self.confirm.value = False  # a second unlock has to be confirmed again
        self.refresh()
        return self.report

    def _export_runners(self) -> exports.ExportRunners:
        """This panel's injected side effects, handed to `colabsd.ui.exports` unchanged.

        The export module writes the archive through whatever this panel was given, so a test
        that fakes the report writer fakes it here too.
        """
        return exports.ExportRunners(
            build_report=self.runners.build_report,
            read_unlock_count=self.runners.read_unlock_count,
            download=self.runners.download,
        )

    def _export_performance(self, metric: str) -> exports.PerformanceExport | None:
        """Rewrite the performance archive, now that there are test numbers to put in it.

        The same file name the export cell above writes with validation numbers, and the same
        call: the registry entry goes with it, because the archive has to say which of the
        budget was the user's and the entry is what it was looked up from. Which partition the
        archive describes is `colabsd.report`'s decision, recorded in `report.json` and copied
        into the manifest; this panel does not get a second opinion on that.
        """
        if self.inputs.output_dir is None:
            return None
        work_dir = Path(self.inputs.work_dir or self.inputs.output_dir)
        try:
            return exports.export_performance(
                run_result=self.inputs.run_result,
                spec=self.inputs.spec,
                best=self.inputs.best,
                work_dir=work_dir,
                report_dir=work_dir / "report",
                metric=metric,
                runners=self._export_runners(),
                download=False,
            )
        except Exception as exc:
            self._say(f"the report could not be written: {exc}")
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
    spec: Any = None,
    work_dir: Path | str | None = None,
    best: Any = None,
    metric: str = DEFAULT_METRIC,
    runners: UnlockRunners | None = None,
) -> UnlockPanel:
    """Build the panel and show it. This is the whole fourth cell of the main notebook."""
    inputs = resolve_inputs(
        source,
        run_result=run_result,
        output_dir=output_dir,
        spec=spec,
        work_dir=work_dir,
        best=best,
    )
    return UnlockPanel(inputs=inputs, metric=metric, runners=runners).display()
