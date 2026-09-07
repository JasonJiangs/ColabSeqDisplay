"""The two things a training session hands back: a model bundle and a performance archive.

Both downloads live here, so the export cell of the main notebook and the unlock cell below
it write the same files in the same shape from one place.

* `model_bundle.zip` — `colabsd.bundle`: the weights, and the provenance needed to use them.
* `performance_report.zip` — the numbers: `report.csv`, `report.png` and `report.json`
  exactly as `colabsd.report.build_report` writes them, plus a `performance.json` and a
  `README.txt` that say what those numbers are.

**Why the archive exists.** `build_report` has always been able to publish a *validation*
report while the test partition is still locked — that is what its `shown_partition` is for
— but the only call to it sat inside the unlock handler. A user who followed this tool's own
discipline and never unlocked therefore ended up with no report at all. The validation
archive is now the ordinary first export, obtainable with the test set untouched; after an
unlock the same file is rewritten, carrying the test numbers and the count.

**Why the archive describes itself.** A CSV of numbers with no provenance is a trap: months
later nobody can tell whether 0.56 was validation or test, whether the test set had been read
once or five times, which backbone and pooling produced it, or whether the hyperparameters
behind it were tuned or a placeholder. `performance.json` records all four, `README.txt`
says the same in prose for a reader who will not open JSON, and every member carries a
sha256 so a truncated download is caught rather than believed. The provenance block is
`colabsd.bundle.provenance_block`, the one the model bundle stamps, so the two downloads
describe the same run in the same words.

The decision layer is pure — `performance_facts`, `manifest_payload`, `facts_from_manifest`,
`readme_text`, `notices`, `partition_verdict`, `unlock_verdict` — and only `export_bundle`,
`export_performance` and the two archive functions touch a disk. The widgets are in
`colabsd.ui.main_workflow` and `colabsd.ui.unlock`, and are a call each.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colabsd import __version__
from colabsd.bundle import file_sha256, hyperparameter_status, provenance_block, sha256_hex
from colabsd.errors import ColabSDError
from colabsd.ui import theme
from colabsd.ui.core import Message, render_messages

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_FORMAT = "colabsd-performance"

DEFAULT_BUNDLE_NAME = "model_bundle.zip"
DEFAULT_ARCHIVE_NAME = "performance_report.zip"
DEFAULT_REPORT_DIRNAME = "report"
DEFAULT_METRIC = "Spearman"

MANIFEST_NAME = "performance.json"
README_NAME = "README.txt"

#: `colabsd.report.ReportPaths` attribute -> the name it is filed under inside the archive.
REPORT_MEMBERS: dict[str, str] = {"csv": "report.csv", "png": "report.png", "json": "report.json"}

#: What each member is for, in the README and in the manifest, so no one has to guess.
MEMBER_PURPOSE: dict[str, str] = {
    "report.csv": "every tracked metric for every condition, mean +/- sd across the repeated runs",
    "report.png": "the figure: the spread across runs, the one-hot floor, and the unlock count",
    "report.json": "what build_report recorded about the figure it drew",
}


class ExportError(ColabSDError):
    """An export could not be written, or an archive could not be read back."""


# ------------------------------------------------------------------------------------------
# The words. One copy each, shared by the manifest, the README and both notebook cells.
# ------------------------------------------------------------------------------------------


def number(value: Any) -> str:
    """A metric, or the words for one that was never recorded — never a bare `nan`."""
    if value is None:
        return "not recorded"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return "not recorded"
    return "not recorded" if not math.isfinite(as_float) else f"{as_float:.4f}"


def unlock_verdict(unlock_count: int) -> str:
    """What a test number means, given how many times the partition has now been read."""
    if unlock_count <= 0:
        return "The test partition has not been read in this run directory."
    if unlock_count == 1:
        return (
            "Read once, deliberately, after the choices were made. This number means what a held-out number "
            "is supposed to mean."
        )
    return (
        f"This is read number {unlock_count} of the same test partition. Choices made after the first read "
        "were informed by it, so treat this as an optimistic estimate, not a held-out one."
    )


def partition_verdict(partition: str, unlock_count: int) -> str:
    """What the numbers in an archive are, in one or two sentences a biologist can act on."""
    if partition == "test":
        return unlock_verdict(unlock_count)
    if unlock_count <= 0:
        return (
            "These are validation numbers, and the test partition has never been read. That is the right state "
            "to be in while you are still choosing a backbone, a pooling or a set of hyperparameters: nothing "
            "here has spent the held-out set."
        )
    return (
        f"These are validation numbers, but the test partition of this run directory has already been read "
        f"{unlock_count}x. The held-out set has been spent; a later test report from here is an optimistic "
        "estimate rather than a held-out one."
    )


# ------------------------------------------------------------------------------------------
# The facts an archive is built from and read back into.
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceFacts:
    """Everything a reader of the archive needs in order to trust — or distrust — the numbers.

    Four questions have to be answerable from the archive alone: which partition, how often
    the test set was read, which backbone and pooling, and whether the hyperparameters were
    tuned or a placeholder. Every field below exists to answer one of them, or to say who
    wrote the answer.
    """

    partition: str = "validation"
    reported_partitions: tuple[str, ...] = ()
    unlock_count: int = 0
    unlock_source: str = "not recorded"
    metric: str = DEFAULT_METRIC
    headline: dict[str, Any] = field(default_factory=dict)
    one_hot_floor: dict[str, Any] | None = None
    model_name: str = "this model"
    adapter_name: str = ""
    pooling: str = ""
    is_provisional: bool = False
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    best_config_meta: dict[str, Any] = field(default_factory=dict)
    conditions: tuple[str, ...] = ()
    n_runs: int = 0
    n_sequences: int = 0
    split_seeds: tuple[int, ...] = ()
    model_seeds: tuple[int, ...] = ()
    selection_metric: str = ""
    run_created_utc: str = ""
    notes: str | None = None
    created_utc: str = ""
    colabsd_version: str = __version__

    @property
    def hyperparameter_status(self) -> str:
        """`"tuned"` or `"PROVISIONAL"`, in `colabsd.bundle`'s vocabulary."""
        return hyperparameter_status(self.is_provisional)

    @property
    def describes_test(self) -> bool:
        return self.partition == "test"

    @property
    def test_untouched(self) -> bool:
        return self.unlock_count <= 0

    def headline_line(self) -> str:
        """The metric on show, mean +/- sd, over however many runs."""
        mean = number(self.headline.get("mean"))
        sd = number(self.headline.get("sd"))
        runs = int(self.headline.get("n") or self.n_runs or 0)
        return f"{self.metric} {mean} +/- {sd} over {runs} run(s)"

    def floor_line(self) -> str:
        """Where the one-hot floor sits, and how far above it this model is."""
        floor = self.one_hot_floor or {}
        if not floor:
            return (
                f"no one-hot floor for {self.metric} — without it there is nothing here to say whether a "
                "language model was worth it"
            )
        head = f"{floor.get('head', '?')} on {floor.get('partition', self.partition)}: {number(floor.get('mean'))}"
        mean, floor_mean = self.headline.get("mean"), floor.get("mean")
        if mean is None or floor_mean is None or not math.isfinite(float(mean)):
            return head
        margin = float(mean) - float(floor_mean)
        return f"{head} ({number(abs(margin))} {'above' if margin >= 0 else 'BELOW'} it)"

    def describe(self) -> str:
        """One line: what these numbers are, and what produced them."""
        return (
            f"{self.metric} on the {self.partition} partition · {self.model_name} · {self.pooling} · "
            f"{self.hyperparameter_status} hyperparameters · test unlocked {self.unlock_count}x · "
            f"written {self.created_utc or 'unknown'}"
        )


def _attr(obj: Any, *names: str) -> Any:
    """The first of *names* the object carries.

    Duck-typed over the three shapes these exports are handed: a `RunResult` or `BestConfig`
    (attributes), a `run_result.json` read back off disk (a mapping), and a
    `colabsd.ui.core.WizardState`, whose wizard-specific choices live behind `get()` rather
    than on the dataclass.
    """
    if obj is None:
        return None
    getter = getattr(obj, "get", None)
    for name in names:
        found = None
        if callable(getter):
            try:
                found = getter(name)
            except TypeError:  # a `get` that is not a lookup; fall back to the attribute
                getter = None
        if found is None:
            found = getattr(obj, name, None)
        if found is not None:
            return found
    return None


def _ints(values: Any) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in values or ())
    except (TypeError, ValueError):
        return ()


def _strings(values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in values or ())


def _mapping(value: Any) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()} if isinstance(value, Mapping) else {}


def _headline_source(summary: Mapping[str, Any], model_name: str) -> str:
    """Which source in `report.json` is the model, rather than the one-hot floor."""
    sources = _mapping(summary.get("sources"))
    if model_name in sources:
        return model_name
    for name in sources:
        if not name.startswith("one-hot"):
            return name
    return next(iter(sources), model_name)


def performance_facts(
    report_summary: Mapping[str, Any],
    *,
    run_result: Any = None,
    best: Any = None,
    spec: Any = None,
    notes: str | None = None,
    created_utc: str | None = None,
) -> PerformanceFacts:
    """Read the facts out of a `report.json` summary and the run that produced it.

    The partition and the unlock count come from the report rather than from anywhere here:
    `colabsd.report` already decides which partition it is allowed to show and already reads
    the persisted counter, and a second copy of either rule is a second chance to disagree.
    The one addition is a `RunResult` that remembers *more* unlocks than the report found —
    the honest number is the larger one.
    """
    from colabsd.bundle import utc_now

    partition = str(report_summary.get("shown_partition") or "validation")
    counted = int(report_summary.get("unlock_count") or 0)
    remembered = int(_attr(run_result, "unlock_count") or 0)
    model_name = str(_attr(run_result, "model_name", "model") or _headline_source(report_summary, "this model"))
    metric = str(report_summary.get("headline_metric") or DEFAULT_METRIC)
    source = _headline_source(report_summary, model_name)
    headline = _mapping(_mapping(_mapping(report_summary.get("sources")).get(source)).get(partition)).get(metric)
    provisional = _attr(run_result, "is_provisional")
    if provisional is None:
        provisional = _attr(best, "is_provisional")
    return PerformanceFacts(
        partition=partition,
        reported_partitions=_strings(report_summary.get("reported_partitions") or (partition,)),
        unlock_count=max(counted, remembered),
        unlock_source=str(report_summary.get("unlock_source") or "not recorded"),
        metric=metric,
        headline=_mapping(headline),
        one_hot_floor=_mapping(report_summary.get("one_hot_floor")) or None,
        model_name=model_name,
        adapter_name=str(_attr(run_result, "adapter_name") or model_name),
        pooling=str(_attr(run_result, "pooling") or _attr(best, "pooling") or "not recorded"),
        is_provisional=bool(provisional),
        hyperparameters=_mapping(_attr(run_result, "params") or _attr(best, "params")),
        training=_mapping(_attr(run_result, "fixed") or _attr(best, "fixed")),
        best_config_meta=_mapping(_attr(best, "meta")),
        conditions=_strings(_attr(run_result, "condition_columns") or _attr(spec, "condition_columns")),
        n_runs=int(_mapping(report_summary.get("n_runs")).get(source) or _attr(run_result, "n_runs") or 0),
        n_sequences=int(_attr(run_result, "n_sequences") or 0),
        split_seeds=_ints(_attr(run_result, "split_seeds")),
        model_seeds=_ints(_attr(run_result, "model_seeds")),
        selection_metric=str(_attr(run_result, "selection_metric") or ""),
        run_created_utc=str(_attr(run_result, "created_utc") or ""),
        notes=notes,
        created_utc=created_utc or utc_now(),
    )


def _json_safe(value: Any) -> Any:
    """NaN and inf are not JSON; write them as null so `performance.json` parses anywhere."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def manifest_payload(facts: PerformanceFacts, files: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """`performance.json`: the four questions, answered, plus a checksum per member."""
    provenance = provenance_block(
        is_provisional=facts.is_provisional,
        unlock_count=facts.unlock_count,
        notes=facts.notes,
        best_config_meta=facts.best_config_meta,
    )
    provenance["created_utc"] = facts.created_utc
    provenance["colabsd_version"] = facts.colabsd_version
    payload = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "format": ARCHIVE_FORMAT,
        "summary": facts.describe(),
        # 1. Which partition do these numbers describe?
        "partition": facts.partition,
        "reported_partitions": list(facts.reported_partitions),
        "describes_test_partition": facts.describes_test,
        "verdict": partition_verdict(facts.partition, facts.unlock_count),
        # 2. How many times was the test set read?
        "unlock": {
            "count": facts.unlock_count,
            "source": facts.unlock_source,
            "verdict": unlock_verdict(facts.unlock_count),
        },
        "headline": {
            "metric": facts.metric,
            "mean": facts.headline.get("mean"),
            "sd": facts.headline.get("sd"),
            "n_runs": facts.headline.get("n"),
        },
        "one_hot_floor": facts.one_hot_floor,
        # 3. Which backbone and pooling produced them?
        "model": {
            "model_name": facts.model_name,
            "adapter_name": facts.adapter_name,
            "pooling": facts.pooling,
        },
        # 4. Were the hyperparameters tuned, or a placeholder? (`provenance` says the same;
        # both are read off one field, so they cannot drift apart.)
        "hyperparameters": {
            "status": facts.hyperparameter_status,
            "is_provisional": facts.is_provisional,
            "params": facts.hyperparameters,
            "training": facts.training,
        },
        "run": {
            "conditions": list(facts.conditions),
            "n_runs": facts.n_runs,
            "n_sequences": facts.n_sequences,
            "split_seeds": list(facts.split_seeds),
            "model_seeds": list(facts.model_seeds),
            "selection_metric": facts.selection_metric,
            "created_utc": facts.run_created_utc,
        },
        "provenance": provenance,
        "files": {str(name): dict(record) for name, record in files.items()},
    }
    return _json_safe(payload)


def facts_from_manifest(manifest: Mapping[str, Any]) -> PerformanceFacts:
    """The inverse of `manifest_payload`, so an archive read back says what it was written with."""
    unlock = _mapping(manifest.get("unlock"))
    headline = _mapping(manifest.get("headline"))
    model = _mapping(manifest.get("model"))
    hyperparameters = _mapping(manifest.get("hyperparameters"))
    run = _mapping(manifest.get("run"))
    provenance = _mapping(manifest.get("provenance"))
    return PerformanceFacts(
        partition=str(manifest.get("partition") or "validation"),
        reported_partitions=_strings(manifest.get("reported_partitions")),
        unlock_count=int(unlock.get("count") or 0),
        unlock_source=str(unlock.get("source") or "not recorded"),
        metric=str(headline.get("metric") or DEFAULT_METRIC),
        headline={
            key: value
            for key, value in (("mean", headline.get("mean")), ("sd", headline.get("sd")), ("n", headline.get("n_runs")))
            if value is not None
        },
        one_hot_floor=_mapping(manifest.get("one_hot_floor")) or None,
        model_name=str(model.get("model_name") or "this model"),
        adapter_name=str(model.get("adapter_name") or ""),
        pooling=str(model.get("pooling") or ""),
        is_provisional=bool(hyperparameters.get("is_provisional", provenance.get("is_provisional", False))),
        hyperparameters=_mapping(hyperparameters.get("params")),
        training=_mapping(hyperparameters.get("training")),
        best_config_meta=_mapping(provenance.get("best_config_meta")),
        conditions=_strings(run.get("conditions")),
        n_runs=int(run.get("n_runs") or 0),
        n_sequences=int(run.get("n_sequences") or 0),
        split_seeds=_ints(run.get("split_seeds")),
        model_seeds=_ints(run.get("model_seeds")),
        selection_metric=str(run.get("selection_metric") or ""),
        run_created_utc=str(run.get("created_utc") or ""),
        notes=provenance.get("notes"),
        created_utc=str(provenance.get("created_utc") or ""),
        colabsd_version=str(provenance.get("colabsd_version") or __version__),
    )


def readme_text(facts: PerformanceFacts, files: Mapping[str, Mapping[str, Any]]) -> str:
    """The same four answers as `performance.json`, for a reader who will not open JSON."""
    seeds = f"split seeds {list(facts.split_seeds) or 'not recorded'}, model seeds {list(facts.model_seeds) or '?'}"
    provisional = (
        "\nThese hyperparameters are a PROVISIONAL placeholder, not a tuned entry: the numbers below are a\n"
        "lower bound on what this backbone and pooling can do, not a benchmark result.\n"
        if facts.is_provisional
        else ""
    )
    lines = [
        "ColabSeqDisplay — performance report",
        "====================================",
        "",
        facts.describe(),
        "",
        "WHAT THESE NUMBERS DESCRIBE",
        f"  partition          {facts.partition}",
        f"  partitions in CSV  {', '.join(facts.reported_partitions) or facts.partition}",
        f"  test set read      {facts.unlock_count}x   (counter read from: {facts.unlock_source})",
        f"  headline           {facts.headline_line()}",
        f"  one-hot floor      {facts.floor_line()}",
        f"  conditions         {', '.join(facts.conditions) or 'not recorded'}",
        f"  repeated runs      {facts.n_runs}  ({seeds})",
        f"  library size       {facts.n_sequences or 'not recorded'} sequences",
        "",
        _wrap(partition_verdict(facts.partition, facts.unlock_count)),
        "",
        "WHAT PRODUCED THEM",
        f"  backbone           {facts.model_name}  (adapter {facts.adapter_name or 'not recorded'})",
        f"  pooling            {facts.pooling or 'not recorded'}",
        f"  hyperparameters    {facts.hyperparameter_status}",
        f"  registry entry     {facts.best_config_meta.get('status', 'not recorded')}",
        f"  selected on        {facts.selection_metric or 'not recorded'} (validation)",
        f"  trained            {facts.run_created_utc or 'not recorded'}",
        provisional,
        "FILES IN THIS ARCHIVE",
    ]
    for name, record in files.items():
        lines.append(f"  {name:<18} {MEMBER_PURPOSE.get(name, 'part of this report')}")
        lines.append(f"  {'':<18} sha256 {record.get('sha256', '?')}")
    lines += [
        f"  {MANIFEST_NAME:<18} everything above, as JSON",
        f"  {README_NAME:<18} this file",
        "",
        f"Written by ColabSeqDisplay {facts.colabsd_version} at {facts.created_utc}.",
    ]
    if facts.notes:
        lines += ["", "NOTES", _wrap(str(facts.notes))]
    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 96) -> str:
    """Fold one sentence-ish string to *width*, so the README reads in a terminal."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)) or text


# ------------------------------------------------------------------------------------------
# Notices, for whichever panel is showing the export.
# ------------------------------------------------------------------------------------------


def notices(facts: PerformanceFacts) -> list[Message]:
    """What to say beside an archive that has just been written."""
    found: list[Message] = []
    if facts.describes_test:
        if facts.unlock_count > 1:
            found.append(
                Message(
                    "archive_test_read_repeatedly",
                    "warning",
                    f"This archive carries **test** numbers from read number {facts.unlock_count} of the same "
                    "partition. The count travels in `performance.json`, and anyone you send it to will see it.",
                )
            )
        else:
            found.append(
                Message(
                    "archive_test_read_once",
                    "info",
                    "This archive carries the **test** numbers, read once. `performance.json` records the "
                    "partition and the count, so the file says which it is without you having to remember.",
                )
            )
    elif facts.test_untouched:
        found.append(
            Message(
                "archive_validation_only",
                "info",
                "This archive carries **validation** numbers and the test partition has never been read — the "
                "state to be in while you are still deciding anything. Nothing here spent the held-out set.",
            )
        )
    else:
        found.append(
            Message(
                "archive_validation_after_unlock",
                "warning",
                f"This archive carries **validation** numbers, but the test partition of this run directory has "
                f"already been read {facts.unlock_count}x. The archive says so.",
            )
        )
    if facts.is_provisional:
        found.append(
            Message(
                "archive_provisional",
                "warning",
                "The hyperparameters behind these numbers are a **PROVISIONAL** placeholder, so this is a lower "
                "bound on what this backbone and pooling can do. The archive is stamped `PROVISIONAL`.",
            )
        )
    if not facts.one_hot_floor:
        found.append(
            Message(
                "archive_no_floor",
                "warning",
                f"No one-hot floor was recorded for {facts.metric}, so nothing in this archive says whether a "
                "language model beat 20·k one-hot features. Run the one-hot baseline and export again.",
            )
        )
    return found


def summary_html(export: PerformanceExport) -> str:
    """The block a notebook shows after writing an archive."""
    files = ", ".join(f"`{name}`" for name in sorted(export.report_paths))
    return theme.note_html(
        f"Written **{export.path.name}** — {export.facts.describe()}\n\n"
        f"{partition_verdict(export.facts.partition, export.facts.unlock_count)}\n\n"
        f"Inside it, beside `{MANIFEST_NAME}` and `{README_NAME}`: {files}."
    ) + render_messages(notices(export.facts))


# ------------------------------------------------------------------------------------------
# Writing and reading the archive.
# ------------------------------------------------------------------------------------------


def report_members(report_paths: Any) -> dict[str, Path]:
    """Normalize a `colabsd.report.ReportPaths` — or a mapping like one — to archive members."""
    found: dict[str, Path] = {}
    for kind, member in REPORT_MEMBERS.items():
        if isinstance(report_paths, Mapping):
            value = report_paths.get(kind, report_paths.get(member))
        else:
            value = getattr(report_paths, kind, None)
        if value is not None:
            found[member] = Path(value)
    if not found:
        raise ExportError(
            "No report files to archive: build_report() returned nothing with a .csv, .png or .json. "
            "Call colabsd.report.build_report(run_result, baseline, spec, out_dir=...) first."
        )
    return found


def write_performance_archive(
    path: str | Path,
    *,
    facts: PerformanceFacts,
    report_paths: Any,
    extra_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write `performance_report.zip` and return its path."""
    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    members = report_members(report_paths)
    for name, source in (extra_files or {}).items():
        members[str(name)] = Path(source)
    missing = [str(source) for source in members.values() if not source.is_file()]
    if missing:
        raise ExportError(
            f"These report files were not written, so the archive would be incomplete: {missing}. "
            "Re-run the report — colabsd.report.build_report() writes all three together."
        )

    files = {
        name: {
            "sha256": file_sha256(source),
            "bytes": int(source.stat().st_size),
            "holds": MEMBER_PURPOSE.get(name, "part of this report"),
        }
        for name, source in members.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        # The README first: whatever opens this, a human reads that file.
        archive.writestr(README_NAME, readme_text(facts, files))
        archive.writestr(MANIFEST_NAME, json.dumps(manifest_payload(facts, files), indent=2))
        for name, source in members.items():
            archive.write(source, name)
    return path


@dataclass(frozen=True)
class PerformanceArchive:
    """A performance archive read back off disk, checksums verified."""

    path: Path
    manifest: dict[str, Any]
    facts: PerformanceFacts
    readme: str
    report: dict[str, Any]
    csv_text: str
    figure: bytes
    members: tuple[str, ...]

    def describe(self) -> str:
        return self.facts.describe()


def read_performance_archive(path: str | Path) -> PerformanceArchive:
    """Read an archive back, verifying the schema and every member's checksum."""
    path = Path(path)
    if not path.is_file():
        raise ExportError(f"No performance archive at {path}. Export one from the main notebook.")
    try:
        handle = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ExportError(f"{path} is not a readable .zip archive ({exc}). Re-download or re-export it.") from exc

    with handle as archive:
        names = set(archive.namelist())
        if MANIFEST_NAME not in names:
            raise ExportError(
                f"{path} has no {MANIFEST_NAME}, so nothing in it says which partition its numbers describe. "
                "It was not written by colabsd.ui.exports; re-export it from the notebook that trained the model."
            )
        manifest = json.loads(archive.read(MANIFEST_NAME))
        version = int(manifest.get("schema_version", 0))
        if version > ARCHIVE_SCHEMA_VERSION:
            raise ExportError(
                f"{path} uses performance-archive schema {version} but this colabsd understands up to "
                f"{ARCHIVE_SCHEMA_VERSION}. Upgrade colabsd (pip install -U colabseqdisplay)."
            )
        if str(manifest.get("format")) != ARCHIVE_FORMAT:
            raise ExportError(
                f"{path} declares itself {manifest.get('format')!r} rather than {ARCHIVE_FORMAT!r}. "
                "It is some other zip; export the performance archive from the main notebook."
            )
        blobs = {}
        for name, record in _mapping(manifest.get("files")).items():
            if name not in names:
                raise ExportError(
                    f"{path} lists {name} in its manifest but does not contain it. The archive is incomplete; "
                    "re-export it."
                )
            blob = archive.read(name)
            expected = str(_mapping(record).get("sha256") or "")
            if expected and sha256_hex(blob) != expected:
                raise ExportError(
                    f"{name} in {path} does not match its manifest checksum, so the numbers in it are not the "
                    "ones that were exported. The archive is corrupt; re-download or re-export it."
                )
            blobs[name] = blob
        readme = archive.read(README_NAME).decode("utf-8") if README_NAME in names else ""

    report = json.loads(blobs["report.json"]) if "report.json" in blobs else {}
    return PerformanceArchive(
        path=path,
        manifest=manifest,
        facts=facts_from_manifest(manifest),
        readme=readme,
        report=report,
        csv_text=blobs.get("report.csv", b"").decode("utf-8"),
        figure=blobs.get("report.png", b""),
        members=tuple(sorted(names)),
    )


# ------------------------------------------------------------------------------------------
# The side effects, injected so both notebook cells can be driven headlessly in tests.
# ------------------------------------------------------------------------------------------


def _build_report(*args: Any, **kwargs: Any) -> Any:
    from colabsd.report import build_report

    return build_report(*args, **kwargs)


def _save_bundle_from_run(path: Path, **kwargs: Any) -> Path:
    from colabsd.bundle import save_bundle_from_run

    return save_bundle_from_run(path, **kwargs)


def _load_bundle(path: Path) -> Any:
    from colabsd.bundle import load_bundle

    return load_bundle(path)


def _read_unlock_count(output_dir: Any) -> int:
    from colabsd.train import read_unlock_count

    return read_unlock_count(output_dir)


def _offer_download(path: Path) -> None:
    """Hand the file to the browser in Colab; do nothing anywhere else."""
    try:
        from google.colab import files
    except ImportError:
        return
    files.download(str(path))


@dataclass(frozen=True)
class ExportRunners:
    """Everything here that touches a disk, a browser or the network of imports.

    The defaults are the real implementations, each importing lazily so that importing this
    module stays free; a test — or a notebook cell that already has its own backend — passes
    its own. Dataclass fields are set on the instance, so these stay plain functions rather
    than becoming bound methods.
    """

    build_report: Callable[..., Any] = _build_report
    save_bundle_from_run: Callable[..., Path] = _save_bundle_from_run
    load_bundle: Callable[[Path], Any] = _load_bundle
    read_unlock_count: Callable[[Any], int] = _read_unlock_count
    download: Callable[[Path], None] = _offer_download


def default_runners() -> ExportRunners:
    """The real implementations."""
    return ExportRunners()


def honest_unlock_count(run_result: Any, *, output_dir: Any = None, runners: ExportRunners | None = None) -> int:
    """The larger of what the run remembers and what `unlock.json` records.

    A `RunResult` sitting in a notebook variable can only ever undercount: it predates any
    unlock taken after it was built. An unreadable counter raises rather than reading as
    zero — claiming "never unlocked" over a partition that has in fact been read is the one
    lie these exports exist to prevent.
    """
    runners = runners or default_runners()
    counts = [int(_attr(run_result, "unlock_count") or 0)]
    directory = output_dir if output_dir is not None else _attr(run_result, "output_dir", "out_dir", "run_dir")
    if directory is not None:
        counts.append(int(runners.read_unlock_count(Path(directory))))
    return max(counts)


# ------------------------------------------------------------------------------------------
# The two exports.
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceExport:
    """A written performance archive: the zip, the facts in it, and the files it gathered."""

    path: Path
    facts: PerformanceFacts
    report_paths: dict[str, Path]
    downloaded: bool = False

    def written(self) -> dict[str, Path]:
        """Everything this export put on disk, archive last, for a "written:" line."""
        return {**self.report_paths, "archive": self.path}

    def describe(self) -> str:
        return self.facts.describe()

    def notices(self) -> list[Message]:
        return notices(self.facts)


def export_performance(
    *,
    work_dir: str | Path,
    run_result: Any = None,
    baseline: Mapping[str, Any] | None = None,
    spec: Any = None,
    best: Any = None,
    report_dir: str | Path | None = None,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    metric: str = DEFAULT_METRIC,
    partition: str | None = None,
    notes: str | None = None,
    runners: ExportRunners | None = None,
    download: bool = True,
) -> PerformanceExport:
    """Build the report and pack it into a self-describing archive. Reads no locked data.

    This is the ordinary first export, and it works with the test partition untouched:
    `colabsd.report.build_report` publishes validation numbers while the test set is locked
    and says so in `report.json`, and that is what the archive then declares. Called again
    after `colabsd.train.unlock_test`, the same file name is rewritten with the test numbers
    and the count — `partition` is left to the report, which knows what it is allowed to show.

    `run_result` may be None when there is nothing but a one-hot baseline to report; what the
    report can be built from is `colabsd.report.build_report`'s rule, not a second one here.
    """
    runners = runners or default_runners()
    work = Path(work_dir)
    out_dir = Path(report_dir) if report_dir is not None else work / DEFAULT_REPORT_DIRNAME
    options: dict[str, Any] = {"out_dir": out_dir, "metric": metric}
    if partition is not None:
        options["partition"] = partition
    written = runners.build_report(run_result, baseline, spec, **options)

    paths = report_members(written)
    summary_path = paths.get("report.json")
    try:
        summary = json.loads(summary_path.read_text()) if summary_path and summary_path.is_file() else {}
    except ValueError as exc:
        raise ExportError(
            f"{summary_path} is not readable JSON ({exc}), so the archive could not say which partition its "
            "numbers describe. Re-run the report."
        ) from exc

    facts = performance_facts(summary, run_result=run_result, best=best, spec=spec, notes=notes)
    archive = write_performance_archive(work / archive_name, facts=facts, report_paths=paths)
    if download:
        runners.download(archive)
    return PerformanceExport(path=archive, facts=facts, report_paths=paths, downloaded=download)


@dataclass(frozen=True)
class BundleExport:
    """A written model bundle, loaded back so the notebook shows what is actually in it."""

    path: Path
    bundle: Any
    unlock_count: int
    downloaded: bool = False

    def describe(self) -> str:
        describe = getattr(self.bundle, "describe", None)
        return describe() if callable(describe) else str(self.bundle)


def export_bundle(
    *,
    run_result: Any,
    spec: Any,
    best: Any,
    path: str | Path,
    region: Mapping[str, Any] | None = None,
    notes: str | None = None,
    runners: ExportRunners | None = None,
    download: bool = True,
) -> BundleExport:
    """Write `model_bundle.zip`, load it back, and hand over both.

    The unlock count written into the manifest is `honest_unlock_count`, not whatever the
    `RunResult` happens to remember, so a bundle exported after the unlock cell ran carries
    the same count as the performance archive beside it.
    """
    runners = runners or default_runners()
    unlocks = honest_unlock_count(run_result, runners=runners)
    written = Path(
        runners.save_bundle_from_run(
            Path(path),
            run_result=run_result,
            spec=spec,
            best=best,
            region=dict(region) if region else None,
            notes=notes,
            unlock_count=unlocks,
        )
    )
    bundle = runners.load_bundle(written)
    if download:
        runners.download(written)
    return BundleExport(path=written, bundle=bundle, unlock_count=unlocks, downloaded=download)


def export_names(state: Any = None) -> tuple[str, str]:
    """The two file names the notebook offers, so both cells spell them the same way."""
    bundle_name = str(_attr(state, "bundle_name") or DEFAULT_BUNDLE_NAME).strip() or DEFAULT_BUNDLE_NAME
    archive_name = str(_attr(state, "archive_name") or DEFAULT_ARCHIVE_NAME).strip() or DEFAULT_ARCHIVE_NAME
    return bundle_name, archive_name
