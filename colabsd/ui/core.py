"""The shared machinery every ColabSeqDisplay wizard is made of.

The ColabPLM notebooks put one 236 KB program in one cell and toggle
`layout.display` 126 times inside it. The interaction is right; the packaging is
not testable. So the *decisions* live here as pure functions over a
`WizardState` — which sections are visible, which fields inside them, which
contextual messages fire — and the widget layer below them does nothing but call
those functions and assign `layout.display`.

The rule is: if you cannot write ``assert plan(state).message_keys == (...)`` for
it, it does not belong in an observer.

Facts come from the repository, never from a number typed here:
`colabsd.backbones.registry` says how long a backbone takes on a T4, whether it
fits one at all, and whether it needs a 3Di string; `colabsd.bestconfig` says
whether a (backbone, pooling) pair has tuned hyperparameters or a placeholder;
`colabsd.structure` says how long a wild type ESMFold will fold on a free T4.

Plain `observe()` / `on_click()`, not `jupyter_ui_poll`. The ColabPLM cell needs
`ui_events` because the whole program lives inside one still-running cell that
drives training on a background thread and uses the cell's own run-button as its
stop button. Our notebook cell installs, imports and launches, then ends; Colab
keeps the widget comm alive afterwards, so ordinary observers fire, exceptions
surface in an `Output` instead of being swallowed by the poll loop, and every
handler is callable directly from a test.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from colabsd.backbones.registry import BACKBONES, BackboneEntry
from colabsd.ui import theme
from colabsd.ui.theme import Severity

DEFAULT_WORK_DIR = "colabsd_work"
"""Where a wizard writes when the notebook cell does not say. Relative on purpose: an
absolute fallback would silently point outside whatever Drive folder the user mounted."""

#: The three wizards. `train` is the main notebook, `prepare` builds a 3Di string
#: and a pooling region for a new protein, `predict` scores variants from a bundle.
MODES: tuple[str, ...] = ("train", "prepare", "predict")

#: Where a wild-type 3Di string can come from, matching `colabsd.structure`.
THREE_DI_SOURCES: tuple[str, ...] = (
    "none",
    "bundled_example",
    "paste",
    "upload_3di",
    "upload_structure",
    "esmfold",
)

#: A Colab session drops well before this; SaprotHub's own notebooks warn above
#: two hours of training. Not a repository fact — a fact about Colab.
COLAB_SESSION_MINUTES: float = 120.0

#: Past this much work, losing the machine hurts enough to be worth a Drive mount.
PERSISTENCE_MINUTES: float = 30.0

#: A free-tier T4 reports ~15 GB; anything at or under this is T4-class.
FREE_T4_MEMORY_GB: float = 16.0

#: An L4 reports ~22.5 GB and a V100 16 GB; past this a card is A100-class.
L4_MEMORY_GB: float = 26.0

#: Fallback for `reference_library_variants()` when the bundled example is not
#: installed. When it is installed, the count is read from `library.csv` itself.
REFERENCE_LIBRARY_VARIANTS: int = 16424

#: Fallback for `esmfold_safe_length()` when `colabsd.structure` cannot be imported;
#: it is that module that owns the real limit.
ESMFOLD_FALLBACK_LENGTH: int = 700

GPU_TIERS: tuple[str, ...] = ("none", "t4", "l4", "a100", "other")

SECTION_ORDER: tuple[str, ...] = (
    "data",
    "structure",
    "model",
    "hyperparameters",
    "training",
    "region",
    "bundle",
    "variants",
    "storage",
    "run",
)

SECTION_TITLES: dict[str, str] = {
    "data": "Your variant library",
    "structure": "Wild-type structure (3Di)",
    "model": "Backbone and pooling",
    "hyperparameters": "Hyperparameters for this pair",
    "training": "How many runs",
    "region": "Pooling region",
    "bundle": "The trained model",
    "variants": "Variants to score",
    "storage": "Where the results are kept",
    "run": "Run it",
}


# --------------------------------------------------------------------------- state


@dataclass
class WizardState:
    """Every choice a user has made, in one object the pure functions read.

    The typed fields are the ones the visibility and warning rules depend on;
    anything a single wizard needs and the rules do not goes in `extra`, so
    `set()`/`get()` work uniformly for an observer that does not know which is which.
    """

    mode: str = "train"

    data_source: str = "bundled_example"
    n_variants: int = 0
    wt_length: int = 0

    backbone: str = "ESM2-35M"
    pooling: str = "cosine_p90_mean"
    dtype: str = "float32"

    three_di_source: str = "none"
    wt_3di_length: int = 0

    n_split_seeds: int = 1
    n_model_seeds: int = 1

    show_advanced: bool = False

    use_drive: bool = False
    drive_folder: str = "ColabSeqDisplay"
    output_dir: str = "colabsd_work"

    bundle_path: str = ""
    variant_source: str = "library_head"

    unlock_count: int = 0
    unlock_requested: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    def set(self, name: str, value: Any) -> None:
        """Write a choice, whether it is a declared field or a wizard-specific one."""
        if name in _STATE_FIELDS:
            setattr(self, name, value)
        else:
            self.extra[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Read a choice from the declared fields or from `extra`."""
        if name in _STATE_FIELDS:
            return getattr(self, name)
        return self.extra.get(name, default)

    def with_changes(self, **changes: Any) -> WizardState:
        """A copy with `changes` applied — for table-driven tests and for undo."""
        return replace(self, **changes)


_STATE_FIELDS: frozenset[str] = frozenset(item.name for item in fields(WizardState))


# ------------------------------------------------------------------- repository facts


def backbone_entry(name: str) -> BackboneEntry | None:
    """Registry metadata for `name`, or None when the name is not registered."""
    return BACKBONES.get(name)


def colab_backbones() -> list[str]:
    """Registered backbones that have a Colab adapter, cheapest T4 run first."""
    names = [name for name, entry in BACKBONES.items() if entry.tier != "local_only"]
    return sorted(names, key=_backbone_sort_key)


def _backbone_sort_key(name: str) -> tuple[int, int, str]:
    entry = BACKBONES[name]
    minutes = entry.approx_lora_minutes_t4
    return (1, 0, name) if minutes is None else (0, minutes, name)


def sequence_only_backbones() -> list[str]:
    """Colab backbones that need no 3Di string — the answer to 'what else can I pick?'."""
    return [name for name in colab_backbones() if not BACKBONES[name].needs_structure]


def structure_backbones() -> list[str]:
    """Colab backbones that do need a 3Di string."""
    return [name for name in colab_backbones() if BACKBONES[name].needs_structure]


def backbone_label(name: str) -> str:
    """One dropdown line: the name plus the two things that decide whether to pick it."""
    entry = backbone_entry(name)
    if entry is None:
        return f"{name} — not registered"
    if entry.tier == "local_only":
        return f"{name} — not available in Colab"
    parts = ["needs a 3Di string" if entry.needs_structure else "sequence only"]
    minutes = entry.approx_lora_minutes_t4
    parts.append("needs an L4 or A100" if minutes is None else f"~{minutes} min per run on a T4")
    if entry.tier == "extra":
        parts.append("one extra install")
    return f"{name} — {', '.join(parts)}"


def backbone_choices(*, include_unavailable: bool = False) -> list[tuple[str, str]]:
    """`(label, value)` pairs for a backbone dropdown, cheapest first."""
    names = colab_backbones()
    if include_unavailable:
        names = names + sorted(name for name, entry in BACKBONES.items() if entry.tier == "local_only")
    return [(backbone_label(name), name) for name in names]


def backbone_summary(name: str) -> str:
    """A short markdown paragraph about one backbone, for the biologist reading the form."""
    entry = backbone_entry(name)
    if entry is None:
        return f"**{name}** is not in the backbone registry. Pick one of: {', '.join(colab_backbones())}."
    structure = "needs a wild-type 3Di string" if entry.needs_structure else "sequence only, no structure needed"
    lines = [
        f"**{name}** — {entry.family} family, {entry.embed_dim}-d pooled feature"
        + (f", `{entry.hf_id}`" if entry.hf_id else "")
        + ".",
        f"Structure: {structure}.",
    ]
    minutes = entry.approx_lora_minutes_t4
    if minutes is None:
        lines.append("Runtime: does **not** fit a free T4 — this one needs an L4 or an A100.")
    else:
        lines.append(
            f"Runtime: the registry estimates **{minutes} min per run** on a T4 for a library the size of the "
            f"bundled example ({reference_library_variants():,d} variants)."
        )
    lines.append(entry.notes)
    return "\n\n".join(lines)


def pooling_choices(backbone: str | None = None, *, root: str | Path | None = None) -> list[str]:
    """Poolings the best-config registry has an entry for, optionally for one backbone."""
    from colabsd import bestconfig

    return bestconfig.available_poolings(backbone, root)


def reference_library_variants() -> int:
    """Rows in the bundled example library — the size the registry's minute estimates assume."""
    global _REFERENCE_VARIANTS
    if _REFERENCE_VARIANTS is None:
        _REFERENCE_VARIANTS = _count_example_rows()
    return _REFERENCE_VARIANTS


_REFERENCE_VARIANTS: int | None = None


def _count_example_rows() -> int:
    from colabsd import EXAMPLES_ROOT

    path = EXAMPLES_ROOT / "slugcas9_5nnk" / "library.csv"
    try:
        with path.open("rb") as handle:
            rows = sum(1 for line in handle if line.strip())
    except OSError:
        return REFERENCE_LIBRARY_VARIANTS
    return max(rows - 1, 1)


def esmfold_safe_length() -> int:
    """Longest wild type ESMFold folds on a free T4, as `colabsd.structure` has it."""
    try:
        from colabsd import structure
    except ImportError:
        return ESMFOLD_FALLBACK_LENGTH
    for name in ("ESMFOLD_SAFE_LENGTH_T4", "_ESMFOLD_SAFE_LENGTH_T4"):
        value = getattr(structure, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return ESMFOLD_FALLBACK_LENGTH


@dataclass(frozen=True)
class ConfigStatus:
    """What the best-config registry has to say about one (backbone, pooling) pair."""

    model: str
    pooling: str
    found: bool
    provisional: bool
    description: str
    poolings: tuple[str, ...] = ()
    error: str | None = None


def config_status(model: str, pooling: str, *, root: str | Path | None = None) -> ConfigStatus:
    """Look the pair up, turning every failure into data rather than an exception."""
    from colabsd import bestconfig

    try:
        best = bestconfig.load_best_config(model, pooling, root)
    except bestconfig.BestConfigNotFound as exc:
        return ConfigStatus(
            model=model,
            pooling=pooling,
            found=False,
            provisional=True,
            description=str(exc),
            poolings=tuple(pooling_choices(model, root=root)),
        )
    except Exception as exc:  # unreadable YAML, or upstream not importable
        return ConfigStatus(
            model=model,
            pooling=pooling,
            found=True,
            provisional=True,
            description=str(exc),
            poolings=tuple(pooling_choices(model, root=root)),
            error=f"{type(exc).__name__}: {exc}",
        )
    return ConfigStatus(
        model=model,
        pooling=pooling,
        found=True,
        provisional=best.is_provisional,
        description=best.describe(),
        poolings=tuple(pooling_choices(model, root=root)),
    )


# ------------------------------------------------------------------------- estimates


@dataclass(frozen=True)
class RuntimeEstimate:
    """How long the training in `state` should take, and how sure that is."""

    n_runs: int
    per_run_minutes: int | None
    minutes: float | None
    reference_variants: int
    n_variants: int

    @property
    def known(self) -> bool:
        return self.minutes is not None

    @property
    def scaled(self) -> bool:
        """True when the estimate was scaled to a library whose size we actually know."""
        return self.minutes is not None and self.n_variants > 0

    @property
    def fits_session(self) -> bool:
        """True when the estimate is unknown or comfortably inside a Colab session."""
        return self.minutes is None or self.minutes <= COLAB_SESSION_MINUTES

    @property
    def runs_phrase(self) -> str:
        return "1 run" if self.n_runs == 1 else f"{self.n_runs} runs"

    def describe(self) -> str:
        if self.per_run_minutes is None:
            return f"{self.runs_phrase}; the registry has no T4 estimate for this backbone."
        scaled = "" if self.n_variants <= 0 else f", scaled to your {self.n_variants:,d} variants"
        return (
            f"about {format_minutes(self.minutes or 0.0)} for {self.runs_phrase} — the registry's "
            f"{self.per_run_minutes} min/run on a T4 for a {self.reference_variants:,d}-variant library{scaled}. "
            "It is an estimate, not a measurement."
        )


def n_runs(state: WizardState) -> int:
    """Runs a fine-tune will do: split seeds x model seeds."""
    return max(1, int(state.n_split_seeds)) * max(1, int(state.n_model_seeds))


def estimate_runtime(state: WizardState) -> RuntimeEstimate:
    """Scale the registry's per-run T4 estimate by the run count and the library size."""
    entry = backbone_entry(state.backbone)
    per_run = entry.approx_lora_minutes_t4 if entry else None
    runs = n_runs(state)
    reference = reference_library_variants()
    minutes: float | None = None
    if per_run is not None:
        minutes = float(per_run) * runs
        if state.n_variants > 0:
            minutes = max(minutes * state.n_variants / float(reference), 1.0)
    return RuntimeEstimate(
        n_runs=runs,
        per_run_minutes=per_run,
        minutes=minutes,
        reference_variants=reference,
        n_variants=int(state.n_variants),
    )


def format_minutes(minutes: float) -> str:
    """`"18 min"`, `"2 h 05 min"` — never a bare float in front of a user."""
    total = int(round(max(minutes, 0.0)))
    if total < 60:
        return f"{total} min"
    return f"{total // 60} h {total % 60:02d} min"


# ---------------------------------------------------------------- visibility engine


Predicate = Callable[["WizardState"], bool]


@dataclass(frozen=True)
class FieldRule:
    """One form field, the section it lives in, and when it is visible."""

    key: str
    section: str
    when: Predicate

    def visible(self, state: WizardState) -> bool:
        return bool(self.when(state))


def _all(*predicates: Predicate) -> Predicate:
    return lambda state: all(predicate(state) for predicate in predicates)


def _mode(*modes: str) -> Predicate:
    return lambda state: state.mode in modes


def _always(_state: WizardState) -> bool:
    return True


def needs_structure(state: WizardState) -> bool:
    """True when the chosen backbone reads structure as well as sequence."""
    if state.mode == "predict":
        return False
    entry = backbone_entry(state.backbone)
    return bool(entry and entry.needs_structure)


def has_three_di(state: WizardState) -> bool:
    """True when a 3Di string is attached, or a source for one has been chosen."""
    return state.wt_3di_length > 0 or state.three_di_source not in ("", "none")


def _data_section(state: WizardState) -> bool:
    if state.mode in ("train", "prepare"):
        return True
    return state.mode == "predict" and state.variant_source == "library_head"


def _structure_section(state: WizardState) -> bool:
    return state.mode == "prepare" or needs_structure(state)


def _three_di_is(source: str) -> Predicate:
    return lambda state: _structure_section(state) and state.three_di_source == source


def _upload_library(state: WizardState) -> bool:
    return state.data_source != "bundled_example"


def _cosine_pooling(state: WizardState) -> bool:
    return state.pooling.startswith("cosine_")


def _advanced(state: WizardState) -> bool:
    return bool(state.show_advanced)


def _drive(state: WizardState) -> bool:
    return bool(state.use_drive)


def _upload_variants(state: WizardState) -> bool:
    return state.variant_source not in ("library_head", "")


#: The whole form, declared once. Adding a field means adding a line here and a
#: widget with the same key; nothing else knows about visibility.
FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule("data_source", "data", _data_section),
    FieldRule("library_csv", "data", _all(_data_section, _upload_library)),
    FieldRule("wt_sequence", "data", _all(_data_section, _upload_library)),
    FieldRule("positions_1based", "data", _data_section),
    FieldRule("mutation_columns", "data", _data_section),
    FieldRule("condition_columns", "data", _data_section),
    FieldRule("three_letter_residues", "data", _all(_data_section, _advanced)),
    FieldRule("count_column", "data", _all(_data_section, _advanced)),
    FieldRule("min_count", "data", _all(_data_section, _advanced)),
    FieldRule("three_di_source", "structure", _structure_section),
    FieldRule("three_di_text", "structure", _three_di_is("paste")),
    FieldRule("three_di_file", "structure", _three_di_is("upload_3di")),
    FieldRule("structure_file", "structure", _three_di_is("upload_structure")),
    FieldRule("chain", "structure", _three_di_is("upload_structure")),
    FieldRule("esmfold_note", "structure", _three_di_is("esmfold")),
    FieldRule("backbone", "model", _mode("train", "prepare")),
    FieldRule("pooling", "model", _mode("train")),
    FieldRule("region_source", "model", _all(_mode("train"), _cosine_pooling)),
    FieldRule("region_filename", "model", _all(_mode("train"), _cosine_pooling)),
    FieldRule("dtype", "model", _all(_mode("train", "prepare"), _advanced)),
    FieldRule("hyperparameters", "hyperparameters", _mode("train")),
    FieldRule("n_split_seeds", "training", _mode("train")),
    FieldRule("n_model_seeds", "training", _mode("train")),
    FieldRule("run_name", "training", _all(_mode("train"), _advanced)),
    FieldRule("resume_finished_runs", "training", _all(_mode("train"), _advanced)),
    FieldRule("region_percentile", "region", _mode("prepare")),
    FieldRule("region_n_sample", "region", _mode("prepare")),
    FieldRule("bundle_path", "bundle", _mode("predict")),
    FieldRule("variant_source", "variants", _mode("predict")),
    FieldRule("variants_csv", "variants", _all(_mode("predict"), _upload_variants)),
    FieldRule("top_n", "variants", _mode("predict")),
    FieldRule("use_drive", "storage", _always),
    FieldRule("drive_folder", "storage", _drive),
    FieldRule("output_dir", "storage", _advanced),
    FieldRule("show_advanced", "run", _always),
    FieldRule("run_button", "run", _always),
)

FIELD_KEYS: tuple[str, ...] = tuple(rule.key for rule in FIELD_RULES)
FIELD_SECTIONS: dict[str, str] = {rule.key: rule.section for rule in FIELD_RULES}


def field_visibility(state: WizardState) -> dict[str, bool]:
    """Every declared field mapped to whether it should be on screen."""
    return {rule.key: rule.visible(state) for rule in FIELD_RULES}


def visible_fields(state: WizardState) -> tuple[str, ...]:
    """The visible field keys, in declaration order."""
    return tuple(rule.key for rule in FIELD_RULES if rule.visible(state))


def visible_sections(state: WizardState) -> tuple[str, ...]:
    """Sections holding at least one visible field, in `SECTION_ORDER`."""
    live = {FIELD_SECTIONS[key] for key in visible_fields(state)}
    return tuple(section for section in SECTION_ORDER if section in live)


# ------------------------------------------------------------------- warning engine


@dataclass(frozen=True)
class Message:
    """One contextual message: a stable key, a severity, and what to tell the user."""

    key: str
    severity: Severity
    text: str

    def html(self) -> str:
        return theme.message_html(self.text, self.severity)


def _ordered(messages: Iterable[Message]) -> list[Message]:
    seen: set[str] = set()
    unique: list[Message] = []
    for item in messages:
        if item.key in seen:
            continue
        seen.add(item.key)
        unique.append(item)
    unique.sort(key=lambda item: theme.SEVERITY_RANK[item.severity])
    return unique


def _backbone_is_usable(state: WizardState) -> bool:
    """True when the chosen backbone is registered and can run in Colab at all."""
    entry = backbone_entry(state.backbone)
    return entry is not None and entry.tier != "local_only"


def backbone_messages(state: WizardState, runtime: Runtime | None = None) -> list[Message]:
    """What the backbone registry has to say about the chosen backbone.

    `runtime` only ever silences a message: a backbone the registry marks as too
    big for a T4 is fine once an L4 or an A100 has actually been detected. Whether
    the backbone fits the card is this function's business alone, so that fact has
    exactly one message key.
    """
    name = state.backbone
    if state.mode == "predict" or not name:
        return []
    entry = backbone_entry(name)
    if entry is None:
        return [
            Message(
                "unknown_backbone",
                "stop",
                f"`{name}` is not a registered backbone. Pick one of: {', '.join(colab_backbones())}.",
            )
        ]
    if entry.tier == "local_only":
        return [Message("backbone_not_in_colab", "stop", f"**{name}** cannot run in Colab. {entry.notes}")]
    out: list[Message] = []
    if entry.approx_lora_minutes_t4 is None and t4_class(runtime):
        card = ""
        if runtime is not None and runtime.has_gpu:
            memory = f", {runtime.gpu_memory_gb:g} GB" if runtime.gpu_memory_gb else ""
            card = f" This session has a {runtime.gpu_name}{memory}."
        out.append(
            Message(
                "backbone_needs_big_gpu",
                "stop",
                f"**{name}** does not fit a free T4.{card} {entry.notes} Switch to an L4 or A100 (Colab Pro), or "
                f"pick a backbone that does fit: {', '.join(_fits_t4())}.",
            )
        )
    if entry.tier == "extra":
        out.append(
            Message(
                "backbone_extra_install",
                "info",
                f"**{name}** needs one extra package. {entry.notes} The loader says which one if it is missing.",
            )
        )
    return out


def t4_class(runtime: Runtime | None) -> bool:
    """True unless something better than a free T4 has actually been detected."""
    return runtime is None or not runtime.has_gpu or runtime.is_t4


def _fits_t4() -> list[str]:
    return [name for name in colab_backbones() if BACKBONES[name].approx_lora_minutes_t4 is not None]


def structure_messages(state: WizardState) -> list[Message]:
    """Whether the 3Di question has been answered, and whether the answer is affordable.

    Every message here is about the structure section, so none of them fires while that
    section is off screen: a 3Di string loaded for SaProt must not go on blocking the run
    after the backbone has been changed to a sequence-only one that never reads it.
    """
    entry = backbone_entry(state.backbone)
    if entry is not None and entry.tier == "local_only":
        return []
    out: list[Message] = []
    if needs_structure(state) and not has_three_di(state):
        out.append(
            Message(
                "three_di_missing",
                "stop",
                f"**{state.backbone}** reads structure alongside sequence, so it needs one Foldseek 3Di letter "
                "per residue of your wild type. Choose a 3Di source above, or pick a sequence-only backbone "
                f"({', '.join(sequence_only_backbones())}).",
            )
        )
    if state.three_di_source == "esmfold" and _structure_section(state):
        limit = esmfold_safe_length()
        if state.wt_length > limit:
            out.append(
                Message(
                    "esmfold_too_long",
                    "stop",
                    f"Your wild type is {state.wt_length:,d} residues. ESMFold is the most memory-hungry step in "
                    f"this notebook and runs out of memory on a free T4 past about {limit:,d}. Download a real "
                    "structure (`.pdb` / `.cif`) from the PDB or AlphaFold instead — it is better and it is free.",
                )
            )
        else:
            out.append(
                Message(
                    "esmfold_expensive",
                    "warning",
                    "Folding the wild type with ESMFold is the last resort: it is the slowest and most "
                    "memory-hungry step here. A real structure from the PDB or AlphaFold gives a better 3Di "
                    "string in seconds.",
                )
            )
    mismatched = state.wt_3di_length and state.wt_length and state.wt_3di_length != state.wt_length
    if mismatched and _structure_section(state):
        out.append(
            Message(
                "three_di_length_mismatch",
                "stop",
                f"The 3Di string is {state.wt_3di_length:,d} states long but the wild type is "
                f"{state.wt_length:,d} residues. They must match one-to-one, or every position shifts.",
            )
        )
    return out


def config_messages(state: WizardState, status: ConfigStatus, estimate: RuntimeEstimate | None = None) -> list[Message]:
    """What the best-config registry has to say about this (backbone, pooling) pair.

    Silent when the backbone itself is the problem. An unregistered name has no
    registry entry either, and METL cannot run here at all: `backbone_messages`
    has already said the one thing that matters, and a second red box about
    missing hyperparameters only buries it.
    """
    if state.mode != "train" or not _backbone_is_usable(state):
        return []
    if status.error is not None:
        return [
            Message(
                "config_unreadable",
                "stop",
                f"The hyperparameters for **{status.model}** / `{status.pooling}` could not be read: "
                f"{status.error}",
            )
        ]
    if not status.found:
        known = ", ".join(f"`{name}`" for name in status.poolings) or "none"
        return [
            Message(
                "config_missing",
                "stop",
                f"No hyperparameters are registered for **{status.model}** with `{status.pooling}` pooling. "
                f"Poolings available for {status.model}: {known}.",
            )
        ]
    if not status.provisional:
        return []
    long_run = estimate is not None and estimate.minutes is not None and estimate.minutes >= PERSISTENCE_MINUTES
    tail = ""
    if long_run:
        tail = (
            f" This run is about {format_minutes(estimate.minutes or 0.0)} — a long time to spend on a "
            "placeholder when a tuned pair costs the same."
        )
    return [
        Message(
            "config_provisional",
            "stop" if long_run else "warning",
            f"{status.description} Nobody has tuned this pair and nobody has measured what it scores, so read "
            f"every number it produces as a lower bound rather than a result.{tail}",
        )
    ]


def schedule_messages(state: WizardState, estimate: RuntimeEstimate) -> list[Message]:
    """Whether this much training survives one Colab session."""
    if state.mode != "train" or estimate.fits_session:
        return []
    return [
        Message(
            "run_exceeds_session",
            "stop",
            f"This is {estimate.describe()} A Colab session disconnects long before that — SaprotHub's own "
            "notebooks warn above two hours — and a disconnect ends the run. Train fewer seeds, choose a faster "
            "backbone, or split the work across sessions.",
        )
    ]


def persistence_messages(state: WizardState, estimate: RuntimeEstimate) -> list[Message]:
    """Long run + no Drive = an hour that Colab will delete when the session ends."""
    if state.use_drive:
        return [
            Message(
                "drive_on",
                "info",
                "Google Drive is mounted: the model weights, the foldseek binary and everything this run writes "
                "land in your Drive, so a disconnect costs you nothing but time.",
            )
        ]
    if state.mode != "train" or estimate.minutes is None or estimate.minutes < PERSISTENCE_MINUTES:
        return []
    return [
        Message(
            "no_persistence",
            "stop" if estimate.scaled else "warning",
            f"About {format_minutes(estimate.minutes)} of training, and nothing is being saved to Google Drive. "
            "Colab wipes this machine when the session ends: the model weights download again next time and an "
            "unfinished run is gone. Tick **Save to Google Drive** above before you start.",
        )
    ]


def reproducibility_messages(state: WizardState) -> list[Message]:
    """One run is a number, not a measurement."""
    if state.mode != "train" or n_runs(state) > 1:
        return []
    return [
        Message(
            "single_run",
            "warning",
            "One split seed x one model seed is a single run, and a single run cannot show reproducibility: the "
            "report writes its standard deviation as `NaN`, not `0`. Two or three of each is what the registry's "
            "own protocol uses.",
        )
    ]


def runtime_messages(runtime: Runtime | None) -> list[Message]:
    """What this particular machine will and will not do.

    Whether the chosen backbone fits the detected card belongs to
    `backbone_messages`; this one is only about the machine itself.
    """
    if runtime is None:
        return []
    out: list[Message] = []
    if not runtime.has_gpu:
        out.append(
            Message(
                "no_gpu",
                "stop",
                "This runtime has no GPU, so nothing here will train. **Runtime → Change runtime type → T4 GPU**, "
                "then run this cell again. The one-hot floor and the report still work without one.",
            )
        )
    if not runtime.in_colab:
        out.append(
            Message(
                "not_colab",
                "info",
                "This is not a Colab session, so the Google Drive mount and the free-tier GPU advice do not apply.",
            )
        )
    return out


def test_unlock_messages(unlock_count: int, *, requested: bool = False) -> list[Message]:
    """The one deliberate step: reading the test partition, and counting every read.

    Separate from everything else on purpose. Model and hyperparameter choices are
    made on validation data; the test partition is touched once, deliberately, so the
    number that gets reported means what it says.
    """
    count = max(0, int(unlock_count))
    if requested and count == 0:
        return [
            Message(
                "test_first_unlock",
                "warning",
                "This reads the test partition for the first time. Do it once, when you have stopped changing "
                "things: the count goes into `unlock.json` and is printed in the report.",
            )
        ]
    if requested and count > 0:
        return [
            Message(
                "test_reunlock",
                "stop",
                f"This test set has already been read {count} time(s). Reading it again, after seeing the last "
                f"number, is how a test set stops being a test set. The report will say {count + 1}.",
            )
        ]
    if count > 0:
        return [
            Message(
                "test_already_unlocked",
                "warning",
                f"This test set has been read {count} time(s), and the report says so.",
            )
        ]
    return [
        Message(
            "test_locked",
            "info",
            "The test partition is locked. Every number on this page is validation — which is the point: you can "
            "change your mind as often as you like up here.",
        )
    ]


def messages_for(
    state: WizardState,
    *,
    runtime: Runtime | None = None,
    status: ConfigStatus | None = None,
    root: str | Path | None = None,
) -> list[Message]:
    """Every contextual message this state deserves, worst first, one per key.

    The test-set messages only join in once an unlock has been asked for or has
    already happened; the unlock step calls `test_unlock_messages` itself.
    """
    estimate = estimate_runtime(state)
    if status is None and state.mode == "train" and _backbone_is_usable(state):
        status = config_status(state.backbone, state.pooling, root=root)
    collected: list[Message] = []
    collected += runtime_messages(runtime)
    collected += backbone_messages(state, runtime)
    collected += structure_messages(state)
    if status is not None:
        collected += config_messages(state, status, estimate)
    collected += schedule_messages(state, estimate)
    collected += persistence_messages(state, estimate)
    collected += reproducibility_messages(state)
    if state.mode == "train" and (state.unlock_requested or state.unlock_count > 0):
        collected += test_unlock_messages(state.unlock_count, requested=state.unlock_requested)
    return _ordered(collected)


# --------------------------------------------------------------------------- runtime


@dataclass(frozen=True)
class Runtime:
    """What this session actually is. Built by `detect_runtime`; never raises."""

    in_colab: bool = False
    has_gpu: bool = False
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    torch_version: str | None = None
    drive_mounted: bool = False

    @property
    def tier(self) -> str:
        return gpu_tier(self.gpu_name, self.gpu_memory_gb)

    @property
    def is_t4(self) -> bool:
        """T4-class: the free tier, and the card the registry's minute estimates assume."""
        return self.tier == "t4"

    def summary(self) -> str:
        return runtime_summary(self)


def gpu_tier(name: str | None, memory_gb: float | None) -> str:
    """Classify a GPU by name first, then by memory. Pure, so the table is testable."""
    lowered = (name or "").lower()
    if not lowered and memory_gb is None:
        return "none"
    for marker, tier in (("a100", "a100"), ("h100", "a100"), ("l4", "l4"), ("t4", "t4"), ("v100", "l4")):
        if marker in lowered:
            return tier
    if memory_gb is None:
        return "other"
    if memory_gb <= FREE_T4_MEMORY_GB:
        return "t4"
    if memory_gb <= L4_MEMORY_GB:
        return "l4"
    return "a100"


def _in_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return True
    if "google.colab" in sys.modules:
        return True
    try:
        from importlib.util import find_spec

        return find_spec("google.colab") is not None
    except Exception:
        return False


def detect_runtime(*, drive_mount_point: str | Path = "/content/drive") -> Runtime:
    """Look at the machine: Colab or not, GPU or not, which one, how much memory."""
    name: str | None = None
    memory: float | None = None
    has_gpu = False
    torch_version: str | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        if torch.cuda.is_available():
            has_gpu = True
            properties = torch.cuda.get_device_properties(0)
            name = str(properties.name)
            memory = round(properties.total_memory / (1024**3), 1)
    except Exception:
        pass
    return Runtime(
        in_colab=_in_colab(),
        has_gpu=has_gpu,
        gpu_name=name,
        gpu_memory_gb=memory,
        torch_version=torch_version,
        drive_mounted=Path(drive_mount_point, "MyDrive").is_dir(),
    )


def backbone_fit(runtime: Runtime | None) -> dict[str, str]:
    """Every registered backbone mapped to `ok` / `needs_bigger_gpu` / `needs_gpu` / `unavailable`."""
    verdicts: dict[str, str] = {}
    for name, entry in BACKBONES.items():
        if entry.tier == "local_only":
            verdicts[name] = "unavailable"
        elif runtime is None or not runtime.has_gpu:
            verdicts[name] = "needs_gpu"
        elif entry.approx_lora_minutes_t4 is None and runtime.is_t4:
            verdicts[name] = "needs_bigger_gpu"
        else:
            verdicts[name] = "ok"
    return verdicts


def runtime_summary(runtime: Runtime | None) -> str:
    """Plain language: what this machine is, and what will and will not run on it."""
    if runtime is None:
        return "Nothing is known about this runtime yet."
    where = "Google Colab" if runtime.in_colab else "a local Jupyter session"
    if not runtime.has_gpu:
        return (
            f"You are on {where} with **no GPU**. Fine-tuning needs one: "
            "**Runtime → Change runtime type → T4 GPU**. The one-hot floor and the report run without it."
        )
    memory = f", {runtime.gpu_memory_gb:g} GB" if runtime.gpu_memory_gb else ""
    verdicts = backbone_fit(runtime)
    ok = sorted(name for name, verdict in verdicts.items() if verdict == "ok")
    big = sorted(name for name, verdict in verdicts.items() if verdict == "needs_bigger_gpu")
    lines = [f"You are on {where} with a **{runtime.gpu_name}**{memory}."]
    if ok:
        lines.append(f"Trains here: {', '.join(ok)}.")
    if big:
        lines.append(f"Too big for this card: {', '.join(big)} — those need an L4 or an A100 (Colab Pro).")
    unavailable = sorted(name for name, verdict in verdicts.items() if verdict == "unavailable")
    if unavailable:
        lines.append(f"Not available in Colab at all: {', '.join(unavailable)}.")
    return "\n\n".join(lines)


# ----------------------------------------------------------------------- Google Drive


@dataclass(frozen=True)
class DriveMount:
    """The result of trying to mount Drive. `mounted=False` is normal, never fatal."""

    mounted: bool
    reason: str
    root: Path | None = None
    hf_cache: Path | None = None
    output_dir: Path | None = None

    def describe(self) -> str:
        if not self.mounted:
            return f"Google Drive is not mounted: {self.reason} Results stay on this machine."
        return (
            f"Google Drive is mounted at `{self.root}`. Model weights cache in `{self.hf_cache}` and results are "
            f"written to `{self.output_dir}`, so a disconnect costs you nothing but time."
        )


def drive_folder_name(folder: str) -> str:
    """The folder a typed name means, with anything that climbs out of MyDrive dropped.

    A typed `../..` would otherwise put the cache and the results outside the Drive
    the user thinks they mounted, which is the one place a notebook must not write.
    """
    parts = [part for part in folder.strip().split("/") if part.strip() not in ("", ".", "..")]
    return "/".join(parts) or "ColabSeqDisplay"


def drive_paths(mount_point: str | Path, folder: str = "ColabSeqDisplay") -> dict[str, Path]:
    """Where a mounted Drive keeps the cache and the outputs. Pure."""
    base = Path(mount_point) / "MyDrive" / drive_folder_name(folder)
    return {
        "base": base,
        "hf_cache": base / "hf_cache",
        "output": base / "work",
        "colabsd_cache": base / "cache",
    }


def drive_env(paths: Mapping[str, Path]) -> dict[str, str]:
    """Environment that points the HuggingFace and foldseek caches at Drive. Pure.

    `COLABSD_CACHE_DIR` is what `colabsd.structure.default_cache_dir` reads, so the
    foldseek binary survives a disconnect along with the model weights.
    """
    return {
        "HF_HOME": str(paths["hf_cache"]),
        "HUGGINGFACE_HUB_CACHE": str(Path(paths["hf_cache"]) / "hub"),
        "COLABSD_CACHE_DIR": str(paths["colabsd_cache"]),
    }


DRIVE_NOTE = (
    "Colab deletes this machine when the session ends. Mounting Google Drive keeps three things that would "
    "otherwise be lost: the downloaded model weights, the foldseek binary, and whatever a long fine-tune has "
    "finished so far. It is off unless you tick it, and it does nothing outside Colab."
)


def drive_widgets(*, folder: str = "ColabSeqDisplay") -> dict[str, Any]:
    """The one checkbox that turns persistence on, plus the folder it writes to.

    Keyed by the field names `FIELD_RULES` knows, so `bind_all` and
    `apply_field_visibility` pick them up with no further wiring.
    """
    import ipywidgets

    checkbox = ipywidgets.Checkbox(value=False, description="Save to Google Drive", indent=False)
    text = ipywidgets.Text(value=folder, description="Drive folder:", style={"description_width": "initial"})
    set_display(text, False)
    return {"use_drive": checkbox, "drive_folder": text}


def mount_drive(
    *,
    enable: bool = False,
    folder: str = "ColabSeqDisplay",
    mount_point: str | Path = "/content/drive",
    environ: dict[str, str] | None = None,
) -> DriveMount:
    """Mount Drive and point the caches and the output directory at it.

    Off by default and silent outside Colab: a `DriveMount` with `mounted=False`
    and a reason, never an exception. Mounting fixes the three things a Colab
    session costs you — the wiped machine, the re-downloaded model weights, and
    the fine-tune that dies with the session.
    """
    if not enable:
        return DriveMount(False, "you did not switch it on.")
    if not _in_colab():
        return DriveMount(False, "this is not a Colab session, so there is nothing to mount.")
    try:
        from google.colab import drive as colab_drive

        colab_drive.mount(str(mount_point))
        paths = drive_paths(mount_point, folder)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        (environ if environ is not None else os.environ).update(drive_env(paths))
        return DriveMount(True, "mounted.", root=paths["base"], hf_cache=paths["hf_cache"], output_dir=paths["output"])
    except Exception as exc:
        return DriveMount(False, f"the mount failed ({type(exc).__name__}: {exc}).")


# ------------------------------------------------------------------------ widget layer


def set_display(widget: Any, visible: bool) -> None:
    """Show or hide one widget — the whole of the ColabPLM interaction, in one line."""
    widget.layout.display = None if visible else "none"


class Section:
    """A titled group of widgets that shows and hides as a unit.

    The box is built on first use, so a `Section` can be declared and toggled
    before ipywidgets is touched.
    """

    def __init__(
        self,
        key: str,
        title: str | None = None,
        widgets: Sequence[Any] = (),
        *,
        description: str = "",
        visible: bool = True,
    ) -> None:
        self.key = key
        self.title = title if title is not None else SECTION_TITLES.get(key, key)
        self.widgets: list[Any] = list(widgets)
        self.description = description
        self.visible = bool(visible)
        self._box: Any = None

    def box(self) -> Any:
        """The `VBox` holding the heading, the note and the fields."""
        if self._box is None:
            import ipywidgets

            children = [theme.heading(self.title)]
            if self.description:
                children.append(theme.note(self.description))
            children.extend(self.widgets)
            self._box = ipywidgets.VBox(children)
            set_display(self._box, self.visible)
        return self._box

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self._box is not None:
            set_display(self._box, self.visible)

    def show(self) -> None:
        self.set_visible(True)

    def hide(self) -> None:
        self.set_visible(False)

    def __repr__(self) -> str:
        return f"Section(key={self.key!r}, visible={self.visible})"


def render_messages(messages: Iterable[Message]) -> str:
    """Every message as one HTML block, worst first."""
    return "".join(item.html() for item in messages)


class MessageBoard:
    """One `ipywidgets.HTML` that redraws itself from a list of `Message`."""

    def __init__(self) -> None:
        self.widget = theme.html("")
        self.keys: tuple[str, ...] = ()

    def update(self, messages: Iterable[Message]) -> tuple[str, ...]:
        """Redraw and return the keys shown, so a caller (or a test) can assert on them."""
        items = list(messages)
        self.widget.value = render_messages(items)
        self.keys = tuple(item.key for item in items)
        set_display(self.widget, bool(items))
        return self.keys


def bind(
    widget: Any,
    state: WizardState,
    name: str,
    *,
    on_change: Callable[[WizardState], Any] | None = None,
) -> Callable[[dict], None]:
    """Write `widget.value` into `state.name` on every change, then call `on_change`.

    Returns the handler, so a test can drive it without a browser.
    """

    def handler(change: dict) -> None:
        state.set(name, change["new"])
        if on_change is not None:
            on_change(state)

    widget.observe(handler, names="value")
    return handler


def bind_all(
    widgets: Mapping[str, Any],
    state: WizardState,
    *,
    on_change: Callable[[WizardState], Any] | None = None,
) -> dict[str, Callable[[dict], None]]:
    """`bind` every widget that has a `value`, keyed by field name."""
    handlers: dict[str, Callable[[dict], None]] = {}
    for name, widget in widgets.items():
        if hasattr(widget, "observe") and hasattr(widget, "value"):
            handlers[name] = bind(widget, state, name, on_change=on_change)
    return handlers


def run_guarded(output: Any, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run `function` with its output and its failure inside an `ipywidgets.Output`.

    A traceback raised inside an observer is invisible in Colab; a printed message
    that says what to do about it is not.
    """
    with output:
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")
            return None


def on_click(button: Any, handler: Callable[[], Any], *, output: Any | None = None) -> Callable[[Any], Any]:
    """Attach a no-argument handler to a button, optionally guarded by an `Output`."""

    def _clicked(_button: Any) -> Any:
        if output is None:
            return handler()
        return run_guarded(output, handler)

    button.on_click(_clicked)
    return _clicked


@dataclass(frozen=True)
class Plan:
    """Everything the widget layer needs to redraw itself, computed without widgets."""

    sections: tuple[str, ...]
    fields: tuple[str, ...]
    messages: tuple[Message, ...]
    estimate: RuntimeEstimate

    @property
    def message_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.messages)

    @property
    def blocking(self) -> tuple[Message, ...]:
        """The `stop` messages — a wizard refuses to start while any of these stand."""
        return tuple(item for item in self.messages if item.severity == "stop")

    @property
    def can_run(self) -> bool:
        return not self.blocking


def plan(
    state: WizardState,
    *,
    runtime: Runtime | None = None,
    status: ConfigStatus | None = None,
    root: str | Path | None = None,
) -> Plan:
    """The whole decision, as data. This is the function the tests hold to a table."""
    return Plan(
        sections=visible_sections(state),
        fields=visible_fields(state),
        messages=tuple(messages_for(state, runtime=runtime, status=status, root=root)),
        estimate=estimate_runtime(state),
    )


def apply_visibility(sections: Mapping[str, Section], state: WizardState) -> tuple[str, ...]:
    """Show the sections this state calls for, hide the rest; return the visible keys."""
    live = set(visible_sections(state))
    for key, section in sections.items():
        section.set_visible(key in live)
    return tuple(key for key in SECTION_ORDER if key in live and key in sections)


def apply_field_visibility(widgets: Mapping[str, Any], state: WizardState) -> tuple[str, ...]:
    """Same for individual fields. A widget with no rule stays visible."""
    visibility = field_visibility(state)
    shown: list[str] = []
    for key, widget in widgets.items():
        visible = visibility.get(key, True)
        set_display(widget, visible)
        if visible:
            shown.append(key)
    return tuple(shown)


def refresh(
    state: WizardState,
    *,
    sections: Mapping[str, Section] | None = None,
    widgets: Mapping[str, Any] | None = None,
    board: MessageBoard | None = None,
    runtime: Runtime | None = None,
    status: ConfigStatus | None = None,
    root: str | Path | None = None,
) -> Plan:
    """Compute the plan, then push it into the widgets. The only thing an observer calls."""
    computed = plan(state, runtime=runtime, status=status, root=root)
    if widgets is not None:
        apply_field_visibility(widgets, state)
    if sections is not None:
        apply_visibility(sections, state)
    if board is not None:
        board.update(computed.messages)
    return computed
