"""Score variants with a saved model bundle: upload, look at what you uploaded, rank, download.

No training happens here. The one thing this interface refuses to be quiet about is the
bundle: somebody scoring variants a month after training will not remember which backbone it
was, which residues it averaged, whether its hyperparameters were tuned or placeholders, or
whether its test set was ever unlocked. `describe_bundle` reads that out of the manifest and
`bundle_html` puts it on screen before a single variant is scored.

Everything said about the backbone comes from `colabsd.backbones.registry`, not from the
bundle's name. `is_offered()` asks the registry whether the notebooks still list one rather
than keeping a copy of the list, and `colabsd.ui.core.offered_backbones_phrase()` is what
names the alternatives, so a bundle built from a backbone they stopped offering is told so and
scored anyway: narrowing a dropdown must not break work already saved.

Two archives come out of one export step, so the commonest upload mistake is the wrong one of
the two. `identify_upload` reads the zip and says which it is, rather than letting
`load_bundle` complain about a missing `manifest.json`.

Every decision is a pure function of a `PredictState` and the `BundleFacts` read off the
bundle; `PredictWizard` only wires those to ipywidgets. One call blocks the kernel —
`google.colab.files.upload()` — so every call site says what it is waiting for before it opens
(`upload_wait_text`): a panel that freezes with every callback dead and says nothing looks
broken. Presentation is `colabsd.ui.theme` and `colabsd.ui.core.Message`.
"""

from __future__ import annotations

import html as _html
import re
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from colabsd import bundle as bundle_module
from colabsd.backbones import registry
from colabsd.backbones.registry import BACKBONES, BackboneEntry
from colabsd.bundle import HEAD_NAME, LORA_NAME, MANIFEST_NAME
from colabsd.errors import BackboneError
from colabsd.ui import core, exports, theme
from colabsd.ui.core import (
    COLAB_SESSION_MINUTES,
    UNDER_A_MINUTE,
    Message,
    counted,
    format_minutes,
    render_messages,
    set_display,
)

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

ASSUMED_LENGTH = 1000
RANKINGS = ("pred_mean", "pred_min")

#: What the ranked table is called when the box is left alone. A name with a folder in it is
#: honored -- the folder is made under the working directory -- rather than lost.
DEFAULT_OUTPUT_NAME = "ranked_variants.csv"

VARIANT_SOURCES = ("upload_variants_csv", "random_combinations", "rows_of_a_library_csv")

#: Where the bundle comes from, in the order the radio group offers it.
BUNDLE_SOURCES = ("upload_model_bundle_zip", "file_in_the_working_folder")

#: What a reader reads beside each button of those three groups. The values above are what
#: `PredictState` carries and what every branch in this module tests against; these are the only
#: thing on screen. The training panel has labeled every one of its controls in English since
#: it was written -- `("Random combinations of residues", "random_combinations")` -- while this
#: panel put the bare identifiers in front of a biologist and asked them to choose between
#: `rows_of_a_library_csv` and `upload_variants_csv`. Written out per value rather than
#: prettified from the identifier, because "Rows of a library csv" is not a sentence anybody
#: would have written, and `pred_min` does not become "the worst condition" by title-casing.
BUNDLE_SOURCE_LABELS: dict[str, str] = {
    "upload_model_bundle_zip": f"Upload a {exports.DEFAULT_BUNDLE_NAME}",
    "file_in_the_working_folder": "A file already in the working folder",
}

VARIANT_SOURCE_LABELS: dict[str, str] = {
    "upload_variants_csv": "A variants CSV of my own",
    "random_combinations": "Random combinations of residues",
    "rows_of_a_library_csv": "The first rows of a library CSV",
}

#: The same wording the training panel's own *Rank by* dropdown uses, so a reader who has seen
#: both pages is choosing between the same two things twice and not between two vocabularies.
RANKING_LABELS: dict[str, str] = {
    "pred_mean": "Average over conditions (pred_mean)",
    "pred_min": "Worst condition (pred_min)",
}


def labeled(values: Sequence[str], labels: dict[str, str]) -> list[tuple[str, str]]:
    """`[(what a reader reads, what the state carries)]`, for an ipywidgets `options=`.

    A value with no label of its own keeps its identifier rather than dropping off the form, so
    a source added to `VARIANT_SOURCES` and not written out above is a shabby label and not a
    missing option; `tests/test_ui_side.py` holds every value to having one.
    """
    return [(labels.get(value, value), value) for value in values]


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Fields that decide which variants **Prepare the variants** builds. Changing one makes the
#: prepared table describe a different request, so `PredictWizard` throws it away rather
#: than scoring yesterday's rows under today's settings.
VARIANT_INPUTS = frozenset({"variants_source", "how_many", "random_seed"})

#: Fields that decide which bundle **Load the bundle** reads.
BUNDLE_INPUTS = frozenset({"bundle_source", "bundle_filename"})

#: What `colabsd.bundle.save_bundle` writes into a model bundle, taken from that module so
#: the two cannot drift apart.
BUNDLE_MEMBERS: tuple[str, ...] = (MANIFEST_NAME, LORA_NAME, HEAD_NAME)

#: What the other archive the main notebook exports holds, named by `colabsd.ui.exports` —
#: the module that writes it. Taken whole rather than rebuilt from parts: assembled here from
#: the manifest, the README and `REPORT_MEMBERS`, it came to five names for an archive of seven,
#: because the two training-curve files reach the zip through `extra_files` and never appear in
#: `REPORT_MEMBERS`. `exports.ARCHIVE_MEMBERS` is the list the module that writes the archive
#: keeps, and the one the main panel, TUTORIAL.md and the generated notebook all read.
PERFORMANCE_MEMBERS: tuple[str, ...] = exports.ARCHIVE_MEMBERS

#: The three report files inside it, which is what a user calls "the performance report".
REPORT_MEMBERS: tuple[str, ...] = tuple(exports.REPORT_MEMBERS.values())

#: Where the forward-pass estimate this panel shares with the training panel may live. Asked
#: at call time, in this order, so this module uses whichever copy exists rather than keeping
#: a second one.
SHARED_MODULES: tuple[str, ...] = ("colabsd.ui.core", "colabsd.ui.prepare_workflow")

# ----------------------------------------------------------------------------------------
# What the registry offers, and what it merely still knows.
# ----------------------------------------------------------------------------------------


def is_offered(backbone: str) -> bool:
    """True when the notebooks still offer *backbone*. False, not an error, for a stranger."""
    return backbone in BACKBONES and registry.is_offered(backbone)


def withheld_note(backbone: str) -> str:
    """The registry's answer to "where did my model go?", and something sayable if it has none.

    `registry.withheld_note` raises when a family is offered nowhere and explained nowhere.
    That is right for a test of the registry and wrong here: a panel whose whole job is to
    explain a bundle must not fail while explaining one.
    """
    try:
        return registry.withheld_note(backbone)
    except BackboneError:
        return f"`{backbone}` is in this colabsd, but the notebooks do not offer it."


# ----------------------------------------------------------------------------------------
# The shared cost model, borrowed rather than restated.
# ----------------------------------------------------------------------------------------


def estimate_owner() -> ModuleType:
    """The module that owns `forward_pass_minutes` — the one timed pass both panels quote."""
    for module_name in SHARED_MODULES:
        try:
            module = import_module(module_name)
        except ImportError:  # pragma: no cover - only a partial install gets here
            continue
        if hasattr(module, "forward_pass_minutes"):
            return module
    raise RuntimeError(
        "None of " + ", ".join(SHARED_MODULES) + " defines forward_pass_minutes(). Point SHARED_MODULES "
        "at wherever it went rather than writing a second table of per-backbone rates here."
    )


def forward_pass_minutes(
    backbone: str, *, n_sequences: int, length: int, has_gpu: bool
) -> tuple[float, float] | None:
    """Minutes for `n_sequences` frozen forward passes of `length` residues, low to high.

    Scoring a variant is one frozen forward pass, exactly what the training panel already
    costs, so it delegates to `estimate_owner()`: one timed run, one set of registry ratios.
    A second table of rates here would disagree with that one within a release.
    """
    owner = estimate_owner()
    return owner.forward_pass_minutes(backbone, n_sequences=n_sequences, length=length, has_gpu=has_gpu)


# ----------------------------------------------------------------------------------------
# Uploading: the one call that stops the panel dead.
# ----------------------------------------------------------------------------------------


def upload_wait_text(what: str) -> str:
    """What to say *before* `files.upload()` takes the kernel, so the freeze is explained.

    Every callback is dead while the picker is open, so the sentence is useless said
    afterwards. `colabsd.ui.core`'s words: the main panel blocks on the same call.
    """
    return core.upload_notice(what)


def upload_canceled_text(what: str) -> str:
    """What to say when the picker came back with nothing in it.

    `colabsd.ui.core`'s words again: two wordings for one outcome drift.
    """
    return core.upload_canceled_notice(what)


# ----------------------------------------------------------------------------------------
# Which of the two archives did they just hand us?
# ----------------------------------------------------------------------------------------


def archive_kind(names: Sequence[str]) -> str:
    """What a zip's member list says the zip is: `model_bundle`, `performance_report`, `unknown`.

    Compared on base names so a zip that nests its files under a folder is still recognized,
    which is what a user who unzipped and rezipped one hands over.
    """
    found = {PurePosixPath(str(name)).name.lower() for name in names if not str(name).endswith("/")}
    if all(member.lower() in found for member in BUNDLE_MEMBERS):
        return "model_bundle"
    if found & {member.lower() for member in PERFORMANCE_MEMBERS}:
        return "performance_report"
    return "unknown"


def identify_upload(path: Path | str) -> str:
    """Read *path* far enough to name it: `missing`, `not_a_zip`, or an `archive_kind`."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    try:
        with zipfile.ZipFile(path) as archive:
            return archive_kind(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return "not_a_zip"


def load_failure_text(path: Path | str, error: Exception) -> str:
    """Why a file could not be loaded as a bundle, in terms of what the file actually is.

    Both archives are zips written by the same export step, so uploading the wrong one is the
    mistake to expect; "missing manifest.json" is nothing the holder of one can act on.

    `error` is escaped once, here, because what comes back is rendered as HTML by the notice
    board: a loader complaining about `<manifest.json>` had that word eaten by the browser.
    The file names stay unescaped inside their backticks -- both renderers escape the contents
    of a code span themselves, and escaping as well prints `&lt;` on the page.
    """
    path = Path(path)
    reason = theme.as_text(error)
    kind = identify_upload(path)
    if kind == "performance_report":
        also_called = "" if path.name == exports.DEFAULT_ARCHIVE_NAME else f" (`{exports.DEFAULT_ARCHIVE_NAME}`)"
        return (
            f"`{path.name}` is the **performance archive**{also_called}, not the model bundle: it holds "
            f"`{'`, `'.join(PERFORMANCE_MEMBERS)}` and no model weights. Upload the other archive that export "
            f"step wrote, `{exports.DEFAULT_BUNDLE_NAME}`, which holds `{'`, `'.join(BUNDLE_MEMBERS)}`."
        )
    if kind == "missing":
        return (
            f"There is no file at `{path}`. Check the name, or switch **Bundle** back to "
            f"**{BUNDLE_SOURCE_LABELS['upload_model_bundle_zip']}** and upload it."
        )
    if kind == "not_a_zip":
        return (
            f"`{path.name}` is not a readable .zip. Re-download the bundle, in case the transfer "
            "truncated it."
        )
    if kind == "unknown":
        return (
            f"`{path.name}` is a zip, but it carries none of `{'`, `'.join(BUNDLE_MEMBERS)}`, so it was "
            f"not written by ColabSeqDisplay's export step ({reason})."
        )
    return f"`{path.name}` looks like a model bundle but could not be loaded: {reason}"


@dataclass(frozen=True)
class BundleFacts:
    """What a bundle actually contains, read out of its manifest.

    Everything here comes from the bundle; nothing is inferred from what the user typed.
    """

    backbone: str
    model_name: str
    n_pooled_positions: int
    conditions: tuple[str, ...]
    mutation_columns: tuple[str, ...]
    positions_1based: tuple[int, ...]
    wt_length: int
    three_letter: bool
    carries_three_di: bool
    is_provisional: bool
    unlock_count: int
    has_label_scaler: bool
    created_utc: str
    colabsd_version: str
    torch_version: str
    notes: str | None
    hyperparameters: tuple[tuple[str, Any], ...]
    #: `colabsd.bundle.budget_line` over the manifest's `training_budget`. Rendered here rather
    #: than left to the template because `effective_batch_size` is both a hyperparameter and one
    #: of the four fields a user may set, so the hyperparameters row above can label the user's
    #: own number a PROVISIONAL placeholder. This row is where the manifest says whose it was.
    budget: str = ""
    #: `colabsd.bundle.training_status_line` over the manifest's `training_status`. Nothing in
    #: `colabsd.ui` read it, so the one screen a colleague opens a bundle on could not say that
    #: these weights came from a run stopped by hand at 2 of 20 epochs.
    epochs_trained: str = ""
    source_path: str | None = None
    entry: BackboneEntry | None = None

    @property
    def k(self) -> int:
        """Number of mutated sites."""
        return len(self.mutation_columns)

    @property
    def registered(self) -> bool:
        """True when `colabsd.backbones.registry` knows the backbone this bundle names."""
        return self.entry is not None

    @property
    def offered(self) -> bool:
        """True when the notebooks would still let somebody choose this backbone today.

        The narrowing is about what can be trained from the panel, not about what a saved
        model is allowed to do: a bundle from a withheld backbone still loads and scores.
        """
        return self.registered and is_offered(self.backbone)

    @property
    def needs_structure(self) -> bool:
        """True when this backbone cannot be tokenized without a wild-type 3Di string.

        `registry.needs_wt_3di`, not `BackboneEntry.needs_structure`: METL needs a structure
        too, but a Rosetta-relaxed one no step here produces, and it has no adapter to feed.
        """
        return self.registered and registry.needs_wt_3di(self.backbone)

    @property
    def runnable(self) -> bool:
        """True when `registry.create_adapter` can rebuild this backbone to score with."""
        return self.registered and registry.has_adapter(self.backbone)

    @property
    def has_t4_estimate(self) -> bool:
        """True when the registry carries a T4 figure for this backbone.

        That figure is `approx_lora_minutes_t4`: a LoRA *training* verdict, and the only thing
        the cost model can scale a scoring run from. Its absence says nothing about whether a
        frozen forward pass fits a T4, so nothing here should read it as an inference limit.
        """
        return bool(self.entry and self.entry.approx_lora_minutes_t4 is not None)

    @property
    def backbone_line(self) -> str:
        """The registry's one-line description of the backbone, or a note that it has none."""
        if self.entry is None:
            return "not in this colabsd's backbone registry"
        parts = [f"{self.entry.family} family", f"{self.entry.embed_dim}-d pooled feature"]
        if self.entry.hf_id:
            parts.append(self.entry.hf_id)
        if not self.offered:
            # Only a runnable backbone earns "still scored here"; METL and its like cannot be.
            parts.append(
                "no longer offered in the notebooks — still loaded and scored here"
                if self.runnable
                else "no longer offered in the notebooks, and no adapter here to run it"
            )
        return ", ".join(parts)

    @property
    def hyperparameter_status(self) -> str:
        """The one word a reader needs about the numbers behind this model."""
        return "PROVISIONAL placeholders" if self.is_provisional else "tuned"


def span_of(low_minutes: float, high_minutes: float) -> str:
    """A low-to-high duration in one register.

    "under a minute to 2 min" was two registers in one range -- words on one side of it and a
    unit on the other -- and "under a minute to 1 min" was barely a range at all. When the low
    end has no number to give, the range has only an upper bound, so say that.
    """
    if high_minutes < 1.0:
        return UNDER_A_MINUTE
    low, high = format_minutes(low_minutes), format_minutes(high_minutes)
    if low == UNDER_A_MINUTE or low == high:
        return f"up to {high}"
    return f"{low} to {high}"


@dataclass(frozen=True)
class RuntimeEstimate:
    """An order-of-magnitude cost for scoring, in minutes."""

    n_variants: int
    low_minutes: float | None
    high_minutes: float | None
    device: str
    size_label: str
    length: int

    @property
    def known(self) -> bool:
        return self.high_minutes is not None

    def sentence(self) -> str:
        """One line a biologist can act on."""
        if self.n_variants <= 0:
            return "No variants chosen yet."
        # `counted` for the variants, which is a count of them; not for the length, which is
        # an approximation of one variant's own size -- "of ~287 residues" is describing each
        # sequence, not counting residues in this sentence.
        #
        # Grouped all the same. Not pluralizing it and not grouping it are two decisions, and
        # only the first one follows from the above: `bundle_html` prints the same wild type two
        # rows up as "1,054 residues", so leaving this one bare showed a reader the same protein
        # as "1,054" and "1054" on one screen -- the exact pair of renderings that read as two
        # different proteins across the two panels before `counted` was given the separator.
        about = (
            f"{counted(self.n_variants, 'variant')} of ~{self.length:,} residues through "
            f"{self.size_label} on the {self.device}"
        )
        if self.low_minutes is None or self.high_minutes is None:
            return (
                f"Rough cost: unknown for {about} — the backbone registry has no T4 estimate for "
                f"{self.size_label} to scale from."
            )
        span = span_of(self.low_minutes, self.high_minutes)
        return f"Rough cost: {span} to score {about}. An order of magnitude, not a measurement."


@dataclass
class PredictState:
    """Everything scoring is decided from. No widgets, no side effects."""

    bundle_source: str = "upload_model_bundle_zip"
    #: The export step names the file; this only has to agree with it.
    bundle_filename: str = exports.DEFAULT_BUNDLE_NAME
    variants_source: str = "upload_variants_csv"
    how_many: int = 200
    random_seed: int = 0
    rank_by: str = "pred_mean"
    top_n_to_show: int = 20
    score_batch_size: int = 8
    output_name: str = DEFAULT_OUTPUT_NAME
    has_gpu: bool = False
    facts: BundleFacts | None = None
    n_variants: int = 0
    variant_columns: tuple[str, ...] = ()
    #: Why the last **Load the bundle** was refused, in words a user can act on. Empty until
    #: something goes wrong; cleared as soon as a bundle loads.
    load_problem: str = ""


def parse_list(text: str) -> list[str]:
    """Split a comma- or semicolon-separated field into stripped, non-empty items."""
    return [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]


def describe_bundle(bundle: Any) -> BundleFacts:
    """Read a `colabsd.bundle.Bundle` into the facts this interface shows.

    Everything is read defensively: an older bundle that is missing a manifest key must
    still be describable, because refusing to describe it is worse than saying "unknown".
    """
    manifest = getattr(bundle, "manifest", {}) or {}
    provenance = manifest.get("provenance", {}) or {}
    spec = bundle.spec
    positions = tuple(int(value) for value in getattr(spec, "positions_1based", ()) or ())
    pooled = getattr(bundle, "pooling_positions_0based", None) or []
    return BundleFacts(
        backbone=str(getattr(bundle, "adapter_name", "unknown")),
        model_name=str(getattr(bundle, "model_name", getattr(bundle, "adapter_name", "unknown"))),
        n_pooled_positions=len(pooled),
        conditions=tuple(str(name) for name in getattr(spec, "condition_columns", ()) or ()),
        mutation_columns=tuple(str(name) for name in getattr(spec, "mutation_columns", ()) or ()),
        positions_1based=positions,
        wt_length=len(getattr(spec, "wt_sequence", "") or ""),
        three_letter=bool(getattr(spec, "three_letter", True)),
        carries_three_di=bool(getattr(spec, "wt_3di", None)),
        is_provisional=bool(getattr(bundle, "is_provisional", False)),
        unlock_count=int(getattr(bundle, "unlock_count", 0) or 0),
        has_label_scaler=bool(getattr(bundle, "label_scaler", None)),
        created_utc=str(provenance.get("created_utc", "unknown")),
        colabsd_version=str(provenance.get("colabsd_version", "unknown")),
        torch_version=str(provenance.get("torch_version", "unknown")),
        notes=provenance.get("notes"),
        hyperparameters=tuple(sorted((manifest.get("hyperparameters", {}) or {}).items())),
        budget=bundle_module.budget_line(getattr(bundle, "training_budget", None)),
        epochs_trained=bundle_module.training_status_line(getattr(bundle, "training_status", None)),
        source_path=None if getattr(bundle, "path", None) is None else str(bundle.path),
        entry=BACKBONES.get(str(getattr(bundle, "adapter_name", ""))),
    )


def estimate_scoring_runtime(
    *, n_variants: int, facts: BundleFacts | None, has_gpu: bool
) -> RuntimeEstimate:
    """Cost the scoring run before it starts, from the registry and the wild-type length.

    A backbone the registry cannot cost gets no number at all rather than an invented one.
    """
    backbone = "" if facts is None else facts.backbone
    length = ASSUMED_LENGTH if facts is None or facts.wt_length <= 0 else facts.wt_length
    band = forward_pass_minutes(
        backbone, n_sequences=max(0, n_variants), length=length, has_gpu=has_gpu
    )
    return RuntimeEstimate(
        n_variants=max(0, n_variants),
        low_minutes=None if band is None else band[0],
        high_minutes=None if band is None else band[1],
        device="GPU" if has_gpu else "CPU",
        size_label=backbone or "no backbone",
        length=length,
    )


def visible_fields(state: PredictState) -> frozenset[str]:
    """The field keys the form should be showing right now."""
    shown = {"bundle_source", "load_bundle_button"}
    if state.bundle_source != "upload_model_bundle_zip":
        shown.add("bundle_filename")
    if state.facts is None:
        return frozenset(shown)
    shown |= {"variants_source", "prepare_button", "rank_by", "top_n_to_show", "score_batch_size", "output_name"}
    if state.variants_source in ("upload_variants_csv", "rows_of_a_library_csv"):
        # Both of these ask the user for a file carrying the bundle's mutated-site columns,
        # so both get the template. `random_combinations` builds its own rows and needs none.
        shown.add("example_button")
    if state.variants_source in ("random_combinations", "rows_of_a_library_csv"):
        shown.add("how_many")
    if state.variants_source == "random_combinations":
        shown.add("random_seed")
    if state.n_variants > 0:
        shown.add("score_button")
    return frozenset(shown)


def missing_columns(facts: BundleFacts | None, columns: Sequence[str]) -> list[str]:
    """The bundle's mutation columns that a variants table does not carry."""
    if facts is None:
        return []
    present = {str(name) for name in columns}
    return [name for name in facts.mutation_columns if name not in present]


def scale_note(facts: BundleFacts | None) -> str:
    """What the predicted numbers are on the scale of."""
    if facts is None:
        return ""
    if facts.has_label_scaler:
        return "the assay's own units — this bundle carries the label scaler it was trained with"
    return (
        "z-scored training units, because this bundle carries no label scaler. The ranking is "
        "meaningful; the numbers are not comparable to your measurements"
    )


def ranking_note(rank_by: str) -> str:
    """What the chosen ranking column means in the lab."""
    if rank_by == "pred_min":
        return (
            "`pred_min` ranks by the **worst** condition — what you want when a hit has to work everywhere."
        )
    return "`pred_mean` ranks by average predicted activity across the conditions."


def notices(state: PredictState, *, estimate: RuntimeEstimate | None = None) -> list[Message]:
    """Every contextual message the current bundle and choices earn."""
    facts = state.facts
    out: list[Message] = []
    if facts is None:
        if state.load_problem:
            out.append(Message("bundle_rejected", "stop", state.load_problem))
        else:
            out.append(
                Message(
                    "no_bundle",
                    "stop",
                    "No bundle loaded yet. Upload `model_bundle.zip` — the archive with the weights in "
                    "it, not the performance archive ColabSeqDisplay.ipynb exports beside it.",
                )
            )
        return out

    if not facts.registered:
        out.append(
            Message(
                "backbone_not_registered",
                "stop",
                f"This bundle names the backbone `{facts.backbone}`, which is not in this colabsd's registry, "
                "so nothing here can rebuild it and it cannot be scored. Upgrading colabsd is the usual fix; "
                f"this one offers {core.offered_backbones_phrase()}.",
            )
        )
    elif facts.needs_structure and not facts.carries_three_di:
        out.append(
            Message(
                "three_di_missing",
                "stop",
                f"`{facts.backbone}` needs a wild-type 3Di string and this bundle's spec carries none, so it "
                "cannot be scored. Re-export it from a run whose spec had `wt_3di` set — the 3Di step of "
                "ColabSeqDisplay.ipynb writes that string.",
            )
        )
    if facts.registered and not facts.offered and facts.runnable:
        out.append(
            Message(
                "backbone_not_offered",
                "info",
                withheld_note(facts.backbone)
                + " ColabSeqDisplay.ipynb offers "
                + core.offered_backbones_phrase()
                + " now. This bundle still loads and "
                "scores exactly as it did; what you cannot do any more is train a new one on this backbone.",
            )
        )
    if facts.registered and not facts.runnable:
        out.append(
            Message(
                "no_adapter",
                "stop",
                withheld_note(facts.backbone)
                + " This bundle cannot be scored: re-training on one of "
                + core.offered_backbones_phrase()
                + " is the only route to a bundle this panel can run.",
            )
        )
    if facts.registered and facts.runnable and not facts.has_t4_estimate:
        out.append(
            Message(
                "no_t4_estimate",
                "warning",
                f"The registry has no T4 figure for `{facts.backbone}`, so nothing here can cost this run. "
                "That figure is a LoRA-training verdict, not an inference one — scoring may well fit where "
                "training does not. If the T4 runs out of memory, switch to an L4 or A100 runtime.",
            )
        )
    if facts.is_provisional:
        out.append(
            Message(
                "provisional",
                "warning",
                "The hyperparameters behind this model were **PROVISIONAL placeholders**, not tuned. Rank "
                "variants with it if you like, but do not quote its numbers as a result.",
            )
        )
    else:
        out.append(
            Message(
                "tuned",
                "info",
                "The hyperparameters behind this model were **tuned** for this backbone.",
            )
        )
    if facts.unlock_count == 0:
        out.append(
            Message(
                "never_unlocked",
                "warning",
                "This bundle's test set was **never unlocked**: no held-out number stands behind these "
                "predictions, only validation scores.",
            )
        )
    elif facts.unlock_count > 1:
        out.append(
            Message(
                "unlocked_repeatedly",
                "warning",
                f"This bundle's test set was unlocked **{facts.unlock_count} times**, so it is no longer "
                "held out. Treat its reported numbers as optimistic.",
            )
        )
    if not facts.has_label_scaler:
        out.append(Message("zscore", "warning", f"Predictions will be on {scale_note(facts)}."))
    if facts.carries_three_di:
        out.append(
            Message(
                "three_di",
                "info",
                "This bundle carries its own wild-type 3Di string, so there is nothing else to upload.",
            )
        )
    if state.variants_source == "random_combinations" and facts.k:
        space = 20**facts.k
        out.append(
            Message(
                "random_space",
                "info",
                f"Random combinations are drawn from the {space:,} possible residue combinations at "
                f"{counted(facts.k, 'site')}; duplicates are dropped, so this is a sample, not a screen.",
            )
        )
    if state.n_variants <= 0 and not state.variant_columns:
        out.append(
            Message(
                "no_variants",
                "stop",
                "No variants chosen yet. Pick where they come from and press **Prepare the variants**: a CSV "
                "with this bundle's columns, the first rows of a library CSV, or random combinations at the "
                "mutated sites.",
            )
        )
    elif state.n_variants <= 0:
        # A table that parsed and holds nothing is not the same as nothing chosen. A header-only
        # CSV used to be reported as "No variants chosen yet", which sends the reader back to a
        # step they already took instead of to the file that is empty.
        out.append(
            Message(
                "empty_variants",
                "stop",
                f"The variants table has its columns -- `{'`, `'.join(state.variant_columns)}` -- and no rows. "
                "A header-only CSV loads without complaint; nothing was scored because there is nothing in it. "
                "Add the variants under that header and press **Prepare the variants** again.",
            )
        )
    if state.n_variants > 0 and missing_columns(facts, state.variant_columns):
        missing = missing_columns(facts, state.variant_columns)
        out.append(
            Message(
                "column_mismatch",
                "stop",
                # The count in front of the noun rather than a `column(s)` after the names: one
                # missing column is the common case and it was being told it had "column(s)".
                f"Your variants table is missing {counted(len(missing), 'column')}: "
                f"`{'`, `'.join(missing)}`. This bundle was trained on "
                f"`{', '.join(facts.mutation_columns)}`; rename your columns to match.",
            )
        )
    cost_already_said = estimate is not None and estimate.n_variants > 0 and not state.has_gpu
    if cost_already_said:
        out.append(
            Message(
                "no_gpu",
                "warning",
                "No GPU in this runtime, so scoring falls back to the CPU. " + estimate.sentence(),
            )
        )
    if estimate is not None and estimate.high_minutes is not None and estimate.high_minutes > COLAB_SESSION_MINUTES:
        # The cost estimate is said once. On a CPU runtime both of these fire, and this one used
        # to parenthesize the same sentence the warning directly above it had just made -- the
        # identical hundred-odd characters, twice, one line apart. When the CPU warning has
        # already given the numbers, point at them instead of repeating them.
        cost = (
            "The estimate above runs past it."
            if cost_already_said
            else estimate.sentence()
        )
        out.append(
            Message(
                "long_run",
                "warning",
                f"This could outlast a Colab session. {cost} Score fewer variants. "
                "There is no progress display, so a long run looks like a frozen cell until it finishes.",
            )
        )
    # Worst first, as every other board on both pages is. The red Stop that blocks Score was
    # last, under three infos and a warning, because this one function appended in narrative
    # order and nothing sorted it. list.sort is stable, so the narrative order survives inside
    # each severity.
    out.sort(key=lambda item: theme.SEVERITY_RANK[item.severity])
    return out


def blocking(found: Sequence[Message]) -> list[Message]:
    """The messages that refuse to let scoring start."""
    return [message for message in found if message.severity == "stop"]


# ----------------------------------------------------------------------------------------
# HTML rendering.
# ----------------------------------------------------------------------------------------

def intro_html() -> str:
    """The heading and the one-paragraph explanation at the top of the wizard."""
    return theme.heading_html("Score variants with a saved model") + theme.note_html(
        "Upload a `model_bundle.zip` and a table of variants; get a ranked prediction table back. The "
        "bundle needs nothing else: it carries its backbone, LoRA weights, head, library spec, mutated "
        "sites and provenance.\n\n"
        f"ColabSeqDisplay.ipynb exports two archives. This step wants `{exports.DEFAULT_BUNDLE_NAME}`; the "
        f"other, `{exports.DEFAULT_ARCHIVE_NAME}`, holds the performance report and the training curves "
        "(`" + "`, `".join(PERFORMANCE_MEMBERS) + "`) and carries no weights."
    )


def _row(label: str, value: str) -> str:
    return (
        f"<tr><td style='padding:2px 12px 2px 0;color:#555;vertical-align:top'>{label}</td>"
        f"<td style='padding:2px 0'>{value}</td></tr>"
    )


def bundle_html(facts: BundleFacts) -> str:
    """What is in this bundle, in full, before anything is scored."""
    # Only the answers worth arguing with are colored; the rest inherits the notebook theme.
    status_color = theme.SEVERITY_COLOR["stop"] if facts.is_provisional else "inherit"
    unlock_color = theme.SEVERITY_COLOR["stop"] if facts.unlock_count == 0 else "inherit"
    readout = (
        f"mean of the embeddings at {counted(facts.n_pooled_positions, 'mutated site')}"
        if facts.n_pooled_positions
        else "no mutated sites recorded"
    )
    parameters = (
        ", ".join(f"{name}={value}" for name, value in facts.hyperparameters)
        if facts.hyperparameters
        else "not recorded"
    )
    # `Bundle.adapter_name` is the registry key; `model_name` is whatever the manifest
    # additionally called it, and is usually the same string.
    alias = f" (recorded as {facts.model_name})" if facts.model_name != facts.backbone else ""
    rows = [
        _row("backbone", f"<b>{facts.backbone}</b>{alias}<br><span style='color:#555'>{facts.backbone_line}</span>"),
        _row("read out at", readout),
        _row(
            "hyperparameters",
            f"<b style='color:{status_color}'>{facts.hyperparameter_status}</b><br>"
            f"<span style='font-family:monospace;font-size:12px'>{parameters}</span>",
        ),
        _row("trained with", f"<span style='font-family:monospace;font-size:12px'>{facts.budget}</span>"),
        _row("epochs trained", facts.epochs_trained),
        _row("test set", f"<b style='color:{unlock_color}'>unlocked {facts.unlock_count}x</b> during training"),
        _row("conditions", ", ".join(facts.conditions) or "none recorded"),
        _row(
            "variant columns",
            ", ".join(f"<code>{name}</code>" for name in facts.mutation_columns)
            + f" &nbsp;→ positions {core.positions_phrase(facts.positions_1based)}"
            + (" &nbsp;(residues written as Asn)" if facts.three_letter else " &nbsp;(residues written as N)"),
        ),
        _row(
            "wild type",
            # Grouped, because the training panel's own note about the same wild type says
            # "1,054 residues" and this row said "1054": one protein read as two.
            counted(facts.wt_length, "residue") + (" · carries a 3Di string" if facts.carries_three_di else ""),
        ),
        _row("prediction scale", scale_note(facts)),
        _row("written", f"{facts.created_utc} by colabsd {facts.colabsd_version} (torch {facts.torch_version})"),
    ]
    if facts.notes:
        rows.append(_row("notes", str(facts.notes)))
    if facts.source_path:
        rows.append(_row("file", f"<code>{facts.source_path}</code>"))
    return (
        "<div style='border:1px solid rgba(128,128,128,0.35);padding:10px 12px;margin:6px 0;line-height:1.5'>"
        "<b>What is in this bundle</b>"
        f"<table style='border-collapse:collapse;margin-top:6px'>{''.join(rows)}</table></div>"
    )


def result_html(*, n_ranked: int, rank_by: str, facts: BundleFacts, top_value: float, median_value: float) -> str:
    """The line that says what was just produced and what its numbers mean."""
    return theme.note_html(
        # The *How many* box is a BoundedIntText with min=1, so "Ranked **1** variants" is one
        # click away here just as it was in the training panel's own scoring outlet.
        f"Ranked **{counted(n_ranked, 'variant')}** by `{rank_by}`. {ranking_note(rank_by)}\n\n"
        f"Top {rank_by} {top_value:.4f}, median {median_value:.4f}, on {scale_note(facts)}.\n\n"
        "These are predictions from a model fitted to one library: they rank variants, they do not "
        "measure them."
    )


# ----------------------------------------------------------------------------------------
# The widget layer.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictRunners:
    """The side effects, injected so the wizard can be driven headlessly in tests."""

    load_bundle: Callable[[Path], Any]
    score_variants: Callable[..., Any]
    read_variant_csv: Callable[[Path], Any]
    one_to_three_letter: Callable[[], dict[str, str]]
    #: `upload(what, announce=None)`. `announce` carries the "this is about to freeze the
    #: panel" sentence, so the code that blocks is the code that prints the explanation.
    upload: Callable[..., Path]
    download: Callable[[Path], None]
    show_table: Callable[[Any], None]


def default_runners() -> PredictRunners:
    """The real implementations, imported on call so importing this module stays cheap."""
    from colabsd.bundle import load_bundle
    from colabsd.data import one_to_three_letter, read_variant_csv
    from colabsd.predict import score_variants

    def upload(what: str, announce: Callable[[str], None] | None = None) -> Path:
        try:
            from google.colab import files  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(core.upload_needs_colab_notice(what)) from None
        # Immediately before the call and never after it: files.upload() holds the kernel until
        # the browser answers. Colab flushes this to the output before the picker opens.
        (announce or print)(upload_wait_text(what))
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError(upload_canceled_text(what))
        return Path(next(iter(uploaded))).resolve()

    def download(path: Path) -> None:
        try:
            from google.colab import files  # noqa: PLC0415
        except ImportError:
            return
        files.download(str(path))

    def show_table(frame: Any) -> None:
        from IPython.display import display

        display(frame)

    return PredictRunners(
        load_bundle=load_bundle,
        score_variants=score_variants,
        read_variant_csv=read_variant_csv,
        one_to_three_letter=one_to_three_letter,
        upload=upload,
        download=download,
        show_table=show_table,
    )


class PredictWizard:
    """The form itself. Every visibility rule and every message comes from the pure layer."""

    def __init__(
        self,
        *,
        work_dir: Path | str,
        has_gpu: bool = False,
        runners: PredictRunners | None = None,
    ) -> None:
        import ipywidgets

        self.work_dir = Path(work_dir)
        self.has_gpu = bool(has_gpu)
        self.runners = runners or default_runners()
        self.bundle: Any = None
        self.facts: BundleFacts | None = None
        self.variants: Any = None
        self.ranked: Any = None
        self.ranked_path: Path | None = None
        self.load_problem: str = ""

        style = {"description_width": "initial"}

        def wide() -> Any:
            # One Layout instance per widget: a shared Layout would hide every widget holding it.
            return ipywidgets.Layout(width="520px")

        self.intro = theme.html(intro_html())
        self.fields: dict[str, Any] = {
            "bundle_source": ipywidgets.RadioButtons(
                options=labeled(BUNDLE_SOURCES, BUNDLE_SOURCE_LABELS),
                value=BUNDLE_SOURCES[0],
                description="Bundle:",
                style=style,
                layout=ipywidgets.Layout(width="max-content"),
            ),
            "bundle_filename": ipywidgets.Text(
                value=exports.DEFAULT_BUNDLE_NAME, description="File name:", style=style, layout=wide()
            ),
            "variants_source": ipywidgets.RadioButtons(
                options=labeled(VARIANT_SOURCES, VARIANT_SOURCE_LABELS),
                value=VARIANT_SOURCES[0],
                description="Variants:",
                style=style,
                layout=ipywidgets.Layout(width="max-content"),
            ),
            "how_many": ipywidgets.BoundedIntText(
                value=200, min=1, max=1_000_000, description="How many:", style=style, layout=wide()
            ),
            "random_seed": ipywidgets.BoundedIntText(
                value=0, min=0, max=10_000, description="Seed:", style=style, layout=wide()
            ),
            "rank_by": ipywidgets.Dropdown(
                options=labeled(RANKINGS, RANKING_LABELS),
                value="pred_mean",
                description="Rank by:",
                style=style,
                layout=wide(),
            ),
            "top_n_to_show": ipywidgets.BoundedIntText(
                value=20, min=1, max=1000, description="Rows to show:", style=style, layout=wide()
            ),
            "score_batch_size": ipywidgets.BoundedIntText(
                value=8, min=1, max=32, description="Batch size:", style=style, layout=wide()
            ),
            "output_name": ipywidgets.Text(
                value=DEFAULT_OUTPUT_NAME, description="Output file:", style=style, layout=wide()
            ),
        }
        self.load_bundle_button = ipywidgets.Button(
            description="Load the bundle", button_style="info", layout=ipywidgets.Layout(width="220px")
        )
        self.prepare_button = ipywidgets.Button(
            description="Prepare the variants", button_style="", layout=ipywidgets.Layout(width="220px")
        )
        self.score_button = ipywidgets.Button(
            description="Score and rank", button_style="success", layout=ipywidgets.Layout(width="220px")
        )
        # The columns a variants CSV needs are the bundle's, not ours to print in a label,
        # so the answer to "what shape is this file" is a file. `visible_fields` keeps it off
        # the page until a bundle has been read, because until then there is nothing to write.
        self.example_button = ipywidgets.Button(
            description="Download an example variants CSV",
            button_style="",
            layout=ipywidgets.Layout(width="280px"),
        )
        self.fields["load_bundle_button"] = self.load_bundle_button
        self.fields["prepare_button"] = self.prepare_button
        self.fields["score_button"] = self.score_button
        self.fields["example_button"] = self.example_button

        self.bundle_box = theme.html("")
        self.notice_box = theme.html("")
        self.result_box = theme.html("")
        self.log = ipywidgets.Output()

        for key, widget in self.fields.items():
            if hasattr(widget, "observe") and not key.endswith("_button"):
                widget.observe(self._observer(key), names="value")
        core.on_click(self.load_bundle_button, self._guarded(self.on_load_bundle))
        core.on_click(self.prepare_button, self._guarded(self.on_prepare_variants))
        core.on_click(self.score_button, self._guarded(self.on_score))
        core.on_click(self.example_button, self._guarded(self.on_example_variants))
        self.refresh()

    # -- state ---------------------------------------------------------------------------

    def state(self) -> PredictState:
        """Read the widgets into the plain state the pure functions take."""
        values = {key: widget.value for key, widget in self.fields.items() if hasattr(widget, "value")}
        columns: tuple[str, ...] = ()
        n_variants = 0
        if self.variants is not None:
            columns = tuple(str(name) for name in self.variants.columns)
            n_variants = len(self.variants)
        return PredictState(
            **values,
            has_gpu=self.has_gpu,
            facts=self.facts,
            n_variants=n_variants,
            variant_columns=columns,
            load_problem=self.load_problem,
        )

    def shown(self) -> frozenset[str]:
        """The field keys currently displayed, read back off the widgets."""
        return frozenset(
            key for key, widget in self.fields.items() if getattr(widget.layout, "display", None) != "none"
        )

    def log_text(self) -> str:
        """Everything printed into the log area, as one string."""
        parts = []
        for entry in self.log.outputs:
            parts.append(entry.get("text", "") if isinstance(entry, dict) else getattr(entry, "text", ""))
        return "".join(parts)

    def _say(self, message: str) -> None:
        self.log.append_stdout(message + "\n")

    def _guarded(self, action: Callable[[], Any]) -> Callable[[], Any]:
        """Run a button's step, and put whatever it raises in the log instead of nowhere.

        ipywidgets hands an exception raised inside a callback to `IPython.showtraceback()`,
        which reaches no cell in Colab: the button returns, the page does not move, and the
        work is gone with no message. Each step already reports the failures it expects; this
        is the net under the ones it does not, and under Colab's stop button, which raises a
        `KeyboardInterrupt` that is not an `Exception` and so is caught by nothing else.
        """

        def handle() -> Any:
            try:
                return action()
            except KeyboardInterrupt:
                self._say("stopped: you interrupted this step before it finished. Press the button again.")
            except Exception as exc:  # noqa: BLE001 - the message is the product here
                self._say(f"that did not work: {type(exc).__name__}: {exc}")
            self.refresh()
            return None

        return handle

    # -- rendering -----------------------------------------------------------------------

    def _observer(self, key: str) -> Callable[[Any], None]:
        """One observer per field, so a change can drop what it has just invalidated."""

        def observer(_change: Any = None) -> None:
            if key in BUNDLE_INPUTS and self.bundle is not None:
                self.forget_bundle()
            elif key in VARIANT_INPUTS and self.variants is not None:
                self.forget_variants()
            self.refresh()

        return observer

    def _on_change(self, _change: Any = None) -> None:
        self.refresh()

    def forget_variants(self) -> None:
        """Drop the prepared variants once the choices that built them change.

        Without this, switching the source from an uploaded CSV to random combinations leaves
        **Score and rank** scoring the uploaded rows: what is on screen would not be what runs.
        """
        self.variants = None
        self.result_box.value = ""
        self._say("that change invalidated the prepared variants; press Prepare the variants again")

    def forget_bundle(self) -> None:
        """Drop the loaded bundle, and with it everything read out of it."""
        self.bundle = None
        self.facts = None
        self.variants = None
        self.load_problem = ""
        self.result_box.value = ""
        self._say("that change invalidated the loaded bundle; press Load the bundle again")

    def refresh(self) -> None:
        """Re-apply visibility, the bundle description and the notices."""
        state = self.state()
        shown = visible_fields(state)
        for key, widget in self.fields.items():
            set_display(widget, key in shown)
        estimate = estimate_scoring_runtime(
            n_variants=state.n_variants, facts=state.facts, has_gpu=state.has_gpu
        )
        self.bundle_box.value = "" if state.facts is None else bundle_html(state.facts)
        self.notice_box.value = render_messages(notices(state, estimate=estimate))

    def display_box(self) -> Any:
        """The whole form as one `VBox`, built but not shown, so a test can walk its children."""
        import ipywidgets

        order = [
            "bundle_source",
            "bundle_filename",
            "load_bundle_button",
            "variants_source",
            "how_many",
            "random_seed",
            "example_button",
            "prepare_button",
            "rank_by",
            "top_n_to_show",
            "score_batch_size",
            "output_name",
            "score_button",
        ]
        return ipywidgets.VBox(
            [self.intro]
            + [self.fields[key] for key in order[:3]]
            + [self.bundle_box]
            + [self.fields[key] for key in order[3:]]
            + [self.notice_box, self.result_box, self.log]
        )

    def display(self) -> PredictWizard:
        """Show the form. Returns self so a notebook cell can keep the handle."""
        from IPython.display import display

        display(self.display_box())
        return self

    # -- actions -------------------------------------------------------------------------

    def _ask_for_file(self, what: str) -> Path:
        """Say what is being waited on, on screen and in the log, then block on the upload.

        Both surfaces on purpose: the banner is what somebody staring at a dead panel reads,
        the log line is the transcript showing the explanation came first.
        """
        self.notice_box.value = theme.message_html(upload_wait_text(what), "warning") + self.notice_box.value
        return Path(self.runners.upload(what, announce=lambda text: self._say(plain_text(text))))

    def on_example_variants(self, _button: Any = None) -> Path | None:
        """Write a variants template with this bundle's columns and offer it for download."""
        from colabsd import templates

        if self.bundle is None or self.facts is None:
            self._say("load a bundle first: the columns of the template come from it.")
            return None
        try:
            spec = self.bundle.spec
            wt = str(getattr(spec, "wt_sequence", "") or "")
            positions = list(self.facts.positions_1based)
            residues = (
                [wt[position - 1] for position in positions]
                if wt and positions and max(positions) <= len(wt)
                else None
            )
            # The bundle records which notation its library was written in, and the template
            # has to match it or the loader refuses the file it just handed over.
            target = templates.write_variants_template(
                self.work_dir,
                self.facts.mutation_columns,
                wt_residues=residues,
                three_letter=bool(self.facts.three_letter),
            )
        except Exception as exc:  # a template is a convenience; it may not break the panel
            self._say(f"could not write the example: {type(exc).__name__}: {exc}")
            return None
        self._say(f"wrote {target}")
        self.runners.download(target)
        return target

    def on_load_bundle(self, _button: Any = None) -> BundleFacts | None:
        """Load the bundle and put everything it contains on the screen."""
        state = self.state()
        self.load_problem = ""
        try:
            path = (
                self._ask_for_file(f"the `{exports.DEFAULT_BUNDLE_NAME}` written by ColabSeqDisplay.ipynb")
                if state.bundle_source == "upload_model_bundle_zip"
                else self.work_dir / state.bundle_filename.strip()
            )
        except Exception as exc:
            self.bundle, self.facts = None, None
            # `load_problem` becomes a Stop on the notice board, which is HTML; the log line is
            # plain text. Escape the board's copy only -- entities in the log would be printed
            # literally. `load_failure_text` below escapes its own, so each value is escaped
            # exactly once and `notices()` does not escape again.
            self.load_problem = theme.as_text(exc)
            self._say(plain_text(str(exc)))
            self.refresh()
            return None
        try:
            self.bundle = self.runners.load_bundle(Path(path))
            self.facts = describe_bundle(self.bundle)
            self.variants = None
            self._say(f"loaded {path}")
        except Exception as exc:
            self.bundle, self.facts = None, None
            # What the file actually is, rather than which key load_bundle missed.
            self.load_problem = load_failure_text(Path(path), exc)
            self._say("could not load the bundle: " + plain_text(self.load_problem))
        self.refresh()
        return self.facts

    def on_prepare_variants(self, _button: Any = None) -> Any:
        """Build the table of variants to score, from whichever source is chosen."""
        state = self.state()
        if self.facts is None:
            self._say("load a bundle first: the variant columns are read from it.")
            return None
        try:
            self.variants = self._build_variants(state)
            self._say(f"{counted(len(self.variants), 'variant')} ready")
            self.runners.show_table(self.variants.head(10))
        except Exception as exc:
            self.variants = None
            self._say(f"could not prepare the variants: {exc}")
        self.refresh()
        return self.variants

    def _build_variants(self, state: PredictState) -> Any:
        import numpy as np
        import pandas as pd

        facts = state.facts
        assert facts is not None
        columns = list(facts.mutation_columns)
        if state.variants_source == "upload_variants_csv":
            return self.runners.read_variant_csv(self._ask_for_file(f"a variants CSV with the columns {columns}"))
        if state.variants_source == "rows_of_a_library_csv":
            frame = self.runners.read_variant_csv(self._ask_for_file(f"a library CSV with the columns {columns}"))
            return frame.head(max(1, state.how_many)).reset_index(drop=True)
        residues = list(AMINO_ACIDS)
        if facts.three_letter:
            mapping = self.runners.one_to_three_letter()
            residues = [mapping[letter] for letter in residues]
        rng = np.random.default_rng(state.random_seed)
        draw = rng.choice(residues, size=(max(1, state.how_many), max(1, facts.k)))
        return pd.DataFrame(draw, columns=columns).drop_duplicates().reset_index(drop=True)

    def on_score(self, _button: Any = None) -> Any:
        """Score, rank, write the CSV and offer it for download, unless a `stop` notice refuses."""
        state = self.state()
        estimate = estimate_scoring_runtime(
            n_variants=state.n_variants, facts=state.facts, has_gpu=state.has_gpu
        )
        refusals = blocking(notices(state, estimate=estimate))
        if refusals:
            for message in refusals:
                self._say("refused: " + plain_text(message.text))
            return None
        assert state.facts is not None
        self._say(estimate.sentence())
        try:
            ranked = self.runners.score_variants(
                self.bundle,
                self.variants,
                device="cuda" if self.has_gpu else "cpu",
                batch_size=state.score_batch_size,
            )
            ranked = ranked.sort_values(state.rank_by, ascending=False).reset_index(drop=True)
            ranked.insert(0, "rank", ranked.index + 1)
        except Exception as exc:
            self._say(f"scoring failed: {exc}")
            return None
        # Everything from here down used to run bare, after the model had already run: a name
        # with a folder in it ("results/ranked.csv") made pandas raise where nothing was
        # catching it, and the whole scoring pass was lost without a word. The folder the user
        # named is made rather than refused, and what is left over is reported.
        path = self.work_dir / (state.output_name.strip() or DEFAULT_OUTPUT_NAME)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ranked.to_csv(path, index=False)
        except OSError as exc:
            # Nothing kept: no file, and no table on the page either, so the panel does not
            # hold a result the user cannot see or download.
            self.ranked, self.ranked_path = None, None
            self._say(f"could not write {path}: {type(exc).__name__}: {exc}")
            self._say("nothing was saved. Correct 'Output file:' and press Score and rank again.")
            return None
        self.ranked, self.ranked_path = ranked, path
        self.result_box.value = result_html(
            n_ranked=len(ranked),
            rank_by=state.rank_by,
            facts=state.facts,
            top_value=float(ranked[state.rank_by].iloc[0]),
            median_value=float(ranked[state.rank_by].median()),
        )
        self._say(f"written: {path}")
        self.runners.show_table(ranked.head(max(1, state.top_n_to_show)))
        self.runners.download(path)
        return ranked


def plain_text(text: str) -> str:
    """A message's markdown, flattened for a plain-text log line.

    Entities are turned back into the characters they stand for. A note's text is written for
    the notice board, which is HTML, so a value that carries `<` arrives here already escaped
    by `theme.as_text`; this log is `ipywidgets.Output.append_stdout`, which is plain text and
    would print the entity itself. Escaping belongs on the HTML side of the seam and has to
    come back off on this one.
    """
    flat = _MARKDOWN_LINK.sub(r"\1 (\2)", text)
    return _html.unescape(flat.replace("**", "").replace("`", "").replace("*", ""))


def launch(
    *,
    work_dir: Path | str | None = None,
    has_gpu: bool | None = None,
    runners: PredictRunners | None = None,
) -> PredictWizard:
    """Build the wizard and show it. This is the whole Predict notebook.

    Every argument is optional so `launch()` works from a bare cell, and the GPU is detected
    rather than assumed absent.
    """
    if has_gpu is None:
        has_gpu = core.detect_runtime().has_gpu
    return PredictWizard(
        work_dir=work_dir if work_dir is not None else core.DEFAULT_WORK_DIR,
        has_gpu=has_gpu,
        runners=runners,
    ).display()
