"""Score variants with a saved model bundle: upload, look at what you uploaded, rank, download.

Short and unfussy — no training happens here. The one thing this interface refuses to be
quiet about is the bundle itself. Somebody scoring variants a month after training will not
remember which backbone it was, which pooling it froze, whether the hyperparameters behind
it were tuned or placeholders, or whether its test set was ever unlocked. `describe_bundle`
reads all of that out of the manifest and `bundle_html` puts it on the screen before a
single variant is scored.

Everything said about the backbone comes from `colabsd.backbones.registry` rather than from
the bundle's name: whether it needs a 3Di string (and therefore whether this bundle can be
scored at all), how wide its pooled feature is, whether it fits a free card, and what a run
will cost. A backbone the registry does not know is called that, not guessed at.

As in `colabsd.ui.prepare_workflow`, every decision is a pure function of a `PredictState`
(plus the `BundleFacts` read off the bundle): `visible_fields`, `notices`,
`estimate_scoring_runtime`, `missing_columns`, `scale_note`. `PredictWizard` only wires
those to ipywidgets with `observe()`/`on_click()`, which is all Colab needs — nothing here
blocks a cell waiting for an answer, so `jupyter_ui_poll` would buy nothing. Presentation is
`colabsd.ui.theme` and `colabsd.ui.core.Message`, the vocabulary the main wizard uses.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colabsd.backbones.registry import BACKBONES, BackboneEntry
from colabsd.ui import core, theme
from colabsd.ui.core import (
    COLAB_SESSION_MINUTES,
    Message,
    format_minutes,
    render_messages,
    set_display,
)

# Scoring a variant is the same frozen forward pass region discovery makes, so it is costed
# by the same function, off the same timed run, with the same registry ratios. Importing it
# rather than restating it is the point: a second table of per-backbone rates here is a
# table that would disagree with that one within a release.
from colabsd.ui.prepare_workflow import forward_pass_minutes

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

ASSUMED_LENGTH = 1000
RANKINGS = ("pred_mean", "pred_min")

VARIANT_SOURCES = ("upload_variants_csv", "random_combinations", "rows_of_a_library_csv")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Fields that decide which variants **Prepare the variants** builds. Changing one makes the
#: prepared table describe a different request, so `PredictWizard` throws it away rather
#: than scoring yesterday's rows under today's settings.
VARIANT_INPUTS = frozenset({"variants_source", "how_many", "random_seed"})

#: Fields that decide which bundle **Load the bundle** reads.
BUNDLE_INPUTS = frozenset({"bundle_source", "bundle_filename"})


@dataclass(frozen=True)
class BundleFacts:
    """What a bundle actually contains, read out of its manifest.

    This exists because the answer to "what is this model?" has to survive a month of not
    thinking about it. Everything here comes from the bundle; nothing is inferred from what
    the user typed.
    """

    backbone: str
    model_name: str
    pooling: str
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
    region_name: str | None
    region_source_model: str | None
    region_n_variants_scored: int | None
    hyperparameters: tuple[tuple[str, Any], ...]
    source_path: str | None
    entry: BackboneEntry | None

    @property
    def k(self) -> int:
        """Number of mutated sites."""
        return len(self.mutation_columns)

    @property
    def registered(self) -> bool:
        """True when `colabsd.backbones.registry` knows the backbone this bundle names."""
        return self.entry is not None

    @property
    def needs_structure(self) -> bool:
        """True when the registry says this backbone reads a 3Di string as well as residues."""
        return bool(self.entry and self.entry.needs_structure)

    @property
    def fits_a_free_card(self) -> bool:
        """True when the registry has a T4 estimate, which is how it says 'this fits'."""
        return bool(self.entry and self.entry.approx_lora_minutes_t4 is not None)

    @property
    def backbone_line(self) -> str:
        """The registry's one-line description of the backbone, or a note that it has none."""
        if self.entry is None:
            return "not in this colabsd's backbone registry"
        parts = [f"{self.entry.family} family", f"{self.entry.embed_dim}-d pooled feature"]
        if self.entry.hf_id:
            parts.append(self.entry.hf_id)
        return ", ".join(parts)

    @property
    def hyperparameter_status(self) -> str:
        """The one word a reader needs about the numbers behind this model."""
        return "PROVISIONAL placeholders" if self.is_provisional else "tuned"


def span_of(low_minutes: float, high_minutes: float) -> str:
    """A low-to-high duration, collapsed when the whole range is under a minute."""
    if high_minutes < 1.0:
        return "under a minute"
    return f"{format_minutes(low_minutes)} to {format_minutes(high_minutes)}"


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
        about = (
            f"{self.n_variants:,} variants of ~{self.length} residues through {self.size_label} "
            f"on the {self.device}"
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
    bundle_filename: str = "model_bundle.zip"
    variants_source: str = "upload_variants_csv"
    how_many: int = 200
    random_seed: int = 0
    rank_by: str = "pred_mean"
    top_n_to_show: int = 20
    score_batch_size: int = 8
    output_name: str = "ranked_variants.csv"
    has_gpu: bool = False
    facts: BundleFacts | None = None
    n_variants: int = 0
    variant_columns: tuple[str, ...] = ()


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
    region = getattr(bundle, "region", None) or {}
    positions = tuple(int(value) for value in getattr(spec, "positions_1based", ()) or ())
    pooled = getattr(bundle, "pooling_positions_0based", None) or []
    return BundleFacts(
        backbone=str(getattr(bundle, "adapter_name", "unknown")),
        model_name=str(getattr(bundle, "model_name", getattr(bundle, "adapter_name", "unknown"))),
        pooling=str(getattr(bundle, "pooling", "unknown")),
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
        region_name=region.get("region_name"),
        region_source_model=region.get("region_source_model"),
        region_n_variants_scored=region.get("n_variants_scored"),
        hyperparameters=tuple(sorted((manifest.get("hyperparameters", {}) or {}).items())),
        source_path=None if getattr(bundle, "path", None) is None else str(bundle.path),
        entry=BACKBONES.get(str(getattr(bundle, "adapter_name", ""))),
    )


def estimate_scoring_runtime(
    *, n_variants: int, facts: BundleFacts | None, has_gpu: bool
) -> RuntimeEstimate:
    """Cost the scoring run before it starts, from the registry and the wild-type length.

    `prepare_workflow.forward_pass_minutes` owns the arithmetic and the one timed pass it
    is scaled from. A backbone the registry cannot cost gets no number at all rather than
    an invented one.
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
        out.append(
            Message(
                "no_bundle",
                "stop",
                "No bundle loaded yet. Upload the `model_bundle.zip` written by the export step of "
                "ColabSeqDisplay.ipynb — it carries its own backbone, pooling, spec and weights, so nothing "
                "else has to match.",
            )
        )
        return out

    if not facts.registered:
        out.append(
            Message(
                "backbone_not_registered",
                "warning",
                f"This bundle names the backbone `{facts.backbone}`, which is not in this colabsd's registry. "
                "It can still be rebuilt if the bundle records a HuggingFace id, but nothing here can tell you "
                "what it needs or what it will cost. Upgrading colabsd is the usual fix.",
            )
        )
    elif facts.needs_structure and not facts.carries_three_di:
        out.append(
            Message(
                "three_di_missing",
                "stop",
                f"`{facts.backbone}` reads a wild-type 3Di string alongside the residues, and this bundle's "
                "spec carries none — scoring would fail while rebuilding the backbone. Re-export the bundle "
                "from a run whose spec had `wt_3di` set (the Prepare notebook writes that string).",
            )
        )
    if facts.registered and not facts.fits_a_free_card:
        notes = facts.entry.notes if facts.entry else ""
        out.append(
            Message(
                "needs_a_bigger_card",
                "warning",
                f"The registry has no free-T4 estimate for `{facts.backbone}` — {notes} Scoring is lighter than "
                "training, but a T4 may still run out of memory, and nothing here can cost the run.",
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
                "The hyperparameters behind this model were **tuned** for this backbone and pooling pair, "
                "not typed in by hand.",
            )
        )
    if facts.unlock_count == 0:
        out.append(
            Message(
                "never_unlocked",
                "warning",
                "This bundle's test set was **never unlocked**, so no held-out number stands behind these "
                "predictions. They are a ranking from a model whose only reported scores were validation ones.",
            )
        )
    elif facts.unlock_count > 1:
        out.append(
            Message(
                "unlocked_repeatedly",
                "warning",
                f"This bundle's test set was unlocked **{facts.unlock_count} times**. A test set read "
                "repeatedly while things were still being changed is no longer held out: treat its reported "
                "numbers as optimistic.",
            )
        )
    if not facts.has_label_scaler:
        out.append(Message("zscore", "warning", f"Predictions will be on {scale_note(facts)}."))
    if facts.carries_three_di:
        out.append(
            Message(
                "three_di",
                "info",
                "This bundle carries its own wild-type 3Di string, so a structure-aware backbone scores "
                "without you uploading anything else.",
            )
        )
    if facts.pooling.startswith("cosine") and facts.n_pooled_positions:
        out.append(
            Message(
                "frozen_region",
                "info",
                f"Pooling is frozen over the {facts.n_pooled_positions} residue positions recorded at training "
                "time, so a rediscovered region cannot silently change what this model averages.",
            )
        )
    if state.variants_source == "random_combinations" and facts.k:
        space = 20**facts.k
        out.append(
            Message(
                "random_space",
                "info",
                f"Random combinations are drawn from the {space:,} possible residue combinations at "
                f"{facts.k} sites, so duplicates are dropped and you get a sample, not a screen.",
            )
        )
    if state.n_variants <= 0:
        out.append(
            Message(
                "no_variants",
                "stop",
                "No variants chosen yet. Pick where they come from and press **Prepare the variants** — upload "
                "a CSV with this bundle's columns, take the first rows of a library CSV, or draw random "
                "combinations at the mutated sites.",
            )
        )
    if state.n_variants > 0 and missing_columns(facts, state.variant_columns):
        missing = ", ".join(missing_columns(facts, state.variant_columns))
        out.append(
            Message(
                "column_mismatch",
                "stop",
                f"Your variants table has no `{missing}` column(s). This bundle was trained on "
                f"`{', '.join(facts.mutation_columns)}`; rename your columns to match.",
            )
        )
    if estimate is not None and estimate.n_variants > 0 and not state.has_gpu:
        out.append(
            Message(
                "no_gpu",
                "warning",
                "No GPU in this runtime, so scoring falls back to the CPU. " + estimate.sentence(),
            )
        )
    if estimate is not None and estimate.high_minutes is not None and estimate.high_minutes > COLAB_SESSION_MINUTES:
        out.append(
            Message(
                "long_run",
                "warning",
                "This could outlast a Colab session (" + estimate.sentence() + ") Score fewer variants, or "
                "keep the tab open and watch the progress counter for your own rate.",
            )
        )
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
        "Upload a `model_bundle.zip` and a table of variants; get a ranked prediction table back. No training "
        "happens here, and the bundle needs nothing else: it carries its backbone, its LoRA weights, its head, "
        "its library spec, its frozen pooling coordinates and its provenance."
    )


def _row(label: str, value: str) -> str:
    return (
        f"<tr><td style='padding:2px 12px 2px 0;color:#555;vertical-align:top'>{label}</td>"
        f"<td style='padding:2px 0'>{value}</td></tr>"
    )


def bundle_html(facts: BundleFacts) -> str:
    """What is in this bundle, in full, before anything is scored."""
    # Only the answers worth arguing with are coloured; the rest inherits the notebook theme.
    status_color = theme.SEVERITY_COLOR["stop"] if facts.is_provisional else "inherit"
    unlock_color = theme.SEVERITY_COLOR["stop"] if facts.unlock_count == 0 else "inherit"
    pooling = facts.pooling
    if facts.n_pooled_positions:
        pooling += f" over {facts.n_pooled_positions} frozen residue positions"
    if facts.region_name:
        scored = facts.region_n_variants_scored
        pooling += (
            f"<br><span style='color:#555'>region <code>{facts.region_name}</code>"
            + (f", found with {facts.region_source_model}" if facts.region_source_model else "")
            + (f" over {scored:,} variants" if isinstance(scored, int) else "")
            + "</span>"
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
        _row("pooling", pooling),
        _row(
            "hyperparameters",
            f"<b style='color:{status_color}'>{facts.hyperparameter_status}</b><br>"
            f"<span style='font-family:monospace;font-size:12px'>{parameters}</span>",
        ),
        _row("test set", f"<b style='color:{unlock_color}'>unlocked {facts.unlock_count}x</b> during training"),
        _row("conditions", ", ".join(facts.conditions) or "none recorded"),
        _row(
            "variant columns",
            ", ".join(f"<code>{name}</code>" for name in facts.mutation_columns)
            + f" &nbsp;→ positions {list(facts.positions_1based)}"
            + (" &nbsp;(residues written as Asn)" if facts.three_letter else " &nbsp;(residues written as N)"),
        ),
        _row(
            "wild type",
            f"{facts.wt_length} residues" + (" · carries a 3Di string" if facts.carries_three_di else ""),
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
        f"Ranked **{n_ranked:,}** variants by `{rank_by}`. {ranking_note(rank_by)}\n\n"
        f"Top {rank_by} {top_value:.4f}, median {median_value:.4f}, on {scale_note(facts)}.\n\n"
        "These are predictions from a model fitted to one library. They rank variants; they do not measure them."
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
    upload: Callable[[str], Path]
    download: Callable[[Path], None]
    show_table: Callable[[Any], None]


def default_runners() -> PredictRunners:
    """The real implementations, imported on call so importing this module stays cheap."""
    from colabsd.baseline import one_to_three_letter
    from colabsd.bundle import load_bundle
    from colabsd.data import read_variant_csv
    from colabsd.predict import score_variants

    def upload(what: str) -> Path:
        try:
            from google.colab import files  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                f"Uploading needs Colab. Outside it, put {what} in the working folder and name it in the form."
            ) from None
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError(f"No file was uploaded for {what}.")
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

        style = {"description_width": "initial"}

        def wide() -> Any:
            # One Layout instance per widget: a shared Layout would hide every widget holding it.
            return ipywidgets.Layout(width="520px")

        self.intro = theme.html(intro_html())
        self.fields: dict[str, Any] = {
            "bundle_source": ipywidgets.RadioButtons(
                options=["upload_model_bundle_zip", "file_in_the_working_folder"],
                value="upload_model_bundle_zip",
                description="Bundle:",
                style=style,
                layout=ipywidgets.Layout(width="max-content"),
            ),
            "bundle_filename": ipywidgets.Text(
                value="model_bundle.zip", description="File name:", style=style, layout=wide()
            ),
            "variants_source": ipywidgets.RadioButtons(
                options=list(VARIANT_SOURCES),
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
                options=list(RANKINGS), value="pred_mean", description="Rank by:", style=style, layout=wide()
            ),
            "top_n_to_show": ipywidgets.BoundedIntText(
                value=20, min=1, max=1000, description="Rows to show:", style=style, layout=wide()
            ),
            "score_batch_size": ipywidgets.BoundedIntText(
                value=8, min=1, max=32, description="Batch size:", style=style, layout=wide()
            ),
            "output_name": ipywidgets.Text(
                value="ranked_variants.csv", description="Output file:", style=style, layout=wide()
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
        self.fields["load_bundle_button"] = self.load_bundle_button
        self.fields["prepare_button"] = self.prepare_button
        self.fields["score_button"] = self.score_button

        self.bundle_box = theme.html("")
        self.notice_box = theme.html("")
        self.result_box = theme.html("")
        self.log = ipywidgets.Output()

        for key, widget in self.fields.items():
            if hasattr(widget, "observe") and not key.endswith("_button"):
                widget.observe(self._observer(key), names="value")
        self.load_bundle_button.on_click(self.on_load_bundle)
        self.prepare_button.on_click(self.on_prepare_variants)
        self.score_button.on_click(self.on_score)
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

        Without this, switching the source from an uploaded CSV to random combinations
        leaves **Score and rank** on screen and scores the uploaded rows: the table the
        user is looking at would not be the table that gets scored.
        """
        self.variants = None
        self.result_box.value = ""
        self._say("that change invalidated the prepared variants; press Prepare the variants again")

    def forget_bundle(self) -> None:
        """Drop the loaded bundle, and with it everything read out of it."""
        self.bundle = None
        self.facts = None
        self.variants = None
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
        """The whole form as one `VBox`, built but not shown — see `PrepareWizard.display_box`."""
        import ipywidgets

        order = [
            "bundle_source",
            "bundle_filename",
            "load_bundle_button",
            "variants_source",
            "how_many",
            "random_seed",
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

    def on_load_bundle(self, _button: Any = None) -> BundleFacts | None:
        """Load the bundle and put everything it contains on the screen."""
        state = self.state()
        try:
            path = (
                self.runners.upload("model_bundle.zip")
                if state.bundle_source == "upload_model_bundle_zip"
                else self.work_dir / state.bundle_filename.strip()
            )
            self.bundle = self.runners.load_bundle(Path(path))
            self.facts = describe_bundle(self.bundle)
            self.variants = None
            self._say(f"loaded {path}")
        except Exception as exc:
            self.bundle, self.facts = None, None
            self._say(f"could not load the bundle: {exc}")
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
            self._say(f"{len(self.variants)} variants ready")
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
            return self.runners.read_variant_csv(self.runners.upload(f"a variants CSV with the columns {columns}"))
        if state.variants_source == "rows_of_a_library_csv":
            frame = self.runners.read_variant_csv(self.runners.upload(f"a library CSV with the columns {columns}"))
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
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = self.work_dir / (state.output_name.strip() or "ranked_variants.csv")
        ranked.to_csv(path, index=False)
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
    """A message's markdown, flattened for a plain-text log line."""
    flat = _MARKDOWN_LINK.sub(r"\1 (\2)", text)
    return flat.replace("**", "").replace("`", "").replace("*", "")


def launch(
    *,
    work_dir: Path | str | None = None,
    has_gpu: bool | None = None,
    runners: PredictRunners | None = None,
) -> PredictWizard:
    """Build the wizard and show it. This is the whole Predict notebook.

    Every argument is optional so `launch()` works from a bare notebook cell, and the GPU is
    detected rather than assumed absent.
    """
    if has_gpu is None:
        has_gpu = core.detect_runtime().has_gpu
    return PredictWizard(
        work_dir=work_dir if work_dir is not None else core.DEFAULT_WORK_DIR,
        has_gpu=has_gpu,
        runners=runners,
    ).display()
