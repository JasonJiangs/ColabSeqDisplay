"""The "run once per new protein" wizard: a wild-type 3Di string, and a pooling region.

Two independent jobs. Producing a 3Di string matters only if you will pick a SaProt
backbone; discovering a pooling region matters only if you will pick `cosine_p90_mean`
pooling. Tick either, both, or neither.

Every rule the interface enforces lives in a pure function here — `visible_fields`,
`notices`, `subsample_preview`, `estimate_region_runtime`, `output_files` — each taking a
`PrepareState` and returning plain data. `PrepareWizard` is the thin ipywidgets layer that
calls them: it owns no decision of its own, so every decision is unit-testable without a
browser, a kernel or a GPU.

Region discovery is the expensive job, so its subsampling is shown before it starts (how
many variants, which rows, which seed) together with an order-of-magnitude runtime
estimate, and the wizard says what each written file is for and where it has to go.

That estimate is not invented here. It is scaled off the one forward pass of this shape
that has actually been timed (recorded in `TIMED_PASS` below) by the per-backbone figures
in `colabsd.backbones.registry`, so a backbone added to the registry is costed by the
registry, and a backbone the registry cannot cost is reported as uncosted rather than
guessed at. `predict_workflow` scores variants with the same function, because scoring a
variant and scoring it for region discovery are the same frozen forward pass.

Presentation is `colabsd.ui.theme` and `colabsd.ui.core.Message`, the same vocabulary the
main wizard uses, so the four interfaces read as one program. Plain `observe()`/`on_click()`
is enough here: Colab keeps widget callbacks alive after the cell finishes, and nothing in
this wizard has to block a cell waiting for an answer, which is the one thing
`jupyter_ui_poll` buys.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colabsd.backbones.registry import BACKBONES
from colabsd.ui import core, theme
from colabsd.ui.core import (
    COLAB_SESSION_MINUTES,
    Message,
    esmfold_safe_length,
    format_minutes,
    render_messages,
    set_display,
)

ASSUMED_LENGTH = 1000
SMALL_CARD_GB = 20.0

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

JOB_THREE_DI = "three_di"
JOB_REGION = "region"

THREE_DI_SOURCES = (
    "upload_structure_pdb_or_cif",
    "upload_3di_text_file",
    "bundled_example_3di",
    "fold_the_wild_type_with_esmfold",
)

#: `colabsd.region.build_region_record` selects at or above this percentile, and refuses
#: anything outside `[0, 100)`. The form has to refuse the same range *before* the forward
#: pass, because discover_region scores every sampled variant first and only then builds
#: the records — an out-of-range percentile would throw away the whole run.
PERCENTILE_RANGE = (0.0, 100.0)


# ----------------------------------------------------------------------------------------
# What one frozen forward pass costs. Every number here is a repository fact, and the two
# that are not measurements say so. `predict_workflow` imports this section: scoring a
# variant and scoring it for region discovery are the same forward pass through the same
# frozen backbone, so there is exactly one cost model for both.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedPass:
    """The one forward pass over a real library that this repository has actually timed."""

    backbone: str
    n_sequences: int
    length: int
    seconds: float
    source: str


#: ESM2-650M over all 16,424 SlugCas9 variants of 1054 residues, float16 — the dtype
#: `colabsd.region` uses on a GPU — in 319 s. It is the only forward pass of this shape
#: that has been stopwatched, and it was run on a B200, not in Colab.
TIMED_PASS = TimedPass(
    backbone="ESM2-650M",
    n_sequences=16_424,
    length=1054,
    seconds=319.0,
    source="one timed float16 region-discovery pass over the bundled library, on a B200",
)

#: How much slower a Colab GPU is than the B200 that pass was timed on. No Colab session
#: has ever been timed (`README.md`, Reproducibility), so this is a deliberately wide band
#: and the only invented number in the estimate; the progress counter reports the real rate.
GPU_SLOWDOWN: tuple[float, float] = (5.0, 30.0)

#: And a CPU is tens of times slower again than that T4.
CPU_SLOWDOWN: tuple[float, float] = (200.0, 1200.0)

#: Region discovery runs `transformers.AutoModel` and reads `last_hidden_state`, one vector
#: per token, so only the ESM2 family is offered — `colabsd.region.cosine_variability_scores`
#: says so in as many words.
REGION_FAMILY = "ESM2"


def _anchor_minutes() -> int:
    entry = BACKBONES[TIMED_PASS.backbone]
    assert entry.approx_lora_minutes_t4 is not None
    return entry.approx_lora_minutes_t4


def relative_cost(backbone: str) -> float | None:
    """How expensive one forward pass through *backbone* is, relative to the timed one.

    The ratio of the registry's own `approx_lora_minutes_t4` figures, so a backbone added to
    `colabsd.backbones.registry` is costed from the registry's estimate rather than from a
    table here that would drift away from it. `None` when the registry has no T4 estimate,
    which is its way of saying the backbone does not fit a free card at all.
    """
    entry = BACKBONES.get(backbone)
    if entry is None or entry.approx_lora_minutes_t4 is None:
        return None
    return entry.approx_lora_minutes_t4 / float(_anchor_minutes())


def forward_pass_minutes(
    backbone: str, *, n_sequences: int, length: int, has_gpu: bool
) -> tuple[float, float] | None:
    """Minutes for `n_sequences` frozen forward passes of `length` residues, low to high.

    Linear in both, scaled off `TIMED_PASS` by the registry's relative cost and by the
    device band. `None` when the registry cannot cost this backbone, which is a better
    answer than a number nobody can defend.
    """
    factor = relative_cost(backbone)
    if factor is None:
        return None
    per_sequence = TIMED_PASS.seconds / TIMED_PASS.n_sequences / TIMED_PASS.length
    seconds = per_sequence * factor * max(0, n_sequences) * max(0, length)
    low, high = GPU_SLOWDOWN if has_gpu else CPU_SLOWDOWN
    return seconds * low / 60.0, seconds * high / 60.0


@dataclass(frozen=True)
class RegionModel:
    """One choice in the region-model dropdown, straight out of the backbone registry."""

    hf_id: str
    label: str

    @property
    def backbone(self) -> str:
        """The registry name, which is also what the dropdown shows."""
        return self.label


def _region_models() -> tuple[RegionModel, ...]:
    """The ESM2 backbones region discovery can load, the registry's most expensive first."""
    entries = [
        (entry.approx_lora_minutes_t4 or 0, name, entry)
        for name, entry in BACKBONES.items()
        if entry.family == REGION_FAMILY and entry.tier != "local_only" and entry.hf_id
    ]
    return tuple(RegionModel(entry.hf_id, name) for _minutes, name, entry in sorted(entries, reverse=True))


REGION_MODELS: tuple[RegionModel, ...] = _region_models()

#: `colabsd.region.DEFAULT_REGION_MODEL`, taken from the registry so the two cannot drift.
DEFAULT_REGION_MODEL = BACKBONES[TIMED_PASS.backbone].hf_id

#: The bundled library, found through `colabsd.EXAMPLES_ROOT` the way `colabsd.ui.core` finds it.
EXAMPLE_DIRNAME = "slugcas9_5nnk"


def default_example_dir() -> Path | None:
    """Where the bundled SlugCas9 example lives, or None when this install has no copy.

    The notebook cell calls `launch(work_dir=..., has_gpu=...)` and passes no `example_dir`,
    so without this the default choice on the form — the bundled example — would fail on the
    user's first click. Found the way `colabsd.ui.core.reference_library_variants` finds it.
    """
    from colabsd import EXAMPLES_ROOT

    candidate = EXAMPLES_ROOT / EXAMPLE_DIRNAME
    return candidate if (candidate / "wt.fasta").is_file() else None


@dataclass(frozen=True)
class SubsamplePreview:
    """Exactly what region discovery would score, worked out before it starts."""

    n_library: int
    n_scored: int
    seed: int
    first_rows: tuple[int, ...]
    is_full_library: bool
    library_loaded: bool


def span_of(low_minutes: float, high_minutes: float) -> str:
    """A low-to-high duration, collapsed when the whole range is under a minute."""
    if high_minutes < 1.0:
        return "under a minute"
    return f"{format_minutes(low_minutes)} to {format_minutes(high_minutes)}"


@dataclass(frozen=True)
class RuntimeEstimate:
    """An order-of-magnitude cost for region discovery, in minutes.

    `low_minutes` and `high_minutes` are `None` when the backbone registry has no T4
    estimate to scale from, which is the honest answer for a backbone nobody has costed.
    """

    n_scored: int
    low_minutes: float | None
    high_minutes: float | None
    device: str
    model_label: str
    length: int
    length_is_assumed: bool

    @property
    def known(self) -> bool:
        return self.high_minutes is not None

    @property
    def is_long(self) -> bool:
        """True when the slow end of the estimate crosses the session-disconnect threshold."""
        return self.high_minutes is not None and self.high_minutes > COLAB_SESSION_MINUTES

    def sentence(self) -> str:
        """One line a biologist can act on."""
        if self.n_scored <= 0:
            return "Nothing to score yet."
        about = f"{self.model_label} over {self.n_scored:,} variants of ~{self.length} residues on the {self.device}"
        tail = " (assuming a 1000-residue protein until you load one)" if self.length_is_assumed else ""
        if self.low_minutes is None or self.high_minutes is None:
            return (
                f"Rough cost: unknown for {about}{tail} — the backbone registry has no T4 estimate for "
                f"{self.model_label} to scale from."
            )
        span = span_of(self.low_minutes, self.high_minutes)
        return f"Rough cost: {span} for {about}{tail}."


@dataclass(frozen=True)
class OutputFile:
    """A file this wizard writes, what needs it, and where it goes next."""

    name: str
    needed_by: str
    where_it_goes: str


@dataclass
class PrepareState:
    """Everything the two jobs are decided from. No widgets, no side effects."""

    make_three_di: bool = True
    make_region: bool = False
    protein_source: str = "bundled_example_SlugCas9_5NNK"
    wt_sequence_or_fasta_url: str = ""
    three_di_source: str = "upload_structure_pdb_or_cif"
    chain: str = ""
    esmfold_risk_accepted: bool = False
    three_di_name: str = "wt_3di.txt"
    positions_1based: str = "984, 985, 990, 1012, 1016"
    mutation_columns: str = "nnk1, nnk2, nnk3, nnk4, nnk5"
    condition_columns: str = "NNGA, NNGT, NNGC, NNGG"
    three_letter_residues: bool = True
    count_column: str = "count"
    min_count: int = 0
    region_model: str = DEFAULT_REGION_MODEL
    n_sample: int = 2000
    seed: int = 0
    batch_size: int = 2
    percentiles: str = "90, 95"
    run_on_cpu_anyway: bool = False
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_gb: float = 0.0
    n_library: int = 0
    wt_length: int = 0
    example_available: bool = True

    @property
    def using_example(self) -> bool:
        return self.protein_source.startswith("bundled_example")

    @property
    def library_loaded(self) -> bool:
        return self.n_library > 0

    @property
    def wt_loaded(self) -> bool:
        """True once **Load the protein** has produced a sequence for the choices on screen."""
        return self.wt_length > 0


#: Fields that decide *what* **Load the protein** reads. Changing any of them makes the
#: loaded wild type and library describe something other than what the form now says, so
#: `PrepareWizard.forget_load` drops them and the refusals come back.
LOAD_INPUTS = frozenset(
    {
        "protein_source",
        "wt_sequence_or_fasta_url",
        "positions_1based",
        "mutation_columns",
        "condition_columns",
        "three_letter_residues",
        "count_column",
        "min_count",
    }
)


def parse_list(text: str) -> list[str]:
    """Split a comma- or semicolon-separated field into stripped, non-empty items."""
    return [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]


def selected_jobs(state: PrepareState) -> tuple[str, ...]:
    """Which of the two jobs are ticked, in the order they would run."""
    jobs = []
    if state.make_three_di:
        jobs.append(JOB_THREE_DI)
    if state.make_region:
        jobs.append(JOB_REGION)
    return tuple(jobs)


def region_model_for(hf_id: str) -> RegionModel:
    """The registry entry behind *hf_id*, or a nameless stand-in for one it does not know."""
    for model in REGION_MODELS:
        if model.hf_id == hf_id:
            return model
    return RegionModel(hf_id=hf_id, label=hf_id)


def visible_fields(state: PrepareState) -> frozenset[str]:
    """The field keys the form should be showing right now.

    Nothing but the two job checkboxes is shown until a job is ticked, and a field only
    appears once the choice above it has made it relevant.
    """
    jobs = selected_jobs(state)
    always = {"make_three_di", "make_region"}
    if not jobs:
        return frozenset(always)

    shown = always | {"protein_source", "load_button", "run_button"}
    if not state.using_example:
        shown.add("wt_sequence_or_fasta_url")

    if JOB_THREE_DI in jobs:
        shown |= {"three_di_source", "three_di_name"}
        if state.three_di_source == "upload_structure_pdb_or_cif":
            shown.add("chain")
        if state.three_di_source == "fold_the_wild_type_with_esmfold":
            shown.add("esmfold_risk_accepted")

    if JOB_REGION in jobs:
        shown |= {"region_model", "n_sample", "seed", "batch_size", "percentiles"}
        if not state.has_gpu:
            shown.add("run_on_cpu_anyway")
        if not state.using_example:
            shown |= {
                "positions_1based",
                "mutation_columns",
                "condition_columns",
                "three_letter_residues",
                "count_column",
                "min_count",
            }
    return frozenset(shown)


def needs_library(state: PrepareState) -> bool:
    """Region discovery scores your variants, so it needs the library; 3Di needs only the WT."""
    return JOB_REGION in selected_jobs(state)


def subsample_preview(state: PrepareState, *, subsample: Callable[..., Any] | None = None) -> SubsamplePreview:
    """Work out exactly which library rows region discovery would score.

    Uses `colabsd.region.subsample`, the same function the run itself calls, so the preview
    cannot drift from what happens. Before the library is loaded it reports the typed
    `n_sample` and says the library is not loaded yet.
    """
    if not state.library_loaded:
        n_scored = state.n_library if state.n_sample <= 0 else state.n_sample
        return SubsamplePreview(
            n_library=state.n_library,
            n_scored=max(0, n_scored),
            seed=state.seed,
            first_rows=(),
            is_full_library=state.n_sample <= 0,
            library_loaded=False,
        )
    if subsample is None:
        from colabsd.region import subsample as subsample_fn
    else:
        subsample_fn = subsample
    rows = [int(row) for row in subsample_fn(state.n_library, state.n_sample, seed=state.seed)]
    return SubsamplePreview(
        n_library=state.n_library,
        n_scored=len(rows),
        seed=state.seed,
        first_rows=tuple(rows[:12]),
        is_full_library=len(rows) >= state.n_library,
        library_loaded=True,
    )


def estimate_region_runtime(
    *,
    n_scored: int,
    wt_length: int,
    region_model: str,
    has_gpu: bool,
) -> RuntimeEstimate:
    """Cost region discovery before the user commits to it.

    `forward_pass_minutes` does the arithmetic: the one timed pass this repository owns,
    scaled by the registry's relative cost for the chosen model and by the device band. It
    is an order of magnitude, not a measurement — the progress counter during the run
    reports the real rate within the first minute.
    """
    model = region_model_for(region_model)
    length_is_assumed = wt_length <= 0
    length = ASSUMED_LENGTH if length_is_assumed else wt_length
    band = forward_pass_minutes(
        model.backbone, n_sequences=max(0, n_scored), length=length, has_gpu=has_gpu
    )
    return RuntimeEstimate(
        n_scored=max(0, n_scored),
        low_minutes=None if band is None else band[0],
        high_minutes=None if band is None else band[1],
        device="GPU" if has_gpu else "CPU",
        model_label=model.label,
        length=length,
        length_is_assumed=length_is_assumed,
    )


def notices(
    state: PrepareState,
    *,
    preview: SubsamplePreview | None = None,
    estimate: RuntimeEstimate | None = None,
) -> list[Message]:
    """Every contextual message the current choices earn, in reading order.

    A `stop` message is a refusal: `blocking(...)` collects them and the run does not start
    while any is present.
    """
    jobs = selected_jobs(state)
    out: list[Message] = []
    if jobs and state.using_example and not state.example_available:
        out.append(
            Message(
                "no_bundled_example",
                "stop",
                "This install has no copy of the bundled SlugCas9 example, so there is nothing to load. Pick "
                "**my_own_protein** and paste your wild-type sequence, or re-clone the repository, which is "
                "what carries `examples/slugcas9_5nnk/`.",
            )
        )
    if jobs and not state.wt_loaded:
        out.append(
            Message(
                "wt_not_loaded",
                "stop",
                "The wild type has not been read yet, and both jobs start from it: a 3Di string is checked "
                "against its length, and region discovery scores variants of it. Click **Load the protein** "
                "above.",
            )
        )
    if not jobs:
        out.append(
            Message(
                "no_job",
                "info",
                "Nothing is ticked, so this notebook has nothing to do. That is a real answer: an ESM2 backbone "
                "with `mutation_site_mean` pooling needs neither file, and you can go straight to "
                "ColabSeqDisplay.ipynb.",
            )
        )
        return out

    if JOB_THREE_DI in jobs:
        out.append(
            Message(
                "three_di_purpose",
                "info",
                "A 3Di string is needed by **SaProt backbones only**. SaProt reads amino acids interleaved with "
                "Foldseek structure states, one state per residue.",
            )
        )
        if state.three_di_source == "bundled_example_3di" and not state.using_example:
            out.append(
                Message(
                    "example_3di_mismatch",
                    "stop",
                    "The bundled 3Di string describes SlugCas9, not the protein you named. Upload a structure of "
                    "your own wild type, or a 3Di file you already have.",
                )
            )
        if state.three_di_source == "fold_the_wild_type_with_esmfold":
            out.append(
                Message(
                    "esmfold_last_resort",
                    "warning",
                    "ESMFold is the last resort, not the default. Its pair representation holds one value per "
                    "residue *pair*, so memory grows with the square of the length. An AlphaFold `.cif` from "
                    "[alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk) is free, faster and more accurate — try "
                    "that first.",
                )
            )
            if not state.has_gpu:
                out.append(
                    Message(
                        "esmfold_needs_gpu",
                        "stop",
                        "ESMFold needs a GPU. Runtime > Change runtime type > T4 GPU, or upload a structure "
                        "instead.",
                    )
                )
            if not state.esmfold_risk_accepted:
                out.append(
                    Message(
                        "esmfold_not_accepted",
                        "stop",
                        "Tick **I accept the ESMFold memory risk** to run it anyway, or pick "
                        "`upload_structure_pdb_or_cif`.",
                    )
                )
            safe = esmfold_safe_length()
            if state.wt_length > safe:
                card = f"a {state.gpu_gb:g} GB card" if state.gpu_gb > 0 else "a card whose memory is not known here"
                crowded = state.gpu_gb <= 0 or state.gpu_gb < SMALL_CARD_GB
                out.append(
                    Message(
                        "esmfold_memory",
                        "warning" if crowded else "info",
                        f"{state.wt_length} residues on {card}. `colabsd.structure` puts the free-T4 ceiling at "
                        f"{safe} residues, past which ESMFold usually runs out of memory: expect a crash on a "
                        "16 GB T4, and switch to an AlphaFold `.cif` rather than paying for an L4 to fold it.",
                    )
                )

    if JOB_REGION in jobs:
        out.append(
            Message(
                "region_purpose",
                "info",
                "A region file is read by `cosine_p90_mean` pooling only. It records the residues whose "
                "embeddings disagree most across your variants — the model's own answer to which part of the "
                "protein your assay moves.",
            )
        )
        if not state.library_loaded:
            out.append(
                Message(
                    "region_needs_library",
                    "stop",
                    "Region discovery scores **your variants**, not just the wild type. Click **Load the "
                    "protein** above so the library is read and the subsample below is real.",
                )
            )
        if not state.has_gpu and not state.run_on_cpu_anyway:
            out.append(
                Message(
                    "region_needs_gpu",
                    "stop",
                    "Region discovery runs a language model over full-length sequences, which needs a GPU. "
                    "Runtime > Change runtime type > T4 GPU, or tick **run on CPU anyway** and wait hours.",
                )
            )
        elif not state.has_gpu and state.run_on_cpu_anyway:
            out.append(
                Message(
                    "region_cpu_override",
                    "warning",
                    "Running on the CPU. This is tens of times slower than a GPU; the estimate below already "
                    "accounts for that.",
                )
            )
        if preview is not None and preview.library_loaded:
            if preview.is_full_library:
                out.append(
                    Message(
                        "region_full_library",
                        "info",
                        f"Every one of the {preview.n_library:,} variants is scored, so the region does not "
                        "depend on the seed.",
                    )
                )
            else:
                out.append(
                    Message(
                        "region_subset_only",
                        "warning",
                        f"The region will be discovered from **{preview.n_scored:,} of {preview.n_library:,} "
                        f"variants** — this subset only, drawn with seed {preview.seed}. Another seed picks "
                        "another subset and therefore a slightly different region. Set `n_sample = 0` to score "
                        "every variant and remove the question.",
                    )
                )
        if estimate is not None and estimate.is_long:
            out.append(
                Message(
                    "region_long_run",
                    "warning",
                    "Your Colab session may disconnect during a long run (this one could exceed two hours). Keep "
                    "the tab open, or cut the cost: a smaller region model, or a smaller `n_sample`, is the "
                    "quickest way.",
                )
            )
        if state.batch_size >= 8 and (state.gpu_gb <= 0 or state.gpu_gb < SMALL_CARD_GB):
            card = f"a {state.gpu_gb:g} GB card" if state.gpu_gb > 0 else "a card whose memory is not known here"
            out.append(
                Message(
                    "region_batch_memory",
                    "warning",
                    f"Batch {state.batch_size} on {card} with full-length sequences has not been measured here. "
                    "If the run dies with an out-of-memory error, halve the batch size; nothing else needs "
                    "changing and the region is unaffected.",
                )
            )
        typed = parse_list(state.percentiles)
        unparseable = [item for item in typed if _as_percentile(item) is None]
        out_of_range = [
            value for value in parsed_percentiles(state) if not PERCENTILE_RANGE[0] <= value < PERCENTILE_RANGE[1]
        ]
        if not typed:
            out.append(
                Message(
                    "no_percentiles",
                    "stop",
                    "No percentile given, so the run would score every variant and then write nothing. `90` is "
                    "the one the main notebook can use.",
                )
            )
        elif unparseable:
            out.append(
                Message(
                    "unparseable_percentiles",
                    "stop",
                    f"`{', '.join(unparseable)}` is not a number. Percentiles are plain numbers separated by "
                    "commas, like `90, 95`.",
                )
            )
        if out_of_range:
            shown = ", ".join(f"{value:g}" for value in out_of_range)
            out.append(
                Message(
                    "percentile_out_of_range",
                    "stop",
                    f"A percentile must be at least 0 and below 100, so `{shown}` would be refused by "
                    "`colabsd.region` — but only *after* every variant had been scored, throwing the whole run "
                    "away. `90` keeps the top decile of residues.",
                )
            )
        if any(abs(value - 95.0) < 1e-9 for value in parsed_percentiles(state)):
            out.append(
                Message(
                    "p95_unused",
                    "info",
                    "`region_p95.json` will be written, but nothing consumes it yet: the main notebook offers "
                    "`cosine_p90_mean` pooling only, and refuses a region file whose name does not match the "
                    "pooling you picked. Keep p95 for comparison.",
                )
            )
    return out


def _as_percentile(item: str) -> float | None:
    """One typed percentile as a float, or None when it is not a number at all."""
    try:
        return float(item)
    except ValueError:
        return None


def parsed_percentiles(state: PrepareState) -> list[float]:
    """The percentiles as floats, in order, without repeats.

    An unparseable entry is dropped rather than crashing the form — `notices` refuses the
    run over it instead. Repeats are dropped because two identical percentiles produce one
    record, not two, and the file list must say what will actually be written.
    """
    values: list[float] = []
    for item in parse_list(state.percentiles):
        value = _as_percentile(item)
        if value is not None and value not in values:
            values.append(value)
    return values


def blocking(found: Sequence[Message]) -> list[Message]:
    """The messages that refuse to let the run start."""
    return [message for message in found if message.severity == "stop"]


def region_file_name(percentile: float) -> str:
    """The file a percentile produces.

    `colabsd.region.build_region_record` names its record `cosine_p{int(percentile)}`, so
    92.5 and 92 write the same file. The form has to promise the name the run will actually
    write, not a prettier one.
    """
    return f"region_p{int(percentile)}.json"


def output_files(state: PrepareState) -> list[OutputFile]:
    """What this run will write, what needs each file, and where it has to go."""
    files: list[OutputFile] = []
    jobs = selected_jobs(state)
    if JOB_THREE_DI in jobs:
        files.append(
            OutputFile(
                name=state.three_di_name.strip() or "wt_3di.txt",
                needed_by="a SaProt backbone, and nothing else",
                where_it_goes="upload it at the wild-type 3Di step of ColabSeqDisplay.ipynb",
            )
        )
    if JOB_REGION in jobs:
        for value in parsed_percentiles(state):
            if not PERCENTILE_RANGE[0] <= value < PERCENTILE_RANGE[1]:
                continue  # `notices` refuses the run over it; do not promise a file for it
            name = region_file_name(value)
            if abs(value - 90.0) < 1e-9:
                files.append(
                    OutputFile(
                        name=name,
                        needed_by="cosine_p90_mean pooling, and nothing else",
                        where_it_goes="upload it at the backbone-and-pooling step of ColabSeqDisplay.ipynb",
                    )
                )
            else:
                files.append(
                    OutputFile(
                        name=name,
                        needed_by="nothing in the main notebook reads it yet",
                        where_it_goes="keep it beside the p90 file for comparison",
                    )
                )
    return files


# ----------------------------------------------------------------------------------------
# Rendering. Pure string builders on top of `colabsd.ui.theme`, so the prose is testable.
# ----------------------------------------------------------------------------------------


def intro_html() -> str:
    """The heading and the one-paragraph explanation at the top of the wizard."""
    return theme.heading_html("Prepare a new protein") + theme.note_html(
        "Run this once per new protein, before the main notebook. It has two independent jobs, and you may "
        "want neither of them:\n\n"
        "- **A wild-type 3Di string** — needed only if you will pick a **SaProt** backbone.\n"
        "- **A pooling region** — needed only if you will pick **cosine_p90_mean** pooling.\n\n"
        "An ESM2 backbone with `mutation_site_mean` pooling needs neither, and that is the default in the "
        "main notebook."
    )


def preview_html(preview: SubsamplePreview, estimate: RuntimeEstimate) -> str:
    """The subsample and its cost, both visible before the user commits to the run."""
    if not preview.library_loaded:
        head = (
            "**Subsample:** the library has not been read yet, so this is what you typed, not what would be "
            f"scored — `n_sample = {preview.n_scored:,}`, seed {preview.seed}."
        )
        rows = ""
    else:
        head = (
            f"**Subsample:** scoring **{preview.n_scored:,}** of {preview.n_library:,} variants "
            f"(seed {preview.seed})."
        )
        rows = (
            "<div style='font-family:monospace;font-size:12px;margin-top:4px'>first sampled rows: "
            + ", ".join(str(row) for row in preview.first_rows)
            + "</div>"
        )
    body = theme.render_markdown_inline(head)
    tail = theme.render_markdown_inline(
        f"**Before you commit:** {estimate.sentence()} That is an order of magnitude, not a measurement — the "
        "progress counter tells you your own rate within the first minute."
    )
    return (
        "<div style='border:1px solid rgba(128,128,128,0.35);padding:8px 10px;margin:6px 0;line-height:1.5'>"
        f"{body}{rows}<div style='margin-top:6px'>{tail}</div></div>"
    )


def outputs_html(files: Sequence[OutputFile]) -> str:
    """What each written file is for and where it has to go next."""
    if not files:
        return ""
    rows = "".join(
        "<tr>"
        f"<td style='padding:3px 10px 3px 0'><code>{item.name}</code></td>"
        f"<td style='padding:3px 10px 3px 0'>{item.needed_by}</td>"
        f"<td style='padding:3px 0'>{item.where_it_goes}</td>"
        "</tr>"
        for item in files
    )
    return (
        "<div style='margin:6px 0;line-height:1.5'><b>This run will write:</b>"
        "<table style='border-collapse:collapse;margin-top:4px'>"
        "<tr style='text-align:left;border-bottom:1px solid rgba(128,128,128,0.35)'>"
        "<th style='padding-right:10px'>file</th><th style='padding-right:10px'>what needs it</th>"
        "<th>where it goes</th></tr>"
        f"{rows}</table></div>"
    )


def runtime_html(*, has_gpu: bool, gpu_name: str, gpu_gb: float) -> str:
    """One line about the card, in the same register as the main wizard's runtime line."""
    if has_gpu:
        return theme.message_html(
            f"GPU: **{gpu_name or 'present'}** ({gpu_gb:g} GB). Both jobs can run here.", "info"
        )
    return theme.message_html(
        "No GPU in this runtime. Runtime > Change runtime type > T4 GPU. Reading a structure into 3Di still "
        "works without one; region discovery and ESMFold do not.",
        "warning",
    )


# ----------------------------------------------------------------------------------------
# The widget layer. Everything below only wires the functions above to ipywidgets.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepareRunners:
    """The side effects, injected so the wizard can be driven headlessly in tests."""

    load_wt_sequence: Callable[..., str]
    load_library: Callable[..., tuple[Any, list[str], Any]]
    make_spec: Callable[..., Any]
    load_three_di: Callable[..., str]
    three_di_from_structure: Callable[..., str]
    three_di_from_esmfold: Callable[..., str]
    discover_region: Callable[..., dict]
    save_region: Callable[..., None]
    upload: Callable[[str], Path]
    download: Callable[[Path], None]
    fetch: Callable[[str, Path], Path]


def default_runners() -> PrepareRunners:
    """The real implementations, imported on call so importing this module stays cheap."""
    from colabsd.data import load_library, load_wt_sequence
    from colabsd.region import discover_region, save_region
    from colabsd.spec import LibrarySpec
    from colabsd.structure import load_three_di, three_di_from_esmfold, three_di_from_structure

    def upload(what: str) -> Path:
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError(
                f"Uploading needs Colab. Outside it, put {what} in the working folder and pass its path."
            ) from None
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError(f"No file was uploaded for {what}.")
        return Path(next(iter(uploaded))).resolve()

    def download(path: Path) -> None:
        try:
            from google.colab import files
        except ImportError:
            return
        files.download(str(path))

    def fetch(url: str, destination: Path) -> Path:
        import urllib.request

        urllib.request.urlretrieve(url, destination)
        return destination

    return PrepareRunners(
        load_wt_sequence=load_wt_sequence,
        load_library=load_library,
        make_spec=LibrarySpec,
        load_three_di=load_three_di,
        three_di_from_structure=three_di_from_structure,
        three_di_from_esmfold=three_di_from_esmfold,
        discover_region=discover_region,
        save_region=save_region,
        upload=upload,
        download=download,
        fetch=fetch,
    )


class PrepareWizard:
    """The form itself. Every visibility rule and every message comes from the pure layer."""

    def __init__(
        self,
        *,
        work_dir: Path | str,
        example_dir: Path | str | None = None,
        has_gpu: bool = False,
        gpu_name: str = "",
        gpu_gb: float = 0.0,
        runners: PrepareRunners | None = None,
    ) -> None:
        import ipywidgets

        self.work_dir = Path(work_dir)
        self.example_dir = Path(example_dir) if example_dir is not None else default_example_dir()
        self.has_gpu = bool(has_gpu)
        self.gpu_name = gpu_name
        self.gpu_gb = float(gpu_gb)
        self.runners = runners or default_runners()
        self.spec: Any = None
        self.sequences: list[str] = []
        self.n_library = 0
        self.wt_sequence = ""
        self.written: dict[str, Path] = {}

        style = {"description_width": "initial"}

        def wide() -> Any:
            # One Layout instance per widget: ipywidgets shares a Layout that is passed
            # twice, and `layout.display` would then hide every widget holding it.
            return ipywidgets.Layout(width="560px")

        self.intro = theme.html(intro_html())
        self.fields: dict[str, Any] = {
            "make_three_di": ipywidgets.Checkbox(
                value=True, description="Make a wild-type 3Di string (SaProt backbones only)", indent=False
            ),
            "make_region": ipywidgets.Checkbox(
                value=False, description="Discover a pooling region (cosine_p90_mean pooling only)", indent=False
            ),
            "protein_source": ipywidgets.RadioButtons(
                options=["bundled_example_SlugCas9_5NNK", "my_own_protein"],
                value="bundled_example_SlugCas9_5NNK",
                description="Protein:",
                style=style,
                layout=ipywidgets.Layout(width="max-content"),
            ),
            "wt_sequence_or_fasta_url": ipywidgets.Textarea(
                value="",
                placeholder="Paste the wild-type amino-acid sequence, or the URL of its FASTA file",
                description="Wild type:",
                style=style,
                layout=ipywidgets.Layout(width="560px", height="70px"),
            ),
            "three_di_source": ipywidgets.Dropdown(
                options=list(THREE_DI_SOURCES),
                value=THREE_DI_SOURCES[0],
                description="3Di from:",
                style=style,
                layout=wide(),
            ),
            "chain": ipywidgets.Text(
                value="",
                placeholder="chain id, e.g. A (blank = the only chain)",
                description="Chain:",
                style=style,
                layout=wide(),
            ),
            "esmfold_risk_accepted": ipywidgets.Checkbox(
                value=False, description="I accept the ESMFold memory risk", indent=False
            ),
            "three_di_name": ipywidgets.Text(
                value="wt_3di.txt", description="3Di file name:", style=style, layout=wide()
            ),
            "positions_1based": ipywidgets.Text(
                value="984, 985, 990, 1012, 1016",
                description="Mutated positions (1-based):",
                style=style,
                layout=wide(),
            ),
            "mutation_columns": ipywidgets.Text(
                value="nnk1, nnk2, nnk3, nnk4, nnk5", description="Mutation columns:", style=style, layout=wide()
            ),
            "condition_columns": ipywidgets.Text(
                value="NNGA, NNGT, NNGC, NNGG", description="Condition columns:", style=style, layout=wide()
            ),
            "three_letter_residues": ipywidgets.Checkbox(
                value=True, description="Residues are written as Asn, not N", indent=False
            ),
            "count_column": ipywidgets.Text(
                value="count", description="Read-count column:", style=style, layout=wide()
            ),
            "min_count": ipywidgets.BoundedIntText(
                value=0, min=0, max=10_000_000, description="Minimum read count:", style=style, layout=wide()
            ),
            "region_model": ipywidgets.Dropdown(
                options=[(f"{model.label}  ({model.hf_id})", model.hf_id) for model in REGION_MODELS],
                value=DEFAULT_REGION_MODEL,
                description="Region model:",
                style=style,
                layout=wide(),
            ),
            "n_sample": ipywidgets.BoundedIntText(
                value=2000,
                min=0,
                max=10_000_000,
                description="Variants to score (0 = all):",
                style=style,
                layout=wide(),
            ),
            "seed": ipywidgets.BoundedIntText(
                value=0, min=0, max=10_000, description="Seed:", style=style, layout=wide()
            ),
            "batch_size": ipywidgets.BoundedIntText(
                value=2, min=1, max=16, description="Batch size:", style=style, layout=wide()
            ),
            "percentiles": ipywidgets.Text(
                value="90, 95", description="Percentiles:", style=style, layout=wide()
            ),
            "run_on_cpu_anyway": ipywidgets.Checkbox(
                value=False, description="Run region discovery on the CPU anyway", indent=False
            ),
        }
        self.load_button = ipywidgets.Button(
            description="Load the protein", button_style="info", layout=ipywidgets.Layout(width="240px")
        )
        self.run_button = ipywidgets.Button(
            description="Run the ticked jobs", button_style="success", layout=ipywidgets.Layout(width="240px")
        )
        self.fields["load_button"] = self.load_button
        self.fields["run_button"] = self.run_button

        self.gpu_line = theme.html(runtime_html(has_gpu=self.has_gpu, gpu_name=self.gpu_name, gpu_gb=self.gpu_gb))
        self.notice_box = theme.html("")
        self.preview_box = theme.html("")
        self.outputs_box = theme.html("")
        self.log = ipywidgets.Output()

        for key, widget in self.fields.items():
            if hasattr(widget, "observe") and key not in ("load_button", "run_button"):
                widget.observe(self._observer(key), names="value")
        self.load_button.on_click(self.on_load)
        self.run_button.on_click(self.on_run)
        self.refresh()

    # -- state ---------------------------------------------------------------------------

    def state(self) -> PrepareState:
        """Read the widgets into the plain state the pure functions take."""
        values = {key: widget.value for key, widget in self.fields.items() if hasattr(widget, "value")}
        return PrepareState(
            **values,
            has_gpu=self.has_gpu,
            gpu_name=self.gpu_name,
            gpu_gb=self.gpu_gb,
            n_library=self.n_library,
            wt_length=len(self.wt_sequence),
            example_available=self.example_dir is not None,
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
        """One observer per field, so a change can drop a load it has just invalidated."""

        def observer(_change: Any = None) -> None:
            if key in LOAD_INPUTS and self.loaded:
                self.forget_load()
            self.refresh()

        return observer

    def _on_change(self, _change: Any = None) -> None:
        self.refresh()

    @property
    def loaded(self) -> bool:
        """True while a wild type (and possibly a library) read from disk is being held."""
        return bool(self.wt_sequence) or self.n_library > 0

    def forget_load(self) -> None:
        """Drop the loaded protein.

        Called the moment a field that decided what was loaded changes. Without it the
        wizard would keep a 16,424-variant library it read for the bundled example after
        the user switched to their own protein: every refusal would fall silent, the
        subsample preview would describe rows that are not theirs, and **Run** would
        discover a region of the wrong protein.
        """
        self.spec = None
        self.sequences = []
        self.n_library = 0
        self.wt_sequence = ""
        self._say("that change invalidated the loaded protein; click Load the protein again")

    def refresh(self) -> None:
        """Re-apply visibility, messages, the subsample preview and the output list."""
        state = self.state()
        shown = visible_fields(state)
        for key, widget in self.fields.items():
            set_display(widget, key in shown)
        preview = subsample_preview(state)
        estimate = estimate_region_runtime(
            n_scored=preview.n_scored,
            wt_length=state.wt_length,
            region_model=state.region_model,
            has_gpu=state.has_gpu,
        )
        self.notice_box.value = render_messages(notices(state, preview=preview, estimate=estimate))
        self.preview_box.value = preview_html(preview, estimate) if state.make_region else ""
        self.outputs_box.value = outputs_html(output_files(state))

    def display_box(self) -> Any:
        """The whole form as one `VBox`, built but not shown.

        Separate from `display` so a test can assert that every field really is on the page:
        the order below is written by hand, and a field left out of it would exist, obey
        every rule, and be invisible.
        """
        import ipywidgets

        order = [
            "make_three_di",
            "make_region",
            "protein_source",
            "wt_sequence_or_fasta_url",
            "positions_1based",
            "mutation_columns",
            "condition_columns",
            "three_letter_residues",
            "count_column",
            "min_count",
            "load_button",
            "three_di_source",
            "chain",
            "esmfold_risk_accepted",
            "three_di_name",
            "region_model",
            "n_sample",
            "seed",
            "batch_size",
            "percentiles",
            "run_on_cpu_anyway",
        ]
        return ipywidgets.VBox(
            [self.intro, self.gpu_line]
            + [self.fields[key] for key in order]
            + [self.preview_box, self.notice_box, self.outputs_box, self.run_button, self.log]
        )

    def display(self) -> PrepareWizard:
        """Show the form. Returns self so a notebook cell can keep the handle."""
        from IPython.display import display

        display(self.display_box())
        return self

    # -- actions -------------------------------------------------------------------------

    def _resolve_wt(self, state: PrepareState) -> str:
        if state.using_example:
            if self.example_dir is None:
                raise RuntimeError(
                    "The bundled example is not available in this install; pick my_own_protein and paste your "
                    "wild-type sequence."
                )
            return self.runners.load_wt_sequence(self.example_dir / "wt.fasta")
        typed = state.wt_sequence_or_fasta_url.strip()
        if not typed:
            raise RuntimeError(
                "No wild type given. Paste the amino-acid sequence, or the URL of its FASTA file (an AlphaFold "
                "or UniProt link works)."
            )
        if typed.startswith("http"):
            self.work_dir.mkdir(parents=True, exist_ok=True)
            return self.runners.load_wt_sequence(self.runners.fetch(typed, self.work_dir / "wt_downloaded.fasta"))
        return "".join(typed.split()).upper()

    def on_load(self, _button: Any = None) -> None:
        """Read the wild type, and the library too when region discovery is ticked."""
        state = self.state()
        try:
            self.wt_sequence = self._resolve_wt(state)
            self._say(f"wild type: {len(self.wt_sequence)} residues")
            if needs_library(state):
                self.spec = self.runners.make_spec(
                    wt_sequence=self.wt_sequence,
                    positions_1based=[int(value) for value in parse_list(state.positions_1based)],
                    mutation_columns=parse_list(state.mutation_columns),
                    condition_columns=parse_list(state.condition_columns),
                    three_letter=state.three_letter_residues,
                    count_column=state.count_column.strip() or None,
                )
                csv_path = (
                    self.example_dir / "library.csv"
                    if state.using_example and self.example_dir is not None
                    else self.runners.upload("your library.csv")
                )
                _frame, sequences, _targets = self.runners.load_library(csv_path, self.spec, min_count=state.min_count)
                self.sequences = list(sequences)
                self.n_library = len(self.sequences)
                self._say(f"library: {self.n_library} variants from {csv_path}")
        except Exception as exc:  # surfaced in the notebook, not raised into a dead callback
            self._say(f"could not load: {exc}")
        self.refresh()

    def on_run(self, _button: Any = None) -> dict[str, Path]:
        """Run the ticked jobs, unless a `stop` message refuses."""
        state = self.state()
        preview = subsample_preview(state)
        estimate = estimate_region_runtime(
            n_scored=preview.n_scored,
            wt_length=state.wt_length,
            region_model=state.region_model,
            has_gpu=state.has_gpu,
        )
        refusals = blocking(notices(state, preview=preview, estimate=estimate))
        if refusals:
            for message in refusals:
                self._say("refused: " + plain_text(message.text))
            return {}
        self.work_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        try:
            if state.make_three_di:
                written["three_di"] = self._run_three_di(state)
            if state.make_region:
                written.update(self._run_region(state, preview, estimate))
        except Exception as exc:
            self._say(f"failed: {exc}")
            return written
        self.written.update(written)
        for item in output_files(state):
            self._say(f"{item.name}: {item.where_it_goes}")
        return written

    def _run_three_di(self, state: PrepareState) -> Path:
        wt = self.wt_sequence
        length = len(wt)
        source = state.three_di_source
        if source == "bundled_example_3di":
            three_di = self.runners.load_three_di(self.example_dir / "wt_3di.txt", expected_length=length)
        elif source == "upload_3di_text_file":
            three_di = self.runners.load_three_di(self.runners.upload("your 3Di text file"), expected_length=length)
        elif source == "upload_structure_pdb_or_cif":
            three_di = self.runners.three_di_from_structure(
                self.runners.upload("the wild-type structure (.pdb / .cif)"),
                chain=state.chain.strip() or None,
                expected_length=length,
                expected_sequence=wt,
            )
        else:
            self._say("folding the wild type with ESMFold; an AlphaFold .cif would have been cheaper")
            three_di = self.runners.three_di_from_esmfold(
                wt, device="cuda" if state.has_gpu else "cpu", pdb_out=self.work_dir / "wt_esmfold.pdb"
            )
        path = self.work_dir / (state.three_di_name.strip() or "wt_3di.txt")
        path.write_text(three_di + "\n")
        self._say(f"3Di: {len(three_di)} states for {length} residues -> {path}")
        self.runners.download(path)
        return path

    def _run_region(
        self, state: PrepareState, preview: SubsamplePreview, estimate: RuntimeEstimate
    ) -> dict[str, Path]:
        self._say(
            f"scoring {preview.n_scored} of {preview.n_library} variants (seed {state.seed}); "
            f"first sampled rows {list(preview.first_rows)}"
        )
        self._say(estimate.sentence())
        records = self.runners.discover_region(
            self.sequences,
            self.wt_sequence,
            percentiles=parsed_percentiles(state),
            n_sample=state.n_sample,
            hf_id=state.region_model,
            device="cuda" if state.has_gpu else "cpu",
            batch_size=state.batch_size,
            seed=state.seed,
            progress=self._progress,
        )
        written: dict[str, Path] = {}
        for name, record in records.items():
            path = self.work_dir / (name.replace("cosine_", "region_") + ".json")
            self.runners.save_region(record, path)
            written[name] = path
            selected = record.get("n_selected_positions", len(record.get("selected_positions_1based", [])))
            self._say(f"{name}: {selected} of {record.get('seq_length', '?')} residues -> {path}")
            self.runners.download(path)
        return written

    def _progress(self, done: int, total: int, label: str) -> None:
        if total and (done == total or done % max(1, total // 10) == 0):
            self._say(f"  {done}/{total}  {label}")


def plain_text(text: str) -> str:
    """A message's markdown, flattened for a plain-text log line."""
    flat = _MARKDOWN_LINK.sub(r"\1 (\2)", text)
    return flat.replace("**", "").replace("`", "").replace("*", "")


def launch(
    *,
    work_dir: Path | str | None = None,
    example_dir: Path | str | None = None,
    has_gpu: bool | None = None,
    gpu_name: str | None = None,
    gpu_gb: float | None = None,
    runners: PrepareRunners | None = None,
) -> PrepareWizard:
    """Build the wizard and show it. This is the whole Prepare notebook.

    Every argument is optional so `launch()` works from a bare notebook cell. Runtime facts
    are detected when not supplied: defaulting them to "no GPU" would make every
    hardware-dependent warning fire backwards on a machine that has one.
    """
    if has_gpu is None or gpu_name is None or gpu_gb is None:
        detected = core.detect_runtime()
        has_gpu = detected.has_gpu if has_gpu is None else has_gpu
        gpu_name = (detected.gpu_name or "") if gpu_name is None else gpu_name
        gpu_gb = (detected.gpu_memory_gb or 0.0) if gpu_gb is None else gpu_gb
    wizard = PrepareWizard(
        work_dir=work_dir if work_dir is not None else core.DEFAULT_WORK_DIR,
        example_dir=example_dir,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_gb=gpu_gb,
        runners=runners,
    )
    return wizard.display()
