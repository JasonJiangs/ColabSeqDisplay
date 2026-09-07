"""Preparation: the step of the main panel that makes what the chosen model needs.

There used to be a second notebook here. It asked which backbone and which pooling, wrote
`wt_3di.txt` and `region_p90.json` to the user's downloads folder, and the main notebook
asked the same two questions again and asked for the files back. The two questions decide
everything, so they are now asked once, in `colabsd.ui.main_workflow`, and this module is
what that panel composes underneath them: two preparation *sections* that appear only
because of an answer given above them.

Nothing here is a question of its own. A SaProt backbone needs a wild-type 3Di string;
`cosine_p90_mean` pooling needs a pooling region; ESM2 with `mutation_site_mean` needs
neither, and then no preparation step is built at all. `preparation_steps` is that rule and
it is the only place it lives.

The expensive machinery is unchanged, because it worked: the subsample preview, the
runtime estimate, the notices and the "this step will write" table. The estimate is still
scaled off the one forward pass this repository has actually timed
(`results/region_investigation_full/report.txt`) by the per-backbone figures in
`colabsd.backbones.registry`, so a backbone the registry cannot cost is reported as
uncosted rather than guessed at. `predict_workflow` imports `forward_pass_minutes` from
here, because scoring a variant and scoring it for region discovery are the same frozen
forward pass.

What is new is reuse. The bundled example ships a 3Di string and both region files, and a
region discovered in this session stays in this session: when an artefact that fits is
already here, the step offers it instead of an hour of GPU time (`available_artefacts`),
and says so.

Every decision is a pure function of a `colabsd.ui.core.WizardState` — `preparation_steps`,
`available_artefacts`, `three_di_choices`, `region_choices`, `notices`, `region_preview`,
`estimate_region_runtime`, `output_files`. `ThreeDiSection` and `RegionSection` are the thin
ipywidgets layers above them, built and composed by the main panel; they own no rule.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
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
)

ASSUMED_LENGTH = 1000
SMALL_CARD_GB = 20.0

_COSINE_PERCENTILE = re.compile(r"^cosine_p(\d+(?:\.\d+)?)_mean$")

#: The two things preparation can have to make, in the order they would run.
STEP_THREE_DI = "three_di"
STEP_REGION = "region"
PREPARATION_STEPS: tuple[str, str] = (STEP_THREE_DI, STEP_REGION)

#: Where a region for this run can come from. `available` is an artefact that is already
#: here — the bundled example's, or one discovered earlier in this session.
REGION_SOURCES: tuple[str, ...] = ("available", "upload", "discover")

#: `colabsd.region.build_region_record` selects at or above this percentile and refuses
#: anything outside `[0, 100)`. The percentile is no longer typed — it is read off the
#: pooling name — but a pooling whose name does not yield one is still refused here, before
#: the forward pass, rather than after every variant has been scored.
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


#: `results/region_investigation_full/report.txt`, run `hf_fp16`: ESM2-650M over all 16,424
#: SlugCas9 variants of 1054 residues, float16 — the dtype `colabsd.region` uses on a GPU —
#: in 319 s. It is the only forward pass of this shape anyone here has stopwatched, and it
#: was on a B200, not on Colab.
TIMED_PASS = TimedPass(
    backbone="ESM2-650M",
    n_sequences=16_424,
    length=1054,
    seconds=319.0,
    source="results/region_investigation_full/report.txt",
)

#: How much slower a Colab GPU is than the B200 that was timed. README: "a free T4 is
#: several times slower than a B200". Nobody has timed a T4, so this is a wide band and the
#: only invented number in the estimate; the progress counter reports the real rate.
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


def forward_pass_minutes(backbone: str, *, n_sequences: int, length: int, has_gpu: bool) -> tuple[float, float] | None:
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
#: `tests/test_ui_prepare.py` asserts they are the same string.
DEFAULT_REGION_MODEL = BACKBONES[TIMED_PASS.backbone].hf_id

#: The bundled library, found through `colabsd.EXAMPLES_ROOT` the way `colabsd.ui.core` finds it.
EXAMPLE_DIRNAME = "slugcas9_5nnk"
EXAMPLE_NAME = "SlugCas9 5NNK"


def default_example_dir() -> Path | None:
    """Where the bundled SlugCas9 example lives, or None when this install has no copy."""
    from colabsd import EXAMPLES_ROOT

    candidate = Path(EXAMPLES_ROOT) / EXAMPLE_DIRNAME
    return candidate if (candidate / "wt.fasta").is_file() else None


def region_model_for(hf_id: str) -> RegionModel:
    """The registry entry behind *hf_id*, or a nameless stand-in for one it does not know."""
    for model in REGION_MODELS:
        if model.hf_id == hf_id:
            return model
    return RegionModel(hf_id=hf_id, label=hf_id)


# ----------------------------------------------------------------------------------------
# What this configuration needs prepared. Derived from the backbone and the pooling, never
# asked: these are the same two answers the panel already has.
# ----------------------------------------------------------------------------------------


def needs_three_di(state: core.WizardState) -> bool:
    """True when the chosen backbone reads structure, and so needs a wild-type 3Di string."""
    return core.needs_structure(state)


def needs_region(state: core.WizardState) -> bool:
    """True when the chosen pooling averages a discovered region rather than the mutated sites."""
    return state.mode == "train" and str(state.pooling).startswith("cosine_")


def preparation_steps(state: core.WizardState) -> tuple[str, ...]:
    """Which preparation steps this configuration calls for, in the order they would run.

    Empty is a real answer, and the common one: an ESM2 backbone with `mutation_site_mean`
    pooling needs nothing prepared, so the panel builds no preparation step at all.
    """
    asked = {STEP_THREE_DI: needs_three_di, STEP_REGION: needs_region}
    return tuple(step for step in PREPARATION_STEPS if asked[step](state))


def pooling_percentile(pooling: str) -> float | None:
    """The percentile `cosine_p90_mean` names, or None when the pooling names none.

    Nobody types a percentile any more: the pooling chosen above *is* the percentile, and a
    region discovered under a different one would be pooled under a name it does not have.
    """
    match = _COSINE_PERCENTILE.match(str(pooling))
    if match is None:
        return None
    value = float(match.group(1))
    return value if PERCENTILE_RANGE[0] <= value < PERCENTILE_RANGE[1] else None


def region_file_name(percentile: float) -> str:
    """The file a percentile produces.

    `colabsd.region.build_region_record` names its record `cosine_p{int(percentile)}`, so
    92.5 and 92 write the same file. The form has to promise the name the run will actually
    write, not a prettier one.
    """
    return f"region_p{int(percentile)}.json"


# ----------------------------------------------------------------------------------------
# What is already here. Recomputing an artefact that ships with the package, or one this
# session has already made, is an hour of GPU time spent on a file that exists.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Artefact:
    """A prepared file that already exists, and what it describes.

    `origin` is `bundled_example` for the files that ship with the package and `session` for
    anything this session made. `describes` is the protein it belongs to, because that is
    the question that decides whether it may be reused at all.
    """

    kind: str
    path: Path
    origin: str
    describes: str
    pooling: str = ""
    length: int = 0
    detail: str = ""

    @property
    def label(self) -> str:
        """The radio line: what it is, where it came from, and that it costs nothing."""
        where = "ships with the package" if self.origin == "bundled_example" else "made earlier in this session"
        tail = f" — {self.detail}" if self.detail else ""
        return f"Reuse the one that is already here ({where}, {self.describes}{tail}) — nothing to compute"


def _example_artefacts(state: core.WizardState) -> list[Artefact]:
    """The bundled example's 3Di string and region files, when that library is the one loaded.

    They describe SlugCas9. Offering them for somebody else's protein is how a run silently
    pools one protein's residues under another protein's name, so they are not offered at
    all unless the bundled library is what was read.
    """
    if state.data_source != "bundled_example":
        return []
    example = default_example_dir()
    if example is None:
        return []
    found: list[Artefact] = []
    three_di = example / "wt_3di.txt"
    if three_di.is_file():
        found.append(
            Artefact(
                kind=STEP_THREE_DI,
                path=three_di,
                origin="bundled_example",
                describes=f"the {EXAMPLE_NAME} wild type",
                detail="computed once with foldseek",
            )
        )
    percentile = pooling_percentile(state.pooling)
    if percentile is not None:
        region = example / region_file_name(percentile)
        if region.is_file():
            found.append(
                Artefact(
                    kind=STEP_REGION,
                    path=region,
                    origin="bundled_example",
                    describes=f"the {EXAMPLE_NAME} wild type",
                    pooling=str(state.pooling),
                    detail=f"discovered for {state.pooling}",
                )
            )
    return found


def available_artefacts(
    state: core.WizardState, *, session: Sequence[Artefact] = ()
) -> tuple[Artefact, ...]:
    """Everything this configuration could reuse instead of making, session copies first.

    A session artefact wins over the bundled one: it was made for the library on screen,
    with the model on screen, and the bundled file was not.
    """
    fitting = [item for item in session if _artefact_fits(item, state)]
    return tuple(fitting + [item for item in _example_artefacts(state) if _artefact_fits(item, state)])


def _artefact_fits(artefact: Artefact, state: core.WizardState) -> bool:
    """A region belongs to one pooling; a 3Di string belongs to one wild type.

    Length is what decides the second one: a 3Di string of another length would be read
    against this sequence position by position and be wrong at every one of them.
    """
    if artefact.kind == STEP_REGION:
        return artefact.pooling == str(state.pooling)
    if artefact.length and state.wt_length:
        return artefact.length == int(state.wt_length)
    return True


def artefact_for(kind: str, state: core.WizardState, *, session: Sequence[Artefact] = ()) -> Artefact | None:
    """The one artefact this step would reuse, or None when there is nothing to reuse."""
    for item in available_artefacts(state, session=session):
        if item.kind == kind:
            return item
    return None


# ----------------------------------------------------------------------------------------
# The choices each step offers. A control that is irrelevant to the stated goal is not on
# screen at all, so these lists are built from the state rather than written down once.
# ----------------------------------------------------------------------------------------


def three_di_choices(state: core.WizardState, artefact: Artefact | None = None) -> list[tuple[str, str]]:
    """Where the wild-type 3Di string can come from, for this library.

    The reuse line is only offered when there is something to reuse; the ESMFold line is
    offered last and says what it costs, because it is the last resort and not the default.
    """
    choices: list[tuple[str, str]] = [("Not chosen yet", "none")]
    if artefact is not None:
        choices.append((artefact.label, "bundled_example" if artefact.origin == "bundled_example" else "session"))
    choices += [
        ("Upload a structure of my wild type (.pdb / .cif) and read the shape off it", "upload_structure"),
        ("Upload a 3Di text file I already have", "upload_3di"),
        ("Paste a 3Di string I already have", "paste"),
        ("Fold the wild type here with ESMFold (slow, memory-hungry, last resort)", "esmfold"),
    ]
    return choices


def region_choices(state: core.WizardState, artefact: Artefact | None = None) -> list[tuple[str, str]]:
    """Where the pooling region can come from, for this pooling."""
    choices: list[tuple[str, str]] = []
    if artefact is not None:
        choices.append((artefact.label, "available"))
    choices += [
        ("Upload a region JSON I already have", "upload"),
        ("Discover it now from my own library (the expensive one — see the estimate below)", "discover"),
    ]
    return choices


def default_region_source(state: core.WizardState, artefact: Artefact | None = None) -> str:
    """What the region radio should start on: reuse when there is something to reuse."""
    return "available" if artefact is not None else "discover"


def default_three_di_source(state: core.WizardState, artefact: Artefact | None = None) -> str:
    """What the 3Di radio should start on: reuse when there is something to reuse."""
    if artefact is None:
        return "none"
    return "bundled_example" if artefact.origin == "bundled_example" else "session"


def preparation_field_visibility(
    state: core.WizardState, *, artefact: Artefact | None = None, has_gpu: bool | None = None
) -> dict[str, bool]:
    """The preparation fields `core.FIELD_RULES` does not declare, because only this step has them.

    `core` owns the 3Di source fields; these are the ones the merged panel added — the
    ESMFold acknowledgement and everything region discovery is steered by.
    """
    folding = needs_three_di(state) and state.three_di_source == "esmfold"
    discovering = needs_region(state) and str(state.get("region_source")) == "discover"
    return {
        "esmfold_risk_accepted": folding,
        "region_model": discovering,
        "region_n_sample": discovering,
        "region_seed": discovering,
        "region_batch_size": discovering and bool(state.show_advanced),
        "region_run_on_cpu": discovering and has_gpu is False,
    }


# ----------------------------------------------------------------------------------------
# What region discovery would actually do, worked out before the user commits to it.
# ----------------------------------------------------------------------------------------


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


def subsample_preview(
    *, n_library: int, n_sample: int, seed: int, subsample: Callable[..., Any] | None = None
) -> SubsamplePreview:
    """Work out exactly which library rows region discovery would score.

    Uses `colabsd.region.subsample`, the same function the run itself calls, so the preview
    cannot drift from what happens. Before the library is loaded it reports the typed
    `n_sample` and says the library is not loaded yet.
    """
    n_library = max(0, int(n_library))
    n_sample = max(0, int(n_sample))
    if n_library <= 0:
        return SubsamplePreview(
            n_library=n_library,
            n_scored=max(0, n_sample),
            seed=int(seed),
            first_rows=(),
            is_full_library=n_sample <= 0,
            library_loaded=False,
        )
    if subsample is None:
        from colabsd.region import subsample as subsample_fn
    else:
        subsample_fn = subsample
    rows = [int(row) for row in subsample_fn(n_library, n_sample, seed=int(seed))]
    return SubsamplePreview(
        n_library=n_library,
        n_scored=len(rows),
        seed=int(seed),
        first_rows=tuple(rows[:12]),
        is_full_library=len(rows) >= n_library,
        library_loaded=True,
    )


def region_preview(state: core.WizardState, *, subsample: Callable[..., Any] | None = None) -> SubsamplePreview:
    """`subsample_preview` for the numbers currently on the form."""
    return subsample_preview(
        n_library=int(state.n_variants or 0),
        n_sample=int(state.get("region_n_sample") or 0),
        seed=int(state.get("region_seed") or 0),
        subsample=subsample,
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
    band = forward_pass_minutes(model.backbone, n_sequences=max(0, n_scored), length=length, has_gpu=has_gpu)
    return RuntimeEstimate(
        n_scored=max(0, n_scored),
        low_minutes=None if band is None else band[0],
        high_minutes=None if band is None else band[1],
        device="GPU" if has_gpu else "CPU",
        model_label=model.label,
        length=length,
        length_is_assumed=length_is_assumed,
    )


def region_estimate(state: core.WizardState, *, has_gpu: bool = False, preview: SubsamplePreview | None = None) -> RuntimeEstimate:
    """`estimate_region_runtime` for the numbers currently on the form."""
    shown = preview if preview is not None else region_preview(state)
    return estimate_region_runtime(
        n_scored=shown.n_scored,
        wt_length=int(state.wt_length or 0),
        region_model=str(state.get("region_model") or DEFAULT_REGION_MODEL),
        has_gpu=bool(has_gpu),
    )


# ----------------------------------------------------------------------------------------
# The contextual messages. `core` already owns everything about the 3Di string that is true
# whatever produced it; these are the ones preparation itself earns.
# ----------------------------------------------------------------------------------------


def notices(
    state: core.WizardState,
    *,
    artefacts: Sequence[Artefact] = (),
    preview: SubsamplePreview | None = None,
    estimate: RuntimeEstimate | None = None,
    has_gpu: bool | None = None,
    gpu_gb: float = 0.0,
    region_ready: bool = False,
) -> list[Message]:
    """Every message the preparation step earns, in reading order.

    `has_gpu=None` means nobody has looked at the machine yet, which is not the same as
    "no GPU": the hardware refusals stay silent rather than firing backwards.
    """
    steps = preparation_steps(state)
    out: list[Message] = []
    if STEP_THREE_DI in steps:
        out += _three_di_notices(state, artefacts=artefacts, has_gpu=has_gpu)
    if STEP_REGION in steps:
        out += _region_notices(
            state,
            artefacts=artefacts,
            preview=preview,
            estimate=estimate,
            has_gpu=has_gpu,
            gpu_gb=gpu_gb,
            region_ready=region_ready,
        )
    return out


def _three_di_notices(
    state: core.WizardState, *, artefacts: Sequence[Artefact], has_gpu: bool | None
) -> list[Message]:
    out: list[Message] = []
    artefact = pick(artefacts, STEP_THREE_DI)
    if state.three_di_source in ("bundled_example", "session") and artefact is None:
        out.append(
            Message(
                "three_di_reuse_gone",
                "stop",
                "The 3Di string you were reusing describes another protein — the library changed under it. "
                "One protein's structure states read against another protein's sequence are wrong at every "
                "position, so upload a structure of your own wild type instead, or a 3Di file you already have.",
            )
        )
    computing = state.three_di_source in ("upload_structure", "esmfold")
    if artefact is not None and computing and not state.wt_3di_length:
        out.append(
            Message(
                "three_di_reuse_available",
                "info",
                f"A 3Di string for {artefact.describes} is already here ({artefact.path.name}), so this protein "
                "needs nothing computed: pick the reuse line above and the step is done. Only compute one if "
                "you want a different structure of the same wild type.",
            )
        )
    if state.three_di_source == "esmfold":
        if has_gpu is False:
            out.append(
                Message(
                    "esmfold_needs_gpu",
                    "stop",
                    "ESMFold needs a GPU and this runtime has none. **Runtime → Change runtime type → T4 GPU**, "
                    "or download a structure of your wild type from [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk) "
                    "and upload that instead — it is free, faster and usually better.",
                )
            )
        if not state.get("esmfold_risk_accepted"):
            out.append(
                Message(
                    "esmfold_risk_not_accepted",
                    "stop",
                    "ESMFold holds one value per residue *pair*, so its memory grows with the square of the "
                    "length: on a free T4 it is the step most likely to end your session. Tick **I accept the "
                    "ESMFold memory risk** to run it anyway, or upload a structure instead.",
                )
            )
    # Whether this wild type is too long to fold at all, and that ESMFold is the expensive
    # answer to a question a downloaded structure answers for free, are `core`'s two
    # messages (`esmfold_too_long`, `esmfold_expensive`). Saying either again here would
    # only bury the one that matters.
    return out


def _region_notices(
    state: core.WizardState,
    *,
    artefacts: Sequence[Artefact],
    preview: SubsamplePreview | None,
    estimate: RuntimeEstimate | None,
    has_gpu: bool | None,
    gpu_gb: float,
    region_ready: bool,
) -> list[Message]:
    out: list[Message] = []
    source = str(state.get("region_source") or "")
    artefact = pick(artefacts, STEP_REGION)
    if pooling_percentile(state.pooling) is None:
        out.append(
            Message(
                "region_pooling_unknown",
                "stop",
                f"`{state.pooling}` averages a discovered region, but its name gives no percentile to discover "
                "one at, so nothing here can make the file it needs. Pick `cosine_p90_mean`, or "
                "`mutation_site_mean`, which pools the mutated sites and needs no region at all.",
            )
        )
        return out
    if source == "available" and artefact is None:
        out.append(
            Message(
                "region_reuse_gone",
                "stop",
                "There is no region here to reuse for this pooling any more — the library or the pooling "
                "changed under it. Upload a region JSON for this protein, or discover one below.",
            )
        )
    if artefact is not None and source == "discover" and not region_ready:
        out.append(
            Message(
                "region_reuse_available",
                "warning",
                f"A region for `{state.pooling}` is already here ({artefact.path.name}, {artefact.describes}). "
                "Discovering another one costs the GPU time below and answers the same question. Reuse it "
                "unless you have changed the library or the region model on purpose.",
            )
        )
    if source == "discover":
        out += _discovery_notices(
            state, preview=preview, estimate=estimate, has_gpu=has_gpu, gpu_gb=gpu_gb
        )
    if source == "upload" and not str(state.get("region_filename") or "").strip():
        out.append(
            Message(
                "region_file_missing",
                "stop",
                f"`{state.pooling}` needs a region file and none has been given. Upload the JSON, or type the "
                "path of one on this machine, or discover a region from your own library below.",
            )
        )
    if source == "discover" and not region_ready:
        out.append(
            Message(
                "region_not_ready",
                "stop",
                f"`{state.pooling}` averages a region of residues and you have chosen to discover one, which is "
                "the one step on this page that can outlast a Colab session. Press **Discover the region** "
                "below when you are ready: Train will not start it for you. Reusing a region instead needs no "
                "run at all.",
            )
        )
    return out


def _discovery_notices(
    state: core.WizardState,
    *,
    preview: SubsamplePreview | None,
    estimate: RuntimeEstimate | None,
    has_gpu: bool | None,
    gpu_gb: float,
) -> list[Message]:
    out: list[Message] = []
    on_cpu = bool(state.get("region_run_on_cpu"))
    if has_gpu is False and not on_cpu:
        out.append(
            Message(
                "region_needs_gpu",
                "stop",
                "Region discovery runs a language model over full-length sequences, which needs a GPU. "
                "**Runtime → Change runtime type → T4 GPU**, or tick **run it on the CPU anyway** and wait "
                "hours for the same answer.",
            )
        )
    elif has_gpu is False and on_cpu:
        out.append(
            Message(
                "region_cpu_override",
                "warning",
                "Running region discovery on the CPU. That is tens of times slower than a GPU, and the "
                "estimate below already accounts for it.",
            )
        )
    if preview is not None and preview.library_loaded:
        if preview.is_full_library:
            out.append(
                Message(
                    "region_full_library",
                    "info",
                    f"Every one of the {preview.n_library:,} variants is scored, so the region does not depend "
                    "on the seed.",
                )
            )
        else:
            out.append(
                Message(
                    "region_subset_only",
                    "warning",
                    f"The region will be discovered from **{preview.n_scored:,} of {preview.n_library:,} "
                    f"variants** — this subset only, drawn with seed {preview.seed}. Another seed picks another "
                    "subset and therefore a slightly different region. Set the sample size to 0 to score every "
                    "variant and remove the question.",
                )
            )
    if estimate is not None and estimate.is_long:
        span = span_of(estimate.low_minutes or 0.0, estimate.high_minutes or 0.0)
        out.append(
            Message(
                "region_long_run",
                "warning",
                f"This discovery is {span}, and a free Colab session disconnects long before the slow end of "
                "that — with a fine-tune still to run afterwards in the same session. A disconnect leaves "
                "nothing behind unless **Save to Google Drive** is ticked in the storage step, so tick it "
                "first, or cut the cost: a smaller region model, or a smaller sample. Reusing a region that "
                "already exists costs nothing at all.",
            )
        )
    batch = int(state.get("region_batch_size") or 0)
    if batch >= 8 and (gpu_gb <= 0 or gpu_gb < SMALL_CARD_GB):
        card = f"a {gpu_gb:g} GB card" if gpu_gb > 0 else "a card whose memory is not known here"
        out.append(
            Message(
                "region_batch_memory",
                "warning",
                f"Batch {batch} on {card} with full-length sequences has not been measured here. If the run "
                "dies with an out-of-memory error, halve the batch size; nothing else needs changing and the "
                "region is unaffected.",
            )
        )
    return out


def pick(artefacts: Iterable[Artefact], kind: str) -> Artefact | None:
    """The artefact of one kind out of a list, or None. The list is short and ordered."""
    for item in artefacts:
        if item.kind == kind:
            return item
    return None


def blocking(found: Sequence[Message]) -> list[Message]:
    """The messages that refuse to let a step start."""
    return [message for message in found if message.severity == "stop"]


# ----------------------------------------------------------------------------------------
# What each step will write, and what stops its button.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputFile:
    """A file this step writes, what needs it, and where it goes."""

    name: str
    needed_by: str
    where_it_goes: str


def output_files(state: core.WizardState, *, work_dir: str | Path = core.DEFAULT_WORK_DIR) -> list[OutputFile]:
    """What preparation will write for this configuration, and what each file is for.

    Nothing has to leave the session any more — the panel that made these files is the panel
    that trains — so the "where it goes" column says where it is kept, not what to upload
    where. They are still written, because a Drive mount is what survives a disconnect.
    """
    files: list[OutputFile] = []
    steps = preparation_steps(state)
    kept = f"kept in this session and written to `{Path(work_dir)}`, so a Drive mount survives a disconnect"
    if STEP_THREE_DI in steps:
        files.append(
            OutputFile(
                name="wt_3di.txt",
                needed_by=f"{state.backbone}, and nothing else on this page",
                where_it_goes=kept,
            )
        )
    if STEP_REGION in steps:
        percentile = pooling_percentile(state.pooling)
        if percentile is not None:
            files.append(
                OutputFile(
                    name=region_file_name(percentile),
                    needed_by=f"`{state.pooling}` pooling, and nothing else on this page",
                    where_it_goes=kept,
                )
            )
    return files


def three_di_blockers(state: core.WizardState, artefact: Artefact | None = None) -> list[str]:
    """What stops the get-the-3Di button, each said as the thing to go and fix."""
    source = state.three_di_source
    problems: list[str] = []
    if source in {"none", ""}:
        problems.append("Choose where the 3Di string should come from first.")
    if source in {"bundled_example", "session"} and artefact is None:
        problems.append(
            "There is no 3Di string here to reuse for this protein. Upload a structure of your own wild type, "
            "or a 3Di file you already have."
        )
    if source == "paste" and not str(state.get("three_di_text") or "").strip():
        problems.append("The 3Di box is empty. Paste the string, one lowercase letter per residue.")
    if source == "upload_3di" and not str(state.get("three_di_file") or "").strip():
        problems.append("No 3Di file yet. Upload one with the button above.")
    if source == "upload_structure" and not str(state.get("structure_file") or "").strip():
        problems.append("No structure yet. Upload a .pdb or .cif of your wild type.")
    if source == "esmfold":
        if not state.get("esmfold_risk_accepted"):
            problems.append(
                "ESMFold is the memory-hungry last resort. Tick the box that accepts that risk, or upload a "
                "structure instead."
            )
        if state.wt_length > esmfold_safe_length():
            problems.append(
                f"Your wild type is {state.wt_length:,d} residues and ESMFold runs out of memory on a free T4 "
                f"past about {esmfold_safe_length():,d}. Download a structure from the PDB or AlphaFold and "
                "upload it instead — it is better, and it is free."
            )
    return problems


def region_blockers(state: core.WizardState, artefact: Artefact | None = None) -> list[str]:
    """What stops the get-the-region button."""
    source = str(state.get("region_source") or "")
    problems: list[str] = []
    if pooling_percentile(state.pooling) is None:
        problems.append(f"`{state.pooling}` names no percentile to discover a region at.")
        return problems
    if source == "available" and artefact is None:
        problems.append("There is no region here to reuse for this pooling. Upload one, or discover it below.")
    if source == "upload" and not str(state.get("region_filename") or "").strip():
        problems.append("No region JSON yet. Upload one with the button above, or type the path of a file here.")
    if source == "discover":
        if int(state.n_variants or 0) <= 0:
            problems.append(
                "Region discovery scores your variants, so the library has to be read first. Press "
                "**Check my library** in step 1."
            )
        if int(state.wt_length or 0) <= 0:
            problems.append("The wild-type length is not known yet, so nothing can be scored against it.")
    return problems


# ----------------------------------------------------------------------------------------
# Rendering. Pure string builders on top of `colabsd.ui.theme`, so the prose is testable.
# ----------------------------------------------------------------------------------------


def step_note(state: core.WizardState) -> str:
    """The one paragraph at the top of the preparation step, written for whoever is here."""
    if not preparation_steps(state):
        return ""
    return (
        f"**{state.backbone}** and `{state.pooling}` pooling need something made before training, and this is "
        "where it is made. Nothing here is a new question: these are the two answers you have already given, "
        "and everything below follows from them."
    )


def block_note(state: core.WizardState, key: str) -> str:
    """What one block is for, said beside its own controls.

    Nobody has to know what a 3Di string is to decide whether they need one. They chose a
    backbone and a pooling; this says what those choices mean and what the block will do
    about it.
    """
    if key == STEP_THREE_DI:
        return (
            f"**{state.backbone}** reads your protein's *shape* as well as its sequence. The shape is written "
            "as one letter per residue — a Foldseek 3Di string — and this block gets one. The usual answer is "
            "a structure file you already have or can download from the PDB or "
            "[AlphaFold](https://alphafold.ebi.ac.uk); folding it here is the last resort, not the default. "
            "The string stays in this session: there is nothing to download and upload back."
        )
    if key == STEP_REGION:
        return (
            f"`{state.pooling}` reads the model out over the residues whose embeddings disagree most across "
            "your variants, rather than over the mutated sites alone — the model's own answer to which part of "
            "the protein your assay moves. Which residues those are is discovered once from your own library, "
            "and reused every time after that."
        )
    return ""


def preview_html(preview: SubsamplePreview, estimate: RuntimeEstimate) -> str:
    """The subsample and its cost, both visible before the user commits to the run."""
    if not preview.library_loaded:
        head = (
            "**Subsample:** the library has not been read yet, so this is what you typed, not what would be "
            f"scored — sample {preview.n_scored:,}, seed {preview.seed}."
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
    """What each written file is for and where it is kept."""
    if not files:
        return ""
    rows = "".join(
        "<tr>"
        f"<td style='padding:3px 10px 3px 0'><code>{item.name}</code></td>"
        f"<td style='padding:3px 10px 3px 0'>{item.needed_by}</td>"
        f"<td style='padding:3px 0'>{theme.render_markdown_inline(item.where_it_goes)}</td>"
        "</tr>"
        for item in files
    )
    return (
        "<div style='margin:6px 0;line-height:1.5'><b>This step will write:</b>"
        "<table style='border-collapse:collapse;margin-top:4px'>"
        "<tr style='text-align:left;border-bottom:1px solid rgba(128,128,128,0.35)'>"
        "<th style='padding-right:10px'>file</th><th style='padding-right:10px'>what needs it</th>"
        "<th>where it is kept</th></tr>"
        f"{rows}</table></div>"
    )


# ----------------------------------------------------------------------------------------
# The widget layer: two sections the main panel composes. They own no decision — every
# question above was answered by a pure function, and these only build and read widgets.
# ----------------------------------------------------------------------------------------

_LABEL = {"description_width": "initial"}


@dataclass(frozen=True)
class SectionTools:
    """The main panel's own widget factories, so a composed section looks like the rest of it.

    Passing them in rather than importing them keeps the upload row — and with it the
    "this is about to block the kernel" notice the main panel prints before
    `files.upload()` — in exactly one place.
    """

    text: Callable[..., Any]
    upload_row: Callable[..., Any]
    button: Callable[..., Any]


class ThreeDiSection:
    """Source a wild-type 3Di string. On screen only because a SaProt backbone was chosen."""

    key = STEP_THREE_DI

    def __init__(self, state: core.WizardState, tools: SectionTools) -> None:
        import ipywidgets

        self.fields: dict[str, Any] = {}
        choices = three_di_choices(state, None)
        values = [value for _label, value in choices]
        self.fields["three_di_source"] = ipywidgets.RadioButtons(
            options=choices,
            value=state.three_di_source if state.three_di_source in values else "none",
            layout={"width": "max-content"},
            style=_LABEL,
        )
        self.fields["structure_file"] = tools.upload_row(
            "structure_file", "Upload .pdb / .cif", "the wild-type structure", "structure"
        )
        self.fields["chain"] = tools.text("chain", "Chain to read:", "A — leave empty for the first chain")
        self.fields["three_di_file"] = tools.upload_row(
            "three_di_file", "Upload wt_3di.txt", "your 3Di file", "structure"
        )
        self.fields["three_di_text"] = ipywidgets.Textarea(
            value=str(state.get("three_di_text") or ""),
            placeholder="dpvqlvvcccd… one lowercase letter per residue",
            description="3Di string:",
            layout={"width": "560px", "height": "80px"},
            style=_LABEL,
        )
        self.fields["esmfold_note"] = theme.note(
            "ESMFold folds the wild type here, on this GPU. If your protein is in the AlphaFold database "
            "([alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk)), downloading that model and uploading it "
            "above is faster, cheaper and at least as accurate."
        )
        self.fields["esmfold_risk_accepted"] = ipywidgets.Checkbox(
            value=bool(state.get("esmfold_risk_accepted")),
            description="I accept the ESMFold memory risk",
            indent=False,
            style=_LABEL,
        )
        self.note = theme.note("")
        self.button = tools.button("Get the 3Di string", "primary")

    def children(self) -> list[Any]:
        """The widgets, in reading order. The board, the button and the log are the panel's."""
        return [
            self.note,
            self.fields["three_di_source"],
            self.fields["structure_file"],
            self.fields["chain"],
            self.fields["three_di_file"],
            self.fields["three_di_text"],
            self.fields["esmfold_note"],
            self.fields["esmfold_risk_accepted"],
        ]

    def sync(self, state: core.WizardState, artefact: Artefact | None) -> str | None:
        """Rebuild the source list for what is actually available; return a forced value.

        The reuse line only exists while there is something to reuse, so switching to your
        own library has to take the choice away rather than leave a dead option selected.
        """
        choices = three_di_choices(state, artefact)
        values = [value for _, value in choices]
        widget = self.fields["three_di_source"]
        if tuple(widget.options) == tuple(choices):
            return None
        wanted = state.three_di_source if state.three_di_source in values else "none"
        widget.options = choices
        widget.value = wanted
        return wanted

    def written_artefact(self, state: core.WizardState, path: Path, three_di: str) -> Artefact:
        """Register the 3Di string this step just wrote, so re-reading the library is free.

        `on_check_library` drops the attached 3Di, because a re-read library may be a
        different protein. When it is the same one, the file is still here and the answer to
        "get the 3Di string" is a click rather than another ESMFold run.
        """
        return Artefact(
            kind=STEP_THREE_DI,
            path=Path(path),
            origin="session",
            describes=f"your {len(three_di):,d}-residue wild type",
            length=len(three_di),
            detail="made by the step above",
        )

    def resolve(
        self,
        state: core.WizardState,
        backend: Any,
        *,
        wt_sequence: str,
        artefact: Artefact | None,
        work_dir: Path,
        has_gpu: bool,
    ) -> str:
        """Produce the 3Di string this run will be trained with. Raises with what to fix."""
        problems = three_di_blockers(state, artefact)
        if problems:
            raise ValueError(" ".join(problems))
        length = len(wt_sequence)
        source = state.three_di_source
        if source in {"bundled_example", "session"}:
            assert artefact is not None  # three_di_blockers refused this above
            return backend.load_three_di(artefact.path, expected_length=length)
        if source == "paste":
            return backend.validate_three_di(str(state.get("three_di_text") or ""), length)
        if source == "upload_3di":
            return backend.load_three_di(
                Path(str(state.get("three_di_file")).strip()), expected_length=length
            )
        if source == "upload_structure":
            return backend.three_di_from_structure(
                Path(str(state.get("structure_file")).strip()),
                chain=str(state.get("chain") or "").strip() or None,
                expected_length=length,
                expected_sequence=wt_sequence,
            )
        return backend.three_di_from_esmfold(
            wt_sequence,
            device="cuda" if has_gpu else "cpu",
            pdb_out=Path(work_dir) / "wt_esmfold.pdb",
        )


class RegionSection:
    """Get the pooling region `cosine_*` pooling reads: reuse one, upload one, or discover one."""

    key = STEP_REGION

    def __init__(self, state: core.WizardState, tools: SectionTools) -> None:
        import ipywidgets

        wide = {"width": "560px"}
        self.fields: dict[str, Any] = {}
        choices = region_choices(state, None)
        values = [value for _label, value in choices]
        current = str(state.get("region_source") or "")
        self.fields["region_source"] = ipywidgets.RadioButtons(
            options=choices,
            value=current if current in values else "upload",
            layout={"width": "max-content"},
            style=_LABEL,
        )
        self.fields["region_filename"] = tools.upload_row(
            "region_filename", "Upload region JSON", "a pooling-region JSON", "region"
        )
        self.fields["region_model"] = ipywidgets.Dropdown(
            options=[(f"{model.label}  ({model.hf_id})", model.hf_id) for model in REGION_MODELS],
            value=str(state.get("region_model") or DEFAULT_REGION_MODEL),
            description="Read the embeddings from:",
            style=_LABEL,
            layout=wide,
        )
        self.fields["region_n_sample"] = ipywidgets.BoundedIntText(
            value=int(state.get("region_n_sample") or 0),
            min=0,
            max=10_000_000,
            description="Variants to score (0 = all):",
            style=_LABEL,
            layout=wide,
        )
        self.fields["region_seed"] = ipywidgets.BoundedIntText(
            value=int(state.get("region_seed") or 0),
            min=0,
            max=10_000,
            description="Sampling seed:",
            style=_LABEL,
            layout=wide,
        )
        self.fields["region_batch_size"] = ipywidgets.BoundedIntText(
            value=int(state.get("region_batch_size") or 2),
            min=1,
            max=16,
            description="Batch size:",
            style=_LABEL,
            layout=wide,
        )
        self.fields["region_run_on_cpu"] = ipywidgets.Checkbox(
            value=bool(state.get("region_run_on_cpu")),
            description="Run region discovery on the CPU anyway",
            indent=False,
            style=_LABEL,
        )
        self.note = theme.note("")
        self.preview = theme.html("")
        self.button = tools.button("Use this region", "primary")

    def children(self) -> list[Any]:
        """The widgets, in reading order."""
        return [
            self.note,
            self.fields["region_source"],
            self.fields["region_filename"],
            self.fields["region_model"],
            self.fields["region_n_sample"],
            self.fields["region_seed"],
            self.fields["region_batch_size"],
            self.fields["region_run_on_cpu"],
            self.preview,
        ]

    def sync(
        self,
        state: core.WizardState,
        artefact: Artefact | None,
        *,
        preview: SubsamplePreview,
        estimate: RuntimeEstimate,
    ) -> str | None:
        """Rebuild the source list, the estimate and the button label. Returns a forced value."""
        forced: str | None = None
        choices = region_choices(state, artefact)
        values = [value for _, value in choices]
        widget = self.fields["region_source"]
        if tuple(widget.options) != tuple(choices):
            current = str(state.get("region_source") or "")
            forced = current if current in values else default_region_source(state, artefact)
            widget.options = choices
            widget.value = forced
        discovering = str(state.get("region_source")) == "discover"
        self.preview.value = preview_html(preview, estimate) if discovering else ""
        core.set_display(self.preview, discovering)
        self.button.description = (
            f"Discover the region ({span_of(estimate.low_minutes or 0.0, estimate.high_minutes or 0.0)})"
            if discovering and estimate.known
            else ("Discover the region" if discovering else "Use this region")
        )
        return forced

    def resolve(
        self,
        state: core.WizardState,
        backend: Any,
        *,
        artefact: Artefact | None,
        sequences: Sequence[str],
        wt_sequence: str,
        work_dir: Path,
        has_gpu: bool,
        progress: Callable[[int, int, str], None] | None = None,
        say: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, Any], Path]:
        """Produce the region record this run will pool over, and the file it lives in."""
        problems = region_blockers(state, artefact)
        if problems:
            raise ValueError(" ".join(problems))
        source = str(state.get("region_source") or "")
        if source == "available":
            assert artefact is not None  # region_blockers refused this above
            return backend.load_region(artefact.path), artefact.path
        if source == "upload":
            path = Path(str(state.get("region_filename")).strip())
            return backend.load_region(path), path
        percentile = pooling_percentile(state.pooling)
        assert percentile is not None  # region_blockers refused a pooling without one
        if say is not None:
            preview = region_preview(state)
            estimate = region_estimate(state, has_gpu=has_gpu, preview=preview)
            say(
                f"scoring {preview.n_scored:,} of {preview.n_library:,} variants (seed {preview.seed}); "
                f"{estimate.sentence()}"
            )
        records = backend.discover_region(
            list(sequences),
            wt_sequence,
            percentiles=[percentile],
            n_sample=int(state.get("region_n_sample") or 0),
            hf_id=str(state.get("region_model") or DEFAULT_REGION_MODEL),
            device="cuda" if has_gpu else "cpu",
            batch_size=int(state.get("region_batch_size") or 2),
            seed=int(state.get("region_seed") or 0),
            progress=progress,
        )
        name = f"cosine_p{int(percentile)}"
        record = records.get(name) or next(iter(records.values()))
        path = Path(work_dir) / region_file_name(percentile)
        backend.save_region(record, path)
        return record, path

    def discovered_artefact(self, state: core.WizardState, path: Path, record: dict[str, Any]) -> Artefact:
        """Register what discovery just wrote, so the next refresh offers it instead of a rerun."""
        selected = record.get("n_selected_positions", len(record.get("selected_positions_1based", []) or []))
        return Artefact(
            kind=STEP_REGION,
            path=Path(path),
            origin="session",
            describes=f"your {int(state.wt_length):,d}-residue wild type",
            pooling=str(state.pooling),
            detail=f"{selected} residues, from {record.get('n_variants_scored', '?')} variants",
        )
