"""The wizard for `ColabSeqDisplay.ipynb` — the notebook users spend their time in.

The notebook cell stays thin: install, import, `launch()`. Everything the user sees is
built here, and every decision it makes is a pure function so it can be tested without a
browser.

`colabsd.ui.core` owns the machinery this shares with the Predict notebook: the
`WizardState`, the field-visibility rules, the contextual messages, runtime detection and
the thin ipywidgets wrappers. This module owns what is specific to training: turning the
form into a `LibrarySpec`, assembling the exact call into `colabsd.train.finetune`,
presenting the looked-up hyperparameters, and the outlets that only exist once something
has been trained — the results, the two exports, and a scored table.

**Preparation is part of this page.** It used to be a notebook of its own, which meant the
backbone question — the one whose answer decides everything else — was asked twice, with a
`wt_3di.txt` downloaded between them. It is asked once now, in step 2, and step 3 is
whatever that answer implies: a SaProt backbone reads structure and needs a wild-type 3Di
string, which `colabsd.ui.prepare_workflow` builds the section for and which stays in the
session. An ESM2 backbone needs nothing, and then there is no step 3 at all.

Two deliberate differences from the ColabPLM notebooks we are otherwise copying:

* **The modelling choices are read-only.** They hand the user LoRA and trainer widgets; we
  look the values up from a study that was already run and show them with their provenance
  and their measured test Spearman, or, for a placeholder entry, in red. A hyperparameter
  re-tuned while you watch your own validation score has quietly eaten your test set. How
  the model is read out is decided the same way and is not on the form either: the pooled
  feature is the mean of the embeddings at the mutated sites. Three settings are the
  exception — the epoch ceiling, the early-stopping patience and the micro batch — because
  they are budget rather than modelling: how long the user is willing to train, and what
  fits on the card they were given. They are prefilled from the same lookup, validated
  before the run, and whatever they end up as is what the bundle and the archive record.
* **The test set is not unlocked here.** There is no widget on this page that can read it.
  That is a separate cell, run on purpose, which counts every unlock. The performance
  archive is written from validation numbers with that partition still locked, because the
  report is what a user should be reading while they are still deciding anything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from html import escape
from pathlib import Path
from typing import Any

from colabsd.backbones.registry import BACKBONES
from colabsd.spec import LibrarySpec
from colabsd.ui import core, exports, theme
from colabsd.ui import prepare_workflow as prep

# --------------------------------------------------------------------------------------
# What is on the page
# --------------------------------------------------------------------------------------

#: Page order, which is also everywhere a contextual message can be put. The keys are
#: `core.SECTION_ORDER`'s, so a section this page shows and a section `core` has a rule for
#: are the same section under the same name.
PAGE_ORDER: tuple[str, ...] = (
    "data",
    "model",
    "structure",
    "hyperparameters",
    "training",
    "storage",
    "run",
    "results",
    "export",
    "score",
)

#: Sections this module adds to `core.SECTION_ORDER`, gated on progress rather than choice.
OUTLET_SECTIONS: tuple[str, ...] = ("results", "export", "score")

#: Fields this wizard adds to `core.FIELD_KEYS`. `core.apply_field_visibility` leaves an
#: unknown key visible, so these are toggled from `outlet_field_visibility` instead.
OUTLET_FIELDS: tuple[str, ...] = (
    "example_note",
    "bundle_name",
    "archive_name",
    "bundle_notes",
    "score_source",
    "score_top_n",
    "score_random_seed",
    "score_variants_csv",
    "score_rank_by",
)

#: Section names without their numbers: which step a section *is* depends on which other
#: sections this configuration has, and `numbered_titles` works that out at every refresh.
#: A page that skips from 2 to 4 reads as a step the user has failed to find.
SECTION_TITLES: dict[str, str] = {
    "data": "Your variant library",
    "model": "The backbone",
    "structure": "The wild-type shape, as a 3Di string",
    "hyperparameters": "The hyperparameters for this backbone",
    "training": "How many runs",
    "storage": "Where the results are kept",
    "run": "Train",
    "results": "What came back",
    "export": "The two things you take away",
    "score": "Score some variants",
}

SECTION_NOTES: dict[str, str] = {
    "data": (
        "One row per variant: the residue at each mutated site, and what you measured for it. Full-length "
        "sequences are **built** by substituting those residues into your wild type, never read from a FASTA."
    ),
    "model": (
        "Which protein language model reads each sequence. The default reads sequence only, fits a free T4 "
        "and needs nothing prepared. Every backbone here is read out the same way: the mean of the "
        "embeddings at your mutated sites. A SaProt backbone adds a preparation step below; ESM2 adds none."
    ),
    "structure": "",  # `prepare_workflow.section_note` writes this one: it names the backbone that caused it.
    "hyperparameters": (
        "Looked up for the backbone above and shown with its provenance. The learning rates, the LoRA "
        "settings and the batch that trains were selected against a held-out score and are not editable. "
        "The three boxes at the bottom are yours: how long you train, and what fits on your card."
    ),
    "training": "A run is one split seed x one model seed, and the ± you report is the spread across them.",
    "storage": core.DRIVE_NOTE,
    "run": (
        "LoRA adapters on the attention projections plus a small head; the backbone stays frozen. Each run "
        "trains on the training split and early-stops on validation. **Validation numbers only** come back: "
        "the test partition is moved out of reach as each run finishes, and only the last cell of this "
        "notebook can open it, which counts every unlock."
    ),
    "results": (
        "What the fine-tuned model scored on the **validation** partition of every run: the macro average "
        "over conditions, and one row per condition and metric. The test partition is untouched and stays "
        "that way until the last cell of this notebook."
    ),
    "export": (
        "- **The model**, as one `.zip` the Predict notebook reads: the LoRA weights, the head, your library "
        "description, the *frozen* pooled coordinates, the hyperparameters and the provenance.\n"
        "- **The performance**, as a second `.zip`: the report CSV, the figure, and a `performance.json` "
        "saying which partition the numbers describe and how many times the test set has been read. It is "
        "written from **validation** numbers with the test partition still locked; unlock it later and press "
        "again for the same archive carrying the test numbers and the count."
    ),
    "score": (
        "The smallest useful prediction outlet. For a real screen open "
        "**ColabSeqDisplay_Predict.ipynb**, which needs nothing but the `.zip` above."
    ),
}

#: Which section a contextual message belongs beside, right under the control that caused it.
#: A message whose section is hidden falls back to the board above the train button, so
#: nothing is ever silently dropped.
MESSAGE_SECTIONS: dict[str, str] = {
    "min_count_without_count_column": "data",
    "library_not_loaded": "data",
    "unknown_backbone": "model",
    "backbone_not_in_colab": "model",
    "backbone_needs_big_gpu": "model",
    "backbone_extra_install": "model",
    "backbone_withdrawn": "model",
    "three_di_missing": "structure",
    "three_di_not_attached": "structure",
    "three_di_length_mismatch": "structure",
    "three_di_reuse_available": "structure",
    "three_di_reuse_gone": "structure",
    "esmfold_too_long": "structure",
    "esmfold_expensive": "structure",
    "esmfold_needs_gpu": "structure",
    "esmfold_risk_not_accepted": "structure",
    "config_provisional": "hyperparameters",
    "config_missing": "hyperparameters",
    "config_unreadable": "hyperparameters",
    "budget_refused": "hyperparameters",
    "single_run": "training",
    "run_exceeds_session": "training",
    "no_persistence": "storage",
    "drive_on": "storage",
    "results_stale": "run",
    "scoring_on_cpu": "score",
}

#: `core` marks a message `stop` when it means "this is a bad idea"; only some of those make
#: the run impossible. An unsaved run lost to a disconnect is a bad idea; a missing 3Di
#: string is a crash three minutes in. The button refuses the second kind only, and shows
#: both. The refusals that belong to the preparation *button* — an unaccepted ESMFold risk —
#: are not here: they stop that step, and the step not having finished is what stops the run.
BLOCKING_KEYS: frozenset[str] = frozenset(
    {
        "library_not_loaded",
        "no_gpu",
        "unknown_backbone",
        "backbone_not_in_colab",
        "backbone_needs_big_gpu",
        "backbone_withdrawn",
        "three_di_missing",
        "three_di_not_attached",
        "three_di_length_mismatch",
        "three_di_reuse_gone",
        "esmfold_too_long",
        "config_missing",
        "config_unreadable",
        # A budget the training loop cannot honour: zero epochs, a patience nothing can reach,
        # a micro batch that does not divide the effective one or does not fit the training
        # split. Every one of those is a crash or an empty run a minute in, so the button
        # refuses rather than warns, and `colabsd.bestconfig` supplies the sentence.
        "budget_refused",
    }
)

#: The bundled example, described once here and read from its own files everywhere else.
EXAMPLE_NAME = "MG8 PETases"
EXAMPLE_POSITIONS = "22, 42, 49, 72, 78, 79, 80, 130, 131, 132, 150, 198, 199, 205, 206, 215, 216, 225, 279"
EXAMPLE_MUTATION_COLUMNS = (
    "p22, p42, p49, p72, p78, p79, p80, p130, p131, p132, p150, p198, p199, p205, p206, p215, p216, p225, p279"
)
EXAMPLE_CONDITION_COLUMNS = "activity"
EXAMPLE_DIR = ("mg8_petases",)


def example_path(*parts: str) -> Path:
    """A file inside the bundled example, found through the package rather than a machine path."""
    from colabsd import EXAMPLES_ROOT

    return Path(EXAMPLES_ROOT).joinpath(*EXAMPLE_DIR, *parts)


def example_wt_length() -> int:
    """Residues in the bundled wild type, counted from `wt.fasta`. 0 when it cannot be read."""
    try:
        text = example_path("wt.fasta").read_text()
    except OSError:
        return 0
    return len("".join(line.strip() for line in text.splitlines() if not line.startswith(">")))


def example_variants() -> int:
    """Rows in the bundled example library, counted from the CSV. 0 when it cannot be read."""
    try:
        with example_path("library.csv").open("rb") as handle:
            rows = sum(1 for line in handle if line.strip())
    except OSError:
        return 0
    return max(rows - 1, 0)


def example_test_rows() -> int:
    """How many of the bundled variants an 8:1:1 split leaves to test on.

    `create_split` needs a directory to write into, so the sentence repeats its arithmetic
    rather than the answer: a typed-in count survives the CSV it was counted from.
    """
    n = example_variants()
    return n - int(0.8 * n) - int(0.1 * n)


def example_note() -> str:
    """The bundled example described from the files themselves, so the prose cannot drift."""
    variants = example_variants()
    length = example_wt_length()
    size = f"{variants:,d} variants" if variants else "the variants"
    protein = f"a {length:,d}-residue PET hydrolase" if length else "a PET hydrolase"
    sites = len(parse_list(EXAMPLE_MUTATION_COLUMNS))
    note = (
        f"The bundled **{EXAMPLE_NAME}** library: {size} of {protein}, {sites} mutated sites against one "
        f"measured condition, `{EXAMPLE_CONDITION_COLUMNS}` — the log of the assay reading. It runs, end "
        "to end, inside one Colab session, and it is far too small to measure a backbone with"
    )
    rows = example_test_rows()
    return note + (f": an 8:1:1 split leaves {rows} variants to test on." if rows > 0 else ".")


def withdrawn_note() -> str:
    """Where the backbones that used to be on this list went, one line per name.

    Somebody who came here for ProtT5 or ESMC deserves an answer rather than a shorter list
    and no explanation, and the answer is not "they were deleted". It is also not one
    sentence: this note used to say every withdrawn backbone had a tested adapter and no
    tuned hyperparameters, and that was wrong four times over — METL has no adapter at all,
    and ProtT5-XL, ESMC-300M, ESMDance and METL each ship a tuned `cosine_p90_mean` entry in
    `config/best/` (the half `colabsd.bestconfig` never opens, so `hyperparameter_state`
    still calls all four placeholders).
    So the reason is asked of `colabsd.backbones.registry` per name, which is where it is
    written and where the notebook's own guide reads it from.
    """
    from colabsd.backbones import registry

    withdrawn = core.withdrawn_backbones()
    if not withdrawn:
        return ""
    families = " and ".join(registry.OFFERED_FAMILIES)
    lines = [
        f"**Only {families} are offered here** — read the sequence, or read the sequence and the shape. "
        f"The other {len(withdrawn)} of the {len(BACKBONES)} are in the package and not on this form:"
    ]
    for name in withdrawn:
        tail = (
            "still builds from Python"
            if registry.has_adapter(name)
            else "no adapter here, so there is nothing to run it with"
        )
        try:
            reason = registry.withheld_reason(name) or "it is offered"
        except Exception:
            # A family in neither list is a registry bug. It must not stop the page building.
            reason = "no reason is recorded for it in the registry, which is a bug worth reporting"
        lines.append(f"- `{name}` — {reason}; {tail}.")
    lines.append(
        "To put one back on the form, move its family out of `WITHHELD_FAMILY_REASONS` and into "
        "`OFFERED_FAMILIES` in `colabsd/backbones/registry.py`."
    )
    return "\n".join(lines)


LIBRARY_NOTE = (
    "- **Mutated positions** — where the mutated residues sit in the wild type, counted from 1.\n"
    "- **Mutation columns** — the CSV columns holding the residue at each of those positions: column *i* "
    "names position *i*.\n"
    "- **Condition columns** — the CSV columns holding what you measured; one per condition, all "
    "predicted at once.\n"
    "- **Read-count column** — name it if your table has sequencing counts, and the box below can drop "
    "the variants seen too few times. Leave it empty if there is none."
)


#: Everything this wizard keeps in `WizardState.extra`, and what it starts as.
EXTRA_DEFAULTS: dict[str, Any] = {
    "library_loaded": False,
    "trained": False,
    "exported": False,
    "library_csv": "",
    "wt_sequence": "",
    "positions_1based": EXAMPLE_POSITIONS,
    "mutation_columns": EXAMPLE_MUTATION_COLUMNS,
    "condition_columns": EXAMPLE_CONDITION_COLUMNS,
    "three_letter_residues": False,
    "count_column": "",
    "min_count": 0,
    "three_di_text": "",
    "three_di_file": "",
    "structure_file": "",
    "chain": "",
    "esmfold_risk_accepted": False,
    "run_name": "run",
    "resume_finished_runs": True,
    "bundle_name": exports.DEFAULT_BUNDLE_NAME,
    "archive_name": exports.DEFAULT_ARCHIVE_NAME,
    "bundle_notes": "",
    "score_source": "library_head",
    "score_variants_csv": "",
    "score_top_n": 32,
    "score_random_seed": 0,
    "score_rank_by": "pred_mean",
    "score_batch_size": 8,
    "trained_fingerprint": None,
}

SCORE_SOURCES: tuple[str, ...] = ("library_head", "random_combinations", "upload_csv")

#: Only used when a registry entry declares no evaluation protocol of its own; every entry
#: in the shipped registry does, and `training_seeds` prefers those.
FALLBACK_SPLIT_SEEDS: tuple[int, ...] = (1, 2, 3)
FALLBACK_MODEL_SEEDS: tuple[int, ...] = (11, 22, 33)

#: Widgets that only ever display something `refresh` computes. They have a `.value`, so
#: binding them would write rendered HTML into the state and re-enter `refresh`.
DISPLAY_ONLY: frozenset[str] = frozenset({"hyperparameters", "esmfold_note", "example_note"})

#: The three looked-up settings this panel hands back to the user, in the order they are
#: asked. A subset of `colabsd.bestconfig.BUDGET_FIELDS`, which also holds
#: `effective_batch_size`: that one is the batch the optimiser averages over, a study selected
#: it against a held-out score, and gradient accumulation makes it up out of whatever micro
#: batch the card can hold. So it stays looked up, and what a user can move is time and memory.
BUDGET_KEYS: tuple[str, ...] = ("max_epochs", "early_stopping_patience", "micro_batch_size")

#: The training share of the 8:1:1 split `colabsd.engine.splits.create_split` cuts.
TRAIN_FRACTION: float = 0.8

#: The label on each budget box, in `BUDGET_KEYS` order. These labels are the whole
#: explanation: two of them say time and the third says memory, in the words of the thing it
#: decides rather than the name of the field in the YAML. The one a user reaches for after an
#: out-of-memory error is the one that carries "(memory)".
BUDGET_LABELS: dict[str, str] = {
    "max_epochs": "Epochs, at most:",
    "early_stopping_patience": "Give up after this many epochs with no gain:",
    "micro_batch_size": "Sequences on the GPU at once (memory):",
}

#: Ceilings on what the three boxes accept. High enough never to be the answer to a real
#: question, and finite so a typed digit cannot become a number of epochs nobody meant.
BUDGET_MAXIMA: dict[str, int] = {"max_epochs": 1000, "early_stopping_patience": 1000, "micro_batch_size": 4096}

#: The metrics the results table shows, in the order they are worth reading. Which of them a
#: given run may show is `results_metrics`: a ranking cut-off longer than the partition it
#: ranks is not the metric it names.
RESULTS_METRICS: tuple[str, ...] = ("Spearman", "R2", "NDCG@50")

#: One text row, tall enough for the progress line and no taller.
PROGRESS_ROW_HEIGHT = "1.6em"

#: The progress line is clipped, never wrapped: `colabsd.train.finetune` rewrites it several
#: times a second and every number in it changes width, so a line allowed to wrap would move
#: the page under the reader's eyes. Tabular figures keep the digits from dancing too.
PROGRESS_STYLE = (
    "font-family:monospace;font-variant-numeric:tabular-nums;white-space:nowrap;"
    f"overflow:hidden;text-overflow:ellipsis;height:{PROGRESS_ROW_HEIGHT};line-height:{PROGRESS_ROW_HEIGHT}"
)


def new_state(**changes: Any) -> core.WizardState:
    """A `WizardState` in train mode, prefilled with the bundled example's description."""
    state = core.WizardState(mode="train")
    state.extra.update(EXTRA_DEFAULTS)
    for name, value in changes.items():
        state.set(name, value)
    return state


# --------------------------------------------------------------------------------------
# Pure decisions -- what the user can see
# --------------------------------------------------------------------------------------


def training_fingerprint(state: core.WizardState) -> tuple[Any, ...]:
    """Everything that decides *what* a run is, so a change to any of it invalidates one.

    Where the results are written is deliberately not in here: mounting Drive half way
    through does not make the numbers on screen belong to a different experiment.
    """
    return (
        state.data_source,
        str(state.get("library_csv") or ""),
        str(state.get("wt_sequence") or ""),
        str(state.get("positions_1based") or ""),
        str(state.get("mutation_columns") or ""),
        str(state.get("condition_columns") or ""),
        bool(state.get("three_letter_residues")),
        str(state.get("count_column") or ""),
        int(state.get("min_count") or 0),
        state.backbone,
        state.dtype,
        state.three_di_source,
        int(state.wt_3di_length),
        int(state.n_split_seeds),
        int(state.n_model_seeds),
        int(state.max_epochs),
        int(state.early_stopping_patience),
        int(state.micro_batch_size),
    )


def results_are_stale(state: core.WizardState) -> bool:
    """True when the form has been changed since the run whose numbers are on screen.

    Without this the page will happily show one backbone's validation table under another
    backbone's name, and -- worse -- write the second one's hyperparameters into the `.zip`
    the first one's weights are in.
    """
    if not state.get("trained"):
        return False
    recorded = state.get("trained_fingerprint")
    return recorded is not None and tuple(recorded) != training_fingerprint(state)


def has_results(state: core.WizardState) -> bool:
    """True when there is a finished run that still describes what the form says."""
    return bool(state.get("trained")) and not results_are_stale(state)


def has_bundle(state: core.WizardState) -> bool:
    """True when there is a written bundle that still describes what the form says."""
    return bool(state.get("exported")) and not results_are_stale(state)


def section_visibility(state: core.WizardState) -> dict[str, bool]:
    """Which sections are on screen: `core`'s rules, gated on how far the user has got.

    `core` decides which sections this *configuration* calls for. This adds the one thing it
    has no opinion about: nothing below the library appears until the library has loaded,
    because the runtime estimate, the ESMFold length check and the pooled coordinates all
    need its size and its wild-type length.
    """
    live = set(core.visible_sections(state))
    loaded = bool(state.get("library_loaded"))
    trained = has_results(state)
    visible = {name: (name in live and loaded) for name in core.SECTION_ORDER}
    visible["data"] = True
    visible["results"] = trained
    visible["export"] = trained
    visible["score"] = has_bundle(state)
    return {name: visible.get(name, False) for name in PAGE_ORDER}


def visible_sections(state: core.WizardState) -> list[str]:
    """The sections on screen, in page order."""
    visible = section_visibility(state)
    return [name for name in PAGE_ORDER if visible[name]]


def numbered_titles(state: core.WizardState) -> dict[str, str]:
    """Every section's heading, numbered by where it actually falls on this page.

    A step this configuration does not need is absent, not greyed out, so the numbers close
    up behind it: an ESM2 run has no preparation step, and its hyperparameters are step 3.
    """
    shown = visible_sections(state)
    titles = {name: f"{index} · {SECTION_TITLES[name]}" for index, name in enumerate(shown, start=1)}
    for name in PAGE_ORDER:
        titles.setdefault(name, SECTION_TITLES[name])
    return titles


def outlet_field_visibility(state: core.WizardState) -> dict[str, bool]:
    """The fields `core.FIELD_RULES` does not declare, because only this wizard has them."""
    source = state.get("score_source")
    return {
        "example_note": state.data_source == "bundled_example",
        "bundle_name": True,
        "archive_name": True,
        "bundle_notes": True,
        "score_source": True,
        "score_top_n": source in {"library_head", "random_combinations"},
        "score_random_seed": source == "random_combinations",
        "score_variants_csv": source == "upload_csv",
        "score_rank_by": True,
    }


# --------------------------------------------------------------------------------------
# Pure decisions -- the contextual messages
# --------------------------------------------------------------------------------------


def three_di_is_promised_but_missing(state: core.WizardState) -> bool:
    """A 3Di source has been chosen for a structure backbone, and no string has arrived.

    Silent when the chosen source is itself already refused -- an ESMFold run that will not
    fit, or one whose memory risk has not been accepted, is one message, not two, and
    "press the button" is the wrong advice for it.
    """
    if not core.needs_structure(state) or state.three_di_source in ("", "none"):
        return False
    if state.wt_3di_length:
        return False
    if state.three_di_source == "esmfold":
        too_long = state.wt_length > core.esmfold_safe_length()
        return not (too_long or not state.get("esmfold_risk_accepted"))
    return True


def extra_messages(state: core.WizardState) -> list[core.Message]:
    """The messages this notebook needs and neither `core` nor `prepare_workflow` knows about."""
    out: list[core.Message] = []
    if not state.get("library_loaded"):
        out.append(
            core.Message(
                "library_not_loaded",
                "stop",
                "Your library has not been read yet. Press **Check my library** below.",
            )
        )
    if int(state.get("min_count") or 0) > 0 and not str(state.get("count_column") or "").strip():
        out.append(
            core.Message(
                "min_count_without_count_column",
                "warning",
                f"You asked to drop variants seen fewer than {state.get('min_count')} times, but no read-count "
                "column is named, so every row will be kept. Name the column above, or set the threshold to 0.",
            )
        )
    if three_di_is_promised_but_missing(state):
        out.append(
            core.Message(
                "three_di_not_attached",
                "stop",
                f"You chose where the 3Di string should come from, but nothing has fetched it yet, so "
                f"**{state.backbone}** would be trained with no structure at all. Press **Get the 3Di "
                "string** in the preparation step.",
            )
        )
    refusal = training_budget(state).settle()[1]
    if refusal:
        out.append(core.Message("budget_refused", "stop", f"**That budget cannot train.** {refusal}"))
    if results_are_stale(state):
        out.append(
            core.Message(
                "results_stale",
                "warning",
                "The settings have changed since the last run, so the results, the bundle and the scoring "
                "outlet have been put away. Press **Train** again, or change the settings back.",
            )
        )
    return out


def preparation_messages(
    state: core.WizardState,
    *,
    runtime: core.Runtime | None = None,
    artefact: prep.Artefact | None = None,
) -> list[core.Message]:
    """`prepare_workflow.notices` for this state, with the machine filled in.

    Silent when this backbone needs nothing prepared, which is the point of the merge: the
    step that does not exist says nothing either.
    """
    return prep.notices(
        state,
        artefact=prep.session_artefact(state, artefact),
        has_gpu=None if runtime is None else bool(runtime.has_gpu),
    )


def messages(
    state: core.WizardState,
    *,
    runtime: core.Runtime | None = None,
    status: core.ConfigStatus | None = None,
    root: str | Path | None = None,
    artefact: prep.Artefact | None = None,
) -> list[core.Message]:
    """Every message this state earns, worst first: `core`'s, preparation's, and this page's."""
    collected = list(core.messages_for(state, runtime=runtime, status=status, root=root))
    collected += extra_messages(state)
    collected += preparation_messages(state, runtime=runtime, artefact=artefact)
    if runtime is not None and not runtime.has_gpu and has_bundle(state):
        collected.append(
            core.Message(
                "scoring_on_cpu",
                "info",
                "With no GPU this falls back to the CPU. A few dozen variants is fine; thousands are not.",
            )
        )
    seen: set[str] = set()
    unique = []
    for item in collected:
        if item.key in seen:
            continue
        seen.add(item.key)
        unique.append(item)
    unique.sort(key=lambda item: theme.SEVERITY_RANK[item.severity])
    return unique


def message_keys(state: core.WizardState, **kwargs: Any) -> list[str]:
    """The keys of the messages this state earns; the shape a test usually wants."""
    return [item.key for item in messages(state, **kwargs)]


def messages_by_section(
    items: Iterable[core.Message], visible: dict[str, bool] | None = None
) -> dict[str, list[core.Message]]:
    """Group messages by the section they belong beside, falling back to the run board.

    A message whose section is hidden would otherwise be invisible, which is how a wizard
    ends up refusing to start for a reason nobody can see.
    """
    grouped: dict[str, list[core.Message]] = {name: [] for name in PAGE_ORDER}
    for item in items:
        section = MESSAGE_SECTIONS.get(item.key, "run")
        if visible is not None and not visible.get(section, False):
            section = "run"
        grouped[section].append(item)
    return grouped


def blockers(items: Iterable[core.Message]) -> list[core.Message]:
    """The messages that actually stop the train button."""
    return [item for item in items if item.key in BLOCKING_KEYS]


def can_train(items: Iterable[core.Message]) -> bool:
    """True when nothing among these messages makes the run impossible."""
    return not blockers(items)


# --------------------------------------------------------------------------------------
# Pure decisions -- turning the form into a library
# --------------------------------------------------------------------------------------


def parse_list(text: Any) -> list[str]:
    """Split a comma- or semicolon-separated field into stripped, non-empty items."""
    return [item.strip() for item in str(text or "").replace(";", ",").split(",") if item.strip()]


def parse_positions(text: Any) -> list[int]:
    """Parse the 1-based mutated positions, naming the token that is not a number."""
    positions = []
    for item in parse_list(text):
        try:
            positions.append(int(item))
        except ValueError:
            raise ValueError(
                f"{item!r} is not a residue number. Mutated positions are whole numbers counted from 1 "
                "along the wild-type sequence."
            ) from None
    return positions


def library_spec(state: core.WizardState, wt_sequence: str, *, wt_3di: str | None = None) -> LibrarySpec:
    """Build the `LibrarySpec` the rest of the pipeline runs on. Reads no files."""
    return LibrarySpec(
        wt_sequence=wt_sequence,
        positions_1based=parse_positions(state.get("positions_1based")),
        mutation_columns=parse_list(state.get("mutation_columns")),
        condition_columns=parse_list(state.get("condition_columns")),
        three_letter=bool(state.get("three_letter_residues")),
        count_column=str(state.get("count_column") or "").strip() or None,
        wt_3di=wt_3di,
    )


def load_blockers(state: core.WizardState) -> list[str]:
    """What stops the check-my-library button, each said as the thing to go and fix."""
    problems = []
    if state.data_source != "bundled_example":
        if not str(state.get("library_csv") or "").strip():
            problems.append("No CSV yet. Upload one with the button above, or type the path of a file on this machine.")
        if not str(state.get("wt_sequence") or "").strip():
            problems.append(
                "The wild-type box is empty. Paste the full-length amino-acid sequence, or the URL of a "
                "FASTA file."
            )
    positions = parse_positions(state.get("positions_1based"))
    columns = parse_list(state.get("mutation_columns"))
    if not columns:
        problems.append("No mutation columns named. List one CSV column per mutated site, in position order.")
    if not parse_list(state.get("condition_columns")):
        problems.append("No condition columns named. List at least one CSV column holding a measured activity.")
    if columns and len(columns) != len(positions):
        problems.append(
            f"{len(columns)} mutation columns for {len(positions)} positions. Name exactly one column per "
            "position, in the same order."
        )
    return problems


#: What stops the preparation button is `prepare_workflow`'s decision, because the controls
#: that button reads are its: `prep.three_di_blockers`.


def export_blockers(state: core.WizardState) -> list[str]:
    """What stops the export button."""
    if not state.get("trained"):
        return ["Nothing has been trained in this session, so there is no model to export."]
    if results_are_stale(state):
        return [
            "The settings have changed since this model was trained, so the bundle would carry the old "
            "run's weights under the settings on screen. Press Train again first."
        ]
    return []


def score_blockers(state: core.WizardState) -> list[str]:
    """What stops the score button."""
    problems = []
    if not has_bundle(state):
        problems.append(
            "There is no model bundle in this session that matches the settings on screen. Write one above, "
            "or open ColabSeqDisplay_Predict.ipynb with a model_bundle.zip you saved earlier."
        )
    if state.get("score_source") == "upload_csv" and not str(state.get("score_variants_csv") or "").strip():
        problems.append("No variants CSV yet. Upload one with the button above.")
    return problems


# --------------------------------------------------------------------------------------
# Pure decisions -- the adapter, and the call into colabsd.train
# --------------------------------------------------------------------------------------


def adapter_kwargs(state: core.WizardState, spec: LibrarySpec) -> dict[str, Any]:
    """The keyword arguments `colabsd.backbones.registry.create_adapter` is called with.

    The pooled coordinates are the mutated sites, which is where the model is read out for
    every backbone: the sites are the only residues the library varies, and they are the
    ones the assay moved. Nothing on the page decides this and nothing else can be chosen.
    """
    return {
        "pooling_positions_0based": list(spec.positions_0based()),
        "wt_3di": spec.wt_3di,
        "dtype": state.dtype,
    }


def training_seeds(state: core.WizardState, best: Any | None = None) -> tuple[list[int], list[int]]:
    """The split and model seeds this run will use.

    They are the first *n* of the registry entry's own evaluation protocol, so a one-run
    smoke test is the first run of the real protocol rather than a different experiment.
    """
    split_seeds = [int(seed) for seed in (getattr(best, "split_seeds", None) or FALLBACK_SPLIT_SEEDS)]
    model_seeds = [int(seed) for seed in (getattr(best, "model_seeds", None) or FALLBACK_MODEL_SEEDS)]
    return (
        split_seeds[: max(1, int(state.n_split_seeds))],
        model_seeds[: max(1, int(state.n_model_seeds))],
    )


def work_dir(state: core.WizardState) -> Path:
    """Where this session writes. Relative by default; the Drive mount replaces it."""
    return Path(str(state.output_dir or core.DEFAULT_WORK_DIR))


def run_dir(state: core.WizardState) -> Path:
    """The directory one fine-tune writes into."""
    return work_dir(state) / (str(state.get("run_name") or "run").strip() or "run")


def split_dir(state: core.WizardState, n_rows: int) -> Path:
    """Where cached train/val/test splits for a library of this size live."""
    return work_dir(state) / "splits" / f"n{int(n_rows)}"


def finetune_kwargs(
    state: core.WizardState,
    *,
    spec: LibrarySpec,
    sequences: Sequence[str],
    targets: Any,
    adapter: Any,
    best: Any,
    splits: dict[int, dict[str, list[int]]],
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Exactly the call `colabsd.train.finetune` is about to receive.

    `budget` is the one part of the registry entry the user may move, and only the settings
    they actually moved go in it: `finetune` resolves the rest from `best`, records what it
    ran with on the `RunResult`, and `colabsd.bundle` and `colabsd.ui.exports` write that
    record into the manifest and the performance archive. So the numbers on the form are the
    numbers that train and the numbers that are reported, through one object rather than three.
    """
    _, model_seeds = training_seeds(state, best)
    return {
        "spec": spec,
        "sequences": sequences,
        "targets": targets,
        "adapter": adapter,
        "best": best,
        "splits": splits,
        "output_dir": run_dir(state),
        "model_seeds": model_seeds,
        "budget": training_budget(state).overrides(),
        "progress": progress,
        "resume": bool(state.get("resume_finished_runs")),
    }


# --------------------------------------------------------------------------------------
# Pure decisions -- presenting what came back
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HyperparameterView:
    """A registry entry as this wizard presents it: a settled result, or a placeholder."""

    headline: str
    status: str
    is_provisional: bool
    provenance: str
    test_spearman: str
    protocol: str
    lora_rows: list[tuple[str, str]]
    training_rows: list[tuple[str, str]]
    source_path: str


def _format_value(value: Any) -> str:
    return f"{value:.10g}" if isinstance(value, float) else str(value)


def hyperparameter_view(best: Any, budget: TrainingBudget | None = None) -> HyperparameterView:
    """Present a registry entry as a settled result, or as a placeholder that says so.

    `budget` is what the form is set to, when there is a form. The three settings it covers
    are in the training block below, and a reader who has moved one would otherwise find the
    looked-up number in this table and the number they typed in a box further down with
    nothing joining them: those rows are shown as `20 → 40`, which is how
    `colabsd.bestconfig.BudgetSettings.describe_changes` says the same thing.
    """
    mean = getattr(best, "test_spearman", None)
    sd = getattr(best, "test_spearman_sd", None)
    n_runs = getattr(best, "n_runs", None)
    if best.is_provisional or mean is None:
        spearman = "unknown — nothing has been measured for this pair"
    else:
        spearman = f"{mean:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")
        if n_runs:
            spearman += f" over {n_runs} re-evaluation runs"
    meta = dict(getattr(best, "meta", {}) or {})
    provenance = str(meta.get("source") or "not recorded")
    if meta.get("optuna_trial") is not None:
        provenance += f", Optuna trial {meta['optuna_trial']}"
    split_seeds = getattr(best, "split_seeds", []) or []
    model_seeds = getattr(best, "model_seeds", []) or []
    return HyperparameterView(
        headline=best.describe(),
        status="provisional" if best.is_provisional else "tuned",
        is_provisional=bool(best.is_provisional),
        provenance=provenance,
        test_spearman=spearman,
        protocol=(
            f"{len(split_seeds)} split seeds × {len(model_seeds)} model seeds = "
            f"{len(split_seeds) * len(model_seeds)} runs, selected on {getattr(best, 'objective', 'unknown')}"
        ),
        lora_rows=[(name, _format_value(value)) for name, value in sorted(dict(best.params).items())],
        training_rows=[
            (name, _looked_up_value(name, value, budget)) for name, value in sorted(dict(best.fixed).items())
        ],
        source_path=str(getattr(best, "path", "") or "the bundled registry"),
    )


def _looked_up_value(name: str, value: Any, budget: TrainingBudget | None) -> str:
    """One training-block value, saying what the form made of it when that is not the same."""
    shown = _format_value(value)
    if budget is None or name not in budget.changed:
        return shown
    return f"{shown} → {int(getattr(budget, name))}"


def _rows_html(rows: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f"<tr><td style='padding:1px 14px 1px 0'><code>{name}</code></td><td>{value}</td></tr>"
        for name, value in rows
    )
    return f"<table>{cells}</table>"


def hyperparameter_html(view: HyperparameterView) -> str:
    """The looked-up hyperparameters, as a settled result or as a red placeholder."""
    if view.is_provisional:
        # `view.headline` is `BestConfig.describe()`, which is also `core.config_provisional`'s
        # text on the board below this table. Printing it here as well put the same sentence in
        # two coloured boxes a few lines apart, so this banner says only what it alone can: the
        # numbers directly under it were not selected. Where they came from is the provenance
        # line, and what to do about it is core's message.
        banner = theme.message_html(
            "**PROVISIONAL.** No study selected the numbers below for this backbone.",
            "stop",
        )
    else:
        banner = theme.note_html(
            f"**{view.headline}** Selected once and re-evaluated: **test Spearman {view.test_spearman}**."
        )
    return (
        banner
        + "<div style='margin-top:8px'><b>LoRA hyperparameters</b>"
        + _rows_html(view.lora_rows)
        + "</div><div style='margin-top:8px'><b>Training block</b>"
        + _rows_html(view.training_rows)
        + "</div>"
        + theme.note_html(
            f"Provenance: {view.provenance}\n"
            f"- Evaluation protocol: {view.protocol}\n"
            f"- Read from `{view.source_path}`"
        )
    )


def no_hyperparameters_html(backbone: str, status: core.ConfigStatus | None) -> str:
    """The hyperparameter section when the registry has nothing for this backbone.

    It has to say something, because leaving the previous backbone's numbers on screen is
    how a user reads seven settled hyperparameters under a heading that has none.
    """
    detail = status.description if status is not None else "The registry has not been read yet."
    return theme.message_html(
        f"**No hyperparameters for {backbone}.** {detail} Pick a backbone the registry has an entry for.",
        "stop",
    )


#: The one word beside a box that has been moved off the lookup, coloured like a warning
#: rather than a failure: a budget set by hand is allowed, and only has to be visible.
BUDGET_TAG_STYLE = f"color:{theme.SEVERITY_COLOR['warning']};font-size:90%;margin-left:8px"


@dataclass(frozen=True)
class TrainingBudget:
    """The three settings on the form, beside the four numbers they were looked up from.

    A tolerant view, unlike `colabsd.bestconfig.BudgetSettings`: it holds whatever is in the
    boxes, including what cannot train, because the panel has to draw a refused number before
    it can refuse it. `settle()` is where it is handed to `colabsd.bestconfig` to be judged.
    """

    max_epochs: int
    early_stopping_patience: int
    micro_batch_size: int
    effective_batch_size: int
    lookup: dict[str, int] = field(default_factory=dict)

    #: Rows the smallest training split will hold; 0 when no library has been read yet.
    n_train: int = 0

    @property
    def known(self) -> bool:
        """True once a registry entry has been read, which is what the boxes start from."""
        return bool(self.lookup)

    @property
    def changed(self) -> tuple[str, ...]:
        """Which of the three no longer say what the registry entry says."""
        if not self.known:
            return ()
        return tuple(key for key in BUDGET_KEYS if int(self.lookup.get(key, 0)) != getattr(self, key))

    def overrides(self) -> dict[str, int]:
        """What to hand `colabsd.train.finetune` as `budget`: only the settings the user moved.

        A box still holding the value it was prefilled with is not a choice the user made, so
        it is left out and the run records itself as the looked-up one. `BudgetOverrides` also
        accepts a field set deliberately to the same number; this panel cannot tell that apart
        from a box nobody touched, and does not pretend to.
        """
        return {key: int(getattr(self, key)) for key in self.changed}

    def settle(self) -> tuple[Any | None, str]:
        """`(BudgetSettings, "")` when this budget can train, `(None, refusal)` when it cannot.

        The judging is `colabsd.bestconfig.resolve_budget`'s: it owns what a budget may be, it
        refuses in sentences that name the field and what to type instead, and `finetune` will
        put the same combination through it a moment later. A second copy of those rules here
        would be the copy that goes stale.
        """
        from colabsd.bestconfig import resolve_budget
        from colabsd.errors import ConfigError

        if not self.known:
            return None, ""
        try:
            settled = resolve_budget(
                {"effective_batch_size": self.effective_batch_size},
                {key: self.lookup[key] for key in BUDGET_KEYS},
                self.overrides(),
                n_train=self.n_train or None,
            )
        except ConfigError as exc:
            return None, str(exc)
        return settled, ""


def training_budget(state: core.WizardState) -> TrainingBudget:
    """What this form will train with: the three boxes, and the entry they were filled from."""
    lookup = {str(key): int(value) for key, value in dict(state.budget_lookup or {}).items()}
    return TrainingBudget(
        max_epochs=int(state.max_epochs),
        early_stopping_patience=int(state.early_stopping_patience),
        micro_batch_size=int(state.micro_batch_size),
        effective_batch_size=int(lookup.get("effective_batch_size", 0)),
        lookup=lookup,
        n_train=training_rows(state.n_variants),
    )


def training_rows(n_variants: int) -> int:
    """Rows an 8:1:1 split leaves to train on, which is what a micro batch has to fit inside.

    `colabsd.engine.splits.create_split` takes `int(0.8 * n)` of them. Recomputed here only so
    the panel can refuse an impossible micro batch *before* the splitter is called; the run
    itself is handed the splitter's own indices and checks against those.
    """
    return int(TRAIN_FRACTION * max(0, int(n_variants)))


def budget_lookup(best: Any) -> dict[str, int]:
    """The four budget numbers one registry entry declares, as `colabsd.bestconfig` reads them.

    Asked of `resolve_budget` with nothing overridden, so `micro_batch_size: auto` resolves to
    the number that would actually run and the panel prefills from the same reading of the
    entry that the run will use. Empty for an entry that declares no budget at all, which
    takes the three boxes off the page rather than filling them with zeros.
    """
    if best is None:
        return {}
    from colabsd.bestconfig import resolve_budget
    from colabsd.errors import ConfigError

    try:
        settled = resolve_budget(dict(getattr(best, "params", {}) or {}), dict(getattr(best, "fixed", {}) or {}))
    except ConfigError:
        return {}
    return dict(settled.looked_up)


def changed_tag_html(used: int, looked_up: int) -> str:
    """What sits beside one budget box: nothing, or that it no longer holds the looked-up value.

    At the point of change on purpose. One line elsewhere saying "you changed two things"
    makes the user hunt for which two.
    """
    if int(used) == int(looked_up):
        return ""
    return f'<span style="{BUDGET_TAG_STYLE}">changed · {int(looked_up)} looked up</span>'


def budget_note(budget: TrainingBudget) -> str:
    """The line that keeps the two batch sizes apart, under the boxes that confuse them.

    There are two and they do different jobs: the one on the form is how many sequences sit on
    the card at once, and the one in the table above it is the batch the optimiser averages
    over, which gradient accumulation makes up out of the first. Someone who has just hit an
    out-of-memory error needs the first; someone who reads "batch size" as the second would
    otherwise change it and see no difference but the speed.

    Both are named in YAML as well as in English, because the table above spells them
    `micro_batch_size` and `effective_batch_size` and the box below spells one of them
    "Sequences on the GPU at once". A reader looking for "the batch size" has to be able to
    tell which row the editable box is.
    """
    settled, _refusal = budget.settle()
    accumulation = int(getattr(settled, "gradient_accumulation", 0) or 0)
    if settled is None:
        # These numbers do not make a batch at all. The refusal directly below says so; this
        # line must not answer "out of how many passes?" with one it has invented.
        made_of = ""
    else:
        made_of = f", {accumulation} × {budget.micro_batch_size}" if accumulation > 1 else ", one pass"
    return (
        "**Sequences on the GPU at once (`micro_batch_size`) is memory, not optimisation.** Gradient "
        f"accumulation still trains in batches of {budget.effective_batch_size} (`effective_batch_size`"
        f"{made_of}), so lowering it for an out-of-memory error changes what fits on the card and not what "
        "is learned."
    )


def out_of_memory_advice(exc: BaseException, budget: TrainingBudget) -> str:
    """What to do about a training run the GPU could not hold, named as a box on this form.

    A CUDA out-of-memory error ends in an allocation size and a list of reserved blocks, and
    names no setting anybody can reach. This is the one moment the panel can say which of the
    two batch sizes is the one that helps — `micro_batch_size` is memory and
    `effective_batch_size` is not — so it says it here rather than leaving the reader to guess
    from a traceback. Empty for anything that is not an out-of-memory error, and for a panel
    that has not read a registry entry yet and so has no box to point at.
    """
    if not budget.known:
        return ""
    text = str(exc).lower()
    if type(exc).__name__ != "OutOfMemoryError" and "out of memory" not in text:
        return ""
    box = BUDGET_LABELS["micro_batch_size"].rstrip(":")
    if budget.micro_batch_size <= 1:
        return (
            f"**The GPU ran out of memory.** *{box}* is already 1, the smallest it can be, so this "
            "backbone does not fit this card at all: pick a smaller one above, or switch to an L4 or "
            "A100 runtime."
        )
    return (
        f"**The GPU ran out of memory.** Lower *{box}* — it is {budget.micro_batch_size} now, and it is "
        "the only setting on this form that changes how much memory a run needs. Training still happens "
        f"in batches of {budget.effective_batch_size}, so what is learned does not change."
    )


def metric_cutoff(metric: str) -> int | None:
    """The *k* in `NDCG@50` or `P@10`; `None` for a metric that ranks nothing.

    `colabsd.engine.metrics` spells every ranking metric this way, so the number a column
    promises is read off the column rather than written down a second time here.
    """
    _, at, tail = str(metric).partition("@")
    return int(tail) if at and tail.isdigit() else None


def truncated_metrics(n_validation: int) -> list[str]:
    """Those of `RESULTS_METRICS` whose cut-off is larger than the partition they rank.

    `colabsd.engine.metrics.ndcg_k` and `precision_k` score the top `min(k, n)`, so on a
    partition shorter than *k* the column is not the cut it names: every row is inside it.
    """
    rows = int(n_validation)
    if rows <= 0:
        return []
    return [metric for metric in RESULTS_METRICS if (metric_cutoff(metric) or 0) > rows]


def results_metrics(n_validation: int) -> list[str]:
    """The metrics the results table may show for a validation partition this size."""
    dropped = set(truncated_metrics(n_validation))
    return [metric for metric in RESULTS_METRICS if metric not in dropped]


def truncation_note(dropped: Sequence[str], n_validation: int) -> str:
    """Why a metric the run did record is missing from the table. Empty when none is."""
    if not dropped:
        return ""
    names = ", ".join(f"`{metric}`" for metric in dropped)
    cuts = ", ".join(str(metric_cutoff(metric)) for metric in dropped)
    rows = int(n_validation)
    counted = f"{rows} row" if rows == 1 else f"{rows} rows"
    verb, named = ("is", "it names") if len(dropped) == 1 else ("are", "they name")
    return (
        f"{names} {verb} not in this table: the validation partition is **{counted}**, shorter than the "
        f"cut {named} ({cuts}). Every variant falls inside a cut that long, so the number cannot mean what "
        "its name promises — with that few rows even an arbitrary ranking scores a long way above zero. "
        "`report.csv` in the performance archive carries it anyway."
    )


def validation_rows(splits: Any) -> int:
    """How many rows the smallest validation partition of these splits holds; 0 when unknown."""
    try:
        sizes = [len(split["val_idx"]) for split in dict(splits).values()]
    except (AttributeError, KeyError, TypeError, ValueError):
        return 0
    return min(sizes) if sizes else 0


def progress_html(done: int, total: int, label: str) -> str:
    """The training progress line: runs finished, and what the run in flight is doing.

    `done` counts *finished* runs, so it holds still while the label beside it counts the
    epochs and batches of the one being trained. `colabsd.train.finetune` rewrites that label
    several times a second and its length changes with every number in it, so the line is one
    fixed-height row that is clipped rather than wrapped, and carries the whole label as hover
    text. Nothing below it moves while a run reports.
    """
    shown = str(label).strip()
    return (
        f"<div title=\"{escape(shown)}\" style=\"{PROGRESS_STYLE}\">"
        f"<code>{int(done)}/{int(total)} runs finished</code>"
        f"{' · ' + escape(shown) if shown else ''}</div>"
    )


def validation_headline(summary: dict[str, Any] | None) -> str:
    """The macro validation Spearman line, honest about a single run."""
    macro = (summary or {}).get("Spearman", {})
    mean = float(macro.get("mean", float("nan")))
    sd = float(macro.get("sd", float("nan")))
    n_runs = int(macro.get("n_runs", 0))
    line = f"Validation Spearman, averaged over conditions: {mean:.4f} ± {sd:.4f} ({n_runs} run(s))"
    if n_runs < 2:
        line += " — one run shows no spread, so the ± is `nan`."
    return line


def plan_line(state: core.WizardState, best: Any | None = None) -> str:
    """What pressing Train is about to do, in one paragraph."""
    split_seeds, model_seeds = training_seeds(state, best)
    estimate = core.estimate_runtime(state)
    line = (
        f"Split seeds {split_seeds} × model seeds {model_seeds} = **{estimate.n_runs} run(s)**. "
        f"{estimate.describe()}"
    )
    protocol = len(getattr(best, "split_seeds", []) or []) * len(getattr(best, "model_seeds", []) or [])
    if protocol and estimate.n_runs < protocol:
        line += (
            f" The registry entry was selected over {protocol} runs; {estimate.n_runs} says less about "
            "reproducibility."
        )
    return line


# --------------------------------------------------------------------------------------
# The backend seam
# --------------------------------------------------------------------------------------


class Backend:
    """Every call this wizard makes out of itself, in one overridable place.

    The wizard never imports `colabsd.train` or `torch` directly; it goes through here, so a
    test subclasses this, records the arguments and returns a stub. That is how the whole
    flow is driven without a GPU.
    """

    def load_library(self, csv_path: Path, spec: LibrarySpec, *, min_count: int = 0) -> Any:
        from colabsd.data import load_library

        return load_library(csv_path, spec, min_count=min_count)

    def load_wt_sequence(self, path: Path) -> str:
        from colabsd.data import load_wt_sequence

        return load_wt_sequence(path)

    def read_variant_csv(self, path: Path) -> Any:
        from colabsd.data import read_variant_csv

        return read_variant_csv(path)

    def make_splits(self, n: int, seeds: list[int], out_dir: Path) -> Any:
        from colabsd.data import make_splits

        return make_splits(n, seeds, out_dir)

    def load_best_config(self, backbone: str) -> Any:
        from colabsd.bestconfig import load_best_config

        return load_best_config(backbone)

    def create_adapter(self, backbone: str, **kwargs: Any) -> Any:
        from colabsd.backbones.registry import create_adapter

        return create_adapter(backbone, **kwargs)

    def load_three_di(self, path: Path, *, expected_length: int | None = None) -> str:
        from colabsd.structure import load_three_di

        return load_three_di(path, expected_length=expected_length)

    def validate_three_di(self, text: str, expected_length: int | None = None) -> str:
        from colabsd.structure import validate_three_di

        return validate_three_di(text, expected_length)

    def three_di_from_structure(self, path: Path, **kwargs: Any) -> str:
        from colabsd.structure import three_di_from_structure

        return three_di_from_structure(path, **kwargs)

    def three_di_from_esmfold(self, wt_sequence: str, **kwargs: Any) -> str:
        from colabsd.structure import three_di_from_esmfold

        return three_di_from_esmfold(wt_sequence, **kwargs)

    def finetune(self, **kwargs: Any) -> Any:
        from colabsd.train import finetune

        return finetune(**kwargs)

    def save_bundle_from_run(self, path: Path, **kwargs: Any) -> Path:
        from colabsd.bundle import save_bundle_from_run

        return save_bundle_from_run(path, **kwargs)

    def load_bundle(self, path: Path) -> Any:
        from colabsd.bundle import load_bundle

        return load_bundle(path)

    def build_report(self, *args: Any, **kwargs: Any) -> Any:
        from colabsd.report import build_report

        return build_report(*args, **kwargs)

    def read_unlock_count(self, output_dir: Any) -> int:
        from colabsd.train import read_unlock_count

        return read_unlock_count(output_dir)

    def score_variants(self, bundle: Any, variants: Any, **kwargs: Any) -> Any:
        from colabsd.predict import score_variants

        return score_variants(bundle, variants, **kwargs)

    def upload_file(self, what: str) -> Path:
        """Ask the browser for one file and return where it landed.

        `files.upload()` holds the kernel until the browser answers, so every widget callback
        on this page is dead while the picker is open; `MainWizard._take_upload` writes
        `core.upload_notice` onto the page on the line before it calls this, and the test
        reads that notice from inside the picker.

        Both ways out of the call are `colabsd.ui.core`'s words rather than this module's: the
        Predict panel blocks on the same picker, cancels the same way and is just as unable to
        open one outside Colab, and two wordings for one freeze drift apart. An `ImportError`
        on `google.colab` in particular is true and useless — every upload button here sits
        beside a text field that takes a path, and that is what the message names.
        """
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError(core.upload_needs_colab_notice(what)) from None
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError(core.upload_cancelled_notice(what))
        return Path(next(iter(uploaded))).resolve()

    def offer_download(self, path: Path) -> None:
        """Hand a written file to the browser, when there is one."""
        try:
            from google.colab import files
        except ImportError:
            return
        files.download(str(path))

    def download_url(self, url: str, target: Path) -> Path:
        import urllib.request

        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)  # noqa: S310 - the user typed this URL
        return target


def random_variants(spec: LibrarySpec, how_many: int, seed: int) -> Any:
    """Draw random residue combinations at the mutated sites, in the spec's own notation."""
    import numpy as np
    import pandas as pd

    from colabsd.data import one_to_three_letter

    residues = list("ACDEFGHIKLMNPQRSTVWY")
    if spec.three_letter:
        mapping = one_to_three_letter()
        residues = [mapping[letter] for letter in residues]
    rng = np.random.default_rng(int(seed))
    draw = rng.choice(residues, size=(max(1, int(how_many)), spec.k))
    return pd.DataFrame(draw, columns=list(spec.mutation_columns)).drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------------------
# The widget layer
# --------------------------------------------------------------------------------------

_LABEL = {"description_width": "initial"}
_WIDE = {"width": "560px"}


class MainWizard:
    """The training wizard: `core`'s decisions pushed into `ipywidgets`, and nothing else.

    Events are plain `observe()` and `on_click()`. `jupyter_ui_poll` exists so that a
    linear, blocking script can wait for a widget; this cell installs, imports, launches
    and ends, and Colab keeps the widget comm alive afterwards, so ordinary observers fire
    and every handler is callable directly from a test.
    """

    def __init__(
        self,
        state: core.WizardState | None = None,
        backend: Backend | None = None,
        runtime: core.Runtime | None = None,
    ) -> None:
        self.state = state if state is not None else new_state()
        self.backend = backend if backend is not None else Backend()
        self.runtime = runtime

        self.spec: LibrarySpec | None = None
        self.frame: Any = None
        self.sequences: list[str] | None = None
        self.targets: Any = None
        self.session_three_di: prep.Artefact | None = None
        self.best: Any = None
        self.adapter: Any = None
        self.splits: Any = None
        self.run: Any = None
        self.bundle: Any = None
        self.bundle_path: Path | None = None
        self.performance: exports.PerformanceExport | None = None
        self._best_key: str | None = None
        self._refreshing = False
        self.trained_best: Any = None
        self.trained_spec: LibrarySpec | None = None
        self.scores: Any = None
        self.plan: core.Plan | None = None

        self._build()
        self.refresh()

    # -- construction ------------------------------------------------------------------

    def _build(self) -> None:
        import ipywidgets

        self.w = ipywidgets
        self.logs = {name: theme.html("") for name in PAGE_ORDER}
        self.boards = {name: core.MessageBoard() for name in PAGE_ORDER}

        self.fields: dict[str, Any] = {}
        self.outlets: dict[str, Any] = {}
        self.uploads: dict[str, Any] = {}

        self._build_data()
        self._build_model()
        self._build_structure()
        self._build_hyperparameters()
        self._build_training()
        self._build_storage()
        self._build_run()
        self._build_results()
        self._build_export()
        self._build_score()

        self.sections = {
            name: core.Section(name, SECTION_TITLES[name], self._children(name), description=SECTION_NOTES[name])
            for name in PAGE_ORDER
        }
        self.root = self.w.VBox([self.sections[name].box() for name in PAGE_ORDER])
        self._bind()

    def _children(self, section: str) -> list[Any]:
        return self._layout[section]

    # -- the individual sections -------------------------------------------------------

    def _build_data(self) -> None:
        self._layout: dict[str, list[Any]] = {}
        self.fields["data_source"] = self.w.RadioButtons(
            options=[
                (f"Use the bundled {EXAMPLE_NAME} example", "bundled_example"),
                ("Use my own library.csv", "upload_csv"),
            ],
            value=self.state.data_source,
            layout={"width": "max-content"},
            style=_LABEL,
        )
        self.outlets["example_note"] = theme.note(example_note())
        self.fields["library_csv"] = self._upload_row("library_csv", "Upload library.csv", "your library.csv", "data")
        self.fields["wt_sequence"] = self.w.Textarea(
            value=str(self.state.get("wt_sequence") or ""),
            placeholder="MSKQ… , or the URL of a FASTA file (an AlphaFold or UniProt link works)",
            description="Wild type:",
            layout={"width": "560px", "height": "80px"},
            style=_LABEL,
        )
        self.fields["positions_1based"] = self._text("positions_1based", "Mutated positions:", EXAMPLE_POSITIONS)
        self.fields["mutation_columns"] = self._text("mutation_columns", "Mutation columns:", EXAMPLE_MUTATION_COLUMNS)
        self.fields["condition_columns"] = self._text(
            "condition_columns", "Condition columns:", EXAMPLE_CONDITION_COLUMNS
        )
        self.fields["three_letter_residues"] = self.w.Checkbox(
            value=bool(self.state.get("three_letter_residues")),
            description="Residues are written as Asn, not N",
            indent=False,
            style=_LABEL,
        )
        self.fields["count_column"] = self._text("count_column", "Read-count column:", "count")
        self.fields["min_count"] = self.w.BoundedIntText(
            value=int(self.state.get("min_count") or 0),
            min=0,
            max=10**9,
            description="Drop variants seen fewer times than:",
            style=_LABEL,
            layout=_WIDE,
        )
        self.check_button = self._button("Check my library", "primary")
        self._layout["data"] = [
            self.fields["data_source"],
            self.outlets["example_note"],
            self.fields["library_csv"],
            self.fields["wt_sequence"],
            theme.note(LIBRARY_NOTE),
            self.fields["positions_1based"],
            self.fields["mutation_columns"],
            self.fields["condition_columns"],
            self.fields["three_letter_residues"],
            self.fields["count_column"],
            self.fields["min_count"],
            self.boards["data"].widget,
            self.check_button,
            self.logs["data"],
        ]

    def _build_model(self) -> None:
        self.fields["backbone"] = self.w.Dropdown(
            options=core.backbone_choices(),
            value=self.state.backbone,
            description="Backbone:",
            style=_LABEL,
            layout={"width": "760px"},
        )
        self.backbone_note = theme.note("")
        self.fields["dtype"] = self.w.Dropdown(
            options=["float32", "float16", "bfloat16"],
            value=self.state.dtype,
            description="Numeric precision:",
            style=_LABEL,
            layout=_WIDE,
        )
        self.withdrawn_note = theme.note(withdrawn_note())
        self._layout["model"] = [
            self.fields["backbone"],
            self.backbone_note,
            self.withdrawn_note,
            self.fields["dtype"],
            self.boards["model"].widget,
            self.logs["model"],
        ]

    def _build_structure(self) -> None:
        """Step 3, which is `prepare_workflow`'s one section and exists only for a SaProt.

        The panel supplies its own widget factories, so the section built here looks like the
        rest of the page and its uploads go through the row that says what a blocking
        `files.upload()` is about to do. `core.needs_structure` decides whether it is on
        screen at all, from the backbone in step 2.
        """
        tools = prep.SectionTools(text=self._text, upload_row=self._upload_row, button=self._button)
        self.three_di = prep.ThreeDiSection(self.state, tools)
        self.fields.update(self.three_di.fields)
        self.outputs_note = theme.html("")
        self._layout["structure"] = [
            *self.three_di.children(),
            self.outputs_note,
            self.boards["structure"].widget,
            self.three_di.button,
            self.logs["structure"],
        ]

    def _build_hyperparameters(self) -> None:
        """The lookup, read-only, and under it the three boxes that start at its own numbers.

        The section keeps its shape: what a study selected is still a table nobody can type
        into. The budget sits below it as ordinary fields, each beside the one word that says
        whether it is still the looked-up value. They start at zero and are off screen until
        `_sync_budget` has an entry to fill them from.
        """
        self.fields["hyperparameters"] = theme.html("")
        self.budget_tags: dict[str, Any] = {}
        self.budget_rows: dict[str, Any] = {}
        for key in BUDGET_KEYS:
            self.fields[key] = self.w.BoundedIntText(
                value=int(self.state.get(key) or 0),
                min=0,
                max=BUDGET_MAXIMA[key],
                description=BUDGET_LABELS[key],
                style=_LABEL,
                layout=_WIDE,
            )
            self.budget_tags[key] = theme.html("")
            self.budget_rows[key] = self.w.HBox([self.fields[key], self.budget_tags[key]])
        self.budget_note = theme.note("")
        self._layout["hyperparameters"] = [
            self.fields["hyperparameters"],
            *self.budget_rows.values(),
            self.budget_note,
            self.boards["hyperparameters"].widget,
            self.logs["hyperparameters"],
        ]

    def _build_training(self) -> None:
        # The ceilings start at one and are widened by `_sync_seed_limits` to whatever
        # protocol the registry entry for the chosen pair declares. Typing a number here
        # would be a second copy of a fact `colabsd.bestconfig` already holds.
        self.fields["n_split_seeds"] = self.w.BoundedIntText(
            value=self.state.n_split_seeds, min=1, max=1, description="Split seeds:", style=_LABEL, layout=_WIDE
        )
        self.fields["n_model_seeds"] = self.w.BoundedIntText(
            value=self.state.n_model_seeds, min=1, max=1, description="Model seeds:", style=_LABEL, layout=_WIDE
        )
        self.fields["run_name"] = self._text("run_name", "Name this run:", "run")
        self.fields["resume_finished_runs"] = self.w.Checkbox(
            value=bool(self.state.get("resume_finished_runs")),
            description="Reuse finished runs if this is re-run",
            indent=False,
            style=_LABEL,
        )
        self.plan_note = theme.note("")
        self._layout["training"] = [
            self.fields["n_split_seeds"],
            self.fields["n_model_seeds"],
            self.fields["run_name"],
            self.fields["resume_finished_runs"],
            self.plan_note,
            self.boards["training"].widget,
            self.logs["training"],
        ]

    def _build_storage(self) -> None:
        self.fields.update(core.drive_widgets(folder=self.state.drive_folder))
        self.fields["output_dir"] = self.w.Text(
            value=self.state.output_dir,
            placeholder="colabsd_work",
            description="Working directory:",
            style=_LABEL,
            layout=_WIDE,
        )
        self._layout["storage"] = [
            self.fields["use_drive"],
            self.fields["drive_folder"],
            self.fields["output_dir"],
            self.boards["storage"].widget,
            self.logs["storage"],
        ]

    def _build_run(self) -> None:
        self.fields["advanced_toggle"] = self.w.Checkbox(
            value=self.state.show_advanced, description="Show the advanced settings", indent=False, style=_LABEL
        )
        self.fields["run_button"] = self._button("Train", "primary")
        self.progress = theme.html("")
        # Reserved before anything is trained: the first progress line then appears in space
        # that is already there rather than pushing the log below it down the page.
        self.progress.layout.min_height = PROGRESS_ROW_HEIGHT
        self._layout["run"] = [
            self.fields["advanced_toggle"],
            self.boards["run"].widget,
            self.fields["run_button"],
            self.progress,
            self.logs["run"],
        ]

    def _build_results(self) -> None:
        self.results = theme.html("")
        self._layout["results"] = [self.results, self.boards["results"].widget, self.logs["results"]]

    def _build_export(self) -> None:
        self.outlets["bundle_name"] = self._text("bundle_name", "Bundle file name:", "model_bundle.zip")
        self.outlets["bundle_notes"] = self.w.Textarea(
            value=str(self.state.get("bundle_notes") or ""),
            placeholder="Anything you will want to read back in six months",
            description="Notes:",
            layout={"width": "560px", "height": "60px"},
            style=_LABEL,
        )
        self.outlets["archive_name"] = self._text(
            "archive_name", "Performance archive name:", exports.DEFAULT_ARCHIVE_NAME
        )
        self.export_button = self._button(f"Write {exports.DEFAULT_BUNDLE_NAME}", "primary")
        self.performance_button = self._button(f"Write {exports.DEFAULT_ARCHIVE_NAME}", "primary")
        self.export_result = theme.html("")
        self.performance_result = theme.html("")
        self._layout["export"] = [
            self.outlets["bundle_name"],
            self.outlets["bundle_notes"],
            self.boards["export"].widget,
            self.export_button,
            self.export_result,
            self.outlets["archive_name"],
            self.performance_button,
            self.performance_result,
            self.logs["export"],
        ]

    def _build_score(self) -> None:
        self.outlets["score_source"] = self.w.RadioButtons(
            options=[
                ("The first rows of the library I loaded", "library_head"),
                ("Random combinations of residues", "random_combinations"),
                ("A variants CSV of my own", "upload_csv"),
            ],
            value=str(self.state.get("score_source")),
            layout={"width": "max-content"},
            style=_LABEL,
        )
        self.outlets["score_top_n"] = self.w.BoundedIntText(
            value=int(self.state.get("score_top_n") or 32),
            min=1,
            max=100000,
            description="How many:",
            style=_LABEL,
            layout=_WIDE,
        )
        self.outlets["score_random_seed"] = self.w.BoundedIntText(
            value=int(self.state.get("score_random_seed") or 0),
            min=0,
            max=10**6,
            description="Random seed:",
            style=_LABEL,
            layout=_WIDE,
        )
        self.outlets["score_variants_csv"] = self._upload_row(
            "score_variants_csv", "Upload variants CSV", "a variants CSV", "score"
        )
        self.outlets["score_rank_by"] = self.w.Dropdown(
            options=[("Average over conditions (pred_mean)", "pred_mean"), ("Worst condition (pred_min)", "pred_min")],
            value=str(self.state.get("score_rank_by")),
            description="Rank by:",
            style=_LABEL,
            layout=_WIDE,
        )
        self.score_button = self._button("Score them", "primary")
        self.score_result = theme.html("")
        self._layout["score"] = [
            self.outlets["score_source"],
            self.outlets["score_top_n"],
            self.outlets["score_random_seed"],
            self.outlets["score_variants_csv"],
            self.outlets["score_rank_by"],
            self.boards["score"].widget,
            self.score_button,
            self.score_result,
            self.logs["score"],
        ]

    # -- small widget builders ---------------------------------------------------------

    def _text(self, name: str, description: str, placeholder: str) -> Any:
        """A text field whose value starts where the state says it does, not blank."""
        return self.w.Text(
            value=str(self.state.get(name) or ""),
            placeholder=placeholder,
            description=description,
            style=_LABEL,
            layout=_WIDE,
        )

    def _button(self, description: str, style: str) -> Any:
        return self.w.Button(description=description, button_style=style, layout=_WIDE)

    def _upload_row(self, name: str, description: str, what: str, section: str) -> Any:
        """An upload button beside the path it produced, so a typed path works too."""
        button = self.w.Button(description=description, button_style="info", layout={"width": "260px"})
        path = self.w.Text(
            value=str(self.state.get(name) or ""),
            placeholder="…or the path of a file on this machine",
            layout={"width": "420px"},
        )
        core.on_click(button, self._guard(section, lambda: self._take_upload(path, what, section)))
        self.uploads[name] = (button, path)
        return self.w.HBox([button, path])

    def _take_upload(self, path_widget: Any, what: str, section: str) -> None:
        """Say what is being waited for, *then* open the picker that freezes the page.

        `files.upload()` blocks the kernel: while it waits every observer on this page is
        dead, so the panel sits in whatever state the button left it in. The explanation has
        to be written before the call, not after it.
        """
        self.logs[section].value = theme.message_html(core.upload_notice(what), "info")
        landed = self.backend.upload_file(what)
        path_widget.value = str(landed)
        self.logs[section].value = theme.note_html(f"Uploaded `{landed}`.")

    # -- wiring ------------------------------------------------------------------------

    def _bind(self) -> None:
        for name, widget in list(self.fields.items()) + list(self.outlets.items()):
            if name in DISPLAY_ONLY:
                continue
            if name in self.uploads:
                core.bind(self.uploads[name][1], self.state, name, on_change=self._changed)
            elif hasattr(widget, "value") and hasattr(widget, "observe"):
                target = "show_advanced" if name == "advanced_toggle" else name
                changed = self._drive_changed if name == "use_drive" else self._changed
                core.bind(widget, self.state, target, on_change=changed)
        core.on_click(self.check_button, self._guard("data", self.on_check_library))
        core.on_click(self.three_di.button, self._guard("structure", self.on_get_three_di))
        core.on_click(self.fields["run_button"], self._guard("run", self.on_train))
        core.on_click(self.export_button, self._guard("export", self.on_export))
        core.on_click(self.performance_button, self._guard("export", self.on_export_performance))
        core.on_click(self.score_button, self._guard("score", self.on_score))

    def _drive_changed(self, state: core.WizardState) -> None:
        """Ticking the box mounts Drive and moves the working directory into it.

        A checkbox that only records an intention would make `core`'s "Google Drive is
        mounted" message a lie the first time something went wrong.
        """
        self._guard("storage", self.mount_drive)()

    def mount_drive(self) -> None:
        """Mount Drive if the box is ticked, and point the working directory at it."""
        if not self.state.use_drive:
            self._say("storage", "Google Drive is off. Everything stays on this machine until the session ends.")
            return
        mount = core.mount_drive(enable=True, folder=self.state.drive_folder)
        if mount.mounted and mount.output_dir is not None:
            self.state.output_dir = str(mount.output_dir)
            self.fields["output_dir"].value = str(mount.output_dir)
        else:
            self.state.use_drive = False
            self.fields["use_drive"].value = False
        self._say("storage", mount.describe())

    def _changed(self, state: core.WizardState) -> None:
        """One observer for every field. Re-entrant calls are dropped, not recursed into.

        `refresh` writes to widgets that are themselves observed -- the 3Di source list and
        the seed maxima -- so without this a single keystroke would re-enter it.
        """
        if state.get("_reloading") or self._refreshing:
            return
        self.refresh()

    def _guard(self, section: str, action: Callable[[], None]) -> Callable[[], None]:
        """Run a step, and put whatever it raises on the page instead of in a traceback.

        The training section gets one sentence more than the exception carries: a run the card
        could not hold is the reason the micro batch is on this form at all, and the error it
        dies with names bytes rather than a field. Only `run` — the folding step above runs out
        of memory too, `colabsd.structure` already says what to do about that, and lowering a
        micro batch would not be it.
        """

        def handle() -> None:
            self.logs[section].value = ""
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - the message is the product here
                text = f"**That did not work.** {type(exc).__name__}: {exc}"
                advice = out_of_memory_advice(exc, training_budget(self.state)) if section == "run" else ""
                self.logs[section].value = theme.message_html(f"{text}\n\n{advice}" if advice else text, "stop")
            self.refresh()

        return handle

    # -- refresh -----------------------------------------------------------------------

    def refresh(self) -> core.Plan:
        """Re-apply every decision to the widgets. The only place `layout.display` is set."""
        self._refreshing = True
        try:
            return self._refresh()
        finally:
            self._refreshing = False

    def _refresh(self) -> core.Plan:
        status = self._config_status()
        self._sync_seed_limits()
        self._sync_budget()
        artefact = self.artefact()
        self._sync_structure(artefact)
        self.plan = core.plan(self.state, runtime=self.runtime, status=status)
        items = messages(self.state, runtime=self.runtime, status=status, artefact=artefact)

        visible = section_visibility(self.state)
        core.apply_field_visibility(self._visibility_targets(), self.state)
        for key, shown in outlet_field_visibility(self.state).items():
            core.set_display(self.outlets[key], shown)
        titles = numbered_titles(self.state)
        for name, section in self.sections.items():
            section.set_visible(visible[name])
            section.set_title(titles[name])

        grouped = messages_by_section(items, visible)
        for name, board in self.boards.items():
            board.update(grouped[name])

        self.backbone_note.value = theme.note_html(core.backbone_summary(self.state.backbone))
        self.plan_note.value = theme.note_html(plan_line(self.state, self.best))
        self.fields["run_button"].disabled = not can_train(items)
        if self.best is not None:
            view = hyperparameter_view(self.best, training_budget(self.state))
            self.fields["hyperparameters"].value = hyperparameter_html(view)
        else:
            self.fields["hyperparameters"].value = no_hyperparameters_html(self.state.backbone, status)
        return self.plan

    def _visibility_targets(self) -> dict[str, Any]:
        """The widget `core.apply_field_visibility` shows or hides for each declared field.

        A budget box lives in an `HBox` beside the tag that says whether it still holds the
        looked-up value; hiding the box alone would leave the tag stranded on the page.
        """
        targets = {key: widget for key, widget in self.fields.items() if key in core.FIELD_KEYS}
        targets.update(self.budget_rows)
        return targets

    def _sync_budget(self) -> None:
        """Fill the three boxes from the entry on screen, then say which no longer match it.

        The prefill happens whenever the numbers the boxes were filled from change: the first
        refresh after a lookup, and again when a new backbone's entry says something different.
        A backbone whose entry says exactly what the last one said is not a new prefill, so a
        micro batch someone lowered after an out-of-memory error survives changing the model —
        the card did not get any bigger. Between prefills the user's numbers stand, and each
        box carries whether it is still the looked-up one.
        """
        lookup = budget_lookup(self.best)
        if lookup and lookup != self.state.budget_lookup:
            self.state.budget_lookup = lookup
            for key in BUDGET_KEYS:
                self.state.set(key, int(lookup[key]))
                self.fields[key].value = int(lookup[key])
        elif not lookup and self.state.budget_lookup:
            # The backbone on screen has no entry at all. Nothing was looked up, so there is
            # nothing for the boxes to mean and `core`'s rule takes them off the page.
            self.state.budget_lookup = {}
        budget = training_budget(self.state)
        for key in BUDGET_KEYS:
            self.budget_tags[key].value = changed_tag_html(getattr(budget, key), budget.lookup.get(key, 0))
        self.budget_note.value = theme.note_html(budget_note(budget)) if budget.known else ""
        core.set_display(self.budget_note, budget.known)

    def artefact(self) -> prep.Artefact | None:
        """The 3Di string this session made, when it still describes the wild type on screen."""
        return prep.session_artefact(self.state, self.session_three_di)

    def _sync_structure(self, artefact: prep.Artefact | None) -> None:
        """Rebuild step 3 from step 2: the source list, the note, and what it will write.

        The source list is built rather than written down because a choice that does not
        apply must not be on screen at all -- "reuse the one this session made" is not
        greyed out after the library has changed under it, it is absent.
        """
        forced = self.three_di.sync(self.state, artefact)
        if forced is not None:
            self.state.three_di_source = forced
        self.three_di.note.value = theme.note_html(prep.section_note(self.state))
        self.outputs_note.value = prep.outputs_html(prep.output_files(self.state, work_dir=work_dir(self.state)))

    def _has_gpu(self) -> bool:
        return bool(self.runtime is not None and self.runtime.has_gpu)

    def _sync_seed_limits(self) -> None:
        """Cap the seed spinners at the protocol the registry entry itself declares."""
        for key, attribute, fallback in (
            ("n_split_seeds", "split_seeds", FALLBACK_SPLIT_SEEDS),
            ("n_model_seeds", "model_seeds", FALLBACK_MODEL_SEEDS),
        ):
            available = len(getattr(self.best, attribute, None) or fallback)
            widget = self.fields[key]
            if widget.max != available:
                widget.max = max(1, available)

    def _config_status(self) -> core.ConfigStatus | None:
        """The registry lookup for the current backbone, read once and cached.

        `refresh` runs on every keystroke, and `self.best` is handed straight to
        `finetune`, so re-reading the YAML each time would both cost a file read and hand
        the training call a different object from the one on screen.
        """
        if not self.state.get("library_loaded"):
            return None
        key = self.state.backbone
        status = core.config_status(key)
        if key == self._best_key and self.best is not None:
            return status
        self._best_key = key
        if status.found and status.error is None:
            try:
                self.best = self.backend.load_best_config(key)
            except Exception:  # noqa: BLE001 - config_status already turned this into a message
                self.best = None
        else:
            self.best = None
        return status

    def _progress(self, done: int, total: int, label: str) -> None:
        """`colabsd.train.finetune`'s progress callback, rendered as one line.

        Called as often as the trainer reports -- several times a second, from inside the
        training loop -- so it does the least a widget update can do: format one string and
        assign it.
        """
        self.progress.value = progress_html(done, total, label)

    def _say(self, section: str, text: str) -> None:
        self.logs[section].value = theme.note_html(text)

    # -- the steps ---------------------------------------------------------------------

    def on_check_library(self) -> None:
        problems = load_blockers(self.state)
        if problems:
            raise ValueError(" ".join(problems))
        csv_path, wt_sequence = self._resolve_library_inputs()
        spec = library_spec(self.state, wt_sequence)
        spec.validate()
        frame, sequences, targets = self.backend.load_library(
            csv_path, spec, min_count=int(self.state.get("min_count") or 0)
        )

        self.spec, self.frame, self.sequences, self.targets = spec, frame, sequences, targets
        self.state.n_variants = len(frame)
        self.state.wt_length = len(spec.wt_sequence)
        self.state.wt_3di_length = 0
        self.state.set("library_loaded", True)
        self._say(
            "data",
            f"**{len(frame):,} variants** from `{csv_path}`.\n"
            f"- Wild type: {len(spec.wt_sequence)} residues.\n"
            f"- {spec.k} mutated sites at {list(spec.positions_1based)}, wild-type residues "
            f"`{spec.wt_residues()}`.\n"
            f"- {spec.n_targets} conditions: {', '.join(spec.condition_columns)}.",
        )

    def _resolve_library_inputs(self) -> tuple[Path, str]:
        if self.state.data_source == "bundled_example":
            return example_path("library.csv"), self.backend.load_wt_sequence(example_path("wt.fasta"))
        csv_path = Path(str(self.state.get("library_csv")).strip())
        typed = str(self.state.get("wt_sequence") or "").strip()
        if typed.lower().startswith(("http://", "https://")):
            target = self.backend.download_url(typed, work_dir(self.state) / "wt_downloaded.fasta")
            return csv_path, self.backend.load_wt_sequence(target)
        return csv_path, "".join(typed.split()).upper()

    def on_get_three_di(self) -> None:
        """The 3Di step: put a wild-type 3Di string in this session. It never leaves it.

        Whatever the source — a file, a structure read by foldseek, or ESMFold — the string is
        attached to the `LibrarySpec` here and written beside the run.
        """
        if self.spec is None:
            raise ValueError("Check the library first: the 3Di string is measured against its wild-type length.")
        three_di = self.three_di.resolve(
            self.state,
            self.backend,
            wt_sequence=self.spec.wt_sequence,
            artefact=self.artefact(),
            work_dir=work_dir(self.state),
            has_gpu=self._has_gpu(),
        )
        self.spec = replace(self.spec, wt_3di=three_di)
        self.spec.validate()
        self.state.wt_3di_length = len(three_di)
        target = work_dir(self.state) / prep.THREE_DI_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(three_di + "\n")
        self.session_three_di = prep.written_artefact(target, three_di)
        length = len(self.spec.wt_sequence)
        self._say(
            "structure",
            f"3Di attached: {len(three_di)} states for {length} residues, and kept in this session.\n"
            f"- sequence `{self.spec.wt_sequence[:60]}…`\n"
            f"- 3Di      `{three_di[:60]}…`\n"
            f"- written to `{target}`",
        )

    def on_train(self) -> None:
        items = messages(self.state, runtime=self.runtime, status=self._config_status())
        stopped = blockers(items)
        if stopped:
            raise RuntimeError(" ".join(item.text for item in stopped))
        if self.spec is None or self.sequences is None or self.best is None:
            raise RuntimeError("The library or the hyperparameters are not ready; press Check my library again.")
        if core.needs_structure(self.state) and not self.spec.wt_3di:
            raise RuntimeError(
                f"{self.state.backbone} reads structure and no 3Di string is attached to this library. "
                "Press **Get the 3Di string** in the preparation step above."
            )

        self.adapter = self.backend.create_adapter(self.state.backbone, **adapter_kwargs(self.state, self.spec))
        split_seeds, _ = training_seeds(self.state, self.best)
        self.splits = self.backend.make_splits(
            len(self.sequences), split_seeds, split_dir(self.state, len(self.sequences))
        )
        self.run = self.backend.finetune(
            **finetune_kwargs(
                self.state,
                spec=self.spec,
                sequences=self.sequences,
                targets=self.targets,
                adapter=self.adapter,
                best=self.best,
                splits=self.splits,
                progress=self._progress,
            )
        )
        # Freeze what this run actually was. `self.best` follows the dropdown; the bundle
        # must not, or it records one backbone's hyperparameters beside another's weights.
        self.trained_best, self.trained_spec = self.best, self.spec
        self.state.set("trained_fingerprint", training_fingerprint(self.state))
        self.state.set("trained", True)
        self.state.set("exported", False)
        self.results.value = self._results_html()
        for warning in getattr(self.run, "warnings", []) or []:
            self.logs["run"].value += theme.message_html(warning, "warning")

    def _results_html(self) -> str:
        """What this run scored on validation, per condition and averaged over them."""
        summary = getattr(self.run, "validation_summary", {}) or {}
        model_label = getattr(self.run, "model_name", self.state.backbone)
        n_validation = validation_rows(self.splits)
        aggregate = getattr(self.run, "aggregate", None)
        if aggregate is not None and len(aggregate):
            table = aggregate[aggregate["metric"].isin(results_metrics(n_validation))].to_html(index=False)
        else:
            table = "<i>no per-condition table</i>"
        note = truncation_note(truncated_metrics(n_validation), n_validation)
        return (
            theme.note_html(
                f"Finished {getattr(self.run, 'n_runs', 0)} run(s) in "
                f"{core.format_minutes(getattr(self.run, 'minutes', 0.0))}.\n"
                f"- **{validation_headline(summary)}**"
            )
            + f"<div style='margin-top:10px'><b>{model_label} — validation</b>{table}</div>"
            + theme.note_html(
                "These are validation numbers: the test partition is still locked."
                + (f"\n\n{note}" if note else "")
            )
        )

    def _export_runners(self) -> exports.ExportRunners:
        """`colabsd.ui.exports`'s side effects, pointed back at this wizard's own backend.

        The panel has one seam for everything that touches a disk or a browser, and the
        exports have another; bridging them here means a test that stubs the backend gets
        both exports stubbed, and the notebook gets the real thing for both.
        """
        return exports.ExportRunners(
            build_report=self.backend.build_report,
            save_bundle_from_run=self.backend.save_bundle_from_run,
            load_bundle=self.backend.load_bundle,
            read_unlock_count=self.backend.read_unlock_count,
            download=self.backend.offer_download,
        )

    def on_export(self) -> None:
        """Write the model bundle: the weights, and everything needed to use them again."""
        problems = export_blockers(self.state)
        if problems:
            raise RuntimeError(" ".join(problems))
        bundle_name, _archive_name = exports.export_names(self.state)
        export = exports.export_bundle(
            run_result=self.run,
            spec=self.trained_spec,
            best=self.trained_best,
            path=work_dir(self.state) / bundle_name,
            notes=str(self.state.get("bundle_notes") or "").strip() or None,
            runners=self._export_runners(),
        )
        self.bundle, self.bundle_path = export.bundle, export.path
        self.state.set("exported", True)
        self.export_result.value = theme.note_html(f"Written `{export.path}`\n\n{export.describe()}")

    def on_export_performance(self) -> None:
        """Write the performance archive — with the test partition still locked.

        The only call to `colabsd.report.build_report` used to be inside the unlock handler,
        so the report was unreachable unless you spent the test set to see it. Pressing this
        again after an unlock rewrites the same file with the test numbers and the count.
        """
        problems = export_blockers(self.state)
        if problems:
            raise RuntimeError(" ".join(problems))
        _bundle_name, archive_name = exports.export_names(self.state)
        export = exports.export_performance(
            work_dir=work_dir(self.state),
            run_result=self.run,
            spec=self.trained_spec,
            best=self.trained_best,
            archive_name=archive_name,
            notes=str(self.state.get("bundle_notes") or "").strip() or None,
            runners=self._export_runners(),
        )
        self.performance = export
        self.performance_result.value = exports.summary_html(export)

    def on_score(self) -> None:
        problems = score_blockers(self.state)
        if problems:
            raise RuntimeError(" ".join(problems))
        variants = self._variants_to_score()
        device = "cuda" if (self.runtime is not None and self.runtime.has_gpu) else "cpu"
        scores = self.backend.score_variants(
            self.bundle, variants, device=device, batch_size=int(self.state.get("score_batch_size") or 8)
        )
        rank_by = str(self.state.get("score_rank_by") or "pred_mean")
        self.scores = scores.sort_values(rank_by, ascending=False).reset_index(drop=True)
        target = work_dir(self.state) / "scored_variants.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.scores.to_csv(target, index=False)
        self.score_result.value = theme.note_html(
            f"Scored {len(self.scores)} variants, ranked by `{rank_by}`. Written to `{target}`."
        ) + self.scores.head(20).to_html(index=False)
        self.backend.offer_download(target)

    def _variants_to_score(self) -> Any:
        if self.spec is None:
            raise ValueError("There is no library in this session to take variants from.")
        source = self.state.get("score_source")
        if source == "upload_csv":
            return self.backend.read_variant_csv(Path(str(self.state.get("score_variants_csv")).strip()))
        how_many = max(1, int(self.state.get("score_top_n") or 1))
        if source == "random_combinations":
            return random_variants(self.spec, how_many, int(self.state.get("score_random_seed") or 0))
        return self.frame.head(how_many).reset_index(drop=True)

    # -- entry point -------------------------------------------------------------------

    def display(self) -> None:
        """Put the wizard on the page."""
        from IPython.display import display

        display(self.root)


def launch(
    *,
    work_dir: Path | str | None = None,
    state: core.WizardState | None = None,
    backend: Backend | None = None,
    runtime: core.Runtime | None = None,
    mount_drive: bool = False,
) -> MainWizard:
    """Build the wizard, put it on the page, and hand it back. What the notebook cell calls.

    `work_dir` is where every run writes. The setup cell computes it after an optional Drive
    mount, so a run started here survives a disconnected session; without it the wizard falls
    back to a directory beside the notebook, which Colab discards when the session ends.
    """
    detected = runtime if runtime is not None else core.detect_runtime()
    resolved = state if state is not None else new_state()
    if work_dir is not None:
        resolved.output_dir = str(work_dir)
    wizard = MainWizard(state=resolved, backend=backend, runtime=detected)
    if mount_drive:
        wizard.fields["use_drive"].value = True
    wizard.display()
    return wizard
