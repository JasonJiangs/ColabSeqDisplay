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

**Preparation is part of this page.** It used to be a notebook of its own, and both
notebooks were about to ask the same two questions — which backbone, which pooling — whose
answers decide everything else. A user answered them once in Prepare, downloaded
`wt_3di.txt` and `region_p90.json`, opened this notebook, answered them again and uploaded
the files back. Now the questions are asked once, in step 2, and step 3 is whatever those
answers imply: `colabsd.ui.prepare_workflow` decides what has to be prepared and builds the
sections that prepare it, and what they produce stays in the session. When nothing has to
be prepared — an ESM2 backbone with `mutation_site_mean` pooling —
there is no step 3 at all.

Two deliberate differences from the ColabPLM notebooks we are otherwise copying:

* **The hyperparameters are read-only.** They hand the user LoRA and trainer widgets; we
  look the values up from a study that was already run and show them with their provenance
  and their measured test Spearman, or, for a placeholder entry, in red. A hyperparameter
  re-tuned while you watch your own validation score has quietly eaten your test set.
* **The test set is not unlocked here.** There is no widget on this page that can read it.
  That is a separate cell, run on purpose, which counts every unlock. The performance
  archive is written from validation numbers with that partition still locked, because the
  report is what a user should be reading while they are still deciding anything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from colabsd.backbones.registry import BACKBONES
from colabsd.spec import LibrarySpec
from colabsd.ui import core, exports, theme
from colabsd.ui import prepare_workflow as prep

# --------------------------------------------------------------------------------------
# What is on the page
# --------------------------------------------------------------------------------------

#: Page order. The user answers the two questions that decide everything — backbone and
#: pooling — in step 2, and step 3 is whatever those answers imply. It is not a question of
#: its own, and for the common pair (ESM2 with `mutation_site_mean`) it is not on the page.
PAGE_ORDER: tuple[str, ...] = (
    "data",
    "model",
    "preparation",
    "hyperparameters",
    "training",
    "storage",
    "run",
    "results",
    "export",
    "score",
)

#: The two blocks inside the preparation step. They are message *slots* rather than page
#: sections, so a 3Di refusal sits under the 3Di controls and a region warning under the
#: region controls instead of both landing in one pile under the step heading.
PREPARATION_SLOTS: tuple[str, ...] = ("structure", "region")

#: Everywhere a contextual message can be put.
MESSAGE_SLOTS: tuple[str, ...] = PAGE_ORDER + PREPARATION_SLOTS

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

#: The preparation fields `core` does not declare either. `prepare_workflow` owns their
#: rules, because they only exist because of an answer given in step 2.
PREPARATION_FIELDS: tuple[str, ...] = (
    "esmfold_risk_accepted",
    "region_model",
    "region_n_sample",
    "region_seed",
    "region_batch_size",
    "region_run_on_cpu",
)

#: Section names without their numbers: which step a section *is* depends on which other
#: sections this configuration has, and `numbered_titles` works that out at every refresh.
#: A page that skips from 2 to 4 reads as a step the user has failed to find.
SECTION_TITLES: dict[str, str] = {
    "data": "Your variant library",
    "model": "Backbone and pooling",
    "preparation": "What that choice needs prepared",
    "hyperparameters": "The hyperparameters for this pair",
    "training": "How many runs",
    "storage": "Where the results are kept",
    "run": "Train",
    "results": "What came back",
    "export": "The two things you take away",
    "score": "Score some variants",
}

#: The two blocks inside the preparation step. They are lettered only when both are there;
#: on their own a block carries the step's own number, because "3b" with no "3a" is a
#: missing half. Whether either exists at all is `prepare_workflow`'s decision.
PREPARATION_TITLES: dict[str, str] = {
    "structure": "The wild-type shape, as a 3Di string",
    "region": "The pooling region",
}

SECTION_NOTES: dict[str, str] = {
    "data": (
        "One row per variant: which residue sits at each mutated site, and what you measured for it. "
        "Full-length sequences are **built** by substituting those residues into your wild type, never "
        "read from a FASTA, so every variant is the same length and every position is checked against "
        "the sequence you gave."
    ),
    "model": (
        "These are the two questions that decide everything below, and they are asked once, here. The "
        "backbone is the protein language model that reads each sequence; the pooling decides *which "
        "residues* it is read out from. Do not agonise over the backbone — the default is tuned, reads "
        "sequence only and fits a free T4. Pooling is the choice that matters. Whatever either answer "
        "needs prepared appears as the next step; for the default pair there is nothing to prepare, and that "
        "step is not on the page at all."
    ),
    "preparation": (
        "A control you cannot see here is one your answers above do not need: choosing ESM2 does not grey the "
        "3Di controls out, it removes them. Change the backbone or the pooling and this step changes with it, "
        "or goes away entirely."
    ),
    "hyperparameters": (
        "Nothing here is editable, on purpose. These seven numbers were selected once, by a search that "
        "was run and recorded, and are looked up for the backbone and pooling above. There are no sliders "
        "because a hyperparameter you re-tune while watching your own validation score is a hyperparameter "
        "that has quietly eaten your test set."
    ),
    "training": (
        "A run is one split seed x one model seed. Repeating over several of each is the only thing that "
        "makes a ± mean anything, because the spread you report is the spread across runs."
    ),
    "storage": core.DRIVE_NOTE,
    "run": (
        "LoRA adapters on the attention projections plus a small head; the backbone itself stays frozen. "
        "Each run trains on the training split and early-stops on validation. **Validation numbers only** "
        "come back — the test partition is moved out of reach as each run finishes, and the last cell of "
        "this notebook is the only thing that can read it."
    ),
    "results": (
        "The language model and the one-hot floor, side by side. The floor is twenty features per mutated "
        "site fitted by ridge and a small MLP on the same splits, and it is the number that says whether "
        "the language model earned its keep: a 650M-parameter model that ties a ridge regression on "
        "one-hot residues has told you the landscape is additive, not that the model is good."
    ),
    "export": (
        "**The model**, as one `.zip` you can keep, share, or hand to the Predict notebook months from now: "
        "the LoRA weights, the head, your library description, the *frozen* pooling coordinates — so scoring "
        "never depends on a region file being around — the hyperparameters and the provenance.\n"
        "- **The performance**, as a second `.zip`: the report CSV, the figure, and a `performance.json` that "
        "says which partition those numbers describe and how many times the test set has been read. It is "
        "written from **validation** numbers with the test partition still locked — which is the state you "
        "should be in while you are still deciding anything. Unlock the test set later and press it again, and "
        "the same archive comes back carrying the test numbers and the count."
    ),
    "score": (
        "The smallest useful prediction outlet, so this page ends with something you can act on. "
        "`pred_min` is the one to rank by when you want a variant that works in **every** condition rather "
        "than on average. For a real screen open **ColabSeqDisplay_Predict.ipynb**, which needs nothing "
        "but the `.zip` above."
    ),
}

#: Which slot a contextual message belongs beside. ColabPLM puts its red text right under the
#: control that caused it; a message whose slot is hidden falls back to the board above the
#: train button, so nothing is ever silently dropped.
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
    "region_not_ready": "region",
    "region_file_missing": "region",
    "region_pooling_unknown": "region",
    "region_reuse_available": "region",
    "region_reuse_gone": "region",
    "region_needs_gpu": "region",
    "region_cpu_override": "region",
    "region_subset_only": "region",
    "region_full_library": "region",
    "region_long_run": "region",
    "region_batch_memory": "region",
    "config_provisional": "hyperparameters",
    "config_missing": "hyperparameters",
    "config_unreadable": "hyperparameters",
    "single_run": "training",
    "run_exceeds_session": "training",
    "no_persistence": "storage",
    "drive_on": "storage",
    "plm_below_floor": "results",
    "results_stale": "run",
    "scoring_on_cpu": "score",
}

#: `core` marks a message `stop` when it means "this is a bad idea"; only some of those
#: make the run impossible. An unsaved run lost to a disconnect is a bad idea and is said
#: loudly; a missing 3Di string is a crash three minutes in. The button refuses the second
#: kind only, and shows both. The refusals that belong to a preparation *button* — an
#: unaccepted ESMFold risk, a CPU-only region discovery — are not here: they stop that step,
#: and the step not having finished is what stops the run.
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
        "region_not_ready",
        "region_file_missing",
        "region_pooling_unknown",
        "region_reuse_gone",
    }
)

POOLING_NOTES: dict[str, str] = {
    "cosine_p90_mean": (
        "**cosine_p90_mean** averages the residues whose embeddings move most across your variants — a "
        "region discovered once and saved to a file. It usually reads out more than the mutated sites "
        "alone, because a mutation changes how the model sees its neighbours too."
    ),
    "cosine_p95_mean": (
        "**cosine_p95_mean** is the same idea as `cosine_p90_mean` over a stricter percentile: about half "
        "as many residues, all of them ones that move a lot."
    ),
    "mutation_site_mean": (
        "**mutation_site_mean** averages the mutated sites themselves and nothing else. It needs no extra "
        "file, and it is the honest default for a protein you have not run region discovery on."
    ),
    "full_mean": "**full_mean** averages every residue in the protein, mutated or not.",
    "last150_mean": "**last150_mean** averages the last 150 residues, whatever is there.",
}

#: The bundled example's name, taken from `prepare_workflow` so the two halves of the page
#: cannot end up calling the same library different things.
EXAMPLE_NAME = prep.EXAMPLE_NAME
EXAMPLE_POSITIONS = "984, 985, 990, 1012, 1016"
EXAMPLE_MUTATION_COLUMNS = "nnk1, nnk2, nnk3, nnk4, nnk5"
EXAMPLE_CONDITION_COLUMNS = "NNGA, NNGT, NNGC, NNGG"
EXAMPLE_DIR = ("slugcas9_5nnk",)


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


def example_note() -> str:
    """The bundled example described from the files themselves, so the prose cannot drift."""
    variants = core.reference_library_variants()
    length = example_wt_length()
    size = f"{variants:,d} variants" if variants else "the variants"
    protein = f"a {length:,d}-residue Cas9" if length else "a Cas9"
    sites = len(parse_list(EXAMPLE_MUTATION_COLUMNS))
    conditions = parse_list(EXAMPLE_CONDITION_COLUMNS)
    return (
        f"The bundled **{EXAMPLE_NAME}** library: {size} of {protein}, {sites} randomised "
        f"sites (`{EXAMPLE_POSITIONS}`) measured against {len(conditions)} PAM conditions "
        f"(`{EXAMPLE_CONDITION_COLUMNS}`). Its wild type, its 3Di string and its pooling regions all ship "
        "with the package, so this runs end to end with nothing to prepare."
    )


def withdrawn_note() -> str:
    """Where the backbones that used to be on this list went, one line per name.

    Somebody who came here for ProtT5 or ESMC deserves an answer rather than a shorter list
    and no explanation, and the answer is not "they were deleted". It is also not one
    sentence: this note used to say every withdrawn backbone had a tested adapter and no
    tuned hyperparameters, and that was wrong four times over — METL has no adapter at all,
    and ProtT5-XL, ESMC-300M, ESMDance and METL each ship a tuned entry in `config/best/`.
    So the reason is asked of `colabsd.backbones.registry` per name, which is where it is
    written and where the notebook's own guide reads it from.
    """
    from colabsd.backbones import registry

    withdrawn = core.withdrawn_backbones()
    if not withdrawn:
        return ""
    families = " and ".join(registry.OFFERED_FAMILIES)
    lines = [
        f"**Only {families} are offered here**, so the two questions above stay a real comparison — read the "
        f"sequence, or read the sequence and the shape — rather than a menu of {len(BACKBONES)}. The other "
        f"{len(withdrawn)} are in the package and not on this form, each for its own reason:"
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
        "Putting one back on the form is one line: its name moves out of `WITHHELD_FAMILY_REASONS` and into "
        "`OFFERED_FAMILIES` in `colabsd/backbones/registry.py`."
    )
    return "\n".join(lines)


LIBRARY_NOTE = (
    "**Mutated positions** — where the randomised residues sit in the wild type, counted from 1, so "
    "`984` is the 984th residue of the sequence above.\n"
    "- **Mutation columns** — the CSV columns holding the residue at each of those positions, in the "
    "same order: column *i* names position *i*.\n"
    "- **Condition columns** — the CSV columns holding what you measured. One per condition; the model "
    "predicts all of them at once.\n"
    "- **Read-count column** — if your table has sequencing counts, name it and you can drop the "
    "variants that were seen too few times to trust. Leave it empty if there is none."
)

SEED_NOTE = (
    "One of each is the fast default, and it reports `nan` for the spread — honestly, because a single "
    "run cannot show reproducibility. The registry's own protocol uses three of each."
)

#: Everything this wizard keeps in `WizardState.extra`, and what it starts as.
EXTRA_DEFAULTS: dict[str, Any] = {
    "library_loaded": False,
    "trained": False,
    "exported": False,
    "floor_cleared": None,
    "library_csv": "",
    "wt_sequence": "",
    "positions_1based": EXAMPLE_POSITIONS,
    "mutation_columns": EXAMPLE_MUTATION_COLUMNS,
    "condition_columns": EXAMPLE_CONDITION_COLUMNS,
    "three_letter_residues": True,
    "count_column": "count",
    "min_count": 0,
    "region_source": "available",
    "region_filename": "",
    "region_model": prep.DEFAULT_REGION_MODEL,
    "region_n_sample": 2000,
    "region_seed": 0,
    "region_batch_size": 2,
    "region_run_on_cpu": False,
    "region_ready": False,
    "region_path": "",
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
        state.pooling,
        state.dtype,
        str(state.get("region_source") or ""),
        str(state.get("region_filename") or ""),
        str(state.get("region_path") or ""),
        state.three_di_source,
        int(state.wt_3di_length),
        int(state.n_split_seeds),
        int(state.n_model_seeds),
    )


def region_key(state: core.WizardState) -> tuple[Any, ...]:
    """What a resolved pooling region has to keep describing to stay usable.

    Not the whole training fingerprint: re-reading the library at the same length does not
    invalidate a region, but changing the pooling, the protein or where the file came from
    does — and a region kept across any of those is pooled under a name it does not have.
    """
    return (
        str(state.pooling),
        int(state.wt_length),
        str(state.get("region_source") or ""),
        str(state.get("region_filename") or ""),
    )


def results_are_stale(state: core.WizardState) -> bool:
    """True when the form has been changed since the run whose numbers are on screen.

    Without this the page will happily show one pair's validation table under another
    pair's name, and -- worse -- write the second pair's hyperparameters into the `.zip`
    the first pair's weights are in.
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

    `core` decides which sections this *configuration* calls for; `prepare_workflow` decides
    whether the configuration needs anything prepared at all. This adds the one thing
    neither has an opinion about: nothing below the library appears until the library has
    actually loaded, because the runtime estimate, the ESMFold length check, the region
    subsample and the pooled coordinates all need to know how big it is and how long the
    wild type is.
    """
    live = set(core.visible_sections(state))
    loaded = bool(state.get("library_loaded"))
    trained = has_results(state)
    visible = {name: (name in live and loaded) for name in core.SECTION_ORDER}
    visible["data"] = True
    visible["preparation"] = loaded and bool(prep.preparation_steps(state))
    visible["results"] = trained
    visible["export"] = trained
    visible["score"] = has_bundle(state)
    return {name: visible.get(name, False) for name in PAGE_ORDER}


def slot_visibility(state: core.WizardState) -> dict[str, bool]:
    """Every message slot: the page sections, plus the two blocks inside step 3."""
    visible = dict(section_visibility(state))
    steps = prep.preparation_steps(state)
    visible["structure"] = visible["preparation"] and prep.STEP_THREE_DI in steps
    visible["region"] = visible["preparation"] and prep.STEP_REGION in steps
    return {name: visible.get(name, False) for name in MESSAGE_SLOTS}


def visible_sections(state: core.WizardState) -> list[str]:
    """The sections on screen, in page order."""
    visible = section_visibility(state)
    return [name for name in PAGE_ORDER if visible[name]]


def numbered_titles(state: core.WizardState) -> dict[str, str]:
    """Every slot's heading, numbered by where it actually falls on this page.

    The steps a configuration does not need are absent, not greyed out, so their numbers
    have to close up behind them: an ESM2 run with `mutation_site_mean` pooling has no
    preparation step, and its hyperparameters are step 3 rather than a step 4 after a
    step 3 nobody can find.
    """
    shown = visible_sections(state)
    titles = {name: f"{index} · {SECTION_TITLES[name]}" for index, name in enumerate(shown, start=1)}
    for name in PAGE_ORDER:
        titles.setdefault(name, SECTION_TITLES[name])
    step = shown.index("preparation") + 1 if "preparation" in shown else 0
    slots = slot_visibility(state)
    blocks = [name for name in PREPARATION_SLOTS if slots[name]]
    for position, name in enumerate(blocks):
        # One block on its own is the step, and the step is already numbered above it.
        prefix = f"{step}{chr(ord('a') + position)} · " if step and len(blocks) > 1 else ""
        titles[name] = f"{prefix}{PREPARATION_TITLES[name]}"
    for name in PREPARATION_SLOTS:
        titles.setdefault(name, PREPARATION_TITLES[name])
    return titles


def field_overrides(state: core.WizardState) -> dict[str, bool]:
    """The declared fields whose rule depends on a radio only this page has.

    `core.FIELD_RULES` shows `region_filename` for any cosine pooling, because `core` does
    not know about this page's "reuse, upload or discover?" radio. Leaving the upload row on
    screen beside "reuse the one that is already here" invites a path that is then silently
    ignored.
    """
    return {
        "region_filename": prep.needs_region(state) and state.get("region_source") == "upload",
    }


def preparation_field_visibility(
    state: core.WizardState, *, artefacts: Sequence[prep.Artefact] = (), runtime: core.Runtime | None = None
) -> dict[str, bool]:
    """The step-3 fields `core` does not declare. The rules are `prepare_workflow`'s.

    `core.FIELD_RULES` declared `region_n_sample` too while a standalone Prepare wizard
    existed, and the live rule here had to be applied after `core`'s to win. That mode is
    gone and so is the duplicate: one field, one rule, and it lives where the step does.
    """
    return prep.preparation_field_visibility(
        state,
        artefact=prep.pick(artefacts, prep.STEP_REGION),
        has_gpu=None if runtime is None else bool(runtime.has_gpu),
    )


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
                "Nothing has been read yet. Press **Check my library** below: the runtime estimate, the "
                "pooled coordinates and the length checks all need to know how big it is.",
            )
        )
    if int(state.get("min_count") or 0) > 0 and not str(state.get("count_column") or "").strip():
        out.append(
            core.Message(
                "min_count_without_count_column",
                "warning",
                f"You asked to drop variants seen fewer than {state.get('min_count')} times, but no read-count "
                "column is named, so nothing can be filtered and every row will be kept.",
            )
        )
    if three_di_is_promised_but_missing(state):
        out.append(
            core.Message(
                "three_di_not_attached",
                "stop",
                f"You chose where the 3Di string should come from, but nothing has fetched it yet, so "
                f"**{state.backbone}** would be trained with no structure at all — silently, and on a model "
                "whose whole point is that it reads one. Press **Get the 3Di string** in the preparation step.",
            )
        )
    if results_are_stale(state):
        out.append(
            core.Message(
                "results_stale",
                "warning",
                "The settings above have changed since the last run, so the results, the bundle and the "
                "scoring outlet have been put away rather than left describing a run that no longer matches "
                "this form. Press **Train** again, or change the settings back.",
            )
        )
    cleared = state.get("floor_cleared")
    if cleared is False and has_results(state):
        out.append(
            core.Message(
                "plm_below_floor",
                "warning",
                "The language model did not clear the one-hot floor. On this library, twenty features per "
                "mutated site do as well as a fine-tuned protein language model, which usually means the "
                "landscape is close to additive. That is a result, not a failure — but do not report the "
                "model as an improvement.",
            )
        )
    return out


def preparation_messages(
    state: core.WizardState,
    *,
    runtime: core.Runtime | None = None,
    artefacts: Sequence[prep.Artefact] | None = None,
) -> list[core.Message]:
    """`prepare_workflow.notices` for this state, with the machine and the preview filled in.

    Silent when this configuration needs nothing prepared, which is the point of the merge:
    the step that does not exist says nothing either.
    """
    if not prep.preparation_steps(state):
        return []
    found = prep.available_artefacts(state) if artefacts is None else list(artefacts)
    has_gpu = None if runtime is None else bool(runtime.has_gpu)
    preview = prep.region_preview(state) if prep.needs_region(state) else None
    estimate = (
        prep.region_estimate(state, has_gpu=bool(has_gpu), preview=preview) if preview is not None else None
    )
    return prep.notices(
        state,
        artefacts=found,
        preview=preview,
        estimate=estimate,
        has_gpu=has_gpu,
        gpu_gb=float(getattr(runtime, "gpu_memory_gb", 0.0) or 0.0),
        region_ready=bool(state.get("region_ready")),
    )


def messages(
    state: core.WizardState,
    *,
    runtime: core.Runtime | None = None,
    status: core.ConfigStatus | None = None,
    root: str | Path | None = None,
    artefacts: Sequence[prep.Artefact] | None = None,
) -> list[core.Message]:
    """Every message this state earns, worst first: `core`'s, preparation's, and this page's."""
    collected = list(core.messages_for(state, runtime=runtime, status=status, root=root))
    collected += extra_messages(state)
    collected += preparation_messages(state, runtime=runtime, artefacts=artefacts)
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
    grouped: dict[str, list[core.Message]] = {name: [] for name in MESSAGE_SLOTS}
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
                f"{item!r} is not a residue number. Mutated positions are whole numbers counted from 1 along "
                f"the wild-type sequence, for example {EXAMPLE_POSITIONS}."
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
                "The wild-type box is empty. Paste the full-length amino-acid sequence, or the URL of its "
                "FASTA file — an AlphaFold or UniProt link works."
            )
    positions = parse_positions(state.get("positions_1based"))
    columns = parse_list(state.get("mutation_columns"))
    if not columns:
        problems.append("No mutation columns named. List one CSV column per mutated site, in position order.")
    if not parse_list(state.get("condition_columns")):
        problems.append("No condition columns named. List at least one CSV column holding a measured activity.")
    if columns and len(columns) != len(positions):
        problems.append(
            f"{len(columns)} mutation columns for {len(positions)} positions. There must be exactly one column "
            "per position, in the same order, so that column i still names position i."
        )
    return problems


#: What stops the two preparation buttons is `prepare_workflow`'s decision, because the
#: controls those buttons read are its: `prep.three_di_blockers` and `prep.region_blockers`.


def export_blockers(state: core.WizardState) -> list[str]:
    """What stops the export button."""
    if not state.get("trained"):
        return ["Nothing has been trained in this session, so there is no model to export."]
    if results_are_stale(state):
        return [
            "The settings have changed since this model was trained, so exporting now would write the "
            "settings on screen into a bundle holding the old run's weights. Press Train again first."
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
# Pure decisions -- region, adapter, and the call into colabsd.train
# --------------------------------------------------------------------------------------


def region_problems(
    region: dict[str, Any], pooling: str, wt_length: int, *, source: str = "the region file"
) -> list[str]:
    """Check a region record against the run it is about to be pooled under.

    Averaging one region under another region's name is silently wrong, and so is a region
    computed for a different protein, so both are refused rather than warned about.
    """
    problems = []
    declared = region.get("pooling") or f"{region.get('region_name', '')}_mean"
    if declared != pooling:
        problems.append(
            f"{source} holds the {declared} region, but this run pools {pooling}. Averaging one region under "
            "another region's name is silently wrong: pick the matching file."
        )
    length = region.get("seq_length")
    if length is not None and int(length) != int(wt_length):
        problems.append(
            f"{source} was computed for a {length}-residue protein, but this wild type is {wt_length} residues. "
            "Discover a region for this one in the preparation step instead."
        )
    return problems


def region_notes(region: dict[str, Any], backbone: str) -> list[str]:
    """What the region file records about its own making, when it bears on this run.

    A region is a list of residues that moved under *some* model's embeddings; the file
    says which. Pooling it under a different backbone is a decision, not an error, so this
    is said once beside the choice rather than made into a refusal.
    """
    source = str(region.get("region_source_model") or "").strip()
    if not source or source == backbone:
        return []
    return [
        f"This region was discovered from **{source}** embeddings and you are training **{backbone}**. The "
        "residues that move most under one model are not necessarily the ones that move most under another, "
        f"so this pools {backbone} over a region {source} chose. Rediscover it in the preparation step, with "
        f"{backbone} as the region model, if you want the region and the backbone to agree."
    ]


def pooling_positions_0based(spec: LibrarySpec, pooling: str, region: dict[str, Any] | None) -> list[int]:
    """The residue offsets the pooled feature is averaged over."""
    if not pooling.startswith("cosine_"):
        return list(spec.positions_0based())
    if not region:
        raise ValueError(
            f"{pooling} averages a discovered region, so it needs one. Finish the pooling-region step above, "
            "or pick mutation_site_mean, which pools the mutated sites and needs nothing prepared."
        )
    return [int(position) - 1 for position in region["positions_1based"]]


def adapter_kwargs(state: core.WizardState, spec: LibrarySpec, region: dict[str, Any] | None) -> dict[str, Any]:
    """The keyword arguments `colabsd.backbones.registry.create_adapter` is called with."""
    return {
        "pooling": state.pooling,
        "pooling_positions_0based": pooling_positions_0based(spec, state.pooling, region),
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
    """Exactly the call `colabsd.train.finetune` is about to receive."""
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


def hyperparameter_view(best: Any) -> HyperparameterView:
    """Present a registry entry as a settled result, or as a placeholder that says so."""
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
        training_rows=[(name, _format_value(value)) for name, value in sorted(dict(best.fixed).items())],
        source_path=str(getattr(best, "path", "") or "the bundled registry"),
    )


def _rows_html(rows: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f"<tr><td style='padding:1px 14px 1px 0'><code>{name}</code></td><td>{value}</td></tr>"
        for name, value in rows
    )
    return f"<table>{cells}</table>"


def hyperparameter_html(view: HyperparameterView) -> str:
    """The looked-up hyperparameters, as a settled result or as a red placeholder."""
    if view.is_provisional:
        banner = theme.message_html(
            f"**PROVISIONAL.** {view.headline} These seven numbers are a placeholder taken from tuned entries "
            "for other backbones. No study selected them for this pair, and no performance number stands "
            "behind them.",
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


def no_hyperparameters_html(backbone: str, pooling: str, status: core.ConfigStatus | None) -> str:
    """Section 4 when the registry has nothing for this pair.

    It has to say something, because leaving the previous pair's numbers on screen is how a
    user reads seven settled hyperparameters under a heading that has none.
    """
    detail = status.description if status is not None else "The registry has not been read yet."
    return theme.message_html(
        f"**No hyperparameters for {backbone} with `{pooling}` pooling.** {detail} Nothing is shown here "
        "because there is nothing to show: pick a pair the registry has an entry for.",
        "stop",
    )


def floor_rows(
    baseline: dict[str, Any] | None, *, metric: str = "Spearman", partition: str = "validation"
) -> list[dict[str, Any]]:
    """The one-hot floor as one row per head, ready to sit beside the model's own table."""
    summary = (baseline or {}).get("summary", {}).get(partition, {})
    rows = []
    for head_name, per_condition in summary.items():
        entry = per_condition.get("mean", {}).get(metric)
        if entry is None:
            continue
        rows.append(
            {
                "head": head_name,
                f"{metric}_mean": float(entry["mean"]),
                f"{metric}_sd": float(entry["sd"]),
                "n_runs": int(entry["n"]),
            }
        )
    return rows


def clears_floor(plm_mean: float | None, floor: dict[str, Any] | None) -> bool | None:
    """Did the language model beat the floor? `None` when there is nothing to compare."""
    if floor is None or plm_mean is None or plm_mean != plm_mean:
        return None
    return float(plm_mean) > float(floor["mean"])


def floor_comparison(model_label: str, plm_mean: float | None, floor: dict[str, Any] | None) -> str:
    """One sentence saying whether the language model earned its keep."""
    if floor is None:
        return "No one-hot floor was computed, so there is nothing to compare this against yet."
    line = (
        f"{floor['head']} reaches validation Spearman {floor['mean']:.4f} ± {floor['sd']:.4f} "
        f"over {floor['n']} run(s)."
    )
    verdict = clears_floor(plm_mean, floor)
    if verdict is None:
        return line
    gap = float(plm_mean) - float(floor["mean"])
    if verdict:
        return (
            f"{line} {model_label} clears it by {gap:.4f} Spearman — the language model found structure that "
            "per-site additivity does not explain."
        )
    return (
        f"{line} {model_label} FAILS to clear it, by {abs(gap):.4f} Spearman. A one-hot encoding of the "
        "mutated residues is as good as the language model on this library."
    )


def validation_headline(summary: dict[str, Any] | None) -> str:
    """The macro validation Spearman line, honest about a single run."""
    macro = (summary or {}).get("Spearman", {})
    mean = float(macro.get("mean", float("nan")))
    sd = float(macro.get("sd", float("nan")))
    n_runs = int(macro.get("n_runs", 0))
    line = f"Validation Spearman, averaged over conditions: {mean:.4f} ± {sd:.4f} ({n_runs} run(s))"
    if n_runs < 2:
        line += " — one run shows no spread, so that ± is `nan` on purpose."
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
            f" The registry entry was selected over {protocol} runs; {estimate.n_runs} is faster and says "
            "correspondingly less about reproducibility."
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

    def load_region(self, path: Path) -> dict[str, Any]:
        from colabsd.region import load_region

        return load_region(path)

    def discover_region(self, sequences: Sequence[str], wt_sequence: str, **kwargs: Any) -> dict[str, Any]:
        from colabsd.region import discover_region

        return discover_region(sequences, wt_sequence, **kwargs)

    def save_region(self, record: dict[str, Any], path: Path) -> Path:
        from colabsd.region import save_region

        path.parent.mkdir(parents=True, exist_ok=True)
        save_region(record, path)
        return path

    def load_best_config(self, backbone: str, pooling: str) -> Any:
        from colabsd.bestconfig import load_best_config

        return load_best_config(backbone, pooling)

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

    def one_hot_baseline(self, *args: Any, **kwargs: Any) -> Any:
        from colabsd.baseline import one_hot_baseline

        return one_hot_baseline(*args, **kwargs)

    def baseline_floor(self, baseline: dict, **kwargs: Any) -> Any:
        from colabsd.baseline import baseline_floor

        return baseline_floor(baseline, **kwargs)

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

    from colabsd.baseline import one_to_three_letter

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
        self.region: dict[str, Any] | None = None
        self.region_path: Path | None = None
        self.session_artefacts: list[prep.Artefact] = []
        self._region_key: tuple[Any, ...] | None = None
        self.best: Any = None
        self.adapter: Any = None
        self.splits: Any = None
        self.run: Any = None
        self.baseline: Any = None
        self.floor: dict[str, Any] | None = None
        self.bundle: Any = None
        self.bundle_path: Path | None = None
        self.performance: exports.PerformanceExport | None = None
        self._best_key: tuple[str, str] | None = None
        self._refreshing = False
        self.trained_best: Any = None
        self.trained_spec: LibrarySpec | None = None
        self.trained_region: dict[str, Any] | None = None
        self.scores: Any = None
        self.plan: core.Plan | None = None

        self._build()
        self.refresh()

    # -- construction ------------------------------------------------------------------

    def _build(self) -> None:
        import ipywidgets

        self.w = ipywidgets
        self.logs = {name: theme.html("") for name in MESSAGE_SLOTS}
        self.boards = {name: core.MessageBoard() for name in MESSAGE_SLOTS}

        self.fields: dict[str, Any] = {}
        self.outlets: dict[str, Any] = {}
        self.uploads: dict[str, Any] = {}

        self._build_data()
        self._build_model()
        self._build_preparation()
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
        self.fields["pooling"] = self.w.Dropdown(
            options=core.pooling_choices(self.state.backbone) or [self.state.pooling],
            value=self.state.pooling,
            description="Pooling:",
            style=_LABEL,
            layout=_WIDE,
        )
        self.pooling_note = theme.note("")
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
            self.fields["pooling"],
            self.pooling_note,
            self.fields["dtype"],
            self.boards["model"].widget,
            self.logs["model"],
        ]

    def _build_preparation(self) -> None:
        """Step 3, composed out of `prepare_workflow`'s two sections.

        The panel supplies its own widget factories, so a section built here looks like the
        rest of the page and its uploads go through the one row that says what a blocking
        `files.upload()` is about to do. Which of the two blocks exists on screen is not
        decided here: `prep.preparation_steps` decides it, from step 2.
        """
        tools = prep.SectionTools(text=self._text, upload_row=self._upload_row, button=self._button)
        self.three_di = prep.ThreeDiSection(self.state, tools)
        self.region_step = prep.RegionSection(self.state, tools)
        self.fields.update(self.three_di.fields)
        self.fields.update(self.region_step.fields)
        self.preparation_note = theme.note("")
        self.outputs_note = theme.html("")
        self.sub_sections = {
            "structure": core.Section(
                "structure",
                PREPARATION_TITLES["structure"],
                [
                    *self.three_di.children(),
                    self.boards["structure"].widget,
                    self.three_di.button,
                    self.logs["structure"],
                ],
            ),
            "region": core.Section(
                "region",
                PREPARATION_TITLES["region"],
                [
                    *self.region_step.children(),
                    self.boards["region"].widget,
                    self.region_step.button,
                    self.logs["region"],
                ],
            ),
        }
        self._layout["preparation"] = [
            self.preparation_note,
            self.sub_sections["structure"].box(),
            self.sub_sections["region"].box(),
            self.outputs_note,
            self.boards["preparation"].widget,
            self.logs["preparation"],
        ]

    def _build_hyperparameters(self) -> None:
        self.fields["hyperparameters"] = theme.html("")
        self._layout["hyperparameters"] = [
            self.fields["hyperparameters"],
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
            theme.note(SEED_NOTE),
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

        `files.upload()` blocks the kernel: while it waits, every observer on this page is
        dead, so the panel sits in whatever state the button left it in and nothing on
        screen says why. The explanation therefore has to be written before the call, not
        after it — after it is an hour too late for the person watching a frozen form.
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
        core.on_click(self.region_step.button, self._guard("region", self.on_get_region))
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

        `refresh` writes to widgets that are themselves observed -- the pooling options and
        the seed maxima -- so without this a single keystroke would re-enter it.
        """
        if state.get("_reloading") or self._refreshing:
            return
        self.refresh()

    def _guard(self, section: str, action: Callable[[], None]) -> Callable[[], None]:
        """Run a step, and put whatever it raises on the page instead of in a traceback."""

        def handle() -> None:
            self.logs[section].value = ""
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - the message is the product here
                self.logs[section].value = theme.message_html(
                    f"**That did not work.** {type(exc).__name__}: {exc}", "stop"
                )
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
        self._sync_pooling_options()
        self._sync_seed_limits()
        self._forget_stale_region()
        artefacts = self.artefacts()
        self._sync_preparation(artefacts)
        self.plan = core.plan(self.state, runtime=self.runtime, status=status)
        items = messages(self.state, runtime=self.runtime, status=status, artefacts=artefacts)

        visible = slot_visibility(self.state)
        core.apply_field_visibility({k: v for k, v in self.fields.items() if k in core.FIELD_KEYS}, self.state)
        for key, shown in field_overrides(self.state).items():
            core.set_display(self.fields[key], shown)
        for key, shown in preparation_field_visibility(
            self.state, artefacts=artefacts, runtime=self.runtime
        ).items():
            core.set_display(self.fields[key], shown)
        for key, shown in outlet_field_visibility(self.state).items():
            core.set_display(self.outlets[key], shown)
        titles = numbered_titles(self.state)
        for name, section in self.sections.items():
            section.set_visible(visible[name])
            section.set_title(titles[name])
        for name, section in self.sub_sections.items():
            section.set_visible(visible[name])
            section.set_title(titles[name])

        grouped = messages_by_section(items, visible)
        for name, board in self.boards.items():
            board.update(grouped[name])

        self.backbone_note.value = theme.note_html(core.backbone_summary(self.state.backbone))
        self.pooling_note.value = theme.note_html(POOLING_NOTES.get(self.state.pooling, ""))
        self.plan_note.value = theme.note_html(plan_line(self.state, self.best))
        self.fields["run_button"].disabled = not can_train(items)
        if self.best is not None:
            self.fields["hyperparameters"].value = hyperparameter_html(hyperparameter_view(self.best))
        else:
            self.fields["hyperparameters"].value = no_hyperparameters_html(
                self.state.backbone, self.state.pooling, status
            )
        return self.plan

    def artefacts(self) -> list[prep.Artefact]:
        """What this session could reuse instead of making: its own output, then the example's."""
        return list(prep.available_artefacts(self.state, session=self.session_artefacts))

    def _sync_preparation(self, artefacts: Sequence[prep.Artefact]) -> None:
        """Rebuild step 3 from step 2: the source lists, the estimate, the button labels.

        The source lists are built rather than written down because a choice that does not
        apply must not be on screen at all -- "reuse the bundled region" is not greyed out
        for somebody else's protein, it is absent.
        """
        forced = self.three_di.sync(self.state, prep.pick(artefacts, prep.STEP_THREE_DI))
        if forced is not None:
            self.state.three_di_source = forced
        if prep.needs_region(self.state):
            # Only worth the subsample when a region is actually going to be pooled; this
            # runs on every keystroke.
            preview = prep.region_preview(self.state)
            estimate = prep.region_estimate(self.state, has_gpu=self._has_gpu(), preview=preview)
            forced = self.region_step.sync(
                self.state, prep.pick(artefacts, prep.STEP_REGION), preview=preview, estimate=estimate
            )
            if forced is not None:
                self.state.set("region_source", forced)
        self.preparation_note.value = theme.note_html(prep.step_note(self.state))
        self.three_di.note.value = theme.note_html(prep.block_note(self.state, prep.STEP_THREE_DI))
        self.region_step.note.value = theme.note_html(prep.block_note(self.state, prep.STEP_REGION))
        self.outputs_note.value = prep.outputs_html(prep.output_files(self.state, work_dir=work_dir(self.state)))

    def _forget_stale_region(self) -> None:
        """Drop a resolved region the form has stopped describing.

        A region is a list of residues of one protein under one pooling. Keeping the one
        that was resolved before the pooling changed is how a run pools `cosine_p95_mean`
        over the p90 residues and reports it under the p95 name.
        """
        if self.region is None:
            return
        if self._region_key == region_key(self.state):
            return
        self.region = None
        self._region_key = None
        self.state.set("region_ready", False)
        self.state.set("region_path", "")

    def _has_gpu(self) -> bool:
        return bool(self.runtime is not None and self.runtime.has_gpu)

    def _sync_pooling_options(self) -> None:
        """Keep the pooling list the registry's, not a snapshot taken when the page was built."""
        choices = list(core.pooling_choices(self.state.backbone)) or [self.state.pooling]
        if tuple(self.fields["pooling"].options) == tuple(choices):
            return
        wanted = self.state.pooling if self.state.pooling in choices else choices[0]
        self.fields["pooling"].options = choices
        self.fields["pooling"].value = wanted
        self.state.pooling = wanted

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
        """The registry lookup for the current pair, read once and cached.

        `refresh` runs on every keystroke, and `self.best` is handed straight to
        `finetune`, so re-reading the YAML each time would both cost a file read and hand
        the training call a different object from the one on screen.
        """
        if not self.state.get("library_loaded"):
            return None
        key = (self.state.backbone, self.state.pooling)
        status = core.config_status(*key)
        if key == self._best_key and self.best is not None:
            return status
        self._best_key = key
        if status.found and status.error is None:
            try:
                self.best = self.backend.load_best_config(*key)
            except Exception:  # noqa: BLE001 - config_status already turned this into a message
                self.best = None
        else:
            self.best = None
        return status

    def _progress(self, done: int, total: int, label: str) -> None:
        self.progress.value = f"<div><code>{done}/{total}</code> {label}</div>"

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
            from colabsd import EXAMPLES_ROOT

            example = Path(EXAMPLES_ROOT) / "slugcas9_5nnk"
            return example / "library.csv", self.backend.load_wt_sequence(example / "wt.fasta")
        csv_path = Path(str(self.state.get("library_csv")).strip())
        typed = str(self.state.get("wt_sequence") or "").strip()
        if typed.lower().startswith(("http://", "https://")):
            target = self.backend.download_url(typed, work_dir(self.state) / "wt_downloaded.fasta")
            return csv_path, self.backend.load_wt_sequence(target)
        return csv_path, "".join(typed.split()).upper()

    def on_get_three_di(self) -> None:
        """The 3Di step: put a wild-type 3Di string in this session. It never leaves it.

        Whatever the source — the example's, a file, a structure read by foldseek, or
        ESMFold — the string is attached to the `LibrarySpec` here and written beside the
        run, so nothing has to be downloaded and uploaded back.
        """
        if self.spec is None:
            raise ValueError("Check the library first: the 3Di string is measured against its wild-type length.")
        three_di = self.three_di.resolve(
            self.state,
            self.backend,
            wt_sequence=self.spec.wt_sequence,
            artefact=prep.pick(self.artefacts(), prep.STEP_THREE_DI),
            work_dir=work_dir(self.state),
            has_gpu=self._has_gpu(),
        )
        self.spec = replace(self.spec, wt_3di=three_di)
        self.spec.validate()
        self.state.wt_3di_length = len(three_di)
        target = work_dir(self.state) / "wt_3di.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(three_di + "\n")
        artefact = self.three_di.written_artefact(self.state, target, three_di)
        self.session_artefacts = [
            item for item in self.session_artefacts if item.kind != prep.STEP_THREE_DI
        ] + [artefact]
        length = len(self.spec.wt_sequence)
        self._say(
            "structure",
            f"3Di attached: {len(three_di)} states for {length} residues, and kept in this session.\n"
            f"- sequence `{self.spec.wt_sequence[:60]}…`\n"
            f"- 3Di      `{three_di[:60]}…`\n"
            f"- written to `{target}` so a Drive mount survives a disconnect",
        )

    def on_get_region(self) -> None:
        """The region step: put a pooling region in this session — reused, uploaded, or discovered.

        Discovery is the one step on this page that can outlast a Colab session, which is
        why it is a button of its own with its estimate printed beside it, and why **Train**
        will never start it for you.
        """
        if self.spec is None:
            raise ValueError("Check the library first: a region is checked against the wild-type length.")
        record, path = self.region_step.resolve(
            self.state,
            self.backend,
            artefact=prep.pick(self.artefacts(), prep.STEP_REGION),
            sequences=self.sequences or [],
            wt_sequence=self.spec.wt_sequence,
            work_dir=work_dir(self.state),
            has_gpu=self._has_gpu(),
            progress=self._progress,
            say=lambda line: self._say("region", line),
        )
        problems = region_problems(record, self.state.pooling, self.state.wt_length, source=str(path))
        if problems:
            raise ValueError(" ".join(problems))
        self._hold_region(record, path)
        selected = record.get("n_selected_positions", len(record.get("selected_positions_1based", []) or []))
        self._say(
            "region",
            f"Region ready: **{selected} of {record.get('seq_length', '?')} residues** for "
            f"`{self.state.pooling}`, from `{path}`. It stays in this session — nothing to download.",
        )
        for note in region_notes(record, self.state.backbone):
            self.logs["region"].value += theme.message_html(note, "warning")

    def _hold_region(self, record: dict[str, Any], path: Path | None) -> None:
        """Keep a resolved region, and offer it back as something to reuse."""
        self.region = record
        self.region_path = Path(path) if path is not None else None
        self._region_key = region_key(self.state)
        self.state.set("region_ready", True)
        self.state.set("region_path", str(path or ""))
        if path is not None and str(self.state.get("region_source")) == "discover":
            artefact = self.region_step.discovered_artefact(self.state, Path(path), record)
            self.session_artefacts = [
                item for item in self.session_artefacts if item.path != artefact.path
            ] + [artefact]

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

        self.region = self._resolve_region()
        self.adapter = self.backend.create_adapter(
            self.state.backbone, **adapter_kwargs(self.state, self.spec, self.region)
        )
        split_seeds, model_seeds = training_seeds(self.state, self.best)
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
        self.baseline = self.backend.one_hot_baseline(
            self.frame,
            self.spec,
            self.targets,
            self.splits,
            model_seeds=model_seeds,
            heads=("ridge", "mlp"),
            progress=self._progress,
        )
        self.floor = self.backend.baseline_floor(self.baseline, metric="Spearman", partition="validation")

        macro = (getattr(self.run, "validation_summary", {}) or {}).get("Spearman", {})
        self.state.set("floor_cleared", clears_floor(macro.get("mean"), self.floor))
        # Freeze what this run actually was. `self.best` follows the dropdowns; the bundle
        # must not, or it records one pair's hyperparameters beside another pair's weights.
        self.trained_best, self.trained_spec, self.trained_region = self.best, self.spec, self.region
        self.state.set("trained_fingerprint", training_fingerprint(self.state))
        self.state.set("trained", True)
        self.state.set("exported", False)
        self.results.value = self._results_html()
        for warning in getattr(self.run, "warnings", []) or []:
            self.logs["run"].value += theme.message_html(warning, "warning")

    def _resolve_region(self) -> dict[str, Any] | None:
        """The region this run pools over: the one the preparation step produced, or a cheap one read now.

        Reading a file that is already here costs nothing, so Train does it rather than
        dead-ending on a button. Discovering one costs a session, so Train refuses instead:
        an hour of GPU time is not something a run should start on your behalf.
        """
        if not prep.needs_region(self.state):
            return None
        if self.region is not None and self._region_key == region_key(self.state):
            return self.region
        if str(self.state.get("region_source")) == "discover":
            raise ValueError(
                "No pooling region has been discovered in this session yet. Press **Discover the region** in "
                "the preparation step — it is the one step here that can outlast a Colab session, so Train "
                "will not start it for you."
            )
        self.on_get_region()
        return self.region

    def _results_html(self) -> str:
        """The validation numbers and the one-hot floor, side by side."""
        import pandas as pd

        summary = getattr(self.run, "validation_summary", {}) or {}
        macro = summary.get("Spearman", {})
        model_label = getattr(self.run, "model_name", self.state.backbone)
        aggregate = getattr(self.run, "aggregate", None)
        if aggregate is not None and len(aggregate):
            model_table = aggregate[aggregate["metric"].isin(["Spearman", "R2", "NDCG@50"])].to_html(index=False)
        else:
            model_table = "<i>no per-condition table</i>"
        rows = floor_rows(self.baseline)
        floor_table = pd.DataFrame(rows).to_html(index=False) if rows else "<i>no floor computed</i>"
        return (
            theme.note_html(
                f"Finished {getattr(self.run, 'n_runs', 0)} run(s) in "
                f"{core.format_minutes(getattr(self.run, 'minutes', 0.0))}.\n"
                f"- **{validation_headline(summary)}**\n"
                f"- {floor_comparison(model_label, macro.get('mean'), self.floor)}"
            )
            + "<div style='display:flex;gap:28px;flex-wrap:wrap;margin-top:10px'>"
            + f"<div><b>{model_label} — validation</b>{model_table}</div>"
            + f"<div><b>One-hot floor — validation</b>{floor_table}</div></div>"
            + theme.note_html(
                "These are validation numbers. The test partition is still locked; the last cell of this "
                "notebook is the only thing that opens it, and it counts every time it does."
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
            region=self.trained_region,
            notes=str(self.state.get("bundle_notes") or "").strip() or None,
            runners=self._export_runners(),
        )
        self.bundle, self.bundle_path = export.bundle, export.path
        self.state.set("exported", True)
        self.export_result.value = theme.note_html(f"Written `{export.path}`\n\n{export.describe()}")

    def on_export_performance(self) -> None:
        """Write the performance archive — with the test partition still locked.

        This is the export a user should have while they are still deciding anything, and
        until now it did not exist: the only call to `colabsd.report.build_report` was
        inside the unlock handler, so the report was unreachable unless you spent the test
        set to see it. Pressing this again after an unlock rewrites the same file with the
        test numbers and the count.
        """
        problems = export_blockers(self.state)
        if problems:
            raise RuntimeError(" ".join(problems))
        _bundle_name, archive_name = exports.export_names(self.state)
        export = exports.export_performance(
            work_dir=work_dir(self.state),
            run_result=self.run,
            baseline=self.baseline,
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
