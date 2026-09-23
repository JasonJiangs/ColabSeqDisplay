"""Backbone metadata registry.

Importing this module stays free: it holds metadata only and never imports
`torch`, `transformers`, `colabsd.engine` or an adapter module. `create_adapter`
imports the adapter module on demand.

Two different questions are answered here, and they are deliberately not the same
question. *Registered* (`BACKBONES`, `available`, `create_adapter`) is what the
package can build: fourteen entries, thirteen of them with a working adapter.
*Offered* (`OFFERED_FAMILIES`, `WITHHELD_MODEL_REASONS`, `offered`, `is_offered`) is
the much shorter list the two notebooks put on screen. Nothing is deleted to shorten a
dropdown — a backbone the notebooks do not list is still one `create_adapter` call away,
and `withheld_reason` says in words why it is not on screen, so a user who goes looking
for their model finds an answer rather than silence.

`embed_dim` is the width of the *pooled feature* the adapter returns, which is
what a downstream head is built with. It is not always the encoder hidden size:
ESMDance pools its 50-dim `res_pred` output, not its 480-dim trunk. The values
here are cross-checked against upstream's `config/models.yaml` in the tests.

`approx_lora_minutes_t4` is an order-of-magnitude wall-clock estimate for one
split seed x one model seed of LoRA fine-tuning on a Colab T4, sized against the
library the tuning study used (16.4k variants of 1054 residues). `None` means the
backbone does not fit a T4 and needs an L4/A100 or a local GPU. The bundled MG8
example is ~486x smaller in tokens, so a run on it costs a small fraction of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from colabsd.errors import BackboneError

if TYPE_CHECKING:
    from colabsd.engine.adapters import SequenceAdapter

TIERS = ("native", "extra", "local_only")


@dataclass(frozen=True)
class BackboneEntry:
    """Everything the notebook needs to describe a backbone before loading it."""

    hf_id: str
    family: str
    embed_dim: int
    tier: str
    needs_structure: bool
    approx_lora_minutes_t4: int | None
    notes: str


BACKBONES: dict[str, BackboneEntry] = {
    "ESM2-8M": BackboneEntry(
        hf_id="facebook/esm2_t6_8M_UR50D",
        family="ESM2",
        embed_dim=320,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=20,
        notes="Smallest and fastest ESM2; the quickest way to put a library through the whole chain.",
    ),
    "ESM2-35M": BackboneEntry(
        hf_id="facebook/esm2_t12_35M_UR50D",
        family="ESM2",
        embed_dim=480,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=40,
        notes="Good accuracy-per-minute trade-off on a T4.",
    ),
    "ESM2-150M": BackboneEntry(
        hf_id="facebook/esm2_t30_150M_UR50D",
        family="ESM2",
        embed_dim=640,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=90,
        notes="Middle of the ESM2 range.",
    ),
    "ESM2-650M": BackboneEntry(
        hf_id="facebook/esm2_t33_650M_UR50D",
        family="ESM2",
        embed_dim=1280,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=240,
        notes="Best tuned sequence-only result upstream; the largest ESM2 that trains on a T4.",
    ),
    "SaProt-35M": BackboneEntry(
        hf_id="westlake-repl/SaProt_35M_AF2",
        family="SaProt",
        embed_dim=480,
        tier="native",
        needs_structure=True,
        approx_lora_minutes_t4=45,
        notes="The smallest structure-aware entry; `colabsd.structure` is what folds the 3Di string.",
    ),
    "SaProt-650M": BackboneEntry(
        hf_id="westlake-repl/SaProt_650M_AF2",
        family="SaProt",
        embed_dim=1280,
        tier="native",
        needs_structure=True,
        approx_lora_minutes_t4=260,
        notes="Structure-aware sibling of ESM2-650M, at the same width.",
    ),
    "SaProt-1.3B": BackboneEntry(
        hf_id="westlake-repl/SaProt_1.3B_AF2",
        family="SaProt",
        embed_dim=1280,
        tier="native",
        needs_structure=True,
        approx_lora_minutes_t4=None,
        notes="66 layers, the deepest entry in the registry; it loads through plain `transformers`.",
    ),
    "ProtT5-XL": BackboneEntry(
        hf_id="Rostlab/prot_t5_xl_uniref50",
        family="ProtT5",
        embed_dim=1024,
        tier="extra",
        needs_structure=False,
        approx_lora_minutes_t4=None,
        notes=(
            "Encoder only (~1.2B parameters), and an 11 GB checkpoint to download. The weights load "
            "through plain `transformers`, but its `sentencepiece` vocabulary does not: `sentencepiece` "
            "reads that vocabulary, and `protobuf` parses it once `transformers` falls back to the slow "
            "T5 tokenizer."
        ),
    ),
    "Ankh-large": BackboneEntry(
        hf_id="ElnaggarLab/ankh-large",
        family="Ankh",
        embed_dim=1536,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=None,
        notes=(
            "Encoder only (~1.2B parameters). Unlike ProtT5 it ships a fast `tokenizer.json`, so plain "
            "`transformers` builds its tokenizer without `sentencepiece` or `protobuf` — and if it ever "
            "does fall back to the slow T5 tokenizer, its own loader error names whichever of the two "
            "is missing rather than guessing at one of them."
        ),
    ),
    "ESMC-300M": BackboneEntry(
        hf_id="EvolutionaryScale/esmc-300m-2024-12",
        family="ESMC",
        embed_dim=960,
        tier="extra",
        needs_structure=False,
        approx_lora_minutes_t4=120,
        notes=(
            "The only offered backbone that loads through the EvolutionaryScale SDK, `esm`, rather "
            "than `transformers` — a separate download from the weights — and the only one whose "
            "960-d trunk sits between the ESM2 sizes. Installing it moves `transformers` too: the "
            "SDK releases that keep LoRA reachable ask for `transformers<4.48.2`, so pip pulls "
            "that in and the runtime has to be restarted before anything loads. Everything here "
            "runs on that release; nothing else on the form needs it."
        ),
    ),
    "ESMC-600M": BackboneEntry(
        hf_id="EvolutionaryScale/esmc-600m-2024-12",
        family="ESMC",
        embed_dim=1152,
        tier="extra",
        needs_structure=False,
        approx_lora_minutes_t4=220,
        notes="The larger of the two ESMC encoders; same `esm` SDK and same input shape as `ESMC-300M`.",
    ),
    "SeqDance": BackboneEntry(
        hf_id="ChaoHou/SeqDance",
        family="SeqDance",
        embed_dim=480,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=45,
        notes="ESM2-35M-shaped encoder trained on conformational dynamics.",
    ),
    "ESMDance": BackboneEntry(
        hf_id="ChaoHou/ESMDance",
        family="ESMDance",
        embed_dim=50,
        tier="native",
        needs_structure=False,
        approx_lora_minutes_t4=45,
        notes=(
            "Dynamics-tuned ESM2-35M; same input shape as SeqDance. Its 50-d feature is its own "
            "`res_pred` prediction of per-residue dynamics, not a trunk embedding — every other backbone "
            "on the form hands the head one of those, 480 to 1280 wide — a different read-out of your "
            "protein, not a smaller one."
        ),
    ),
    "METL": BackboneEntry(
        hf_id="",
        family="METL",
        embed_dim=512,
        tier="local_only",
        needs_structure=True,
        approx_lora_minutes_t4=None,
        notes="Rosetta-pretrained and protein-specific: no HuggingFace weights for anything here to load.",
    ),
}

_FAMILY_ADAPTERS: dict[str, tuple[str, str]] = {
    "ESM2": ("colabsd.backbones.esm2_hf", "ESM2Adapter"),
    "SaProt": ("colabsd.backbones.saprot_hf", "SaProtAdapter"),
    "ProtT5": ("colabsd.backbones.prott5_hf", "ProtT5Adapter"),
    "Ankh": ("colabsd.backbones.ankh_hf", "AnkhAdapter"),
    "ESMC": ("colabsd.backbones.esmc_hf", "ESMCAdapter"),
    "SeqDance": ("colabsd.backbones.dance_hf", "DanceAdapter"),
    "ESMDance": ("colabsd.backbones.dance_hf", "DanceAdapter"),
}

#: The packages an `extra`-tier family needs that neither `transformers` nor Colab brings,
#: keyed by family beside `_FAMILY_ADAPTERS` because it is the same kind of fact: what has to be
#: true before the adapter can be built. `notes` says it in a sentence, which is the right shape
#: for a message and the wrong shape for a table cell; `tests/test_backbone_surface.py` holds the
#: two to naming the same packages. ProtT5 needs two: `sentencepiece` reads the vocabulary and,
#: under transformers 5, the slow T5 tokenizer parses `spiece.model` through `protobuf`. A tuple
#: per family rather than a string, because the adapter's own precheck had learned about the
#: second package and nothing a user reads before choosing had.
_FAMILY_EXTRA_INSTALLS: dict[str, tuple[str, ...]] = {
    "ProtT5": ("sentencepiece", "protobuf"),
    "ESMC": ("esm",),
}


# --- what the notebooks offer ------------------------------------------------

#: The families the notebooks list. The rule is per backbone and this tuple is per family,
#: so a family is listed only when every entry in it clears the same bar — tuned
#: hyperparameters in `config/best/*_mutation_site_mean.yaml` **and** a working adapter —
#: except for a member struck out of it by name in `WITHHELD_MODEL_REASONS` below. The rule
#: itself has no exceptions: nothing reaches the form on placeholder hyperparameters.
#: Not fitting a free T4 is not a reason to withhold anything — the panel says to switch
#: to an A100 — and neither is needing an extra pip install, which `extra_install` names and
#: both the panel and the notebook's backbone table print.
#: This says nothing about what the package can run: `create_adapter` builds every entry
#: that has an adapter, offered or not. Putting a family on screen is one line — move its
#: name out of `WITHHELD_FAMILY_REASONS` and into this tuple.
OFFERED_FAMILIES: tuple[str, ...] = ("ESM2", "SaProt", "ProtT5", "ESMDance", "ESMC")

#: The backbones struck out of a family that is otherwise on the form, and why — the per-model
#: half of the answer `OFFERED_FAMILIES` gives per family, so one untuned member no longer
#: keeps its whole family off screen. A family reaches the form whole unless a name is here.
#: Only a member of an offered family belongs in this dict: a backbone whose family sits in
#: `WITHHELD_FAMILY_REASONS` is withheld already and would be carrying two reasons for one
#: silence. `withheld_reason` reads this before the family's, so the model's own sentence wins.
WITHHELD_MODEL_REASONS: dict[str, str] = {
    "ESM2-8M": (
        "the one ESM2 the study never searched — no LoRA search was run for it at all, so both of its "
        "`config/best/` entries are placeholders synthesized from the ten tuned ones, and a run here "
        "would not be a studied result. Its three siblings were searched and are on the form. It is "
        "still the smallest and fastest entry in the registry, and the only one this package has "
        "fine-tuned end to end — neither of which makes an untuned run a result"
    ),
    "ESMC-600M": (
        "the larger ESMC, which the study never searched — at 220 minutes a run it was never taken "
        "through `mutation_site_mean`, so `config/best/` holds a placeholder for it and a run here "
        "would not be a studied result. Its smaller sibling `ESMC-300M` was searched, clears the bar "
        "on its own and is on the form; the two read a sequence the same way, so the 300M is the one "
        "to reach for while this one is untuned"
    ),
}

#: Why each family with nothing on screen is in the package, phrased for the user who goes
#: looking for their model. Every family in `BACKBONES` is either in `OFFERED_FAMILIES` or
#: here, never both and never neither, and `tests/test_backbone_surface.py` keeps it that way,
#: so a family can never be dropped from the notebooks without leaving a reason behind. One
#: name struck out of a family that stays is not a family question: it goes in
#: `WITHHELD_MODEL_REASONS` above, and its family stays in `OFFERED_FAMILIES`.
WITHHELD_FAMILY_REASONS: dict[str, str] = {
    "Ankh": (
        "a 1.2B-parameter encoder with no tuned entry — the study never searched `mutation_site_mean` "
        "for it, so `config/best/` holds a placeholder and a run here would not be a studied result. "
        "Its size is not the reason: `ProtT5-XL`, the same shape of encoder and equally an L4/A100 "
        "backbone, was searched and is on the form"
    ),
    "SeqDance": (
        "a dynamics-pretrained ESM2-35M with no tuned entry — the study never searched "
        "`mutation_site_mean` for it, so `config/best/` holds a placeholder and a run here would "
        "not be a studied result. Its dynamics-tuned sibling `ESMDance`, which was searched, is "
        "on the form"
    ),
    "METL": "Rosetta-pretrained and protein-specific, with no HuggingFace weights for anything to load",
}

#: Families whose adapter cannot tokenize a variant without a wild-type 3Di string.
#: Narrower than `BackboneEntry.needs_structure`: METL needs a structure too, but a
#: Rosetta-relaxed one that no notebook step produces, and it has no adapter at all.
THREE_DI_FAMILIES: frozenset[str] = frozenset({"SaProt"})

#: The numeric precision a backbone cannot load in anything but, keyed by name. SaProt-1.3B's 66
#: layers need two bytes a weight, not `float16` in particular: in `float32` they run out of memory
#: on every card Colab offers, after the download, and `bfloat16` is the same two bytes and so the
#: same memory. It reads `bfloat16` rather than `float16` because `float16` cannot train anything
#: at all -- AdamW's epsilon, 1e-8, is smaller than the smallest number `float16` holds and rounds
#: to zero there, so the first optimizer step divides by zero and every weight becomes NaN -- and
#: it is refused at the training boundary everywhere: `colabsd.train.finetune` raises on a
#: `float16` adapter and `colabsd.ui.core.backbone_messages` fires `dtype_not_trainable`. Requiring
#: `float16` here therefore left this backbone with no setting the panel allowed that could produce
#: a model: `float32` and `bfloat16` were refused for the precision, `float16` trained NaN and
#: reported it as a validation Spearman of 0.0000. `bfloat16` loads through the identical code
#: path -- SaProt-35M's trunk parameters come back `torch.bfloat16` and its pooled output finite --
#: and it trains. Nothing here refuses `float16`: loading and scoring in it are unaffected, and
#: `colabsd.backbones.base.DTYPES` still offers all three.
#:
#: Recorded here at all because the requirement was stated in three documents and reachable from
#: none of them: README's table cell, the panel's own summary and this entry's `notes` all named a
#: precision while `colabsd.ui.core` had nothing to ask and fired no message, and the control is
#: behind *Show the advanced settings* with `float32` as its default.
REQUIRED_DTYPE: dict[str, str] = {"SaProt-1.3B": "bfloat16"}


def required_dtype(name: str) -> str | None:
    """The precision *name* cannot load without, or None when any precision works."""
    get_entry(name)  # an unknown backbone is a backbone error, not a silent None
    return REQUIRED_DTYPE.get(name)


#: The backbones whose HuggingFace repo publishes weights only as `pytorch_model.bin`. Since
#: CVE-2025-32434 `transformers` refuses to `torch.load` one below torch 2.6 -- the floor
#: `pyproject.toml` declares -- so on an older runtime these have nothing anything here is allowed
#: to load, cache or no cache, and `tests/test_backbones_real_weights.py` skips both of each one's
#: tests saying exactly that. Recorded once because three places counted it differently: README
#: called it one backbone's caveat, `pyproject.toml` said 'half the offered backbones' and
#: `tests/conftest.py` said 'SaProt'. It is four names, and four of the nine `offered()` holds --
#: it was half when the form was eight, and saying 'half' is what went stale when it became nine.
BIN_ONLY_WEIGHTS: frozenset[str] = frozenset(
    {"SaProt-35M", "SaProt-650M", "SaProt-1.3B", "ProtT5-XL"}
)


def bin_only_weights(*, offered_only: bool = False) -> list[str]:
    """Backbones publishing only a `pytorch_model.bin`, in registry order.

    Restricted to the form's list with `offered_only=True`, which is the list a document quoting
    the fraction has to agree with -- count it rather than writing a fraction down, which is how
    'half the offered backbones' outlived the eight-backbone form it was true of.
    """
    return [
        name for name in BACKBONES
        if name in BIN_ONLY_WEIGHTS and (not offered_only or is_offered(name))
    ]


#: What `hyperparameter_state` can answer. "placeholder" is `config/best/`'s own
#: `_meta.status: provisional` — a median of the tuned entries, standing in until a
#: study is run for that backbone.
HYPERPARAMETER_STATES: tuple[str, ...] = ("tuned", "placeholder", "missing")


def as_sentence(text: str) -> str:
    """*text* finished as a sentence: a period added only when it does not already end one.

    The reasons in this module are written as clauses and end without punctuation -- except
    ESMC's, which closes on a parenthesized sentence of its own. Three renderers appended a
    period unconditionally, so every one of them printed "...would not be a reason to withhold
    them.).". It lives here rather than in `colabsd.ui.core`, where it was written, because
    `withheld_note` below needs it and nothing under `colabsd/backbones/` may import the widget
    layer: `colabsd.ui.core` imports this module at module scope, so the reverse is a cycle.
    Closing parentheses, brackets and quotes are stripped before the test, because the
    character that ends the sentence is behind them.
    """
    stripped = text.rstrip()
    if stripped.rstrip(')"\'”’]').endswith((".", "!", "?")):
        return stripped
    return stripped + "."


def get_entry(name: str) -> BackboneEntry:
    """Return registry metadata for *name*."""
    try:
        return BACKBONES[name]
    except KeyError:
        raise BackboneError(f"Unknown backbone '{name}'. Pick one of: {', '.join(BACKBONES)}.") from None


def available(tier: str | None = None) -> list[str]:
    """Return registered backbone names, optionally restricted to one tier.

    This is what the package can build, not what the notebooks show: for the
    dropdown, ask `offered()`.
    """
    if tier is None:
        return list(BACKBONES)
    if tier not in TIERS:
        raise BackboneError(f"Unknown tier '{tier}'. Pick one of: {', '.join(TIERS)}.")
    return [name for name, entry in BACKBONES.items() if entry.tier == tier]


def has_adapter(name: str) -> bool:
    """True when `create_adapter` can build *name* (a `local_only` entry has no adapter)."""
    return get_entry(name).family in _FAMILY_ADAPTERS


def extra_installs(name: str) -> tuple[str, ...]:
    """Every package *name* needs installing before it loads; empty when it needs none.

    The list, where `extra_install` gives the `pip install` argument line. ProtT5 needs two and
    the error message was the only surface that knew it, so a reader who hit that message had a
    documented remedy — one package, and one extra in `pyproject.toml` — that could not work.
    """
    entry = get_entry(name)
    if entry.tier != "extra":
        return ()
    try:
        return _FAMILY_EXTRA_INSTALLS[entry.family]
    except KeyError:
        raise BackboneError(
            f"'{name}' is tier 'extra' but no package is recorded for family '{entry.family}', so "
            f"nothing here can tell a user what to install before picking it. Add '{entry.family}' "
            "to _FAMILY_EXTRA_INSTALLS in colabsd/backbones/registry.py."
        ) from None


#: A version bound on an extra package, keyed by package name, rendered into the `pip install`
#: line every surface prints. Kept apart from `_FAMILY_EXTRA_INSTALLS` because that answers
#: "which packages", which is what a `notes` sentence has to name, while this answers "which
#: releases of them work" -- and so far one package needs the second answer. Quoted when
#: rendered: an unquoted `esm>=3.1,<3.4` is a shell redirect, not an argument.
_EXTRA_INSTALL_PINS: dict[str, str] = {
    # From `esm` 3.4.0 the SDK builds ESM-C's fused `layernorm_qkv` as a class of its own rather
    # than an `nn.Sequential` holding an `nn.Linear`, so `colabsd.engine.lora.inject_lora` reaches
    # `out_proj` and nothing else and the run is refused -- after the weights have downloaded.
    # Verified on the published checkpoint: 3.2.1 covers all four projections, 3.4.0 and 3.4.1 do
    # not. There is no 3.3. `colabsd.backbones.esmc_hf` refuses such a build by name at load time.
    "esm": ">=3.1,<3.4",
}


def pinned_install(package: str) -> str:
    """*package* as it should appear on a `pip install` line, with its bound when it has one."""
    bound = _EXTRA_INSTALL_PINS.get(package)
    return f"'{package}{bound}'" if bound else package


def extra_install(name: str) -> str | None:
    """The package *name* needs installing before it loads, or None when it needs none.

    `tier == "extra"` says that there is one; this says which, so the notebook's backbone table
    can print the package beside the backbone rather than leaving it to a message that only
    appears once the choice has been made. More than one package is one space-separated argument
    line, because every caller renders this as `pip install {package}` and a list would make four
    places each invent their own punctuation for it.
    """
    packages = extra_installs(name)
    return " ".join(pinned_install(package) for package in packages) if packages else None


# --- the notebook surface ----------------------------------------------------


def is_offered(name: str) -> bool:
    """True when the notebooks list *name* among the backbones a user can pick.

    Both halves of the answer: the family is on the form, and this member of it has not been
    struck out of it by name.
    """
    entry = get_entry(name)
    return entry.family in OFFERED_FAMILIES and name not in WITHHELD_MODEL_REASONS


def offered() -> list[str]:
    """The backbones the notebooks offer, in registry order.

    Every panel, dropdown and generated table reads this rather than keeping a literal
    list of its own, so the registry stays the only place the answer changes.
    """
    return [name for name in BACKBONES if is_offered(name)]


def withheld() -> list[str]:
    """Registered backbones the notebooks do not list, in registry order.

    They are withheld from the *dropdown*, not from the package: each one that has an
    adapter is still built by `create_adapter`.
    """
    return [name for name in BACKBONES if not is_offered(name)]


def withheld_reason(name: str) -> str | None:
    """Why *name* is not offered in the notebooks, or None when it is offered.

    Asked of the model first and of its family second, which is the order the two lists are
    written in: a name struck out of an offered family has a reason of its own, and every
    other withheld backbone inherits its family's.
    """
    entry = get_entry(name)
    if is_offered(name):
        return None
    model_reason = WITHHELD_MODEL_REASONS.get(name)
    if model_reason is not None:
        return model_reason
    try:
        return WITHHELD_FAMILY_REASONS[entry.family]
    except KeyError:
        raise BackboneError(
            f"Family '{entry.family}' is in neither OFFERED_FAMILIES nor WITHHELD_FAMILY_REASONS, "
            f"so there is nothing to tell a user who goes looking for '{name}'. "
            "Add the family to one of them in colabsd/backbones/registry.py."
        ) from None


#: What `withheld_note` adds about `config/best/` per `hyperparameter_state`. A withheld
#: family with no entry gets no clause: the note says nothing rather than promising a file.
_CONFIG_CLAUSE: dict[str, str] = {
    "tuned": " `config/best/` holds tuned hyperparameters for it.",
    "placeholder": " `config/best/` holds a placeholder entry for it, not tuned values.",
    "missing": "",
}


def withheld_note(name: str) -> str:
    """One paragraph for a reader who went looking for a backbone the panel does not list.

    The `config/best/` clause is *asked for* per backbone, never asserted for the class:
    having an adapter says nothing about having an entry, and today every withheld entry
    that has one is a placeholder rather than the tuned values the old blanket sentence
    promised. An unreadable registry costs the clause, not the note.
    """
    reason = withheld_reason(name)
    if reason is None:
        return f"`{name}` is one of the backbones the notebooks offer."
    lines = [f"`{name}` is in this package but the notebooks do not offer it: {as_sentence(reason)}"]
    if has_adapter(name):
        try:
            config = _CONFIG_CLAUSE.get(hyperparameter_state(name), "")
        except Exception:
            config = ""
        lines.append(
            f"Its adapter is still here and still tested — `create_adapter({name!r})` builds it from "
            f"Python.{config}"
        )
    else:
        lines.append("It has no adapter in this package, so there is nothing here to run it with.")
    return " ".join(lines)


# --- the two questions the preparation step is derived from ------------------


def needs_wt_3di(name: str) -> bool:
    """True when *name* cannot be tokenized without a wild-type 3Di string.

    This is the question preparation is derived from rather than asked about: a SaProt
    backbone makes the 3Di step appear, and a backbone that reads the sequence alone removes
    it from the panel.
    """
    return get_entry(name).family in THREE_DI_FAMILIES


def hyperparameter_state(name: str, root: str | Path | None = None) -> str:
    """Whether `config/best/` holds `"tuned"` values for *name*, a `"placeholder"`, or `"missing"`.

    `colabsd.bestconfig` owns the answer; this is the join, so no panel has to re-derive
    it from a literal list of backbones. A placeholder is a real, runnable configuration —
    the median of the tuned entries — but no study selected it, so nothing it produces
    is a result.
    """
    from colabsd import bestconfig

    get_entry(name)  # an unknown backbone is a backbone error, not a missing-file report
    try:
        best = bestconfig.load_best_config(name, root)
    except bestconfig.BestConfigNotFound:
        return "missing"
    return "placeholder" if best.is_provisional else "tuned"


def is_tuned(name: str, root: str | Path | None = None) -> bool:
    """True only when `config/best/` carries tuned hyperparameters for *name*."""
    return hyperparameter_state(name, root) == "tuned"


def tuned_backbones(root: str | Path | None = None, *, offered_only: bool = True) -> list[str]:
    """The backbones with tuned hyperparameters, sorted.

    Restricted to offered backbones by default, which is the list a notebook should
    quote; pass `offered_only=False` for every tuned entry the registry holds.
    """
    from colabsd import bestconfig

    return sorted(
        model
        for model in bestconfig.available_models(root)
        if model in BACKBONES and (not offered_only or is_offered(model)) and is_tuned(model, root)
    )


def _adapter_class(module: ModuleType, class_name: str) -> type:
    declared = getattr(module, class_name, None) or getattr(module, "ADAPTER", None)
    if declared is not None:
        return declared
    candidates = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__ and value.__name__.endswith("Adapter")
    ]
    if len(candidates) != 1:
        raise BackboneError(
            f"{module.__name__} does not expose an adapter class named '{class_name}'. "
            "Name the adapter class after its family or set a module-level ADAPTER alias."
        )
    return candidates[0]


def create_adapter(name: str, **kwargs: Any) -> SequenceAdapter:
    """Instantiate the adapter for *name*, importing its module on demand.

    Every registered backbone with an adapter is built here, whether or not the
    notebooks offer it: what the panel shows is `offered()`, not what this can do.
    """
    entry = get_entry(name)
    target = _FAMILY_ADAPTERS.get(entry.family)
    if target is None:
        raise BackboneError(
            f"'{name}' has no Colab adapter (tier '{entry.tier}'): {entry.notes} "
            f"Runnable backbones: {', '.join(sorted(set(available('native')) | set(available('extra'))))}."
        )
    module_name, class_name = target
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise BackboneError(
            f"The adapter for '{name}' ({module_name}) could not be imported: {exc}. "
            "Check that colabsd is installed completely."
        ) from exc
    return _adapter_class(module, class_name)(name, **kwargs)
