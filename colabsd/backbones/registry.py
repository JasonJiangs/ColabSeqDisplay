"""Backbone metadata registry.

Importing this module stays free: it holds metadata only and never imports
`torch`, `transformers`, `colabsd.engine` or an adapter module. `create_adapter`
imports the adapter module on demand.

Two different questions are answered here, and they are deliberately not the same
question. *Registered* (`BACKBONES`, `available`, `create_adapter`) is what the
package can build: fourteen entries, thirteen of them with a working adapter.
*Offered* (`OFFERED_FAMILIES`, `offered`, `is_offered`) is the much shorter list the
two notebooks put on screen. Nothing is deleted to shorten a dropdown — a family
the notebooks do not list is still one `create_adapter` call away, and
`withheld_reason` says in words why it is not on screen, so a user who goes looking
for their model finds an answer rather than silence.

`embed_dim` is the width of the *pooled feature* the adapter returns, which is
what a downstream head is built with. It is not always the encoder hidden size:
ESMDance pools its 50-dim `res_pred` output, not its 480-dim trunk. The values
here are cross-checked against upstream's `config/models.yaml` in the tests.

`approx_lora_minutes_t4` is an order-of-magnitude wall-clock estimate for one
split seed x one model seed of LoRA fine-tuning on the bundled SlugCas9 example
(16.4k variants, 1054 residues) on a Colab T4. `None` means the backbone does
not fit a T4 and needs an L4/A100 or a local GPU.
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
        notes="Fastest sanity-check backbone; use it to validate a library end to end before paying for a big run.",
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
        notes="Structure-aware; needs a wild-type 3Di string (colabsd.structure) as well as the sequence.",
    ),
    "SaProt-650M": BackboneEntry(
        hf_id="westlake-repl/SaProt_650M_AF2",
        family="SaProt",
        embed_dim=1280,
        tier="native",
        needs_structure=True,
        approx_lora_minutes_t4=260,
        notes="Structure-aware sibling of ESM2-650M; needs a wild-type 3Di string.",
    ),
    "SaProt-1.3B": BackboneEntry(
        hf_id="westlake-repl/SaProt_1.3B_AF2",
        family="SaProt",
        embed_dim=1280,
        tier="native",
        needs_structure=True,
        approx_lora_minutes_t4=None,
        notes="66 layers: loads through plain transformers but needs dtype='float16' and an L4/A100; a T4 OOMs.",
    ),
    "ProtT5-XL": BackboneEntry(
        hf_id="Rostlab/prot_t5_xl_uniref50",
        family="ProtT5",
        embed_dim=1024,
        tier="extra",
        needs_structure=False,
        approx_lora_minutes_t4=None,
        notes=(
            "Encoder only (~1.2B parameters); needs an L4/A100. The weights load through plain transformers, "
            "but the slow sentencepiece tokenizer is mandatory for this repo: pip install sentencepiece."
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
            "Encoder only (~1.2B parameters); needs an L4/A100. Unlike ProtT5 it ships a fast tokenizer.json, "
            "so it loads with plain transformers and needs no sentencepiece install."
        ),
    ),
    "ESMC-300M": BackboneEntry(
        hf_id="EvolutionaryScale/esmc-300m-2024-12",
        family="ESMC",
        embed_dim=960,
        tier="extra",
        needs_structure=False,
        approx_lora_minutes_t4=120,
        notes="Needs the EvolutionaryScale SDK: pip install esm.",
    ),
    "ESMC-600M": BackboneEntry(
        hf_id="EvolutionaryScale/esmc-600m-2024-12",
        family="ESMC",
        embed_dim=1152,
        tier="extra",
        needs_structure=False,
        approx_lora_minutes_t4=220,
        notes="Needs the EvolutionaryScale SDK: pip install esm.",
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
            "Dynamics-tuned ESM2-35M; same input shape as SeqDance, but the pooled feature is the 50-dim "
            "res_pred head output, not the 480-dim trunk, so heads are built with embed_dim=50."
        ),
    ),
    "METL": BackboneEntry(
        hf_id="",
        family="METL",
        embed_dim=512,
        tier="local_only",
        needs_structure=True,
        approx_lora_minutes_t4=None,
        notes="Rosetta-pretrained and protein-specific: no HuggingFace weights, so it is out of scope for Colab.",
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


# --- what the notebooks offer ------------------------------------------------

#: The families the notebooks list. The panel asks two questions — which backbone,
#: which pooling — and every later step is derived from the answers, so the first
#: question stays a real comparison (one sequence-only family, one structure-aware
#: one) rather than a menu of fourteen. This says nothing about what the package can
#: run: `create_adapter` builds every entry that has an adapter, offered or not.
#: Putting a family back on screen is one line — move its name out of
#: `WITHHELD_FAMILY_REASONS` and into this tuple.
OFFERED_FAMILIES: tuple[str, ...] = ("ESM2", "SaProt")

#: Why each remaining family is in the package but not on screen, phrased for the user
#: who goes looking for their model. Every family in `BACKBONES` sits in exactly one of
#: these two places, and `tests/test_backbone_surface.py` keeps it that way, so a family
#: can never be dropped from the notebooks without leaving a reason behind.
WITHHELD_FAMILY_REASONS: dict[str, str] = {
    "ProtT5": (
        "a 1.2B-parameter encoder that needs an L4 or A100 and an extra sentencepiece install, "
        "where the notebooks target a free T4 and install nothing beyond colabsd"
    ),
    "Ankh": "a 1.2B-parameter encoder that needs an L4 or A100, where the notebooks target a free T4",
    "ESMC": (
        "it loads only through the EvolutionaryScale SDK (`pip install esm`), an install the "
        "notebooks do not make on a user's behalf"
    ),
    "SeqDance": (
        "a dynamics-pretrained ESM2-35M — a good model, but it answers a narrower question than "
        "the sequence-versus-structure choice the notebooks are built around"
    ),
    "ESMDance": (
        "a dynamics-tuned ESM2-35M whose pooled feature is its 50-dim prediction head rather than "
        "the trunk, so it is not read like the other entries in a single comparison"
    ),
    "METL": "Rosetta-pretrained and protein-specific, with no HuggingFace weights for anything to load",
}

#: Families whose adapter cannot tokenize a variant without a wild-type 3Di string.
#: Narrower than `BackboneEntry.needs_structure`: METL needs a structure too, but a
#: Rosetta-relaxed one that no notebook step produces, and it has no adapter at all.
THREE_DI_FAMILIES: frozenset[str] = frozenset({"SaProt"})

#: What `hyperparameter_state` can answer. "placeholder" is `config/best/`'s own
#: `_meta.status: provisional` — a median of the tuned entries, standing in until a
#: study is run for that pair.
HYPERPARAMETER_STATES: tuple[str, ...] = ("tuned", "placeholder", "missing")


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


# --- the notebook surface ----------------------------------------------------


def is_offered(name: str) -> bool:
    """True when the notebooks list *name* among the backbones a user can pick."""
    return get_entry(name).family in OFFERED_FAMILIES


def offered() -> list[str]:
    """The backbones the notebooks offer, in registry order: the ESM2 ladder, then SaProt.

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
    """Why *name* is not offered in the notebooks, or None when it is offered."""
    entry = get_entry(name)
    if entry.family in OFFERED_FAMILIES:
        return None
    try:
        return WITHHELD_FAMILY_REASONS[entry.family]
    except KeyError:
        raise BackboneError(
            f"Family '{entry.family}' is in neither OFFERED_FAMILIES nor WITHHELD_FAMILY_REASONS, "
            f"so there is nothing to tell a user who goes looking for '{name}'. "
            "Add the family to one of them in colabsd/backbones/registry.py."
        ) from None


def withheld_note(name: str) -> str:
    """One paragraph for a reader who went looking for a backbone the panel does not list."""
    reason = withheld_reason(name)
    if reason is None:
        return f"`{name}` is one of the backbones the notebooks offer."
    lines = [f"`{name}` is in this package but the notebooks do not offer it: {reason}."]
    if has_adapter(name):
        lines.append(
            f"Its adapter is still here and still tested — `create_adapter({name!r}, pooling=...)` builds it "
            "from Python, and `config/best/` still carries its hyperparameters."
        )
    else:
        lines.append("It has no adapter in this package, so there is nothing here to run it with.")
    return " ".join(lines)


# --- the two questions the preparation step is derived from ------------------


def needs_wt_3di(name: str) -> bool:
    """True when *name* cannot be tokenized without a wild-type 3Di string.

    This is the question preparation is derived from rather than asked about: a SaProt
    backbone makes the 3Di step appear, an ESM2 backbone removes it from the panel.
    """
    return get_entry(name).family in THREE_DI_FAMILIES


def hyperparameter_state(name: str, pooling: str, root: str | Path | None = None) -> str:
    """Whether `config/best/` holds `"tuned"` values for this pair, a `"placeholder"`, or `"missing"`.

    `colabsd.bestconfig` owns the answer; this is the join, so no panel has to re-derive
    it from a literal list of pairs. A placeholder is a real, runnable configuration —
    the median of the tuned entries — but no study selected it, so nothing it produces
    is a result.
    """
    from colabsd import bestconfig

    get_entry(name)  # an unknown backbone is a backbone error, not a missing-file report
    try:
        best = bestconfig.load_best_config(name, pooling, root)
    except bestconfig.BestConfigNotFound:
        return "missing"
    return "placeholder" if best.is_provisional else "tuned"


def is_tuned(name: str, pooling: str, root: str | Path | None = None) -> bool:
    """True only when `config/best/` carries tuned hyperparameters for this pair."""
    return hyperparameter_state(name, pooling, root) == "tuned"


def tuned_pairs(root: str | Path | None = None, *, offered_only: bool = True) -> list[tuple[str, str]]:
    """The (backbone, pooling) pairs with tuned hyperparameters, sorted.

    Restricted to offered backbones by default, which is the list a notebook should
    quote; pass `offered_only=False` for every tuned pair the registry holds.
    """
    from colabsd import bestconfig

    pairs = [
        (model, pooling)
        for model, pooling in bestconfig.available_pairs(root)
        if model in BACKBONES and (not offered_only or is_offered(model)) and is_tuned(model, pooling, root)
    ]
    return sorted(pairs)


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
