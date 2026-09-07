"""Backbone metadata registry.

Importing this module stays free: it holds metadata only and never imports
`torch`, `transformers`, `colabsd.engine` or an adapter module. `create_adapter`
imports the adapter module on demand.

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


def get_entry(name: str) -> BackboneEntry:
    """Return registry metadata for *name*."""
    try:
        return BACKBONES[name]
    except KeyError:
        raise BackboneError(f"Unknown backbone '{name}'. Pick one of: {', '.join(BACKBONES)}.") from None


def available(tier: str | None = None) -> list[str]:
    """Return registered backbone names, optionally restricted to one tier."""
    if tier is None:
        return list(BACKBONES)
    if tier not in TIERS:
        raise BackboneError(f"Unknown tier '{tier}'. Pick one of: {', '.join(TIERS)}.")
    return [name for name, entry in BACKBONES.items() if entry.tier == tier]


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
    """Instantiate the adapter for *name*, importing its module on demand."""
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
