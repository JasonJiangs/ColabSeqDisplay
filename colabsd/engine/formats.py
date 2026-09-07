"""Per-family input formatting: how each backbone wants its sequences written.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(Python package ``seqdisplay_opt``), which owns this science and whose published
numbers were produced by it. Original modules, one per family:

* ``seqdisplay_opt/models/backbones/saprot.py`` — ``FOLDSEEK_SEQ_VOCAB``,
  ``FOLDSEEK_STRUC_VOCAB`` and ``saprot_interleaved_sequence``.
* ``seqdisplay_opt/models/backbones/prott5.py`` — ``format_prott5_sequence``.
* ``seqdisplay_opt/models/backbones/ankh.py`` — ``_format_sequence``, public here
  as ``format_ankh_sequence``.
* ``seqdisplay_opt/models/backbones/seqdance.py`` — ``SEQDANCE_CONFIG`` and
  ``_make_seqdance_class``, public here as ``make_seqdance_class``.

See ``ATTRIBUTION.md``.

Left behind: the four ``BackboneLoader`` classes those modules exist to serve.
They load ``fair-esm`` checkpoints and local snapshots out of an HPC model
catalogue, which is exactly the part ``colabsd`` replaces -- its adapters in
``colabsd/backbones/`` load the same weights through ``transformers``. What is
kept is the formatting the weights were trained with, which no adapter may
paraphrase: a SaProt input that is not AA/3Di-interleaved, or a ProtT5 input
without its spaces, tokenizes to something the checkpoint has never seen.

Changed while copying: the two private names became public, because they are now
this package's own API rather than someone else's internals. Nothing else is
touched. ``torch``, ``transformers`` and ``huggingface_hub`` are imported inside
``make_seqdance_class`` so that importing this module stays free of them.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------- SaProt

# The 20 canonical amino acids plus the mask state, and the 20 foldseek 3Di
# states plus the mask state. Their cartesian product, in this order, is SaProt's
# 441-token structure-aware vocabulary; `colabsd.structure` validates generated
# 3Di strings against FOLDSEEK_STRUC_VOCAB.
FOLDSEEK_SEQ_VOCAB = "ACDEFGHIKLMNPQRSTVWY#"
FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"


def saprot_interleaved_sequence(aa_sequence: str, foldseek_sequence: str) -> str:
    """Return the AA/3Di interleaved SaProt input string."""
    if len(aa_sequence) != len(foldseek_sequence):
        raise ValueError(
            "SaProt requires equal-length amino-acid and Foldseek strings: "
            f"{len(aa_sequence)} != {len(foldseek_sequence)}"
        )
    return "".join(aa + state for aa, state in zip(aa_sequence, foldseek_sequence.lower(), strict=False))


# ---------------------------------------------------------------------------- ProtT5


def format_prott5_sequence(sequence: str) -> str:
    """Format an amino-acid sequence for ProtT5 tokenization."""
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return " ".join(sequence)


# ------------------------------------------------------------------------------ Ankh


def format_ankh_sequence(sequence: str) -> str:
    """Format an amino-acid sequence for Ankh tokenization.

    Upstream's ``seqdisplay_opt.models.backbones.ankh._format_sequence``: the rare
    residues ``U``, ``Z``, ``O`` and ``B`` become ``X``, and unlike ProtT5 the
    residues are not separated by spaces.
    """
    return re.sub(r"[UZOB]", "X", sequence)


# -------------------------------------------------------------- SeqDance / ESMDance

SEQDANCE_CONFIG: dict[str, dict[str, Any]] = {
    "training": {"dropout": 0.1},
    "seqdance": {"freeze_esm": False, "randomize_esm": True},
    "esmdance": {"freeze_esm": True, "randomize_esm": False},
    "model_35M": {
        "model_id": "facebook/esm2_t12_35M_UR50D",
        "atten_dim": 240,
        "embed_dim": 480,
        "pair_out_dim": 13,
        "res_out_dim": 50,
    },
}


def make_seqdance_class() -> type:
    """Build the ``ESMwrap`` class the SeqDance and ESMDance checkpoints were saved from.

    The class is defined inside a function, as upstream does, because it needs
    ``transformers`` and ``huggingface_hub`` at definition time. Its parameter
    names are what the published weights are keyed on, so the module structure is
    copied exactly: renaming a submodule here silently drops weights on load.
    """
    import math

    import torch
    import torch.nn as nn
    from huggingface_hub import PyTorchModelHubMixin  # type: ignore[import]
    from transformers import EsmModel  # type: ignore[import]

    class ESMwrap(nn.Module, PyTorchModelHubMixin):
        def __init__(self, esm2_select: str, model_select: str):
            super().__init__()
            cfg = SEQDANCE_CONFIG
            self.esm2 = EsmModel.from_pretrained(cfg[esm2_select]["model_id"])
            self.freeze_esm = cfg[model_select]["freeze_esm"]
            if self.freeze_esm:
                for param in self.esm2.parameters():
                    param.requires_grad = False
                self.esm2.eval()
            if cfg[model_select]["randomize_esm"]:
                self.randomize_model(self.esm2)

            embed_dim = cfg[esm2_select]["embed_dim"]
            res_out_dim = cfg[esm2_select]["res_out_dim"]
            atten_dim = cfg[esm2_select]["atten_dim"]
            pair_out_dim = cfg[esm2_select]["pair_out_dim"]
            dropout = cfg["training"]["dropout"]
            self.res_pred_nn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, res_out_dim),
            )
            self.res_transform_nn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim * 2),
            )
            self.pair_middle_linear = nn.Linear(embed_dim * 2, atten_dim)
            self.pair_pred_linear = nn.Linear(atten_dim + atten_dim, pair_out_dim)
            self._init_bias_zero()

        def randomize_model(self, model: nn.Module) -> None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear | nn.Embedding):
                    if getattr(module, "bias", None) is not None:
                        module.bias.data.zero_()
                    if hasattr(module, "weight"):
                        gain = 1 / math.sqrt(2) if any(x in name for x in ["query", "key", "value"]) else 1
                        nn.init.xavier_uniform_(module.weight, gain=gain)
                elif isinstance(module, nn.LayerNorm):
                    if hasattr(module, "bias"):
                        module.bias.data.zero_()
                    if hasattr(module, "weight"):
                        module.weight.data.fill_(1.0)

        def _init_bias_zero(self) -> None:
            for name, module in self.named_modules():
                if "esm2" not in name and isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, inputs, feature: str = "res_emb"):
            esm_output = self.esm2(**inputs, output_attentions=False)
            res_emb = esm_output["last_hidden_state"]
            if feature == "res_emb":
                return res_emb
            if feature == "res_pred":
                return self.res_pred_nn(res_emb)
            if feature == "res_emb_plus_pred":
                return torch.cat([res_emb, self.res_pred_nn(res_emb)], dim=-1)
            raise ValueError(f"Unknown SeqDance feature '{feature}'")

    return ESMwrap
