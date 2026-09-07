"""The structural contract a fine-tunable backbone has to satisfy.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(`seqdisplay_opt/finetuning/adapters.py`), of which this is the ``Protocol``
only. Upstream's concrete adapters load `fair-esm` checkpoints from a local
model catalogue, which is exactly what `colabsd` replaces: its own adapters live
in `colabsd/backbones/` and load through `transformers`. See ``ATTRIBUTION.md``.

The protocol is structural and deliberately not ``runtime_checkable``, matching
upstream: an adapter satisfies it by having the attributes and methods, never by
inheriting from it.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class SequenceAdapter(Protocol):
    """A backbone that turns raw sequences into one pooled vector per sequence."""

    model_name: str
    embed_dim: int

    def load_model(self) -> nn.Module: ...

    def pooled_forward(
        self,
        model: nn.Module,
        sequences: list[str],
        device: torch.device,
    ) -> torch.Tensor: ...
