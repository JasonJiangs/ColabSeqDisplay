"""Mean-pool the residue embeddings at the library's mutated sites.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from: `select_positions` comes from
`seqdisplay_opt/pooling/regions.py` and `mean_reduce` from
`seqdisplay_opt/pooling/reducers.py`, both verbatim. See `ATTRIBUTION.md`.

Upstream dispatched among five poolings through a registry of `PoolingStrategy`
classes (`seqdisplay_opt/pooling/{base,factory,named_mean}.py`), each naming a
protein region to average. This pipeline pools one way -- the mean over the sites
the library mutates -- so `pool` composes the two reducers directly. There is no
registry, no factory and no strategy name to dispatch on.

`torch` is imported only for type checking: nothing here touches the module at
runtime, so importing this module costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

#: The name every artefact records for this pooling: the `config/best/` filename
#: suffix, `training.pooling` in a run config, and `pooling` in a bundle manifest.
#: It is a label, not a choice -- there is one strategy.
POOLING_NAME = "mutation_site_mean"


def mean_reduce(token_embs: torch.Tensor) -> torch.Tensor:
    """Mean-reduce token embeddings ``(N, L, D)`` to ``(N, D)``."""
    return token_embs.mean(dim=1)


def select_positions(token_embs: torch.Tensor, positions_0based: Sequence[int]) -> torch.Tensor:
    """Select residue positions from full-sequence token embeddings."""
    positions = list(positions_0based)
    if not positions:
        raise ValueError("positions_0based is empty")
    min_pos = min(positions)
    max_pos = max(positions)
    if min_pos < 0 or max_pos >= token_embs.shape[1]:
        raise ValueError(f"Positions [{min_pos}, {max_pos}] are out of bounds for token length {token_embs.shape[1]}")
    return token_embs[:, positions, :]


def pool(token_embs: torch.Tensor, positions_0based: Sequence[int]) -> torch.Tensor:
    """Mean the embeddings at the mutated sites: ``(N, L, D)`` -> ``(N, D)``."""
    return mean_reduce(select_positions(token_embs, positions_0based))
