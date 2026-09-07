"""Pooling strategies that turn per-residue embeddings into one vector per sequence.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from; original modules
`seqdisplay_opt/pooling/{base,factory,reducers,regions,named_mean}.py`, flattened
into this one file. See `ATTRIBUTION.md`.

`torch` is imported only for type checking: nothing here touches the module at
runtime, so importing this module costs nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from colabsd.engine.protein_db import DEFAULT_PROTEIN_ID, resolve_region_positions_0based

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


class PoolingStrategy(ABC):
    """Convert per-residue embeddings ``(N, L, D)`` into sequence features ``(N, D)``."""

    name: str

    @abstractmethod
    def __call__(self, token_embs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Pool token embeddings into one vector per sequence."""


POOLING_REGISTRY: dict[str, type[PoolingStrategy]] = {}


def register_pooling(name: str):
    """Register a ``PoolingStrategy`` class under *name*."""

    def decorator(cls: type[PoolingStrategy]) -> type[PoolingStrategy]:
        POOLING_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def create_pooling(name: str, **kwargs) -> PoolingStrategy:
    """Instantiate a registered pooling strategy."""
    if name not in POOLING_REGISTRY:
        known = sorted(POOLING_REGISTRY)
        raise ValueError(f"Unknown pooling '{name}'. Available: {known}")
    return POOLING_REGISTRY[name](**kwargs)


def pool(name: str, token_embs: torch.Tensor, **kwargs) -> torch.Tensor:
    """Apply a registered pooling strategy by name."""
    return create_pooling(name)(token_embs, **kwargs)


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


def select_named_region(
    token_embs: torch.Tensor,
    region_name: str,
    *,
    positions_0based: Sequence[int] | None = None,
    protein_id: str | None = None,
    protein_database: str | Path | None = None,
) -> torch.Tensor:
    """Select an explicit region or resolve its positions from protein metadata."""
    positions = (
        list(positions_0based)
        if positions_0based is not None
        else resolve_region_positions_0based(
            region_name,
            protein_id=protein_id or DEFAULT_PROTEIN_ID,
            database_path=protein_database,
            sequence_length=token_embs.shape[1],
        )
    )
    return select_positions(token_embs, positions)


class NamedRegionMeanPooling(PoolingStrategy):
    """Mean-pool a region resolved from protein metadata or explicit positions."""

    region_name: str

    def __call__(self, token_embs: torch.Tensor, **kwargs) -> torch.Tensor:
        selected = select_named_region(
            token_embs,
            self.region_name,
            positions_0based=kwargs.get("positions_0based"),
            protein_id=kwargs.get("protein_id"),
            protein_database=kwargs.get("protein_database"),
        )
        return mean_reduce(selected)


@register_pooling("cosine_p90_mean")
class CosineP90MeanPooling(NamedRegionMeanPooling):
    region_name = "cosine_p90"


@register_pooling("cosine_p95_mean")
class CosineP95MeanPooling(NamedRegionMeanPooling):
    region_name = "cosine_p95"


@register_pooling("full_mean")
class FullMeanPooling(NamedRegionMeanPooling):
    region_name = "full"


@register_pooling("last150_mean")
class Last150MeanPooling(NamedRegionMeanPooling):
    region_name = "last_150"


@register_pooling("mutation_site_mean")
class MutationSiteMeanPooling(NamedRegionMeanPooling):
    region_name = "mutation_sites"
