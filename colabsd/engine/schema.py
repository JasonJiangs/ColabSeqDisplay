"""Training hyperparameters shared by the heads and the training loop.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from; original module
`seqdisplay_opt/config/schema.py`. Only `TrainingConfig` is kept: the cached-
embedding selection objects around it (`ModelSpec`, `SelectionConfig` and its
YAML loader) belong to upstream's offline selection CLI, which the notebooks
never reach. See `ATTRIBUTION.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Hyperparameters for the training and selection loop."""

    split_seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    model_seeds: list[int] = field(default_factory=lambda: [11, 22, 33])
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    inner_val_fraction: float = 0.1
