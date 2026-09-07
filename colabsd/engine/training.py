"""Shared training primitives for sequence-level fine-tuning.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(`seqdisplay_opt/finetuning/training.py`). ``train_epoch`` is the loop that
produced the tuned configurations in ``config/best/``, including its exact
sample-weighted gradient accumulation, so it is copied verbatim; only the
``SequenceAdapter`` import is repointed at the vendored protocol. See
``ATTRIBUTION.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from colabsd.engine.adapters import SequenceAdapter


def batch_indices(
    indices: list[int],
    batch_size: int,
    seed: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Build a reproducibly ordered loader over dataset indices."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(indices, batch_size=batch_size, shuffle=shuffle, generator=generator)


def as_float_tensor(
    values: np.ndarray,
    indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Select rows from a NumPy array and place them on a device."""
    return torch.tensor(values[indices], dtype=torch.float32, device=device)


def train_epoch(
    *,
    adapter: SequenceAdapter,
    model: nn.Module,
    head: nn.Module,
    optimizer: Optimizer,
    sequences: Sequence[str],
    scaled_targets: np.ndarray,
    train_indices: list[int],
    micro_batch_size: int,
    accumulation_steps: int,
    seed: int,
    device: torch.device,
    max_grad_norm: float,
    shuffle: bool = True,
) -> float:
    """Train one epoch using exact sample-weighted gradient accumulation."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1")

    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    criterion = nn.MSELoss(reduction="sum")
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    batches = batch_indices(
        train_indices,
        micro_batch_size,
        seed,
        shuffle=shuffle,
    )
    n_batches = len(batches)
    total_loss = 0.0
    total_numel = 0
    window_numel = 0

    for step, batch in enumerate(batches, start=1):
        batch_idx = [int(index) for index in batch]
        pooled = adapter.pooled_forward(
            model,
            [sequences[index] for index in batch_idx],
            device,
        ).float()
        targets = as_float_tensor(scaled_targets, batch_idx, device)
        loss_sum = criterion(head(pooled), targets)
        loss_sum.backward()

        batch_numel = targets.numel()
        total_loss += float(loss_sum.detach().cpu())
        total_numel += batch_numel
        window_numel += batch_numel

        if step % accumulation_steps == 0 or step == n_batches:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_numel)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_numel = 0

    if total_numel == 0:
        raise ValueError("train_indices must contain at least one sample")
    return total_loss / total_numel
