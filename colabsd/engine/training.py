"""Shared training primitives for sequence-level fine-tuning.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(`seqdisplay_opt/finetuning/training.py`). ``train_epoch`` is the loop that
produced the tuned configurations in ``config/best/``, so its exact
sample-weighted gradient accumulation is copied unchanged. Two things differ
from upstream, both recorded in ``ATTRIBUTION.md``: the ``SequenceAdapter``
import is repointed at the vendored protocol, and the loop takes an optional
``on_batch`` callback.

The callback exists because an epoch here runs to thousands of micro-batches and
a notebook user watching one has nothing to look at until it ends. It is called
after a micro-batch's arithmetic is finished, its return value is ignored, and
nothing in the loop reads anything it might touch, so a caller that passes one
trains exactly what a caller that does not trains. It fires for every batch;
deciding how often that reaches a widget is the caller's job.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from colabsd.engine.adapters import SequenceAdapter

#: ``on_batch(step, n_batches, running_loss)`` -- the 1-based micro-batch just finished,
#: how many the epoch holds, and the mean squared error over every target seen so far.
BatchCallback = Callable[[int, int, float], None]


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
    on_batch: BatchCallback | None = None,
) -> float:
    """Train one epoch using exact sample-weighted gradient accumulation.

    Returns the epoch's mean squared error over every target. When *on_batch* is given it
    is called once per finished micro-batch with that same running mean, so the caller can
    report progress from inside the epoch; see :data:`BatchCallback`.
    """
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

        if on_batch is not None:
            on_batch(step, n_batches, total_loss / total_numel)

    if total_numel == 0:
        raise ValueError("train_indices must contain at least one sample")
    return total_loss / total_numel
