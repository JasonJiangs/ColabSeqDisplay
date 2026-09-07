"""Reproducible 8:1:1 dataset splits.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(`seqdisplay_opt/data/splits.py`); the private ``_load_torch`` helper is
`seqdisplay_opt/utils/torch_io.py::load_torch`, inlined here so this module has
no upstream dependency. The split functions are copied verbatim, because the
tuned configurations in ``config/best/`` were selected on the exact index lists
these produce. See ``ATTRIBUTION.md``.

``create_split`` caches on the output directory alone: an existing
``indices.pt`` is returned without checking that it was built for the same
``n``. That is upstream behaviour and is kept; ``colabsd.data.make_splits``
guards against a stale cache on our side.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch

_SUPPORTS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters


def _load_torch(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = False,
) -> Any:
    """Load a PyTorch artifact across versions with and without ``weights_only``."""
    kwargs: dict[str, Any] = {"map_location": map_location}
    if _SUPPORTS_WEIGHTS_ONLY:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)


def create_nested_selection_split(
    split: dict,
    *,
    seed: int,
    inner_val_fraction: float = 0.1,
) -> dict[str, list[int]]:
    """Split the outer training partition for leakage-free model selection.

    The returned ``fit_idx`` and ``inner_val_idx`` are derived exclusively from
    ``split['train_idx']``. The outer validation partition becomes
    ``selection_idx`` and the outer test partition remains locked.
    """
    if not 0.0 < inner_val_fraction < 1.0:
        raise ValueError("inner_val_fraction must be between 0 and 1")

    outer_train = [int(index) for index in split["train_idx"]]
    if len(outer_train) < 2:
        raise ValueError("The outer training partition must contain at least two samples")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(outer_train), generator=generator).tolist()
    n_inner_val = max(1, int(len(outer_train) * inner_val_fraction))
    n_inner_val = min(n_inner_val, len(outer_train) - 1)
    inner_val_positions = set(permutation[:n_inner_val])

    fit_idx = sorted(index for position, index in enumerate(outer_train) if position not in inner_val_positions)
    inner_val_idx = sorted(index for position, index in enumerate(outer_train) if position in inner_val_positions)
    return {
        "fit_idx": fit_idx,
        "inner_val_idx": inner_val_idx,
        "selection_idx": [int(index) for index in split["val_idx"]],
        "locked_test_idx": [int(index) for index in split["test_idx"]],
    }


def create_split(n: int, seed: int, out_dir: Path) -> dict:
    """Create an 8:1:1 train/val/test split and cache it to *out_dir*.

    If *out_dir/indices.pt* already exists the cached split is returned
    without recomputing, so repeated calls with the same seed are idempotent.

    Returns:
        dict with keys ``train_idx``, ``val_idx``, ``test_idx`` (Python lists).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices_path = out_dir / "indices.pt"

    if indices_path.exists():
        blob = _load_torch(indices_path)
        return {
            "train_idx": blob["train_idx"].tolist(),
            "val_idx": blob["val_idx"].tolist(),
            "test_idx": blob["test_idx"].tolist(),
        }

    g = torch.Generator().manual_seed(seed)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    perm = torch.randperm(n, generator=g).tolist()
    train_idx = sorted(perm[:n_train])
    val_idx = sorted(perm[n_train : n_train + n_val])
    test_idx = sorted(perm[n_train + n_val :])

    torch.save(
        {
            "train_idx": torch.tensor(train_idx),
            "val_idx": torch.tensor(val_idx),
            "test_idx": torch.tensor(test_idx),
        },
        indices_path,
    )
    meta = {
        "n_total": n,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "split_ratio": "8:1:1",
        "seed": seed,
    }
    (out_dir / "data_split.json").write_text(json.dumps(meta, indent=2))
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def create_all_splits(n: int, seeds: list[int], base_dir: Path) -> dict[int, dict]:
    """Create splits for all *seeds*, caching under *base_dir/split/seed_<seed>/*."""
    splits = {}
    for seed in seeds:
        out_dir = Path(base_dir) / "split" / f"seed_{seed}"
        splits[seed] = create_split(n, seed, out_dir)
    return splits


def load_split(split_dir: Path) -> dict:
    """Load a previously saved split from *split_dir/indices.pt*."""
    split_dir = Path(split_dir)
    indices_path = split_dir / "indices.pt"
    blob = _load_torch(indices_path)
    return {
        "train_idx": blob["train_idx"].tolist(),
        "val_idx": blob["val_idx"].tolist(),
        "test_idx": blob["test_idx"].tolist(),
    }
