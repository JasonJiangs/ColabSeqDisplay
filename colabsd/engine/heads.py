"""Pluggable regression heads: the registry, the shared AdamW loop, ridge and the MLP.

Vendored from the *SequenceDisplay Workflow Optimization* research package
(``seqdisplay-opt``), which owns this science. Original modules:

* ``seqdisplay_opt/models/heads.py`` — ``HEAD_REGISTRY``, ``register_head``, ``BaseHead``,
  ``TorchHeadBase``, ``MLPHead``, ``RidgeHead`` and the target-standardization helper.
* ``seqdisplay_opt/models/factory.py`` — ``create_head``.
* ``seqdisplay_opt/utils/device.py`` — ``select_safe_device``, a transitive dependency of
  the torch training loop.

Copied so that ColabSeqDisplay installs without the research checkout. Both copies were
fitted on real data side by side: they predict identically for the same seed.

Left behind on purpose: the heads ColabSeqDisplay never reaches. Upstream registers eleven
(``linear``, ``cnn``, ``random_forest``, ``hgb``, ``knn``, ``xgboost``, ``elastic_net``,
``gpr``, ``pca_gpr`` besides these two); the notebooks run ``mlp`` for every fine-tuning
and best-config path and ``ridge`` + ``mlp`` for the one-hot floor, so only those two are
here. Dropping the rest also drops upstream's hard ``xgboost`` dependency.

Changed while copying: ``RidgeHead`` read its alpha-grid size from the ``PLM_RIDGE_N_ALPHAS``
environment variable, an HPC sweep knob no notebook sets. The grid is now the explicit
module constant ``RIDGE_N_ALPHAS``, whose value is upstream's default, so the numbers are
unchanged.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from colabsd.engine.metrics import evaluate_predictions, mean_metric

if TYPE_CHECKING:  # pragma: no cover - typing only
    from colabsd.engine.schema import TrainingConfig

# Upstream's default alpha grid: 17 points of np.logspace(-4, 4).
RIDGE_N_ALPHAS = 17

# A head reads only `lr`, `weight_decay`, `batch_size`, `max_epochs` and `patience` off the
# config, so anything carrying those five numbers works in place of a `TrainingConfig`.


def select_safe_device() -> torch.device:
    """Return CUDA when available and fail clearly for an unsupported GPU build."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    wanted = f"sm_{major}{minor}"
    supported = set(torch.cuda.get_arch_list())
    if supported and wanted not in supported:
        raise RuntimeError(
            f"GPU architecture {wanted} is not supported by this PyTorch build "
            f"({', '.join(sorted(supported))}). Install a compatible PyTorch build "
            "or hide CUDA devices to run explicitly on CPU."
        )
    return torch.device("cuda")


# Registry

HEAD_REGISTRY: dict[str, type[BaseHead]] = {}


def register_head(name: str):
    """Class decorator that adds a ``BaseHead`` subclass to ``HEAD_REGISTRY``."""

    def decorator(cls: type[BaseHead]) -> type[BaseHead]:
        HEAD_REGISTRY[name] = cls
        return cls

    return decorator


def create_head(name: str) -> BaseHead:
    """Create a registered regression head."""
    if name not in HEAD_REGISTRY:
        known = sorted(HEAD_REGISTRY)
        raise ValueError(f"Unknown head '{name}'. Available: {known}")
    return HEAD_REGISTRY[name]()


# Base interface


class BaseHead(ABC):
    """A regression head: fit on train/val, predict on a held-out matrix."""

    @abstractmethod
    def run(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        seed: int,
        cfg: TrainingConfig,
    ) -> tuple[np.ndarray, object, list]:
        """Train on train/val splits and predict on x_test."""


# Shared helpers for PyTorch heads


def _standardize_targets(
    y_train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Z-score targets on the training statistics; return the scaled arrays and the stats."""
    y_mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    y_std = (y_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    scaled = [((y - y_mean) / y_std).astype(np.float32) for y in others]
    return ((y_train - y_mean) / y_std).astype(np.float32), scaled, y_mean, y_std


class TorchHeadBase(BaseHead, ABC):
    """Shared AdamW training loop for nn.Module heads.

    Early stopping selects on validation mean Spearman, not on validation loss, and the
    returned predictions come from the best epoch's weights.
    """

    standardize_y: bool = False
    lr_factor: float = 1.0  # multiplied by cfg.lr; set to 0.3 for LN-based heads

    @abstractmethod
    def build_model(self, embed_dim: int, n_outputs: int = 4) -> nn.Module:
        """Return the module to train for *embed_dim* inputs and *n_outputs* targets."""

    def run(self, x_train, y_train, x_val, y_val, x_test, seed, cfg):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        dev = select_safe_device()
        model = self.build_model(x_train.shape[1], y_train.shape[1]).to(dev)

        y_mean, y_std = None, None
        if self.standardize_y:
            y_tr, [y_v], y_mean, y_std = _standardize_targets(y_train, y_val)
        else:
            y_tr, y_v = y_train, y_val

        lr = cfg.lr * self.lr_factor
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()
        generator = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_v, dtype=torch.float32),
            ),
            batch_size=cfg.batch_size,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(x_test, dtype=torch.float32)),
            batch_size=cfg.batch_size,
        )

        best_val_spearman = -np.inf
        best_state = None
        best_epoch = 0
        patience = 0
        training_log: list[dict] = []

        for epoch in range(cfg.max_epochs):
            model.train()
            losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            model.eval()
            val_losses = []
            val_predictions = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    prediction = model(xb.to(dev))
                    val_losses.append(criterion(prediction, yb.to(dev)).item())
                    val_predictions.append(prediction.cpu())

            train_loss = float(np.mean(losses))
            val_loss = float(np.mean(val_losses))
            val_pred = torch.cat(val_predictions, dim=0).numpy().astype(np.float32)
            if y_mean is not None and y_std is not None:
                val_pred = val_pred * y_std + y_mean
            val_spearman = mean_metric(evaluate_predictions(y_val, val_pred), "Spearman")
            training_log.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_mean_Spearman": round(val_spearman, 6),
                }
            )

            if best_state is None or val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
            if patience >= cfg.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        preds = []
        with torch.no_grad():
            for (xb,) in test_loader:
                preds.append(model(xb.to(dev)).cpu())
        pred_np = torch.cat(preds, dim=0).numpy().astype(np.float32)

        if y_mean is not None and y_std is not None:
            pred_np = pred_np * y_std + y_mean

        training_log.append(
            {
                "best_epoch": best_epoch,
                "best_val_mean_Spearman": round(best_val_spearman, 6),
            }
        )
        return pred_np, best_state, training_log


# Head architectures


@register_head("mlp")
class MLPHead(TorchHeadBase):
    """Default bottleneck MLP used for frozen and LoRA workflows."""

    standardize_y = True
    lr_factor = 0.3

    def build_model(self, embed_dim: int, n_outputs: int = 4) -> nn.Module:
        return nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_outputs),
        )


@register_head("ridge")
class RidgeHead(BaseHead):
    """Ridge regression with alpha selected on validation mean Spearman."""

    def run(self, x_train, y_train, x_val, y_val, x_test, seed, cfg):
        alphas = np.logspace(-4, 4, RIDGE_N_ALPHAS)
        best_score, best_model = -np.inf, None
        log: list[dict] = []
        for alpha in alphas:
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(x_train, y_train)
            score = mean_metric(evaluate_predictions(y_val, model.predict(x_val).astype(np.float32)), "Spearman")
            log.append({"alpha": float(alpha), "val_mean_Spearman": score})
            if score > best_score:
                best_score, best_model = score, model
        return best_model.predict(x_test).astype(np.float32), best_model, log
