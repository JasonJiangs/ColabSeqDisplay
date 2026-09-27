"""LoRA adapters for PyTorch linear layers.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(`seqdisplay_opt/finetuning/lora.py`), whose authors wrote this LoRA injection
and whose published numbers were produced by it. Copied verbatim apart from this
header; the module has no dependency beyond ``torch``. See ``ATTRIBUTION.md``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_TARGET_MODULES = ["query", "key", "value", "output"]

_PROJECTION_ALIASES = {
    "query": {"query", "q_proj", "q"},
    "key": {"key", "k_proj", "k"},
    "value": {"value", "v_proj", "v"},
    "output": {"output", "out_proj", "o_proj", "o"},
}
_FUSED_QKV_ALIASES = {"qkv", "layernorm_qkv"}


class LoRALinear(nn.Module):
    """Linear layer with frozen base weights and trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return self.base(x) + update

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


def target_roles(module_name: str) -> frozenset[str]:
    """Return canonical q/k/v/o roles represented by an attention projection."""
    lowered = module_name.lower()
    if "attention" not in lowered and "attn" not in lowered:
        return frozenset()

    segments = lowered.split(".")
    leaf = segments[-1]
    if any(segment in _FUSED_QKV_ALIASES for segment in segments):
        return frozenset(("query", "key", "value"))

    roles = {role for role, aliases in _PROJECTION_ALIASES.items() if leaf in aliases}
    if leaf == "dense" and "output" in segments:
        roles.add("output")
    return frozenset(roles)


def _matches_target(module_name: str) -> bool:
    return bool(target_roles(module_name))


def _attention_group(module_name: str) -> str:
    segments = module_name.lower().split(".")
    for index, segment in enumerate(segments):
        if "attention" in segment or "attn" in segment:
            return ".".join(segments[: index + 1])
    return module_name.lower()


def validate_qkvo_coverage(module_names: list[str]) -> None:
    """Require every matched attention block to cover query, key, value, and output."""
    grouped: dict[str, set[str]] = {}
    for name in module_names:
        grouped.setdefault(_attention_group(name), set()).update(target_roles(name))
    if not grouped:
        raise RuntimeError("LoRA injection did not find any attention projections.")
    incomplete = {
        group: [role for role in LORA_TARGET_MODULES if role not in roles]
        for group, roles in grouped.items()
        if any(role not in roles for role in LORA_TARGET_MODULES)
    }
    if incomplete:
        examples = list(incomplete.items())[:4]
        raise RuntimeError(
            "LoRA injection did not cover every attention block; "
            f"incomplete groups: {examples}"
        )


def _set_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, nn.Sequential | nn.ModuleList):
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> list[str]:
    """Replace matching ``nn.Linear`` layers with ``LoRALinear``.

    Returns the matched module names. The caller should freeze the model before
    calling this function when only LoRA adapters should train.
    """
    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(module_name):
            continue
        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module_name, module))

    matched_names = [name for _, _, name, _ in replacements]
    validate_qkvo_coverage(matched_names)

    for parent, child_name, _, module in replacements:
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout).to(
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        _set_child(
            parent,
            child_name,
            wrapped,
        )

    return matched_names


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable LoRA parameters."""
    return [
        param
        for name, param in model.named_parameters()
        if ("lora_a" in name or "lora_b" in name) and param.requires_grad
    ]
