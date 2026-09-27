"""Resolve the pooled residue positions and the validation metric from a run configuration.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from; original module
`seqdisplay_opt/config/optuna_space.py`. Only the resolution helpers are kept -- the
Optuna search space (its YAML loader, its validator and `suggest_lora_params`) stays
upstream, because `colabsd` runs one configuration rather than searching for one.

One deliberate departure: upstream resolved the protein region named by
`config["fixed"]["pooling"]`. This pipeline always pools the mutated sites, so the
positions come from the protein record alone and the pooling name is never read.
See `ATTRIBUTION.md`.
"""

from __future__ import annotations

from typing import Any

from colabsd.engine.protein_db import mutated_positions_0based

OBJECTIVE_METRICS = {
    "mean_validation_r2": "R2",
    "mean_validation_pearson": "Pearson",
    "mean_validation_spearman": "Spearman",
}


def validation_metric_name(config: dict[str, Any]) -> str:
    """Return the metric selected by the configured validation objective."""
    objective = str(config["study"]["objective"])
    try:
        return OBJECTIVE_METRICS[objective]
    except KeyError as exc:
        supported = ", ".join(sorted(OBJECTIVE_METRICS))
        raise ValueError(f"Unsupported validation objective {objective!r}; choose from {supported}") from exc


def pooling_positions_0based(config: dict[str, Any]) -> list[int]:
    """Resolve the pooled residue positions from the configured protein database."""
    protein = config.get("protein") or {}
    protein_id = str(protein.get("id") or "")
    if not protein_id:
        raise ValueError("config['protein']['id'] must name the protein the database was written for")
    return mutated_positions_0based(protein_id, protein.get("database"))


def pooling_positions_1based(config: dict[str, Any]) -> list[int]:
    """Resolve the pooled residue positions as one-based protein coordinates."""
    return [position + 1 for position in pooling_positions_0based(config)]
