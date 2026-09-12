"""Synthesize the `proteins.yaml` the engine resolves pooled positions from.

`colabsd.engine.config_space.pooling_positions_1based(config)` reads a protein-
metadata database keyed by protein id. A Colab user has a wild-type sequence and a
`LibrarySpec`, not a database, so we write one for them in the run directory and
point the runtime config at it. The engine itself is never patched.

The record holds the wild type's length and the positions the library mutates,
because those positions are what pooling averages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colabsd.engine.pooling import POOLING_NAME
from colabsd.engine.protein_db import clear_database_cache
from colabsd.errors import ConfigError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from colabsd.bestconfig import BestConfig
    from colabsd.spec import LibrarySpec

DATABASE_FILENAME = "proteins.yaml"
DEFAULT_PROTEIN_ID = "user_protein"


def _validated_spec(spec: LibrarySpec) -> tuple[str, list[int]]:
    """Return the spec's wild-type sequence and its mutated positions, both checked."""
    validate = getattr(spec, "validate", None)
    if callable(validate):
        try:
            validate()
        except ValueError as exc:
            raise DataError(f"The library spec is not usable: {exc}") from exc
    sequence = str(spec.wt_sequence)
    try:
        positions = [int(position) for position in spec.positions_1based]
    except (TypeError, ValueError) as exc:
        raise DataError(f"spec.positions_1based must be 1-based integers: {exc}.") from exc
    if not sequence:
        raise DataError("spec.wt_sequence is empty; load the WT sequence before writing a protein record.")
    if not positions:
        raise DataError("spec.positions_1based is empty; pooling needs at least one mutated position.")
    if len(set(positions)) != len(positions):
        raise DataError("spec.positions_1based repeats a position; every pooled residue must appear once.")
    outside = [position for position in positions if position < 1 or position > len(sequence)]
    if outside:
        raise DataError(
            f"spec.positions_1based has {outside[:5]} outside 1..{len(sequence)}. Positions are 1-based "
            "coordinates in the WT sequence: check that the sequence is the full-length protein."
        )
    return sequence, sorted(positions)


def build_protein_record(spec: LibrarySpec, *, protein_id: str = DEFAULT_PROTEIN_ID) -> dict[str, Any]:
    """Build the one protein record the engine's database loader expects."""
    sequence, positions = _validated_spec(spec)
    return {
        "name": protein_id,
        "sequence_length": len(sequence),
        "generated_by": "colabsd",
        "mutated_positions_1based": positions,
    }


def write_protein_record(
    spec: LibrarySpec,
    out_dir: str | Path = ".",
    *,
    protein_id: str = DEFAULT_PROTEIN_ID,
) -> Path:
    """Write a one-protein `proteins.yaml` and return its absolute path."""
    import yaml

    record = build_protein_record(spec, protein_id=protein_id)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / DATABASE_FILENAME).resolve()
    try:
        payload = yaml.safe_dump({"version": 1, "proteins": {protein_id: record}}, sort_keys=False)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"The protein record for {protein_id!r} cannot be written as YAML: {exc}. "
            "The spec's positions must be plain Python integers, not numpy scalars or arrays."
        ) from exc
    path.write_text(payload)
    clear_database_cache()
    return path


def _config_blocks(params: BestConfig | Mapping[str, Any] | None) -> tuple[str, dict, dict, dict]:
    if params is None:
        raise ConfigError(
            "runtime_config needs the LoRA hyperparameters; pass the BestConfig from "
            "colabsd.bestconfig.load_best_config()."
        )
    if hasattr(params, "fixed") and hasattr(params, "params"):
        return str(params.model), dict(params.params), dict(params.fixed), dict(params.evaluation)
    if not isinstance(params, Mapping):
        raise ConfigError(f"params must be a BestConfig or a mapping, got {type(params).__name__}.")
    model = str(params.get("model") or "colabsd_model")
    lora = _as_block(params.get("parameters") or params.get("params"), "parameters")
    fixed = _as_block(params.get("fixed") or params.get("training"), "training")
    evaluation = _as_block(params.get("evaluation"), "evaluation")
    if not lora and not fixed:
        lora = {key: value for key, value in params.items() if key not in {"model", "evaluation"}}
    return model, lora, fixed, evaluation


def _as_block(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"params[{label!r}] must be a mapping of names to values, got {type(value).__name__}.")
    return dict(value)


def _seed_list(explicit: Sequence[int] | None, fallback: Any, label: str, default: list[int]) -> list[int]:
    values = explicit if explicit is not None else fallback
    if values is None:
        return list(default)
    if isinstance(values, (str, bytes, Mapping)):
        raise ConfigError(f"{label} must be a list of integers, got {type(values).__name__}.")
    try:
        seeds = [int(seed) for seed in values]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a list of integers: {exc}.") from exc
    if not seeds:
        raise ConfigError(f"{label} is empty; give at least one integer seed, or leave it unset to use the default.")
    return seeds


def runtime_config(
    spec: LibrarySpec,
    *,
    params: BestConfig | Mapping[str, Any] | None = None,
    protein_id: str = DEFAULT_PROTEIN_ID,
    database_path: str | Path,
    split_seeds: Sequence[int] | None = None,
    model_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the config dict upstream's `train_eval_config(config=...)` expects.

    `params` is staged as written and handed to upstream's loader to validate, so a caller
    whose user set their own epochs, patience or batch sizes passes the values that will
    actually run -- `colabsd.train.finetune` passes the `parameters`/`training` mapping its
    resolved `BudgetSettings` produced, not the looked-up blocks it started from.
    """
    import tempfile

    import yaml

    from colabsd.engine.config_space import pooling_positions_1based
    from colabsd.engine.train_config import load_lora_best_config

    model, lora_params, fixed, evaluation = _config_blocks(params)
    training = dict(fixed)
    training.setdefault("pooling", POOLING_NAME)
    if str(training["pooling"]) != POOLING_NAME:
        raise ConfigError(
            f"These hyperparameters were tuned for pooling {str(training['pooling'])!r}, but colabsd averages "
            f"the embeddings at the mutated sites and nothing else ({POOLING_NAME!r}). The run would be "
            f"recorded under a pooling it did not use: load the {POOLING_NAME!r} entry for this model."
        )
    if "micro_batch_size" not in training:
        raise ConfigError(
            "training.micro_batch_size must be set explicitly: pass the BestConfig from "
            "colabsd.bestconfig.load_best_config(), or add a training block that sets it. "
            "colabsd.engine.train_config would otherwise resolve it to 'auto', and the run would be "
            "recorded with a micro-batch size nobody chose (see config/best/README.md)."
        )
    if training.get("evaluate_test_during_optimization", False):
        raise ConfigError(
            "training.evaluate_test_during_optimization must stay false: the test partition is locked and is "
            "scored only by colabsd.train.unlock_test(). Remove that flag from the training block."
        )
    raw = {
        "model": model,
        "parameters": lora_params,
        "training": training,
        "evaluation": {
            "selection_objective": str(evaluation.get("selection_objective", "mean_validation_spearman")),
            "split_seeds": _seed_list(split_seeds, evaluation.get("split_seeds"), "split_seeds", [1]),
            "model_seeds": _seed_list(model_seeds, evaluation.get("model_seeds"), "model_seeds", [11]),
        },
        "protein": {"id": protein_id, "database": str(Path(database_path).resolve())},
        "data": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "best_config.yaml"
        try:
            staged.write_text(yaml.safe_dump(raw, sort_keys=False))
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"These hyperparameters cannot be written as YAML: {exc}. "
                "Use plain Python numbers, not numpy scalars."
            ) from exc
        try:
            _, _, config = load_lora_best_config(staged)
        except ValueError as exc:
            raise ConfigError(f"These hyperparameters are not a valid LoRA configuration: {exc}") from exc

    clear_database_cache()
    _check_record_matches_spec(spec, protein_id, raw["protein"]["database"])
    # Resolve the coordinates through the path training itself uses, so an unusable record
    # fails here rather than after the backbone has been downloaded.
    try:
        pooling_positions_1based(config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ConfigError(
            f"The pooled positions could not be resolved from {raw['protein']['database']}: {exc}. "
            "Write the protein record first with colabsd.protein_db.write_protein_record(spec, out_dir)."
        ) from exc
    return config


def _check_record_matches_spec(spec: LibrarySpec, protein_id: str, database_path: str | Path) -> None:
    from colabsd.engine.protein_db import load_protein_record

    try:
        record = load_protein_record(protein_id, database_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ConfigError(
            f"Protein record {protein_id!r} is not usable in {database_path}: {exc}. "
            "Call colabsd.protein_db.write_protein_record(spec, out_dir) first."
        ) from exc
    declared = int(record["sequence_length"])
    actual = len(str(spec.wt_sequence))
    if declared != actual:
        raise ConfigError(
            f"Protein record {protein_id!r} was written for a {declared}-residue sequence but the WT is {actual} "
            "residues. Rewrite the protein record for this WT sequence."
        )
    recorded = sorted(int(position) for position in record["mutated_positions_1based"])
    expected = sorted(int(position) for position in spec.positions_1based)
    if recorded != expected:
        raise ConfigError(
            f"Protein record {protein_id!r} pools mutation sites {recorded[:8]} but this library varies "
            f"{expected[:8]}. The run would average the wrong residues: rewrite the protein record for this "
            "spec with colabsd.protein_db.write_protein_record()."
        )
