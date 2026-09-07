"""Synthesize the `proteins.yaml` the engine resolves pooling coordinates from.

`colabsd.engine.config_space.pooling_positions_1based(config)` reads a protein-
metadata database keyed by protein id. A Colab user has a WT sequence and a
region JSON, not a database, so we write one for them in the run directory and
point the runtime config at it. The engine itself is never patched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colabsd.engine.protein_db import clear_database_cache
from colabsd.errors import ConfigError, DataError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from colabsd.bestconfig import BestConfig
    from colabsd.spec import LibrarySpec

DATABASE_FILENAME = "proteins.yaml"
DEFAULT_PROTEIN_ID = "user_protein"
MUTATION_SITE_REGION = "mutation_sites"
COSINE_REGIONS: tuple[str, ...] = ("cosine_p90", "cosine_p95")
TAIL_REGION_LENGTH = 150
_REGION_EXTRA_KEYS = {"source_model", "region_source_model", "score", "score_definition", "percentile"}

RegionPositions = Sequence[int] | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None


def _region_name(key: Any) -> str:
    name = str(key).strip().lower()
    if name.endswith("_mean"):
        name = name[: -len("_mean")]
    if name.isdigit():
        name = f"cosine_p{name}"
    elif name.startswith("p") and name[1:].isdigit():
        name = f"cosine_{name}"
    elif name.startswith("percentile"):
        digits = "".join(character for character in name if character.isdigit())
        name = f"cosine_p{digits}"
    if name not in COSINE_REGIONS:
        raise ConfigError(
            f"Unknown region key {key!r}. Region positions must be keyed by one of "
            f"{', '.join(COSINE_REGIONS)} (or 90 / 95), because upstream only pools those percentiles."
        )
    return name


def _positions_of(record: Mapping[str, Any]) -> list[int] | None:
    for key in ("positions_1based", "selected_positions_1based"):
        value = record.get(key)
        if value is not None:
            return [int(position) for position in value]
    return None


def _region_body(
    positions: Sequence[int],
    extras: Mapping[str, Any] | None = None,
    *,
    label: str = "region",
) -> dict[str, Any]:
    try:
        values = [int(position) for position in positions]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Region '{label}' must list 1-based integer positions: {exc}.") from exc
    if not values:
        raise ConfigError(
            f"Region '{label}' lists no positions. Upstream cannot pool an empty region: re-run region "
            "discovery for this wild-type sequence, or choose a pooling that does not need it."
        )
    body: dict[str, Any] = {"mode": "positions"}
    for key, value in (extras or {}).items():
        if key in _REGION_EXTRA_KEYS and value is not None:
            body["source_model" if key == "region_source_model" else key] = value
    body["positions_1based"] = values
    return body


def _check_region_length(record: Mapping[str, Any], name: str, expected_length: int | None) -> None:
    declared = record.get("seq_length")
    if expected_length is None or declared is None:
        return
    if int(declared) != int(expected_length):
        raise ConfigError(
            f"Region '{name}' was discovered for a {int(declared)}-residue sequence but this wild type is "
            f"{int(expected_length)} residues. Its 1-based positions do not describe this protein: re-run "
            "colabsd.region.discover_region() on this wild-type sequence."
        )


def _region_from_record(
    record: Mapping[str, Any],
    *,
    name: str | None = None,
    expected_length: int | None = None,
) -> tuple[str, dict[str, Any]]:
    positions = _positions_of(record)
    if positions is None:
        raise ConfigError(
            "A region record must carry 'positions_1based' (or 'selected_positions_1based'); "
            f"got keys {sorted(record)}."
        )
    declared = record.get("region_name") or record.get("pooling") or record.get("percentile")
    resolved = _region_name(declared) if declared is not None else None
    if name is not None and resolved is not None and resolved != name:
        raise ConfigError(
            f"A region filed under {name!r} declares itself {resolved!r}. Region positions are percentile "
            "specific: file each record under the percentile it was discovered at."
        )
    final = name or resolved or COSINE_REGIONS[0]
    _check_region_length(record, final, expected_length)
    return final, _region_body(positions, record, label=final)


def normalize_regions(
    region_positions_1based: RegionPositions,
    *,
    expected_length: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Turn the many shapes a caller may hold into `{region_name: region_body}`."""
    if region_positions_1based is None:
        return {}
    if isinstance(region_positions_1based, Mapping):
        if _positions_of(region_positions_1based) is not None:
            name, body = _region_from_record(region_positions_1based, expected_length=expected_length)
            return {name: body}
        regions = {}
        for key, value in region_positions_1based.items():
            name = _region_name(key)
            if isinstance(value, Mapping):
                _, body = _region_from_record(value, name=name, expected_length=expected_length)
            else:
                body = _region_body(value, label=name)
            regions[name] = body
        return regions
    items = list(region_positions_1based)
    if not items:
        return {}
    if all(isinstance(item, Mapping) for item in items):
        return dict(_region_from_record(item, expected_length=expected_length) for item in items)
    return {COSINE_REGIONS[0]: _region_body(items, label=COSINE_REGIONS[0])}


def _validated_spec(spec: LibrarySpec) -> tuple[str, list[int]]:
    validate = getattr(spec, "validate", None)
    if callable(validate):
        try:
            validate()
        except ValueError as exc:
            raise DataError(f"The library spec is not usable: {exc}") from exc
    sequence = str(spec.wt_sequence)
    positions = [int(position) for position in spec.positions_1based]
    if not sequence:
        raise DataError("spec.wt_sequence is empty; load the WT sequence before writing a protein record.")
    if not positions:
        raise DataError("spec.positions_1based is empty; mutation-site pooling needs at least one position.")
    return sequence, positions


def build_protein_record(
    spec: LibrarySpec,
    region_positions_1based: RegionPositions = None,
    *,
    protein_id: str = DEFAULT_PROTEIN_ID,
) -> dict[str, Any]:
    """Build the one protein record upstream's database loader expects."""
    sequence, mutation_positions = _validated_spec(spec)
    length = len(sequence)
    regions: dict[str, Any] = {"full": {"mode": "full"}}
    if length >= TAIL_REGION_LENGTH:
        regions["last_150"] = {"mode": "tail", "length": TAIL_REGION_LENGTH}
    regions[MUTATION_SITE_REGION] = _region_body(mutation_positions, label=MUTATION_SITE_REGION)
    regions.update(normalize_regions(region_positions_1based, expected_length=length))

    for name, body in regions.items():
        positions = body.get("positions_1based")
        if positions is None:
            continue
        if len(set(positions)) != len(positions):
            raise ConfigError(f"Region '{name}' repeats a position; every pooled residue must appear once.")
        outside = [position for position in positions if position < 1 or position > length]
        if outside:
            raise ConfigError(
                f"Region '{name}' has positions {outside[:5]} outside 1..{length}. "
                "Region positions are 1-based coordinates in the WT sequence: regenerate the region for this WT."
            )
        body["positions_1based"] = sorted(positions)

    return {
        "name": protein_id,
        "sequence_length": length,
        "generated_by": "colabsd",
        "regions": regions,
    }


def write_protein_record(
    spec: LibrarySpec,
    region_positions_1based: RegionPositions = None,
    out_dir: str | Path = ".",
    *,
    protein_id: str = DEFAULT_PROTEIN_ID,
) -> Path:
    """Write a one-protein `proteins.yaml` and return its absolute path."""
    import yaml

    record = build_protein_record(spec, region_positions_1based, protein_id=protein_id)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / DATABASE_FILENAME).resolve()
    try:
        payload = yaml.safe_dump({"version": 1, "proteins": {protein_id: record}}, sort_keys=False)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"The protein record for {protein_id!r} cannot be written as YAML: {exc}. "
            "Region metadata must be plain Python numbers and strings, not numpy scalars or arrays."
        ) from exc
    path.write_text(payload)
    clear_database_cache()
    return path


def _config_blocks(params: BestConfig | Mapping[str, Any] | None) -> tuple[str, dict, dict, dict]:
    if params is None:
        raise ConfigError(
            "runtime_config needs the LoRA hyperparameters; pass the BestConfig from "
            "colabsd.bestconfig.load_best_config(model, pooling)."
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
    pooling: str,
    params: BestConfig | Mapping[str, Any] | None = None,
    protein_id: str = DEFAULT_PROTEIN_ID,
    database_path: str | Path,
    split_seeds: Sequence[int] | None = None,
    model_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the config dict upstream's `train_eval_config(config=...)` expects."""
    import tempfile

    import yaml

    from colabsd.engine.config_space import pooling_positions_1based
    from colabsd.engine.protein_db import POOLING_REGIONS
    from colabsd.engine.train_config import load_lora_best_config

    if pooling not in POOLING_REGIONS:
        raise ConfigError(f"Unknown pooling {pooling!r}; the engine supports: {', '.join(sorted(POOLING_REGIONS))}.")
    model, lora_params, fixed, evaluation = _config_blocks(params)
    training = {**fixed, "pooling": pooling}
    if "micro_batch_size" not in training:
        raise ConfigError(
            "training.micro_batch_size must be set explicitly: pass the BestConfig from "
            "colabsd.bestconfig.load_best_config(model, pooling), or add a training block that sets it. "
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
    _check_record_matches_spec(spec, protein_id, raw["protein"]["database"], pooling)
    try:
        positions = pooling_positions_1based(config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ConfigError(
            f"Pooling {pooling!r} could not be resolved from {raw['protein']['database']}: {exc}. "
            "Write the protein record first with colabsd.protein_db.write_protein_record(), and make sure the "
            "region you picked was discovered for this WT sequence."
        ) from exc
    if not positions:
        raise ConfigError(f"Pooling {pooling!r} resolved to no residue positions for protein {protein_id!r}.")
    return config


def _check_record_matches_spec(
    spec: LibrarySpec,
    protein_id: str,
    database_path: str | Path,
    pooling: str,
) -> None:
    from colabsd.engine.protein_db import POOLING_REGIONS, load_protein_record

    try:
        record = load_protein_record(protein_id, database_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ConfigError(
            f"Protein record {protein_id!r} is not usable in {database_path}: {exc}. "
            "Call colabsd.protein_db.write_protein_record(spec, region, out_dir) first."
        ) from exc
    declared = int(record["sequence_length"])
    actual = len(str(spec.wt_sequence))
    if declared != actual:
        raise ConfigError(
            f"Protein record {protein_id!r} was written for a {declared}-residue sequence but the WT is {actual} "
            "residues. Rewrite the protein record for this WT sequence."
        )
    if POOLING_REGIONS.get(pooling) != MUTATION_SITE_REGION:
        return
    region = record["regions"].get(MUTATION_SITE_REGION) or {}
    recorded = [int(position) for position in region.get("positions_1based", [])]
    expected = sorted(int(position) for position in spec.positions_1based)
    if recorded and recorded != expected:
        raise ConfigError(
            f"Protein record {protein_id!r} pools mutation sites {recorded[:8]} but this library varies "
            f"{expected[:8]}. mutation_site_mean would average the wrong residues: rewrite the protein record "
            "for this spec with colabsd.protein_db.write_protein_record()."
        )
