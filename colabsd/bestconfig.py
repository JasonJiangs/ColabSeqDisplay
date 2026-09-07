"""Best-config registry: one YAML per (backbone, pooling) pair.

Schema validation is upstream's job. `load_lora_best_config` is the single
authority on what a LoRA configuration must contain, so an entry it rejects is
rejected here too; this module only adds the registry lookup and the `_meta`
provenance block the notebook shows next to the backbone dropdown.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colabsd import BEST_CONFIG_ROOT
from colabsd.errors import ConfigError

TUNED = "tuned"
PROVISIONAL = "provisional"

#: Registry filenames are ``<model>_<pooling>.yaml``; model names may contain
#: hyphens and pooling names contain underscores, so the split needs the known
#: pooling suffixes. `tests/test_bestconfig.py` asserts these stay identical to
#: upstream `POOLING_REGIONS`.
POOLING_NAMES: tuple[str, ...] = (
    "cosine_p90_mean",
    "cosine_p95_mean",
    "full_mean",
    "last150_mean",
    "mutation_site_mean",
)

LORA_PARAMETERS: tuple[str, ...] = (
    "adapter_lr",
    "head_lr",
    "adapter_weight_decay",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "effective_batch_size",
)


class BestConfigNotFound(ConfigError, LookupError):
    """No registry entry exists for the requested (model, pooling) pair."""


@dataclass(frozen=True)
class BestConfig:
    """One validated registry entry plus its provenance."""

    model: str
    params: dict[str, Any]
    fixed: dict[str, Any]
    evaluation: dict[str, Any]
    meta: dict[str, Any]
    is_provisional: bool
    path: Path | None = None
    study: dict[str, Any] = field(default_factory=dict)

    @property
    def pooling(self) -> str:
        return str(self.fixed["pooling"])

    @property
    def status(self) -> str:
        return str(self.meta.get("status", PROVISIONAL)).strip().lower()

    @property
    def objective(self) -> str:
        return str(self.evaluation.get("selection_objective", "mean_validation_spearman"))

    @property
    def split_seeds(self) -> list[int]:
        return [int(seed) for seed in self.evaluation.get("split_seeds", [])]

    @property
    def model_seeds(self) -> list[int]:
        return [int(seed) for seed in self.evaluation.get("model_seeds", [])]

    @property
    def test_spearman(self) -> float | None:
        return _as_float(self.meta.get("test_spearman_mean"))

    @property
    def test_spearman_sd(self) -> float | None:
        return _as_float(self.meta.get("test_spearman_sd"))

    @property
    def n_runs(self) -> int | None:
        value = self.meta.get("n_reevaluation_runs")
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def describe(self) -> str:
        """One human-readable line for the notebook form."""
        head = f"{self.model} · {self.pooling}"
        if self.is_provisional:
            return f"{head} · PROVISIONAL — hyperparameters are a placeholder, performance unknown"
        return f"{head} · {self.status}{_performance_suffix(self)}"


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _performance_suffix(config: BestConfig) -> str:
    mean = config.test_spearman
    if mean is None:
        return " (test performance not recorded)"
    parts = [f"test ρ={mean:.4f}"]
    sd = config.test_spearman_sd
    if sd is not None:
        parts[0] += f" ± {sd:.4f}"
    if config.n_runs is not None:
        parts.append(f"{config.n_runs} runs")
    return f" ({', '.join(parts)})"


def _root(root: str | Path | None) -> Path:
    return Path(root) if root is not None else BEST_CONFIG_ROOT


def config_path(model_name: str, pooling: str, root: str | Path | None = None) -> Path:
    """Return the registry path for a (model, pooling) pair; existence is not checked."""
    for label, value in (("model_name", model_name), ("pooling", pooling)):
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ConfigError(f"{label} must be a plain registry name, got {value!r}")
    return _root(root) / f"{model_name}_{pooling}.yaml"


def available_pairs(root: str | Path | None = None) -> list[tuple[str, str]]:
    """List the (model, pooling) pairs present in the registry, sorted."""
    directory = _root(root)
    if not directory.is_dir():
        return []
    pairs = []
    for path in sorted(directory.glob("*.yaml")):
        if not path.is_file():
            continue
        for pooling in POOLING_NAMES:
            model = path.stem[: -len(pooling) - 1]
            if model and path.stem.endswith(f"_{pooling}"):
                pairs.append((model, pooling))
                break
    return sorted(pairs)


def available_models(pooling: str | None = None, root: str | Path | None = None) -> list[str]:
    """List registry models, optionally restricted to those tuned for one pooling."""
    return sorted({model for model, name in available_pairs(root) if pooling is None or name == pooling})


def available_poolings(model_name: str | None = None, root: str | Path | None = None) -> list[str]:
    """List registry poolings, optionally restricted to those available for one model."""
    return sorted({name for model, name in available_pairs(root) if model_name is None or model == model_name})


def _missing_entry_error(model_name: str, pooling: str, path: Path, root: Path) -> BestConfigNotFound:
    pairs = available_pairs(root)
    if not pairs:
        return BestConfigNotFound(
            f"The best-config registry at {root} is empty (looked for {path.name}). "
            "Point `root` at a directory of <model>_<pooling>.yaml files, e.g. config/best/."
        )
    models = available_models(root=root)
    for_model = available_poolings(model_name, root)
    for_pooling = available_models(pooling, root)
    lines = [f"No best config for model {model_name!r} with pooling {pooling!r} in {root}."]
    if for_model:
        lines.append(f"{model_name} is available with pooling: {', '.join(for_model)}.")
    else:
        suggestion = difflib.get_close_matches(model_name, models, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        lines.append(f"Known models: {', '.join(models)}.{hint}")
    if for_pooling:
        lines.append(f"{pooling} is available for: {', '.join(for_pooling)}.")
    else:
        lines.append(f"Known poolings: {', '.join(available_poolings(root=root))}.")
    lines.append(f"Pick a listed pair, or add {path.name} to that directory (see config/best/README.md).")
    return BestConfigNotFound(" ".join(lines))


#: Blocks a registry entry may declare; each must be a YAML mapping when present.
_MAPPING_BLOCKS: tuple[str, ...] = ("_meta", "parameters", "training", "evaluation", "protein", "data")


def _read_entry(path: Path) -> dict[str, Any]:
    """Parse one registry file into a mapping, reporting bad YAML as a `ConfigError`."""
    import yaml

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"{path} could not be read as YAML: {exc}. "
            "Fix the file against config/best/README.md, or pick another (model, pooling) pair."
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must be a YAML mapping with model/parameters/training/evaluation blocks, "
            f"but its top level is a {type(raw).__name__}. See config/best/README.md."
        )
    for block in _MAPPING_BLOCKS:
        value = raw.get(block)
        if value is not None and not isinstance(value, dict):
            raise ConfigError(
                f"{path}: the `{block}:` block must be a mapping, got a {type(value).__name__}. "
                "See config/best/README.md."
            )
    return raw


def load_best_config(model_name: str, pooling: str, root: str | Path | None = None) -> BestConfig:
    """Load and validate one registry entry, keeping its `_meta` provenance."""
    from colabsd.engine.train_config import load_lora_best_config

    directory = _root(root)
    path = config_path(model_name, pooling, directory)
    if not path.is_file():
        raise _missing_entry_error(model_name, pooling, path, directory)

    raw = _read_entry(path)
    try:
        loaded_model, params, runtime = load_lora_best_config(path)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ConfigError(
            f"{path} is not a valid LoRA best-config: {exc}. "
            "Fix the file against config/best/README.md, or pick another (model, pooling) pair."
        ) from exc

    meta = dict(raw.get("_meta") or {})
    fixed = dict(runtime["fixed"])
    if loaded_model != model_name:
        raise ConfigError(
            f"{path} declares model {loaded_model!r} but is filed under {model_name!r}. "
            "Rename the file or fix its `model:` field."
        )
    if str(fixed.get("pooling")) != pooling:
        raise ConfigError(
            f"{path} declares training.pooling {fixed.get('pooling')!r} but is filed under {pooling!r}. "
            "Rename the file or fix its `training.pooling` field."
        )

    return BestConfig(
        model=loaded_model,
        params=dict(params),
        fixed=fixed,
        evaluation=dict(raw.get("evaluation") or {}),
        meta=meta,
        is_provisional=str(meta.get("status", PROVISIONAL)).strip().lower() != TUNED,
        path=path,
        study=dict(runtime["study"]),
    )


def describe(model_name: str, pooling: str, root: str | Path | None = None) -> str:
    """One human-readable line for the notebook form, shouting about provisional entries."""
    return load_best_config(model_name, pooling, root).describe()
