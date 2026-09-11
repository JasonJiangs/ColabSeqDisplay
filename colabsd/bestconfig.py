"""Best-config registry: the committed LoRA hyperparameters for one backbone.

Schema validation is upstream's job. `load_lora_best_config` is the single
authority on what a LoRA configuration must contain, so an entry it rejects is
rejected here too; this module only adds the registry lookup and the `_meta`
provenance block the notebook shows next to the backbone dropdown.

The lookup takes a backbone name and nothing else. `config/best/` holds 28 files
because it is the record of a study that compared pooling strategies, and their
names still carry the strategy that produced them; only the `mutation_site_mean`
half is reachable, because that is what the pipeline pools. `config/best/README.md`
says which is which.
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

#: The filename suffix of every reachable entry, `<model>_<ENTRY_SUFFIX>.yaml`. It is a fact
#: about the files on disk, not a choice offered to anyone: the pipeline averages the
#: embeddings at the mutated sites, so this is the only half of the directory that is read.
#: It is the same label the engine records in every artefact, `colabsd.engine.pooling.
#: POOLING_NAME`; `tests/test_bestconfig.py` asserts the two stay identical.
ENTRY_SUFFIX = "mutation_site_mean"

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
    """No registry entry exists for the requested model."""


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
        if self.is_provisional:
            return f"{self.model} · PROVISIONAL — hyperparameters are a placeholder, performance unknown"
        return f"{self.model} · {self.status}{_performance_suffix(self)}"


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


def config_path(model_name: str, root: str | Path | None = None) -> Path:
    """Return the registry path for *model_name*; existence is not checked."""
    if not model_name or "/" in model_name or "\\" in model_name or model_name in {".", ".."}:
        raise ConfigError(f"model_name must be a plain registry name, got {model_name!r}")
    return _root(root) / f"{model_name}_{ENTRY_SUFFIX}.yaml"


def available_models(root: str | Path | None = None) -> list[str]:
    """List the models the registry can be asked for, sorted.

    A directory may hold entries from a study that compared several pooling strategies;
    only the `<model>_mutation_site_mean.yaml` half is reachable, so only those are listed.
    """
    directory = _root(root)
    if not directory.is_dir():
        return []
    models = set()
    for path in sorted(directory.glob(f"*_{ENTRY_SUFFIX}.yaml")):
        if not path.is_file():
            continue
        model = path.stem[: -len(ENTRY_SUFFIX) - 1]
        if model:
            models.add(model)
    return sorted(models)


def _missing_entry_error(model_name: str, path: Path, root: Path) -> BestConfigNotFound:
    models = available_models(root)
    if not models:
        return BestConfigNotFound(
            f"The best-config registry at {root} holds no entry this pipeline can read (looked for {path.name}). "
            f"Point `root` at a directory of <model>_{ENTRY_SUFFIX}.yaml files, e.g. config/best/."
        )
    suggestion = difflib.get_close_matches(model_name, models, n=1)
    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
    return BestConfigNotFound(
        f"No best config for model {model_name!r} in {root}. Known models: {', '.join(models)}.{hint} "
        f"Pick a listed model, or add {path.name} to that directory (see config/best/README.md)."
    )


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
            "Fix the file against config/best/README.md, or pick another model."
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


def load_best_config(model_name: str, root: str | Path | None = None) -> BestConfig:
    """Load and validate the registry entry for *model_name*, keeping its `_meta` provenance."""
    from colabsd.engine.train_config import load_lora_best_config

    directory = _root(root)
    path = config_path(model_name, directory)
    if not path.is_file():
        raise _missing_entry_error(model_name, path, directory)

    raw = _read_entry(path)
    try:
        loaded_model, params, runtime = load_lora_best_config(path)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ConfigError(
            f"{path} is not a valid LoRA best-config: {exc}. "
            "Fix the file against config/best/README.md, or pick another model."
        ) from exc

    meta = dict(raw.get("_meta") or {})
    if loaded_model != model_name:
        raise ConfigError(
            f"{path} declares model {loaded_model!r} but is filed under {model_name!r}. "
            "Rename the file or fix its `model:` field."
        )
    _require_matching_filename(raw, path)

    return BestConfig(
        model=loaded_model,
        params=dict(params),
        fixed=dict(runtime["fixed"]),
        evaluation=dict(raw.get("evaluation") or {}),
        meta=meta,
        is_provisional=str(meta.get("status", PROVISIONAL)).strip().lower() != TUNED,
        path=path,
        study=dict(runtime["study"]),
    )


def _require_matching_filename(raw: dict[str, Any], path: Path) -> None:
    """Refuse an entry filed under a name its own record contradicts.

    The study these files came from recorded the pooling it used in `training.pooling`, and
    the filename repeats it. A file that says one thing and is named another would hand this
    pipeline the hyperparameters of a run it is not reproducing.
    """
    declared = (raw.get("training") or {}).get("pooling")
    if declared is not None and str(declared) != ENTRY_SUFFIX:
        raise ConfigError(
            f"{path} records training.pooling {str(declared)!r} but is filed as an {ENTRY_SUFFIX} entry. "
            "Those hyperparameters were selected for a different run: rename the file, or fix its "
            "`training.pooling` field."
        )


def describe(model_name: str, root: str | Path | None = None) -> str:
    """One human-readable line for the notebook form, shouting about provisional entries."""
    return load_best_config(model_name, root).describe()
