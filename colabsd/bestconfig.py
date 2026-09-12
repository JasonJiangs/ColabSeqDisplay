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
import math
from collections.abc import Mapping
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


#: The training budget a user may set for themselves, in the order a form shows it. These are
#: not modelling choices: they are how long the run is allowed to take and what fits on the card
#: it runs on. Everything else in `LORA_PARAMETERS` stays looked up. `effective_batch_size` lives
#: in a registry entry's `parameters:` block and the other three in its `training:` block; from a
#: user's point of view they are one group, so this module resolves them together.
BUDGET_FIELDS: tuple[str, ...] = (
    "max_epochs",
    "early_stopping_patience",
    "effective_batch_size",
    "micro_batch_size",
)

#: What each budget field does, short enough for a form label or an error message. The two batch
#: sizes are the thing a user gets wrong -- one is memory, the other is optimisation -- and this
#: is the one place the difference is written down, so every refusal and every panel repeats it
#: in the same words.
BUDGET_MEANINGS: dict[str, str] = {
    "max_epochs": "how many passes over the training set",
    "early_stopping_patience": "how many epochs without improvement before the run stops",
    "effective_batch_size": "how many sequences one optimiser step averages over (optimisation)",
    "micro_batch_size": "how many sequences go on the GPU at once (memory)",
}


class BestConfigNotFound(ConfigError, LookupError):
    """No registry entry exists for the requested model."""


def _whole_number(name: str, value: Any) -> int:
    """Coerce one budget value to a count of at least 1, or refuse it by name."""
    meaning = BUDGET_MEANINGS[name]
    if isinstance(value, (bool, str, bytes)):
        raise ConfigError(
            f"{name} must be a whole number -- {meaning} -- but got {value!r}. Give it a count of 1 or more."
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} must be a whole number -- {meaning} -- but got {value!r} ({exc}). Give it a count of 1 or more."
        ) from exc
    if number != value:
        raise ConfigError(
            f"{name} must be a whole number -- {meaning} -- but got {value!r}. "
            f"Round it to {number} or another count of 1 or more."
        )
    if number < 1:
        raise ConfigError(
            f"{name} must be at least 1, but got {number}: {meaning}, and a run with none of them does nothing. "
            f"Set {name} to 1 or more, or clear it to use the looked-up value."
        )
    return number


@dataclass(frozen=True)
class BudgetOverrides:
    """The budget settings a user set for themselves; `None` is every one they left alone.

    `None` is not the same fact as "equal to the lookup". Someone who types the looked-up
    number has still chosen it, and the report says so, so an untouched field is `None` and
    a chosen one is the number -- even when the two agree.
    """

    max_epochs: int | None = None
    early_stopping_patience: int | None = None
    effective_batch_size: int | None = None
    micro_batch_size: int | None = None

    def __post_init__(self) -> None:
        for name in BUDGET_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _whole_number(name, value))

    @property
    def chosen(self) -> tuple[str, ...]:
        """The field names the user set, in `BUDGET_FIELDS` order."""
        return tuple(name for name in BUDGET_FIELDS if getattr(self, name) is not None)

    def __bool__(self) -> bool:
        return bool(self.chosen)

    def to_dict(self) -> dict[str, int]:
        """Only the fields the user set; an untouched field is absent, not `None`."""
        return {name: int(getattr(self, name)) for name in self.chosen}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BudgetOverrides:
        """Build from a form's `{field: value}` mapping, refusing anything else by name."""
        unknown = sorted(str(key) for key in values if key not in BUDGET_FIELDS)
        if unknown:
            raise ConfigError(
                f"{', '.join(unknown)} cannot be overridden. A run takes its own budget -- "
                f"{', '.join(BUDGET_FIELDS)} -- and nothing else: the learning rates, LoRA rank, alpha and "
                "dropout are looked up for this backbone and are not settings (see config/best/README.md)."
            )
        return cls(**{name: values[name] for name in BUDGET_FIELDS if values.get(name) is not None})

    @classmethod
    def coerce(cls, value: BudgetOverrides | Mapping[str, Any] | None) -> BudgetOverrides:
        """Accept an instance, a form mapping or `None` (nothing overridden)."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        raise ConfigError(
            f"budget must be a colabsd.bestconfig.BudgetOverrides or a mapping of "
            f"{{{', '.join(BUDGET_FIELDS)}}}, got {type(value).__name__}."
        )


@dataclass(frozen=True)
class BudgetSettings:
    """The budget one run actually used, and which of it the user chose.

    `looked_up` is what the registry entry said, so a reader six months later can see both
    numbers; `chosen` names the fields the user set, whether or not they landed on the same
    value. `gradient_accumulation` is derived, never set: it is how many micro-batches make
    up one optimiser step.
    """

    max_epochs: int
    early_stopping_patience: int
    effective_batch_size: int
    micro_batch_size: int
    gradient_accumulation: int
    chosen: tuple[str, ...] = ()
    looked_up: dict[str, int] = field(default_factory=dict)

    @property
    def is_looked_up(self) -> bool:
        """True when the user set nothing and the run is the configuration it was looked up from."""
        return not self.chosen

    def as_fixed(self) -> dict[str, int]:
        """The three values that belong in a runtime config's `fixed` (training) block."""
        return {
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "micro_batch_size": self.micro_batch_size,
        }

    def changes(self) -> dict[str, dict[str, int]]:
        """`{field: {"looked_up": x, "used": y}}` for every field the user set."""
        return {
            name: {"looked_up": self.looked_up[name], "used": int(getattr(self, name))}
            for name in self.chosen
            if name in self.looked_up
        }

    def describe_changes(self) -> list[str]:
        """One line per field the user set, saying what it was and what it became."""
        lines = []
        for name, pair in self.changes().items():
            if pair["looked_up"] == pair["used"]:
                lines.append(f"{name} {pair['used']} (set by hand, the same as the lookup)")
            else:
                lines.append(f"{name} {pair['looked_up']} -> {pair['used']}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        """The whole record: what ran, what the lookup said, and which fields are the user's."""
        payload: dict[str, Any] = {name: int(getattr(self, name)) for name in BUDGET_FIELDS}
        payload["gradient_accumulation"] = int(self.gradient_accumulation)
        payload["chosen_by_user"] = list(self.chosen)
        payload["looked_up"] = dict(self.looked_up)
        return payload


def _looked_up_budget(params: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict[str, int]:
    """Read the four budget values a registry entry declares, resolving upstream's `"auto"`."""
    from colabsd.engine.train_config import resolve_micro_batch_size

    if "effective_batch_size" not in params:
        raise ConfigError(
            "This configuration does not set parameters.effective_batch_size, so there is no batch size to "
            "run or to override. Load the entry with colabsd.bestconfig.load_best_config()."
        )
    for name in ("max_epochs", "early_stopping_patience"):
        if name not in fixed:
            raise ConfigError(
                f"This configuration does not set training.{name}, so there is nothing to run or to override. "
                "Load the entry with colabsd.bestconfig.load_best_config()."
            )
    effective = _whole_number("effective_batch_size", params["effective_batch_size"])
    return {
        "max_epochs": _whole_number("max_epochs", fixed["max_epochs"]),
        "early_stopping_patience": _whole_number("early_stopping_patience", fixed["early_stopping_patience"]),
        "effective_batch_size": effective,
        # `micro_batch_size: auto` is upstream's default and means "whatever fits"; resolving it here
        # is what lets a manifest record the number that actually ran instead of the word.
        "micro_batch_size": _whole_number(
            "micro_batch_size", resolve_micro_batch_size(fixed.get("micro_batch_size", "auto"), effective)
        ),
    }


def _set_by_hand(chosen: tuple[str, ...], *names: str) -> bool:
    """True when the user set any of *names* themselves."""
    return any(name in chosen for name in names)


def resolve_budget(
    params: Mapping[str, Any],
    fixed: Mapping[str, Any],
    overrides: BudgetOverrides | Mapping[str, Any] | None = None,
    *,
    n_train: int | None = None,
) -> BudgetSettings:
    """Resolve the budget one run will use, refusing a combination that cannot train.

    `params` and `fixed` are a registry entry's two blocks; `overrides` is what the user set.
    `n_train` is the smallest training split the run will see, when it is known. Every refusal
    is raised here, before a backbone is downloaded, and names the field and what to do.
    """
    user = BudgetOverrides.coerce(overrides)
    looked_up = _looked_up_budget(params, fixed)
    used = {**looked_up, **user.to_dict()}

    max_epochs = used["max_epochs"]
    patience = used["early_stopping_patience"]
    effective = used["effective_batch_size"]
    micro = used["micro_batch_size"]
    # A combination nobody touched is upstream's to judge -- `load_lora_best_config` already
    # refuses a batch size that cannot accumulate -- and refusing it here would stop a run for
    # something the user did not do and cannot fix from the panel. So each check below runs only
    # when one of its own fields was set by hand.
    by_hand = user.chosen

    if patience > max_epochs and _set_by_hand(by_hand, "max_epochs", "early_stopping_patience"):
        raise ConfigError(
            f"early_stopping_patience is {patience} but the run is only {max_epochs} epochs long, so it could "
            f"never stop early. Lower early_stopping_patience to {max_epochs} or less, or raise max_epochs."
        )
    if micro > effective and _set_by_hand(by_hand, "effective_batch_size", "micro_batch_size"):
        raise ConfigError(
            f"micro_batch_size {micro} is larger than effective_batch_size {effective}. micro_batch_size is "
            f"{BUDGET_MEANINGS['micro_batch_size']} and effective_batch_size is "
            f"{BUDGET_MEANINGS['effective_batch_size']}, so the first cannot exceed the second: lower "
            f"micro_batch_size to {effective} or below, or raise effective_batch_size to a multiple of {micro}."
        )
    if effective % micro and _set_by_hand(by_hand, "effective_batch_size", "micro_batch_size"):
        lower = (effective // micro) * micro
        raise ConfigError(
            f"effective_batch_size {effective} is not a whole number of micro-batches of {micro}: gradient "
            f"accumulation cannot run a fraction of a forward pass. Use {lower} or {lower + micro} for "
            f"effective_batch_size, or set micro_batch_size to a divisor of {effective}."
        )
    if n_train is not None and micro > n_train and _set_by_hand(by_hand, "effective_batch_size", "micro_batch_size"):
        raise ConfigError(
            f"micro_batch_size {micro} is larger than the {n_train} sequences in the smallest training split, "
            f"so a batch could never be filled. Lower micro_batch_size to {n_train} or below, or measure more "
            "variants. micro_batch_size is only "
            f"{BUDGET_MEANINGS['micro_batch_size']}: lowering it does not change what the model learns."
        )

    return BudgetSettings(
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        effective_batch_size=effective,
        micro_batch_size=micro,
        # How many micro-batches make one optimiser step, rounded up exactly as the trainer does.
        gradient_accumulation=max(1, math.ceil(effective / micro)),
        chosen=user.chosen,
        looked_up=looked_up,
    )


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
