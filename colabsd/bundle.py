"""Portable model bundles: one `.zip` holding a manifest plus LoRA and head tensors.

A bundle is what a user downloads at the end of a Colab session and hands to a
colleague a year later. It therefore carries provenance, not just weights: which
backbone produced it, the exact hyperparameters, whether those hyperparameters
were a provisional placeholder, the metrics that were measured, how often the
locked test partition had been unlocked when it was written, and the label scaler
needed to put predictions back on the assay's own scale.

Layout::

    manifest.json   schema version, spec, backbone, hyperparameters,
                    provenance, metrics, per-file sha256
    lora.pt         torch.save of the LoRA state dict (lora_a / lora_b tensors)
    head.pt         torch.save of the MLP head state dict

The residues a model averages are the library's mutated sites, so the manifest
records them once, as `spec.positions_1based`, and `Bundle.pooling_positions_0based`
derives the scoring coordinates from that one copy. Schema 1 bundles instead named a
pooling; this version cannot honour that name, so `load_bundle` refuses them rather
than quietly averaging different residues.

`save_bundle` takes three keyword arguments beyond the contract, all optional:
`label_scaler` (the train-split z-score statistics upstream fits), `unlock_count`,
and `trained_with` -- the run's own record of the hyperparameters and budget it
used, which is what the manifest publishes when it is given. Without a label
scaler `colabsd.predict` returns z-scored predictions and says so in the manifest.

The training budget -- epochs, early-stopping patience, and the two batch sizes --
is a user's to set, so a manifest that reported the looked-up numbers for a run
that used different ones would be a lie. `training_budget` therefore records
`colabsd.bestconfig.BudgetSettings.to_dict()`: what ran, what the registry entry
said, and which fields the user set. A run that never handed one over records
nothing rather than the lookup; `budget_line` says "not recorded" in as many words.

The provenance block is built by `provenance_block`, and `colabsd.ui.exports`
stamps the same block into the performance archive a session downloads beside the
bundle. The two therefore say "tuned" or "PROVISIONAL", count the test-set reads,
and describe the budget in one vocabulary rather than several that can drift apart.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colabsd import __version__
from colabsd.errors import BundleError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from colabsd.bestconfig import BestConfig, BudgetSettings
    from colabsd.spec import LibrarySpec

#: Schema 3 added `training_budget`. Schema 2 bundles still load -- their weights and the
#: residues they average are unchanged -- and report their budget as not recorded, because
#: there is nothing in them that says what it was. See `_require_readable_schema` for the one
#: schema this version refuses.
SCHEMA_VERSION = 3
#: The first schema that fixed the pooling to the library's mutated sites. Anything older named
#: a pooling instead, and this version cannot honour that name.
MUTATION_SITE_SCHEMA = 2
MANIFEST_NAME = "manifest.json"
LORA_NAME = "lora.pt"
HEAD_NAME = "head.pt"

#: What the hyperparameters behind a number are worth, in one word. Every artefact this
#: package writes uses these two strings, so a reader who learns what "PROVISIONAL" means
#: on a bundle already knows what it means on a performance archive.
PROVISIONAL_STATUS = "PROVISIONAL"
TUNED_STATUS = "tuned"


def hyperparameter_status(is_provisional: bool) -> str:
    """`"PROVISIONAL"` or `"tuned"` -- the one phrase every export uses for this."""
    return PROVISIONAL_STATUS if is_provisional else TUNED_STATUS


def utc_now() -> str:
    """The timestamp every artefact written here is stamped with."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 -- datetime.UTC needs 3.11


def sha256_hex(blob: bytes) -> str:
    """The checksum every archived file is recorded under."""
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | Path) -> str:
    """`sha256_hex` of a file on disk, read in chunks so a figure is never held twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_block(
    *,
    is_provisional: bool,
    unlock_count: int,
    notes: str | None = None,
    best_config_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Who wrote this, from which registry entry, and how often the test set had been read.

    Shared with `colabsd.ui.exports` so the model bundle and the performance archive
    downloaded beside it describe the same run in the same words. `hyperparameter_status`
    is spelled out next to the boolean because a reader opening the JSON months later
    should not have to know which way round `is_provisional` points.
    """
    return {
        "created_utc": utc_now(),
        "colabsd_version": __version__,
        "is_provisional": bool(is_provisional),
        "hyperparameter_status": hyperparameter_status(bool(is_provisional)),
        "best_config_meta": dict(best_config_meta or {}),
        "unlock_count": int(unlock_count),
        "notes": notes,
    }


# ------------------------------------------------------------------------------------------
# The training budget, in the one vocabulary every artefact writes it in.
# ------------------------------------------------------------------------------------------

#: The manifest key the budget is recorded under, the same in the bundle manifest, the
#: performance archive's `performance.json` and `report.json`.
BUDGET_KEY = "training_budget"

#: What a run, a wizard or an export may be carrying its resolved budget as. `budget_of` looks
#: for each in turn so that a caller can hand over whatever it has.
BUDGET_ATTRIBUTES: tuple[str, ...] = ("budget", "budget_settings", BUDGET_KEY)

#: What every artefact says when nothing recorded what the run was given. It is not a default:
#: a bundle written before `training_budget` existed has no record, and saying so is the only
#: honest thing to print over it.
BUDGET_NOT_RECORDED = "training budget not recorded"


def _count(value: Any) -> int | None:
    """*value* as a count of one or more, or None -- a word, a fraction and a zero are not one."""
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number == value and number > 0 else None


def budget_payload(source: Any) -> dict[str, Any]:
    """The budget record every artefact writes, out of whatever a caller carries it as.

    Accepts a `colabsd.bestconfig.BudgetSettings`, the mapping its `to_dict()` produced (an
    artefact read back off disk), or something that is neither -- which records as `{}`. That
    empty record is the point: "not recorded" is a fact a reader can act on, and writing the
    looked-up numbers for a run that may not have used them is exactly the lie this block
    exists to prevent.
    """
    from colabsd.bestconfig import BUDGET_FIELDS

    to_dict = getattr(source, "to_dict", None)
    if callable(to_dict) and all(hasattr(source, field) for field in BUDGET_FIELDS):
        source = to_dict()
    if not isinstance(source, Mapping):
        return {}
    used = {field: _count(source.get(field)) for field in BUDGET_FIELDS}
    if any(value is None for value in used.values()):
        return {}
    record: dict[str, Any] = dict(used)
    accumulation = _count(source.get("gradient_accumulation"))
    effective, micro = used["effective_batch_size"], used["micro_batch_size"]
    # How many micro-batches make up one optimiser step. Kept as recorded when the source has
    # it, because that is what the run actually did; derived only when it is absent.
    record["gradient_accumulation"] = accumulation or max(1, -(-effective // micro))
    record["chosen_by_user"] = [
        str(name) for name in source.get("chosen_by_user") or () if str(name) in BUDGET_FIELDS
    ]
    looked_up = source.get("looked_up") or {}
    record["looked_up"] = {
        str(name): _count(value)
        for name, value in (looked_up.items() if isinstance(looked_up, Mapping) else ())
        if str(name) in BUDGET_FIELDS and _count(value) is not None
    }
    return record


def _carried(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def budget_of(*sources: Any) -> dict[str, Any]:
    """The first budget recorded among *sources*, most authoritative first.

    Each source is tried both as a budget itself and as something carrying one under any of
    `BUDGET_ATTRIBUTES` -- so a caller hands over what it has, in the order it trusts it: an
    explicit record, then the `RunResult` the artefact describes, then the wizard or
    `BestConfig` the run was configured from. Nothing found records nothing.
    """
    for source in sources:
        record = budget_payload(source)
        if record:
            return record
        for name in BUDGET_ATTRIBUTES:
            record = budget_payload(_carried(source, name))
            if record:
                return record
    return {}


def budget_settings(record: Any) -> BudgetSettings | None:
    """A recorded budget back as a `colabsd.bestconfig.BudgetSettings`, or None if unrecorded."""
    from colabsd.bestconfig import BUDGET_FIELDS, BudgetSettings

    payload = budget_payload(record)
    if not payload:
        return None
    return BudgetSettings(
        **{field: int(payload[field]) for field in BUDGET_FIELDS},
        gradient_accumulation=int(payload["gradient_accumulation"]),
        chosen=tuple(payload["chosen_by_user"]),
        looked_up=dict(payload["looked_up"]),
    )


def budget_was_set_by_hand(record: Any) -> bool:
    """True when a user set any of it, so the run no longer matches the entry it came from."""
    return bool(budget_payload(record).get("chosen_by_user"))


def budget_changes_line(record: Any) -> str:
    """Which budget fields the user set, and what each one was before -- or "as looked up"."""
    settings = budget_settings(record)
    if settings is None:
        return BUDGET_NOT_RECORDED
    if settings.is_looked_up:
        return "as looked up"
    against_lookup = settings.changes()
    parts = [
        *settings.describe_changes(),
        # A field the user set that the registry entry never declared: there is nothing to
        # compare it against, and saying "-> 30" without a "from" would invent the from.
        *(
            f"{name} {int(getattr(settings, name))} (set by hand)"
            for name in settings.chosen
            if name not in against_lookup
        ),
    ]
    return "set by hand: " + ", ".join(parts)


def budget_line(record: Any) -> str:
    """The full budget in one line: what the run was given, and which of it is the user's.

    The two batch sizes are named for the job each does, because that is the one thing a
    reader gets wrong: the micro-batch is what has to fit on the card, the effective batch is
    what shapes the optimisation.
    """
    settings = budget_settings(record)
    if settings is None:
        return BUDGET_NOT_RECORDED
    return " · ".join(
        [
            _plural(settings.max_epochs, "epoch"),
            f"patience {settings.early_stopping_patience}",
            f"effective batch {settings.effective_batch_size} (optimisation)",
            f"micro-batch {settings.micro_batch_size} on the GPU (memory)",
            f"{_plural(settings.gradient_accumulation, 'micro-batch', 'micro-batches')} per optimiser step",
        ]
    ) + f" — {budget_changes_line(settings)}"


def budget_headline(record: Any) -> str:
    """The compact form, for a line that already carries five other facts."""
    settings = budget_settings(record)
    if settings is None:
        return BUDGET_NOT_RECORDED
    tail = "as looked up" if settings.is_looked_up else "budget set by hand"
    return f"{_plural(settings.max_epochs, 'epoch')}, micro-batch {settings.micro_batch_size} ({tail})"


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {plural or noun + 's'}"


@dataclass(frozen=True)
class Bundle:
    """A loaded bundle. `manifest` is the raw JSON; the rest is unpacked for use."""

    manifest: dict[str, Any]
    spec: LibrarySpec
    adapter_name: str
    params: dict[str, Any]
    fixed: dict[str, Any]
    lora_state: dict[str, torch.Tensor]
    head_state: dict[str, torch.Tensor]
    adapter_dtype: str | None
    adapter_hf_id: str | None
    metrics: dict[str, Any] | None
    label_scaler: dict[str, list[list[float]]] | None
    path: Path | None = None

    @property
    def schema_version(self) -> int:
        return int(self.manifest["schema_version"])

    @property
    def pooling_positions_0based(self) -> list[int]:
        """The residues the model averaged, as string offsets: the library's mutated sites.

        Derived rather than stored, so the manifest cannot end up with two copies of these
        coordinates that disagree about which residues a bundle was fitted on.
        """
        return list(self.spec.positions_0based())

    @property
    def model_name(self) -> str:
        return str(self.manifest.get("model_name", self.adapter_name))

    @property
    def condition_columns(self) -> list[str]:
        return list(self.spec.condition_columns)

    @property
    def is_provisional(self) -> bool:
        return bool(self.manifest.get("provenance", {}).get("is_provisional", False))

    @property
    def unlock_count(self) -> int:
        return int(self.manifest.get("provenance", {}).get("unlock_count", 0))

    @property
    def training_budget(self) -> dict[str, Any]:
        """What this run was given -- epochs, patience, both batch sizes -- and whose choice it was.

        Empty for a bundle written before this schema: those still load, and nothing in them
        records a budget, so nothing here invents one.
        """
        return budget_payload(self.manifest.get(BUDGET_KEY))

    @property
    def budget_was_set_by_hand(self) -> bool:
        """True when the user set part of the budget, so this run is not the looked-up one."""
        return budget_was_set_by_hand(self.training_budget)

    def describe(self) -> str:
        """One human-readable line for a notebook."""
        provenance = self.manifest.get("provenance", {})
        status = f"{hyperparameter_status(self.is_provisional)} hyperparameters"
        return (
            f"{self.model_name} · {len(self.pooling_positions_0based)} mutated sites · {status} · "
            f"{budget_headline(self.training_budget)} · {len(self.condition_columns)} conditions · "
            f"test unlocked {self.unlock_count}x · written {provenance.get('created_utc', 'unknown')}"
        )


def save_bundle(
    path: str | Path,
    *,
    spec: LibrarySpec,
    adapter_name: str,
    best: BestConfig,
    lora_state: dict[str, Any],
    head_state: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    label_scaler: dict[str, Any] | None = None,
    unlock_count: int | None = None,
    notes: str | None = None,
    adapter_dtype: str | None = None,
    adapter_hf_id: str | None = None,
    trained_with: Any = None,
    budget: Any = None,
) -> Path:
    """Write a `.zip` bundle and return its path.

    `best` is the registry entry the run was configured from. `trained_with` is the run's own
    record of what it then did -- a `colabsd.train.RunResult`, or anything carrying `params`
    and `fixed` -- and wins wherever it has something to say, because a manifest has to publish
    the hyperparameters that ran rather than the ones that were looked up. `budget` is the
    resolved `colabsd.bestconfig.BudgetSettings`; without one, `trained_with` and `best` are
    searched for a budget they carry, and a run that recorded none records none.
    """
    import torch

    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    lora = _tensor_state(lora_state, "lora_state")
    head = _tensor_state(head_state, "head_state")
    if not lora:
        raise BundleError(
            "lora_state is empty, so the bundle would hold no adapter weights. Pass the 'lora_state_dict' entry "
            "of a run's best_checkpoint.pt."
        )
    if not head:
        raise BundleError(
            "head_state is empty, so the bundle could not predict anything. Pass the 'head_state_dict' entry "
            "of a run's best_checkpoint.pt."
        )

    lora_blob = _serialize(lora)
    head_blob = _serialize(head)
    if unlock_count is None:
        unlock_count = int((metrics or {}).get("unlock_count", 0))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format": "colabsd-bundle",
        "model_name": str(getattr(best, "model", adapter_name)),
        "adapter_name": str(adapter_name),
        "adapter_dtype": None if adapter_dtype is None else str(adapter_dtype),
        "adapter_hf_id": None if adapter_hf_id is None else str(adapter_hf_id),
        "spec": _spec_payload(spec),
        "hyperparameters": _block(trained_with, best, "params"),
        "training": _block(trained_with, best, "fixed"),
        BUDGET_KEY: budget_of(budget, trained_with, best),
        "evaluation": dict(getattr(best, "evaluation", {}) or {}),
        "metrics": metrics,
        "label_scaler": _label_scaler_payload(label_scaler),
        "provenance": {
            **provenance_block(
                is_provisional=bool(getattr(best, "is_provisional", False)),
                unlock_count=int(unlock_count),
                notes=notes,
                best_config_meta=dict(getattr(best, "meta", {}) or {}),
            ),
            # Only a bundle carries tensors, so only a bundle records what wrote them.
            "torch_version": torch.__version__,
        },
        "tensors": {
            "lora": _tensor_manifest(LORA_NAME, lora, lora_blob),
            "head": _tensor_manifest(HEAD_NAME, head, head_blob),
        },
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        archive.writestr(LORA_NAME, lora_blob)
        archive.writestr(HEAD_NAME, head_blob)
    return path


def _block(trained_with: Any, best: Any, name: str) -> dict[str, Any]:
    """One hyperparameter block, as the run recorded it -- falling back to the entry it came from."""
    for source in (trained_with, best):
        found = _carried(source, name)
        if isinstance(found, Mapping) and found:
            return {str(key): value for key, value in found.items()}
    return {}


def load_bundle(path: str | Path) -> Bundle:
    """Read a bundle back, verifying the schema version and tensor checksums."""
    path = Path(path)
    if not path.is_file():
        raise BundleError(f"No bundle at {path}. Check the path, or re-run save_bundle() to create one.")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise BundleError(f"{path} is not a readable .zip bundle ({exc}). Re-download or re-create it.") from exc

    with archive:
        names = set(archive.namelist())
        missing = [name for name in (MANIFEST_NAME, LORA_NAME, HEAD_NAME) if name not in names]
        if missing:
            raise BundleError(
                f"{path} is missing {missing}. It was not written by colabsd.bundle.save_bundle(); "
                "re-create it from the training run."
            )
        manifest = json.loads(archive.read(MANIFEST_NAME))
        _require_readable_schema(manifest, path)
        lora = _read_tensors(archive, manifest, "lora", LORA_NAME, path)
        head = _read_tensors(archive, manifest, "head", HEAD_NAME, path)

    absent = [field for field in ("spec", "adapter_name") if manifest.get(field) is None]
    if absent:
        raise BundleError(
            f"The manifest in {path} has no {absent}. It was not written by colabsd.bundle.save_bundle(); "
            "re-create the bundle from the training run."
        )
    return Bundle(
        manifest=manifest,
        spec=_spec_from_payload(manifest["spec"]),
        adapter_name=str(manifest["adapter_name"]),
        params=dict(manifest.get("hyperparameters", {})),
        fixed=dict(manifest.get("training", {})),
        lora_state=lora,
        head_state=head,
        adapter_dtype=manifest.get("adapter_dtype"),
        adapter_hf_id=manifest.get("adapter_hf_id"),
        metrics=manifest.get("metrics"),
        label_scaler=manifest.get("label_scaler"),
        path=path,
    )


def _require_readable_schema(manifest: dict[str, Any], path: Path) -> None:
    """Refuse a bundle this version cannot describe, rather than reinterpreting it.

    One schema is refused, and only one. Schema 1 recorded a pooling by name and froze whatever
    coordinates that name resolved to. This version always averages the mutated sites, so it
    cannot tell which residues a schema-1 bundle was actually fitted on -- and averaging the
    wrong ones is a silently wrong prediction table, not an error anyone would notice.

    A schema-2 bundle is a different case and is not refused. Its weights, its spec and the
    residues it averages are exactly what this version writes; the only thing missing is the
    `training_budget` block, which describes the run rather than changing what the bundle does.
    Refusing a usable model over absent provenance would throw away someone's only copy of it,
    so it loads and `Bundle.training_budget` says, in as many words, that it was not recorded.
    """
    version = int(manifest.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise BundleError(
            f"{path} uses bundle schema {version} but this colabsd understands up to {SCHEMA_VERSION}. "
            "Upgrade colabsd (pip install -U colabseqdisplay)."
        )
    recorded = manifest.get("pooling")
    if version >= MUTATION_SITE_SCHEMA and recorded is None:
        return
    wrote = str(manifest.get("provenance", {}).get("colabsd_version") or "an earlier colabsd")
    chose = f", and it chose {str(recorded)!r}" if recorded is not None else ""
    raise BundleError(
        f"{path} is a schema-{version} bundle: it was written when the pooling was a choice{chose}. This "
        "colabsd always averages the embeddings at the library's mutated sites and cannot honour that "
        f"choice. Re-create the bundle from its training run with colabsd.bundle.save_bundle_from_run(), "
        f"or score it with the colabsd that wrote it ({wrote})."
    )


def save_bundle_from_run(
    path: str | Path,
    *,
    run_result: Any,
    spec: LibrarySpec,
    best: BestConfig,
    checkpoint: str | Path | None = None,
    notes: str | None = None,
    unlock_count: int | None = None,
    budget: Any = None,
) -> Path:
    """Bundle the best-validation run of a `colabsd.train.RunResult`.

    `unlock_count` is for a caller that has read `unlock.json` off disk: a `RunResult` held
    in a notebook variable can only ever *undercount*, because it predates any unlock taken
    after it was built, so the larger of the two numbers is the honest one and that is what
    the manifest records. `colabsd.report` reads the counter the same way.

    The run is also what the manifest's hyperparameters, training block and budget are taken
    from: `best` is where the run was looked up, `run_result` is what it then trained with, and
    the two differ the moment a user sets a budget of their own.
    """
    import torch

    run = run_result.best_run() if checkpoint is None else None
    checkpoint_path = Path(checkpoint) if checkpoint is not None else Path(run["checkpoint"])
    if not checkpoint_path.is_file():
        raise BundleError(
            f"No checkpoint at {checkpoint_path}. Keep the finetune() output_dir around, or pass "
            "checkpoint=<path to best_checkpoint.pt>."
        )
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require_checkpoint_positions(blob, spec, checkpoint_path)
    recorded = int(getattr(run_result, "unlock_count", 0) or 0)
    unlocks = recorded if unlock_count is None else max(int(unlock_count), recorded)
    metrics = {
        "validation_summary": run_result.validation_summary,
        "validation_runs": run_result.runs,
        "selection_metric": run_result.selection_metric,
        "n_runs": run_result.n_runs,
        "unlock_count": unlocks,
        "split_seeds": run_result.split_seeds,
        "model_seeds": run_result.model_seeds,
        "source_run": None if run is None else {"split_seed": run["split_seed"], "model_seed": run["model_seed"]},
    }
    if run_result.test_runs is not None:
        metrics["test_runs"] = run_result.test_runs
    return save_bundle(
        path,
        spec=spec,
        adapter_name=run_result.adapter_name,
        best=best,
        lora_state=blob["lora_state_dict"],
        head_state=blob["head_state_dict"],
        metrics=metrics,
        label_scaler=blob.get("label_scaler"),
        unlock_count=unlocks,
        notes=notes,
        adapter_dtype=getattr(run_result, "adapter_dtype", None),
        adapter_hf_id=getattr(run_result, "adapter_hf_id", None),
        trained_with=run_result,
        budget=budget,
    )


def _require_checkpoint_positions(blob: dict[str, Any], spec: LibrarySpec, checkpoint_path: Path) -> None:
    """The residues the run averaged must be the ones this spec calls mutated.

    Upstream records `mutation_positions_1based` in every checkpoint, so a spec that no longer
    matches the trained model is caught here instead of producing quietly wrong predictions.
    """
    recorded = blob.get("mutation_positions_1based")
    if not recorded:
        return
    trained = sorted(int(position) for position in recorded)
    expected = sorted(int(position) for position in spec.positions_1based)
    if trained != expected:
        raise BundleError(
            f"{checkpoint_path} was trained averaging {len(trained)} residues (1-based {trained[:5]}) but this "
            f"spec varies {len(expected)} (1-based {expected[:5]}). The bundle would score residues the model "
            "was never fitted on: pass the LibrarySpec the run was trained with."
        )


def _serialize(state: dict[str, torch.Tensor]) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _read_tensors(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    key: str,
    name: str,
    path: Path,
) -> dict[str, torch.Tensor]:
    import torch

    blob = archive.read(name)
    expected = manifest.get("tensors", {}).get(key, {}).get("sha256")
    if expected and sha256_hex(blob) != expected:
        raise BundleError(
            f"{name} in {path} does not match its manifest checksum. The bundle is corrupt; re-download it."
        )
    state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise BundleError(f"{name} in {path} does not hold a state dict. Re-create the bundle from the run.")
    return state


def _tensor_state(state: dict[str, Any], label: str) -> dict[str, torch.Tensor]:
    import torch

    converted: dict[str, torch.Tensor] = {}
    for name, value in dict(state).items():
        if isinstance(value, torch.Tensor):
            converted[str(name)] = value.detach().cpu().clone()
            continue
        try:
            converted[str(name)] = torch.as_tensor(value).cpu().clone()
        except (TypeError, RuntimeError) as exc:
            raise BundleError(
                f"{label}[{name!r}] is a {type(value).__name__}, which is not a tensor. Pass the state dict from "
                "a run's best_checkpoint.pt unchanged."
            ) from exc
    return converted


def _tensor_manifest(name: str, state: dict[str, torch.Tensor], blob: bytes) -> dict[str, Any]:
    return {
        "file": name,
        "n_tensors": len(state),
        "n_parameters": int(sum(int(tensor.numel()) for tensor in state.values())),
        "keys": sorted(state),
        "dtypes": {key: str(value.dtype) for key, value in sorted(state.items())},
        "sha256": sha256_hex(blob),
    }


def _spec_payload(spec: LibrarySpec) -> dict[str, Any]:
    try:
        payload = {
            "wt_sequence": str(spec.wt_sequence),
            "positions_1based": [int(position) for position in spec.positions_1based],
            "mutation_columns": [str(column) for column in spec.mutation_columns],
            "condition_columns": [str(column) for column in spec.condition_columns],
            "three_letter": bool(spec.three_letter),
            "wt_3di": None if spec.wt_3di is None else str(spec.wt_3di),
        }
    except AttributeError as exc:
        raise BundleError(
            f"spec is not a colabsd.spec.LibrarySpec ({exc}). Pass the LibrarySpec you trained with."
        ) from exc
    _require_mutated_sites(payload["positions_1based"], len(payload["wt_sequence"]))
    return payload


def _require_mutated_sites(positions_1based: list[int], length: int) -> None:
    """The mutated sites are the residues a bundle averages, so they must be usable coordinates.

    `load_bundle` rebuilds a `LibrarySpec` without re-validating it, and the scoring coordinates
    come straight off that spec: an unusable position written here is an unusable position for
    every prediction the bundle ever makes.
    """
    if not positions_1based:
        raise BundleError(
            "The spec lists no mutated positions, so the bundle records no residues to average. Pass the "
            "LibrarySpec you trained with."
        )
    if len(set(positions_1based)) != len(positions_1based):
        raise BundleError(
            "The spec lists the same mutated position twice, so that residue would be weighted twice in the "
            "mean. Give each mutated site one entry in positions_1based."
        )
    outside = [position for position in positions_1based if position < 1 or position > length]
    if outside:
        raise BundleError(
            f"The spec's mutated positions {outside[:5]} fall outside the {length}-residue wild type. "
            "positions_1based are 1-based coordinates in spec.wt_sequence."
        )


def _spec_from_payload(payload: dict[str, Any]) -> LibrarySpec:
    try:
        from colabsd.spec import LibrarySpec
    except ImportError as exc:  # pragma: no cover - colabsd.spec is a hard dependency
        raise BundleError(
            f"colabsd.spec is unavailable, so the bundle's library spec cannot be rebuilt ({exc})."
        ) from exc
    try:
        return LibrarySpec(
            wt_sequence=payload["wt_sequence"],
            positions_1based=[int(position) for position in payload["positions_1based"]],
            mutation_columns=list(payload["mutation_columns"]),
            condition_columns=list(payload["condition_columns"]),
            three_letter=bool(payload.get("three_letter", True)),
            wt_3di=payload.get("wt_3di"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(
            f"The bundle's library spec is incomplete or malformed ({exc}). Re-create the bundle from the "
            "training run."
        ) from exc


def _label_scaler_payload(label_scaler: Any) -> dict[str, list[list[float]]] | None:
    if label_scaler is None:
        return None
    import numpy as np

    if not isinstance(label_scaler, dict) or "mean" not in label_scaler or "std" not in label_scaler:
        raise BundleError(
            "label_scaler must be a {'mean': ..., 'std': ...} mapping, as stored in a run's best_checkpoint.pt."
        )
    return {
        "mean": np.asarray(label_scaler["mean"], dtype=np.float64).reshape(1, -1).tolist(),
        "std": np.asarray(label_scaler["std"], dtype=np.float64).reshape(1, -1).tolist(),
    }
