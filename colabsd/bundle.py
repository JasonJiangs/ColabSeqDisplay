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

`save_bundle` takes two keyword arguments beyond the contract, both optional:
`label_scaler` (the train-split z-score statistics upstream fits) and
`unlock_count`. Without a label scaler `colabsd.predict` returns z-scored
predictions and says so in the manifest.

The provenance block is built by `provenance_block`, and `colabsd.ui.exports`
stamps the same block into the performance archive a session downloads beside the
bundle. The two therefore say "tuned" or "PROVISIONAL", and count the test-set
reads, in one vocabulary rather than two that can drift apart.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from colabsd import __version__
from colabsd.errors import BundleError

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from colabsd.bestconfig import BestConfig
    from colabsd.spec import LibrarySpec

SCHEMA_VERSION = 2
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

    def describe(self) -> str:
        """One human-readable line for a notebook."""
        provenance = self.manifest.get("provenance", {})
        status = f"{hyperparameter_status(self.is_provisional)} hyperparameters"
        return (
            f"{self.model_name} · {len(self.pooling_positions_0based)} mutated sites · {status} · "
            f"{len(self.condition_columns)} conditions · test unlocked {self.unlock_count}x · "
            f"written {provenance.get('created_utc', 'unknown')}"
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
) -> Path:
    """Write a `.zip` bundle and return its path."""
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
        "hyperparameters": dict(getattr(best, "params", {}) or {}),
        "training": dict(getattr(best, "fixed", {}) or {}),
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

    Schema 1 recorded a pooling by name and froze whatever coordinates that name resolved
    to. This version always averages the mutated sites, so it cannot tell which residues a
    schema-1 bundle was actually fitted on -- and averaging the wrong ones is a silently
    wrong prediction table, not an error anyone would notice.
    """
    version = int(manifest.get("schema_version", 0))
    if version > SCHEMA_VERSION:
        raise BundleError(
            f"{path} uses bundle schema {version} but this colabsd understands up to {SCHEMA_VERSION}. "
            "Upgrade colabsd (pip install -U colabseqdisplay)."
        )
    recorded = manifest.get("pooling")
    if version == SCHEMA_VERSION and recorded is None:
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
) -> Path:
    """Bundle the best-validation run of a `colabsd.train.RunResult`.

    `unlock_count` is for a caller that has read `unlock.json` off disk: a `RunResult` held
    in a notebook variable can only ever *undercount*, because it predates any unlock taken
    after it was built, so the larger of the two numbers is the honest one and that is what
    the manifest records. `colabsd.report` reads the counter the same way.
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
