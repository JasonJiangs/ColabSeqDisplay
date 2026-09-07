"""Protein metadata and named residue-region lookup.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from; original module
`seqdisplay_opt/data/protein_database.py`. Only the region-resolution half is
kept. See `ATTRIBUTION.md`.

Two deliberate departures from upstream, both because `colabsd` ships no protein
database of its own -- `colabsd.protein_db` synthesizes one per run:

* `database_path` is required. Upstream falls back to a `proteins.yaml` packaged
  in its repository; here that fallback would silently pool a different protein.
* The cache is cleared through the public `clear_database_cache()` rather than
  through `_load_database.cache_clear`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_PROTEIN_ID = "slugcas9"
POOLING_REGIONS = {
    "cosine_p90_mean": "cosine_p90",
    "cosine_p95_mean": "cosine_p95",
    "full_mean": "full",
    "last150_mean": "last_150",
    "mutation_site_mean": "mutation_sites",
}


def _database_path(path: str | Path | None) -> Path:
    if path is None:
        raise FileNotFoundError(
            "A protein database path is required. colabsd ships no protein database: write one for this "
            "wild type with colabsd.protein_db.write_protein_record(spec, region, out_dir) and pass its path."
        )
    return Path(path).resolve()


@lru_cache(maxsize=16)
def load_database(path: Path) -> dict[str, Any]:
    """Load and cache a `proteins.yaml` payload. *path* must be absolute and resolved."""
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"Protein database not found: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    proteins = payload.get("proteins")
    if not isinstance(proteins, dict) or not proteins:
        raise ValueError(f"Protein database has no 'proteins' records: {path}")
    return payload


def clear_database_cache() -> None:
    """Forget every cached database, so a rewritten `proteins.yaml` is read again."""
    load_database.cache_clear()


def load_protein_record(
    protein_id: str = DEFAULT_PROTEIN_ID,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return one validated protein record from the metadata database."""
    path = _database_path(database_path)
    proteins = load_database(path)["proteins"]
    if protein_id not in proteins:
        raise KeyError(f"Unknown protein '{protein_id}' in {path}; available: {sorted(proteins)}")
    record = dict(proteins[protein_id] or {})
    length = int(record.get("sequence_length", 0))
    if length < 1:
        raise ValueError(f"Protein '{protein_id}' must define a positive sequence_length")
    if not isinstance(record.get("regions"), dict):
        raise ValueError(f"Protein '{protein_id}' must define a regions mapping")
    return record


def resolve_region_positions_0based(
    region_name: str,
    *,
    protein_id: str = DEFAULT_PROTEIN_ID,
    database_path: str | Path | None = None,
    sequence_length: int | None = None,
) -> list[int]:
    """Resolve a named protein region to ordered, zero-based residue positions."""
    record = load_protein_record(protein_id, database_path)
    regions = record["regions"]
    if region_name not in regions:
        raise KeyError(f"Protein '{protein_id}' has no region '{region_name}'; available: {sorted(regions)}")
    region = regions[region_name] or {}
    declared_length = int(record["sequence_length"])
    length = int(sequence_length or declared_length)
    if length != declared_length:
        raise ValueError(f"Token length {length} does not match protein '{protein_id}' length {declared_length}")

    mode = region.get("mode")
    if mode == "full":
        positions = list(range(length))
    elif mode == "tail":
        tail_length = int(region.get("length", 0))
        if tail_length < 1 or tail_length > length:
            raise ValueError(f"Invalid tail length {tail_length} for protein length {length}")
        positions = list(range(length - tail_length, length))
    elif mode == "positions":
        positions_1based = region.get("positions_1based")
        if not isinstance(positions_1based, list) or not positions_1based:
            raise ValueError(f"Region '{region_name}' must define positions_1based")
        positions = [int(position) - 1 for position in positions_1based]
    else:
        raise ValueError(f"Region '{region_name}' has unsupported mode {mode!r}")

    if len(set(positions)) != len(positions):
        raise ValueError(f"Region '{region_name}' contains duplicate positions")
    if any(position < 0 or position >= length for position in positions):
        raise ValueError(f"Region '{region_name}' contains positions outside protein length {length}")
    return positions


def resolve_pooling_positions_0based(
    pooling: str,
    *,
    protein_id: str = DEFAULT_PROTEIN_ID,
    database_path: str | Path | None = None,
    sequence_length: int | None = None,
) -> list[int]:
    """Resolve the protein region associated with a registered mean pooling name."""
    if pooling not in POOLING_REGIONS:
        raise ValueError(f"Pooling '{pooling}' has no protein-region mapping; available: {sorted(POOLING_REGIONS)}")
    return resolve_region_positions_0based(
        POOLING_REGIONS[pooling],
        protein_id=protein_id,
        database_path=database_path,
        sequence_length=sequence_length,
    )
