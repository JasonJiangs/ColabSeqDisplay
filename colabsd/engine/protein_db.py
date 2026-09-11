"""Protein metadata: a wild type's length and the positions its library mutates.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from; original module
`seqdisplay_opt/data/protein_database.py`. See `ATTRIBUTION.md`.

Three deliberate departures from upstream, all because `colabsd` pools one way and
ships no protein database of its own -- `colabsd.protein_db` synthesizes one per run:

* A record carries `mutated_positions_1based` and nothing else. Upstream's records
  hold a `regions` mapping, and its loader resolves a region by name, because it
  pooled five different regions of one protein. Here the pooled residues are the
  mutated sites, so there is no region to name, select or validate.
* `database_path` and `protein_id` are both required. Upstream falls back to a
  `proteins.yaml` packaged in its repository, describing its own wild type; here
  that fallback would silently pool a different protein.
* The cache is cleared through the public `clear_database_cache()` rather than
  through `_load_database.cache_clear`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


def _database_path(path: str | Path | None) -> Path:
    if path is None:
        raise FileNotFoundError(
            "A protein database path is required. colabsd ships no protein database: write one for this "
            "wild type with colabsd.protein_db.write_protein_record(spec, out_dir) and pass its path."
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


def load_protein_record(protein_id: str, database_path: str | Path | None) -> dict[str, Any]:
    """Return one validated protein record from the metadata database."""
    path = _database_path(database_path)
    proteins = load_database(path)["proteins"]
    if protein_id not in proteins:
        raise KeyError(f"Unknown protein '{protein_id}' in {path}; available: {sorted(proteins)}")
    record = dict(proteins[protein_id] or {})
    length = int(record.get("sequence_length", 0))
    if length < 1:
        raise ValueError(f"Protein '{protein_id}' must define a positive sequence_length")
    positions = record.get("mutated_positions_1based")
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"Protein '{protein_id}' must list mutated_positions_1based")
    return record


def mutated_positions_0based(protein_id: str, database_path: str | Path | None) -> list[int]:
    """Resolve a protein's mutated sites to ordered, zero-based residue positions."""
    record = load_protein_record(protein_id, database_path)
    length = int(record["sequence_length"])
    try:
        positions = [int(position) - 1 for position in record["mutated_positions_1based"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Protein '{protein_id}' must list 1-based integer mutated positions: {exc}") from exc
    if len(set(positions)) != len(positions):
        raise ValueError(f"Protein '{protein_id}' lists the same mutated position twice")
    if any(position < 0 or position >= length for position in positions):
        raise ValueError(f"Protein '{protein_id}' lists mutated positions outside 1..{length}")
    return sorted(positions)
