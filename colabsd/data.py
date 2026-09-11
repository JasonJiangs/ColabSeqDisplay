"""Load a variant library and build one full-length sequence per row.

Sequences are always constructed by substituting the row's residues into
`LibrarySpec.wt_sequence`; nothing here reads a per-variant FASTA. That keeps
every sequence exactly as long as the wild type, so the mutated positions are the
same residue offsets in every variant.

This module is also where the package decides what an ambiguous variant table is: a CSV
header that names the same column twice is refused rather than silently mangled by pandas,
and a `min_count` with nothing to filter on says so out loud. `colabsd.baseline` and
`colabsd.predict` inherit both rules through `require_unique_columns`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataError
from .spec import LibrarySpec, SpecError, require_residue_alphabet

THREE_TO_ONE: dict[str, str] = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}
ONE_LETTER_CODES: frozenset[str] = frozenset(THREE_TO_ONE.values())


class LibraryError(DataError, ValueError):
    """The variant table does not match its `LibrarySpec`.

    Subclasses both `ColabSDError` (so notebooks show a friendly message) and
    `ValueError` (so callers that expect the plain Python error still catch it).
    """


def load_library(
    csv_path: str | Path,
    spec: LibrarySpec,
    *,
    min_count: int = 0,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Read a variant CSV and return `(df, sequences, targets)`.

    Args:
        csv_path: variant table, one row per variant.
        spec: library description; `spec.validate()` runs first.
        min_count: drop rows whose `spec.count_column` value is below this. A positive
            threshold with no such column warns and keeps every row.

    Returns:
        The filtered table with a reset index, one full-length amino-acid string
        per row, and a float32 array of shape `(N, len(spec.condition_columns))`.
    """
    spec.validate()
    path = Path(csv_path)
    frame = read_variant_csv(path)

    _require_columns(frame, spec, path)
    if frame.empty:
        raise LibraryError(f"{path} has no rows. The variant table needs at least one variant.")

    frame = _apply_min_count(frame, spec, min_count, path)
    frame = frame.reset_index(drop=True)
    sequences = build_sequences(frame, spec)
    targets = build_targets(frame, spec)
    return frame, sequences, targets


def read_variant_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a variant CSV, refusing a header that names the same column twice.

    `pandas` silently renames repeated headers (`aa1`, `aa1.1`), so a duplicated mutation or
    condition column would be dropped without a word and the run would train on the first
    copy alone. This is the one reader the package uses, so every caller inherits the check.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise LibraryError(f"Variant table not found: {path}. Upload the CSV and pass its path.")
    try:
        header = pd.read_csv(path, header=None, nrows=1, dtype=str).iloc[0].tolist()
        frame = pd.read_csv(path)
    except Exception as exc:
        raise LibraryError(f"Could not read {path} as CSV: {exc}") from exc
    names = ["" if pd.isna(value) else str(value) for value in header]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise LibraryError(
            f"The header of {path} names {repeated} more than once. pandas renames the later copies "
            f"('{repeated[0]}' becomes '{repeated[0]}.1'), so a repeated mutation or condition column would be "
            "silently ignored and the run would use only the first copy. Give every column a unique name."
        )
    return frame


def require_unique_columns(frame: pd.DataFrame, columns: list[str], *, source: str | None = None) -> None:
    """Raise `LibraryError` when any of *columns* labels more than one column of *frame*.

    A frame built by hand, or read with pandas' duplicate mangling turned off, can carry the
    same label twice; `frame[label]` then hands back a DataFrame and every residue lookup
    downstream fails with an unreadable pandas error instead of an actionable one.
    """
    labels = [str(label) for label in frame.columns]
    repeated = sorted({str(column) for column in columns if labels.count(str(column)) > 1})
    if repeated:
        where = f" of {source}" if source else ""
        raise LibraryError(
            f"Columns {repeated}{where} appear more than once in the variant table, so it is ambiguous which "
            "copy holds the residues or the measurements. Give every column a unique name."
        )


def _apply_min_count(frame: pd.DataFrame, spec: LibrarySpec, min_count: int, path: Path) -> pd.DataFrame:
    """Filter on `spec.count_column`, saying so out loud when there is nothing to filter on."""
    threshold = _as_threshold(min_count)
    if not threshold:
        return frame
    column = spec.count_column
    if column is None:
        raise LibraryError(
            f"min_count={min_count!r} needs a read-count column, but this LibrarySpec sets count_column=None. "
            "Name the column holding the read counts, or pass min_count=0 to keep every variant."
        )
    if column not in frame.columns:
        warnings.warn(
            f"min_count={min_count!r} was ignored: {path} has no '{column}' column, so all {len(frame)} variants "
            f"were kept. Its columns are {list(frame.columns)}; set LibrarySpec(count_column=...) to the read-count "
            "column, or pass min_count=0 to say you meant to keep every variant.",
            UserWarning,
            stacklevel=3,
        )
        return frame
    require_unique_columns(frame, [column], source=str(path))
    counts = _numeric_counts(frame, path, column)
    filtered = frame.loc[counts >= threshold]
    if filtered.empty:
        raise LibraryError(
            f"No variant in {path} has {column} >= {min_count}; the highest is {counts.max():g}. Lower min_count."
        )
    return filtered


def build_sequences(frame: pd.DataFrame, spec: LibrarySpec) -> list[str]:
    """Substitute each row's residues into the wild-type sequence."""
    spec.validate()
    if frame.empty:
        raise LibraryError("No variants to build sequences from; the table is empty.")
    _require_columns(frame, spec, None, conditions=False)

    residues = [_one_letter_column(frame, column, spec) for column in spec.mutation_columns]
    wild_type = spec.wt_sequence
    offsets = spec.positions_0based()
    fillers = [wild_type[: offsets[0]]]
    fillers += [wild_type[left + 1 : right] for left, right in zip(offsets, offsets[1:], strict=False)]
    tail = wild_type[offsets[-1] + 1 :]

    sequences = []
    for row in zip(*residues, strict=True):
        parts: list[str] = []
        for filler, residue in zip(fillers, row, strict=True):
            parts.append(filler)
            parts.append(residue)
        parts.append(tail)
        sequences.append("".join(parts))

    from colabsd.engine.sequences import require_uniform_sequence_length

    length = require_uniform_sequence_length(sequences)
    if length != len(wild_type):
        raise LibraryError(f"Built sequences of length {length} from a {len(wild_type)}-residue wild type.")
    return sequences


def build_targets(frame: pd.DataFrame, spec: LibrarySpec) -> np.ndarray:
    """Return the measured conditions as float32 of shape `(N, T)`."""
    spec.validate()
    _require_columns(frame, spec, None, mutations=False)
    block = frame[list(spec.condition_columns)].apply(pd.to_numeric, errors="coerce")
    wide = block.to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(wide)
    if invalid.any():
        raise LibraryError(
            f"{int(invalid.sum())} condition values are missing or not numeric, for example "
            f"{_first_offender(invalid, frame, spec)}. Drop or fill those rows before loading."
        )
    with np.errstate(over="ignore"):
        values = wide.astype(np.float32)
    overflow = ~np.isfinite(values)
    if overflow.any():
        raise LibraryError(
            f"{int(overflow.sum())} condition values are too large for float32, for example "
            f"{_first_offender(overflow, frame, spec)}. Rescale the column (for example take a log) "
            "before loading."
        )
    return values


def _first_offender(mask: np.ndarray, frame: pd.DataFrame, spec: LibrarySpec) -> str:
    row, column = (int(index) for index in np.argwhere(mask)[0])
    name = spec.condition_columns[column]
    return f"column '{name}' at row {row} (index {frame.index[row]!r})"


def load_wt_sequence(path: str | Path) -> str:
    """Read the wild-type amino-acid sequence from a FASTA or plain-text file."""
    source = Path(path)
    if not source.is_file():
        raise LibraryError(f"Wild-type sequence file not found: {source}.")
    text = source.read_text()
    if text.lstrip().startswith(">"):
        from colabsd.engine.loader import load_single_fasta

        try:
            sequence = "".join(load_single_fasta(source).split()).upper()
        except ValueError as exc:
            raise LibraryError(
                f"{source} must hold exactly one FASTA record for the wild-type protein ({exc})."
            ) from exc
    else:
        sequence = "".join(text.split()).upper()
    if not sequence:
        raise LibraryError(f"{source} is empty. It must hold the wild-type amino-acid sequence.")
    try:
        require_residue_alphabet(sequence, str(source))
    except SpecError as exc:
        raise LibraryError(str(exc)) from exc
    return sequence


def make_splits(n: int, seeds: list[int], out_dir: str | Path) -> dict[int, dict]:
    """Create cached 8:1:1 splits, one per seed, under `out_dir/split/seed_<seed>/`.

    Upstream caches by directory alone, so reusing an `out_dir` after the library changed
    size would hand back indices for the *previous* library. Every returned split is checked
    to cover exactly `range(n)` and a stale cache is refused rather than silently used.
    """
    from colabsd.engine.splits import create_all_splits

    rows = int(n)
    if rows < 1:
        raise LibraryError(f"make_splits needs at least one row, got n={n}. Load the library first and pass len(df).")
    base = Path(out_dir)
    splits = create_all_splits(rows, [int(seed) for seed in seeds], base)
    for seed, split in splits.items():
        _require_fresh_split(split, rows, int(seed), base)
    return splits


def _require_fresh_split(split: dict, n: int, seed: int, base_dir: Path) -> None:
    covered = sorted(int(index) for key in ("train_idx", "val_idx", "test_idx") for index in split[key])
    if covered == list(range(n)):
        return
    cached = base_dir / "split" / f"seed_{seed}"
    raise LibraryError(
        f"The cached split in {cached} covers {len(covered)} rows (up to index "
        f"{covered[-1] if covered else 'none'}), not the {n} rows of this library. It was built for a "
        f"different table — a different CSV, or a different min_count. Delete {base_dir / 'split'} or pass a "
        "fresh out_dir so the splits are rebuilt for this library."
    )


def _require_columns(
    frame: pd.DataFrame,
    spec: LibrarySpec,
    path: Path | None,
    *,
    mutations: bool = True,
    conditions: bool = True,
) -> None:
    wanted: list[str] = []
    if mutations:
        wanted += list(spec.mutation_columns)
    if conditions:
        wanted += list(spec.condition_columns)
    missing = [column for column in wanted if column not in frame.columns]
    if missing:
        where = f" in {path}" if path is not None else ""
        raise LibraryError(
            f"Columns {missing} are not{where} in the variant table. "
            f"Available columns: {list(frame.columns)}. Fix the column names in the library specification."
            f"{_mangled_hint(frame, missing)}"
        )
    require_unique_columns(frame, wanted, source=None if path is None else str(path))


def _mangled_hint(frame: pd.DataFrame, missing: list[str]) -> str:
    """Name the pandas duplicate-mangling artefacts, which is what a missing column usually is."""
    labels = [str(label) for label in frame.columns]
    mangled = [label for label in labels for name in missing if label.startswith(f"{name}.") and label[-1].isdigit()]
    if not mangled:
        return ""
    return (
        f" The table does carry {sorted(set(mangled))}, which is how pandas renames a header that names the same "
        "column twice; de-duplicate the header instead of renaming the specification."
    )


def _as_threshold(min_count: object) -> float:
    if isinstance(min_count, bool) or not isinstance(min_count, (int, float, np.integer, np.floating)):
        raise LibraryError(
            f"min_count must be a number of reads, got {min_count!r}. Pass 0 to keep every variant."
        )
    value = float(min_count)
    if not np.isfinite(value):
        raise LibraryError(f"min_count must be a finite number of reads, got {min_count!r}.")
    return max(value, 0.0)


def _numeric_counts(frame: pd.DataFrame, path: Path, column: str) -> pd.Series:
    counts = pd.to_numeric(frame[column], errors="coerce")
    if counts.isna().any():
        rows = counts.index[counts.isna()][:5].tolist()
        raise LibraryError(
            f"The '{column}' column of {path} is not numeric, for example at rows {rows}. "
            "Make it a read count, or pass min_count=0 to skip the filter."
        )
    return counts


def _one_letter_column(frame: pd.DataFrame, column: str, spec: LibrarySpec) -> list[str]:
    text = frame[column].astype(str).str.strip()
    if spec.three_letter:
        mapped = text.str.capitalize().map(THREE_TO_ONE)
        expected = "three-letter codes such as Asn; set three_letter=False if the CSV uses one-letter codes"
    else:
        upper = text.str.upper()
        mapped = upper.where(upper.isin(ONE_LETTER_CODES))
        expected = "one-letter codes such as N; set three_letter=True if the CSV uses three-letter codes"
    unknown = mapped.isna()
    if unknown.any():
        position = int(np.argmax(unknown.to_numpy()))
        raise LibraryError(
            f"Unknown residue {text.iloc[position]!r} in column '{column}', row {position} "
            f"(index {frame.index[position]!r}). Residues must be {expected}."
        )
    return mapped.tolist()
