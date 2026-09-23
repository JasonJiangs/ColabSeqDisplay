"""Load a variant library and build one full-length sequence per row.

Sequences are always constructed by substituting the row's residues into
`LibrarySpec.wt_sequence`; nothing here reads a per-variant FASTA. That keeps
every sequence exactly as long as the wild type, so the mutated positions are the
same residue offsets in every variant.

This module is also where the package decides what an ambiguous variant table is: a CSV
header that names the same column twice is refused rather than silently mangled by pandas,
and a `min_count` with nothing to filter on says so out loud. `colabsd.predict` inherits the
first rule through `require_unique_columns`.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

#: Where `load_library` records what a `min_count` did to the table it hands back: the key
#: `DataFrame.attrs` carries, holding the read-count column, the threshold, the rows read out
#: of the CSV and the rows kept. Plain strings and ints rather than an object of ours, because
#: this travels with the frame and whatever reads it back should need nothing imported from
#: here to understand it. Absent when no filter ran -- `min_count` was zero, the spec named no
#: read-count column, or the CSV had no such column, which warns `MinCountIgnoredWarning`
#: instead. Present with `rows_kept == rows_read` when the threshold ran and nothing fell below
#: it, because a threshold that dropped nothing is a fact a user should be shown, not silence.
MIN_COUNT_RECORD: str = "colabsd_min_count"


def one_to_three_letter() -> dict[str, str]:
    """Invert `THREE_TO_ONE`, the one residue table this package owns.

    `colabsd.ui.predict_workflow` draws example variants in whichever notation the bundle's
    `LibrarySpec` declares, so it needs this table read the other way round. Building it here
    rather than writing a second table out by hand keeps one list of twenty residues.
    """
    return {one: three for three, one in THREE_TO_ONE.items()}


def _quoted(names: Iterable[Any]) -> str:
    """`'p22', 'p42', 'activity'` -- a column list as prose, not as a Python list repr.

    Kept private and duplicated in `colabsd.report` rather than shared: both modules sit below
    the widget layer and cannot import `colabsd.ui.core`, which holds the panels' copy
    (`positions_phrase`). The same reason `_plural` is written out twice.
    """
    return ", ".join(f"'{name}'" for name in names) or "none"


class MinCountIgnoredWarning(UserWarning):
    """`min_count` was asked for but there was no column to filter on, so nothing was dropped.

    A `UserWarning` subclass rather than a bare one so a caller that wants to put this on a
    screen can ask for exactly this -- `warnings.catch_warnings(record=True)` plus
    `warnings.simplefilter("always", MinCountIgnoredWarning)` -- instead of sifting every
    warning numpy, torch and pandas raise during a load. It stays a `UserWarning`, so code
    that already catches that keeps working.
    """


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
            threshold whose column is missing from the table warns and keeps every row;
            a spec that names no read-count column at all raises instead, because there
            is nothing to count. A whole number of reads either way: a fractional
            threshold raises, because the record left on the frame carries a whole one
            and would misreport it.

    Returns:
        The filtered table with a reset index, one full-length amino-acid string
        per row, and a float32 array of shape `(N, len(spec.condition_columns))`.
        When a `min_count` actually filtered, the returned table also carries
        `attrs[MIN_COUNT_RECORD]` -- the column, the threshold, and the rows read and
        kept -- because the count a caller is handed is the post-filter one and the
        pre-filter one is the only thing that exposes a mistyped threshold.
    """
    spec.validate()
    path = Path(csv_path)
    frame = read_variant_csv(path)

    _require_columns(frame, spec, path)
    if frame.empty:
        raise LibraryError(f"{path} has no rows. The variant table needs at least one variant.")

    frame, filtered = _apply_min_count(frame, spec, min_count, path)
    frame = frame.reset_index(drop=True)
    if filtered is not None:
        frame.attrs[MIN_COUNT_RECORD] = filtered
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


def _apply_min_count(
    frame: pd.DataFrame, spec: LibrarySpec, min_count: int, path: Path
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Filter on `spec.count_column`, saying so out loud when there is nothing to filter on.

    Hands back the table and, when a threshold really ran over a column that was really there,
    a record of what it did -- the column, the threshold, and the rows read and kept -- for
    `load_library` to hang on the frame under `MIN_COUNT_RECORD`. The record is `None` on every
    path where no filtering happened, because the two ways a threshold can do nothing already
    announce themselves: a spec with no read-count column raises here, and a table without the
    column it names warns `MinCountIgnoredWarning`.
    """
    threshold = _as_threshold(min_count)
    if not threshold:
        return frame, None
    column = spec.count_column
    if column is None:
        raise LibraryError(
            f"min_count={min_count!r} needs a read-count column, but this LibrarySpec sets count_column=None. "
            "Name the column holding the read counts, or pass min_count=0 to keep every variant."
        )
    if column not in frame.columns:
        warnings.warn(
            f"min_count={min_count!r} was ignored: {path} has no '{column}' column, so all {len(frame)} variants "
            f"were kept. Its columns are {_quoted(frame.columns)}; set LibrarySpec(count_column=...) to the "
            "read-count column, or pass min_count=0 to say you meant to keep every variant.",
            MinCountIgnoredWarning,
            stacklevel=3,
        )
        return frame, None
    require_unique_columns(frame, [column], source=str(path))
    counts = _numeric_counts(frame, path, column)
    filtered = frame.loc[counts >= threshold]
    if filtered.empty:
        raise LibraryError(
            f"No variant in {path} has {column} >= {min_count}; the highest is {counts.max():g}. Lower min_count."
        )
    return filtered, {
        "column": str(column),
        "threshold": int(threshold),
        "rows_read": int(len(frame)),
        "rows_kept": int(len(filtered)),
    }


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
            f"Columns {_quoted(missing)} are not{where} in the variant table. "
            f"Available columns: {_quoted(frame.columns)}. Fix the column names in the library specification."
            f"{_mangled_hint(frame, missing)}"
        )
    require_unique_columns(frame, wanted, source=None if path is None else str(path))


def _mangled_hint(frame: pd.DataFrame, missing: list[str]) -> str:
    """Name the pandas duplicate-mangling artifacts, which is what a missing column usually is."""
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
    # Clamped before the whole-number test, so that the two negatives answer the same way. A
    # threshold below zero is below every read count there can be, so it filters nothing whichever
    # way it is spelled; tested first, `-3` was quietly clamped to 0 while `-0.5` was refused as
    # fractional, and the refusal then suggested "Pass 0 or 1" about a number the caller had
    # written as negative. Nothing above zero is changed by the clamp, so 2.5 and 0.5 still raise.
    value = max(value, 0.0)
    if value != int(value):
        raise LibraryError(
            f"min_count must be a whole number of reads, got {min_count!r}. The record this filter "
            f"leaves on the frame -- and the sentence the panel prints from it -- carries a whole "
            f"threshold, so a fractional one would be reported as {int(value)} while the filter kept "
            f"only rows at or above {min_count!r}: a false claim about a real row. Pass {int(value)} "
            f"or {int(value) + 1}."
        )
    return value


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
    # Two readers, one sentence. The notebook user has a checkbox and no keyword argument in
    # sight, so the checkbox is named first, verbatim, including its comma -- it is
    # `colabsd.ui.main_workflow`'s "Residues are written as Asn, not N", under "Show the
    # advanced settings". The plain-API user gets the constructor argument second. Telling
    # only the second reader, as this did, sent the first one looking for a `three_letter=`
    # that is on no control.
    if spec.three_letter:
        mapped = text.str.capitalize().map(THREE_TO_ONE)
        expected = (
            "three-letter codes such as Asn. Untick **Residues are written as Asn, not N** on the "
            "form -- it is under *Show the advanced settings* -- if the CSV uses one-letter codes, "
            "or build the spec with LibrarySpec(three_letter=False)."
        )
    else:
        upper = text.str.upper()
        mapped = upper.where(upper.isin(ONE_LETTER_CODES))
        expected = (
            "one-letter codes such as N. Tick **Residues are written as Asn, not N** on the form -- "
            "it is under *Show the advanced settings* -- if the CSV uses three-letter codes, or "
            "build the spec with LibrarySpec(three_letter=True)."
        )
    unknown = mapped.isna()
    if unknown.any():
        position = int(np.argmax(unknown.to_numpy()))
        raise LibraryError(
            f"Unknown residue {text.iloc[position]!r} in column '{column}', row {position} "
            f"(index {frame.index[position]!r}). Residues must be {expected}"
        )
    return mapped.tolist()
