"""`LibrarySpec` — the single description of a variant library.

Upstream's loader hardcodes five NNK columns and reads every variant sequence
from a FASTA keyed by its residue names. A `LibrarySpec` replaces both: it names
the mutated positions in the wild-type sequence and the CSV columns that carry
the residues and the measured conditions, so sequences are *built* rather than
read, for any number of sites and any number of conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from .errors import DataError

CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_RESIDUES = frozenset("BJOUXZ")
ALLOWED_RESIDUES = CANONICAL_RESIDUES | AMBIGUOUS_RESIDUES
DEFAULT_COUNT_COLUMN = "count"


class SpecError(DataError, ValueError):
    """A `LibrarySpec` field is missing or inconsistent.

    Subclasses both `ColabSDError` (so notebooks show a friendly message) and
    `ValueError` (so callers that expect the plain Python error still catch it).
    """


class FrozenList(list):
    """A `list` that refuses every in-place change, and hashes like the tuple of its items.

    `LibrarySpec` is a frozen dataclass, but a plain list field leaves it mutable in
    practice: `spec.positions_1based.append(9999)` slips past `validate()` and shifts every
    residue afterwards. A `list` subclass rather than a tuple keeps `spec.positions_1based ==
    [1, 4]` true for callers, keeps the `list[int]` annotations honest, and keeps
    `dataclasses.asdict` / `json.dumps` round trips working unchanged.
    """

    __slots__ = ()

    def _frozen(self, *_args: Any, **_kwargs: Any) -> NoReturn:
        raise SpecError(
            "A LibrarySpec is frozen: its position and column lists cannot be changed in place, because "
            "validate() has already checked them against each other. Build a new LibrarySpec instead."
        )

    append = extend = insert = remove = pop = clear = sort = reverse = _frozen
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _frozen

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from a tuple: `copy.deepcopy` and `pickle` would otherwise `append` item by item."""
        return (self.__class__, (tuple(self),))


def _register_yaml_representer() -> None:
    """Teach PyYAML to dump a `FrozenList` as the list it is.

    PyYAML dispatches on the exact type, so `yaml.safe_dump(spec.positions_1based)` — which is
    how a protein record gets written — would otherwise raise `RepresenterError`.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a declared dependency
        return
    for dumper in (yaml.SafeDumper, yaml.Dumper):
        dumper.add_representer(FrozenList, dumper.represent_list)


_register_yaml_representer()


def _compact(text: str, field: str) -> str:
    if not isinstance(text, str):
        raise SpecError(f"{field} must be a string, got {type(text).__name__}.")
    return "".join(text.split())


def require_residue_alphabet(sequence: str, source: str = "wt_sequence") -> None:
    """Raise `SpecError` unless every character of `sequence` is an amino-acid letter.

    Whitespace is already stripped by the time this runs, so a pasted FASTA header or a
    numbered listing would otherwise be spliced into the sequence and silently shift every
    1-based position.
    """
    offenders = sorted({character for character in sequence if character not in ALLOWED_RESIDUES})
    if not offenders:
        return
    shown = ", ".join(repr(character) for character in offenders[:8])
    hint = "Remove the FASTA '>' description line. " if ">" in offenders else ""
    raise SpecError(
        f"{source} contains {shown}, which are not amino-acid letters. {hint}"
        "Paste only the residue letters (no header line, no numbering, no gaps or '*' stops): "
        "anything else shifts every position in positions_1based onto the wrong residue."
    )


@dataclass(frozen=True)
class LibrarySpec:
    """Everything the pipeline needs to know about one variant library.

    Attributes:
        wt_sequence: full-length wild-type amino-acid sequence.
        positions_1based: mutated residue positions in that sequence, 1-based.
        mutation_columns: CSV columns holding the residue at each position.
        condition_columns: CSV columns holding the measured activities.
        three_letter: residues are written ``Asn`` rather than ``N``.
        wt_3di: lowercase Foldseek 3Di string, same length as `wt_sequence`.
        count_column: CSV column holding per-variant read counts, or None when the
            table has none. `colabsd.data.load_library(min_count=...)` filters on it.

    Whitespace is stripped from `wt_sequence` and `wt_3di` on construction, the
    amino-acid sequence is upper-cased and the 3Di string lower-cased, so a spec
    survives a `dataclasses.asdict` round trip unchanged. The three list fields are
    `FrozenList`s: a validated spec cannot be edited in place afterwards, and it hashes.
    """

    wt_sequence: str
    positions_1based: list[int]
    mutation_columns: list[str]
    condition_columns: list[str]
    three_letter: bool = True
    wt_3di: str | None = None
    count_column: str | None = DEFAULT_COUNT_COLUMN

    def __post_init__(self) -> None:
        object.__setattr__(self, "wt_sequence", _compact(self.wt_sequence, "wt_sequence").upper())
        object.__setattr__(self, "positions_1based", _as_positions(self.positions_1based))
        object.__setattr__(self, "mutation_columns", _as_columns(self.mutation_columns, "mutation_columns"))
        object.__setattr__(self, "condition_columns", _as_columns(self.condition_columns, "condition_columns"))
        object.__setattr__(self, "three_letter", bool(self.three_letter))
        if self.wt_3di is not None:
            object.__setattr__(self, "wt_3di", _compact(self.wt_3di, "wt_3di").lower())
        if self.count_column is not None:
            object.__setattr__(self, "count_column", _as_count_column(self.count_column))

    @property
    def k(self) -> int:
        """Number of mutated sites."""
        return len(self.positions_1based)

    @property
    def n_targets(self) -> int:
        """Number of measured conditions."""
        return len(self.condition_columns)

    def positions_0based(self) -> list[int]:
        """Mutated positions as Python string offsets."""
        return [position - 1 for position in self.positions_1based]

    def wt_residues(self) -> str:
        """One-letter wild-type residues at the mutated positions."""
        return "".join(self.wt_sequence[offset] for offset in self.positions_0based())

    def validate(self) -> None:
        """Raise `SpecError` when the spec cannot describe a real library."""
        if not self.wt_sequence:
            raise SpecError("wt_sequence is empty. Provide the full-length wild-type amino-acid sequence.")
        require_residue_alphabet(self.wt_sequence)
        if self.k < 1:
            raise SpecError("positions_1based is empty. List at least one mutated position, 1-based.")
        if len(self.mutation_columns) != self.k:
            raise SpecError(
                f"{len(self.mutation_columns)} mutation columns for {self.k} positions "
                f"({self.mutation_columns} vs {self.positions_1based}). "
                "Give exactly one CSV column per mutated position, in the same order."
            )
        _reject_duplicates(self.mutation_columns, "mutation_columns")
        if not self.condition_columns:
            raise SpecError(
                "condition_columns is empty. Name at least one CSV column holding a measured activity."
            )
        _reject_duplicates(self.condition_columns, "condition_columns")
        length = len(self.wt_sequence)
        out_of_range = [position for position in self.positions_1based if not 1 <= position <= length]
        if out_of_range:
            raise SpecError(
                f"Positions {out_of_range} are outside the wild-type sequence, which is {length} residues long. "
                "Positions are 1-based; check that the sequence is the full-length protein."
            )
        unsorted = [
            (first, second)
            for first, second in zip(self.positions_1based, self.positions_1based[1:], strict=False)
            if second <= first
        ]
        if unsorted:
            raise SpecError(
                f"positions_1based must be strictly increasing and unique, but {unsorted[0][0]} is followed by "
                f"{unsorted[0][1]} in {self.positions_1based}. Sort the positions and the mutation columns "
                "together so column i still names position i."
            )
        if self.wt_3di is not None:
            stray = sorted(
                {character for character in self.wt_3di if not (character.isascii() and character.isalpha())}
            )
            if stray:
                shown = ", ".join(repr(character) for character in stray[:8])
                raise SpecError(
                    f"wt_3di contains {shown}, which are not 3Di letters. A Foldseek 3Di string is plain "
                    "lowercase letters; strip any FASTA header or numbering before passing it."
                )
            if len(self.wt_3di) != length:
                raise SpecError(
                    f"wt_3di is {len(self.wt_3di)} characters but the wild-type sequence is {length} residues. "
                    "The 3Di string must cover the same chain; regenerate it from a structure of this exact "
                    "sequence."
                )


def _as_count_column(column: object) -> str:
    name = str(column).strip()
    if not name:
        raise SpecError(
            "count_column is empty. Name the CSV column holding the read counts, or pass count_column=None "
            "when the table has none."
        )
    return name


def _as_positions(positions: object) -> FrozenList:
    if isinstance(positions, (str, bytes)) or not hasattr(positions, "__iter__"):
        raise SpecError(f"positions_1based must be a list of integers, got {type(positions).__name__}.")
    result = []
    for position in positions:
        if isinstance(position, bool):
            raise SpecError(f"positions_1based must contain integers, got {position!r}.")
        try:
            value = int(position)
        except (TypeError, ValueError):
            raise SpecError(f"positions_1based must contain integers, got {position!r}.") from None
        if not isinstance(position, (str, bytes)) and value != position:
            raise SpecError(
                f"positions_1based must contain whole numbers, got {position!r}. Rounding it to {value} "
                "would silently point at the wrong residue."
            )
        result.append(value)
    return FrozenList(result)


def _as_columns(columns: object, field: str) -> FrozenList:
    if isinstance(columns, (str, bytes)) or not hasattr(columns, "__iter__"):
        raise SpecError(f"{field} must be a list of column names, got {type(columns).__name__}.")
    return FrozenList(str(column) for column in columns)


def _reject_duplicates(columns: list[str], field: str) -> None:
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise SpecError(f"{field} names {duplicates} more than once. Every column must appear exactly once.")
