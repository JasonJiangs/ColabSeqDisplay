"""Downloadable CSV templates: the shape of a file, next to the button that asks for one.

Two upload buttons on the two notebooks ask for a CSV, and neither can say in a label what
the columns have to be. A variant table's column names are not ours to fix: they are
whatever the user called their mutated sites, and the notebook only learns them from the
form or from a bundle. So the template is *generated*, from the spec in hand, rather than
shipped as one static file that would name somebody else's columns.

Two templates, and the difference between them is the point:

* `write_library_template` is the training input -- one column per mutated site, one per
  measured condition. Before a library is loaded there is no spec to generate from, so this
  one is the bundled MG8 PETases file itself, copied. It is real, it runs, and its header is
  the header a reader has to imitate.

* `write_variants_template` is the scoring input -- the mutated-site columns and nothing
  else. No condition column: those hold what you measured, and the whole point of scoring is
  that you have not measured these yet. Sending a user back with their assay column still in
  the file is the mistake this template exists to prevent.

A template is only useful if the loader that reads it back would accept it, so the
notation is the library's own: `three_letter=True` writes `Asn` where the user's CSV writes
`Asn`, and residues the reader does not take -- the ambiguity codes B/J/O/U/X/Z, which a
wild type is allowed to carry at a randomised site -- are stood in for by a real one.

Both write into the working directory and return the path, so the caller hands it to
`offer_download` the same way it hands over a bundle.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: What the download arrives as. Named for what it is, not for the library it came from: a
#: file called `library.csv` in a downloads folder is indistinguishable from the user's own.
LIBRARY_TEMPLATE_NAME = "example_library.csv"
VARIANTS_TEMPLATE_NAME = "example_variants.csv"

#: How many rows a generated variants template carries. Enough to show that a row is one
#: variant and how a residue is written, few enough to read in one screen.
TEMPLATE_ROWS = 5

#: The residue written where the wild type carries none the loader would take: an ambiguity
#: code (B/J/O/U/X/Z), which `spec.require_residue_alphabet` allows in a wild-type sequence
#: but `colabsd.data._one_letter_column` refuses in a variant column. Alanine, because a
#: template is read as an example of the shape and alanine is the conventional stand-in.
PLACEHOLDER_RESIDUE = "A"

#: A substitution the wild type does not already carry, so each row differs from row one.
_SUBSTITUTION = {"A": "G", "G": "A"}


def write_library_template(target_dir: str | Path, source: str | Path) -> Path:
    """Copy the bundled example library to *target_dir* under a name of its own."""
    origin = Path(source)
    if not origin.is_file():
        raise FileNotFoundError(f"The bundled example library is not where it should be: {origin}")
    destination = Path(target_dir) / LIBRARY_TEMPLATE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, destination)
    return destination


def template_residue(residue: object, column: str | None = None) -> str:
    """The one-letter code a template can write for *residue*, whichever way it is spelled.

    `LibrarySpec.wt_sequence` is always one-letter and always upper-case, so this is mostly a
    strip-and-upper. Two cases are not:

    * `Asn`, should a caller hand over residues already in the library's own notation, is
      folded back to `N` -- the template is built in one letter and converted once, at the
      end, so there is only one place that knows the two alphabets apart.
    * `X` (or B/J/O/U/Z) is a residue `spec.require_residue_alphabet` accepts in a wild-type
      sequence -- a placeholder at a randomised site is a normal way to write a combinatorial
      library -- but `colabsd.data._one_letter_column` refuses in a variant column, because a
      model cannot embed an ambiguity code. Writing it into the template would hand back a
      file this package's own reader rejects, so `PLACEHOLDER_RESIDUE` stands in for it.

    Anything that is not an amino acid at all is a caller bug, and is named as one.
    """
    from colabsd.data import ONE_LETTER_CODES, THREE_TO_ONE
    from colabsd.spec import AMBIGUOUS_RESIDUES

    text = str(residue).strip()
    letter = text.upper()
    if letter in ONE_LETTER_CODES:
        return letter
    if text.capitalize() in THREE_TO_ONE:
        return THREE_TO_ONE[text.capitalize()]
    if letter in AMBIGUOUS_RESIDUES:
        return PLACEHOLDER_RESIDUE
    where = f" for column '{column}'" if column else ""
    raise ValueError(
        f"{text!r}{where} is not an amino acid, so no template row could be written from it. "
        "Wild-type residues are one-letter codes such as N, or three-letter codes such as Asn."
    )


def variants_template_frame(
    columns: Sequence[str],
    *,
    wt_residues: Sequence[str] | None = None,
    rows: int = TEMPLATE_ROWS,
    three_letter: bool = False,
) -> Any:
    """A table with the mutated-site columns of *columns* and a few illustrative rows.

    `wt_residues` is the wild-type residue at each of those sites, when the caller knows it:
    the first row is then the wild type itself and the rest are single substitutions of it,
    so every value in the file is a residue that library actually contains. Without it the
    rows walk a short alphabet, which shows the shape but not the chemistry. A wild type
    carrying an ambiguity code at one of those sites (X, B, J, O, U, Z -- legal in a
    `wt_sequence`, refused in a variant column) gets `PLACEHOLDER_RESIDUE` there instead:
    a template is only worth having if the loader would take it back.

    `three_letter` is the notation of the library this template is for -- `LibrarySpec
    .three_letter`, which is what `colabsd.data` will decode the filled-in file with. It
    defaults to one letter, the notation of `wt_sequence` itself, so a caller that does not
    know gets the same file it always got; a caller that does know must pass it, or it hands
    a three-letter library a template its own loader refuses.
    """
    import pandas as pd

    names = [str(name).strip() for name in columns]
    if not names:
        raise ValueError("A variants template needs at least one mutated-site column.")
    if len(set(names)) != len(names):
        raise ValueError(f"Repeated column name in {names}; each mutated site is named once.")
    if rows < 1:
        raise ValueError(f"A template needs at least one row, not {rows}.")

    if wt_residues is not None and len(wt_residues) != len(names):
        raise ValueError(
            f"{len(wt_residues)} wild-type residues for {len(names)} columns. "
            "There is one residue per mutated site."
        )

    if wt_residues is None:
        base = [PLACEHOLDER_RESIDUE] * len(names)
    else:
        base = [template_residue(residue, name) for residue, name in zip(wt_residues, names, strict=True)]
    built = [list(base)]
    for index in range(1, rows):
        row = list(base)
        site = (index - 1) % len(names)
        row[site] = _SUBSTITUTION.get(row[site], PLACEHOLDER_RESIDUE)
        built.append(row)

    table = built[:rows]
    if three_letter:
        from colabsd.data import one_to_three_letter

        spelled = one_to_three_letter()
        table = [[spelled[letter] for letter in row] for row in table]
    return pd.DataFrame(table, columns=names)


def write_variants_template(
    target_dir: str | Path,
    columns: Sequence[str],
    *,
    wt_residues: Sequence[str] | None = None,
    rows: int = TEMPLATE_ROWS,
    three_letter: bool = False,
) -> Path:
    """Write `variants_template_frame` into *target_dir* and return the path.

    `three_letter` is the library's notation; pass `spec.three_letter` so the file that comes
    back down reads in the same alphabet the loader will decode it with.
    """
    frame = variants_template_frame(
        columns, wt_residues=wt_residues, rows=rows, three_letter=three_letter
    )
    destination = Path(target_dir) / VARIANTS_TEMPLATE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
