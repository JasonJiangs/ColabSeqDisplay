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
#: variant and that residues are single letters, few enough to read in one screen.
TEMPLATE_ROWS = 5


def write_library_template(target_dir: str | Path, source: str | Path) -> Path:
    """Copy the bundled example library to *target_dir* under a name of its own."""
    origin = Path(source)
    if not origin.is_file():
        raise FileNotFoundError(f"The bundled example library is not where it should be: {origin}")
    destination = Path(target_dir) / LIBRARY_TEMPLATE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, destination)
    return destination


def variants_template_frame(
    columns: Sequence[str],
    *,
    wt_residues: Sequence[str] | None = None,
    rows: int = TEMPLATE_ROWS,
) -> Any:
    """A table with the mutated-site columns of *columns* and a few illustrative rows.

    `wt_residues` is the wild-type residue at each of those sites, when the caller knows it:
    the first row is then the wild type itself and the rest are single substitutions of it,
    so every value in the file is a residue that library actually contains. Without it the
    rows walk a short alphabet, which shows the shape but not the chemistry.
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

    base = [str(r).strip().upper() for r in wt_residues] if wt_residues is not None else ["A"] * len(names)
    # A substitution the wild type does not already carry, so each row differs from row one.
    swap = {"A": "G", "G": "A"}
    built = [list(base)]
    for index in range(1, rows):
        row = list(base)
        site = (index - 1) % len(names)
        row[site] = swap.get(row[site], "A")
        built.append(row)
    return pd.DataFrame(built[:rows], columns=names)


def write_variants_template(
    target_dir: str | Path,
    columns: Sequence[str],
    *,
    wt_residues: Sequence[str] | None = None,
    rows: int = TEMPLATE_ROWS,
) -> Path:
    """Write `variants_template_frame` into *target_dir* and return the path."""
    frame = variants_template_frame(columns, wt_residues=wt_residues, rows=rows)
    destination = Path(target_dir) / VARIANTS_TEMPLATE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
