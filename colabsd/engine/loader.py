"""Read the wild-type sequence files a run needs.

Vendored from **SequenceDisplay-Workflow-Optimization** (`seqdisplay_opt`), the
research package this pipeline was published from; original module
`seqdisplay_opt/data/loader.py`. Upstream's `load_targets` and `load_sequences`
are left behind: they read the 5NNK library through hardcoded `nnk1..nnk5` and
PAM columns, and `colabsd.data` builds both from the user's own
`LibrarySpec` instead. See `ATTRIBUTION.md`.
"""

from __future__ import annotations

from pathlib import Path


def read_fasta_records(path: Path) -> dict[str, str]:
    """Read a FASTA file into ``{record id: sequence}``, keyed by the first header word."""
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    with open(path) as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if current is not None:
                    records[current] = "".join(chunks)
                current = text[1:].split()[0]
                chunks = []
            else:
                chunks.append(text)
    if current is not None:
        records[current] = "".join(chunks)
    return records


def load_single_fasta(path: Path) -> str:
    """Load the only sequence from a single-record FASTA file."""
    records = read_fasta_records(Path(path))
    if len(records) != 1:
        raise ValueError(f"Expected one FASTA record in {path}, found {len(records)}")
    return next(iter(records.values()))


def load_foldseek_sequence(foldseek_wt_path: str | Path, data_root: Path) -> str:
    """Load the wild-type Foldseek 3Di string used by SaProt.

    ``foldseek_wt_path`` may point to a plain text file or a FASTA-style file.
    """
    if not foldseek_wt_path:
        raise FileNotFoundError("SaProt requires foldseek_wt_path")
    path = Path(foldseek_wt_path)
    if not path.is_absolute():
        path = data_root / path
    if not path.is_file():
        raise FileNotFoundError(f"Foldseek WT file not found: {path}")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        lines = [line for line in lines if not line.startswith(">")]
    return "".join(lines).lower()
