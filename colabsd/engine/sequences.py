"""Validation helpers for sequence batches.

Vendored from the research project **SequenceDisplay-Workflow-Optimization**
(`seqdisplay_opt/data/sequences.py`), copied verbatim apart from this header.
See ``ATTRIBUTION.md``.
"""

from __future__ import annotations


def require_uniform_sequence_length(sequences: list[str]) -> int:
    """Return the shared sequence length or raise for an empty or ragged batch."""
    if not sequences:
        raise ValueError("At least one sequence is required")

    expected = len(sequences[0])
    mismatched = [index for index, sequence in enumerate(sequences) if len(sequence) != expected]
    if mismatched:
        preview = ", ".join(str(index) for index in mismatched[:5])
        raise ValueError(
            "This extraction path requires equal-length sequences; "
            f"expected length {expected}, but found mismatches at indices {preview}."
        )
    return expected
