"""Amino-acid vocabulary and per-site one-hot encoding of a mutated-residue table.

Vendored from the *SequenceDisplay Workflow Optimization* research package
(``seqdisplay-opt``), original module ``seqdisplay_opt/baselines/one_hot.py``: the ``AA3``
vocabulary, its ``AA3_TO_INDEX`` inverse and the encoder upstream calls
``encode_five_site_one_hot``.

Copied so that ColabSeqDisplay installs without the research checkout.
``tests/test_engine_heads.py`` encodes the bundled example library under both copies and
asserts the arrays are identical whenever that checkout is present.

Left behind on purpose: everything else in that 650-line module — the ``argparse`` CLI, the
YAML config plumbing, ``BaselineContext``, the ``BASELINE_REGISTRY`` of sklearn/torch
predictors and the run harness that writes result directories. ColabSeqDisplay fits its
one-hot floor with ``colabsd.engine.heads`` instead, so none of it is reachable.

Changed while copying: upstream's name promises five sites even though its body already
loops over ``columns``. The encoder is ``encode_site_one_hot`` here and is documented for
any number of sites; the five-site result is byte-identical, which the tests assert both
against upstream and against a hand-built indicator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# The 20 standard amino acids as three-letter names, in the order that fixes the one-hot
# axis. Changing this order changes every encoded feature, so it is fixed for good.
AA3: tuple[str, ...] = (
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
)
AA3_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(AA3)}


def encode_site_one_hot(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Encode residue names as an array with shape ``(N, len(columns), 20)``.

    One column per randomized site, in the order given; the site axis follows that order.
    Any number of sites is accepted, including one.
    """
    missing_columns = [column for column in columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing mutation columns: {missing_columns}")

    encoded = np.zeros((len(frame), len(columns), len(AA3)), dtype=np.float32)
    for site_index, column in enumerate(columns):
        unknown = sorted(set(frame[column].astype(str)).difference(AA3_TO_INDEX))
        if unknown:
            raise ValueError(f"Unknown residues in {column}: {unknown}")
        residue_indices = frame[column].map(AA3_TO_INDEX).to_numpy(dtype=np.int64)
        encoded[np.arange(len(frame)), site_index, residue_indices] = 1.0
    return encoded
