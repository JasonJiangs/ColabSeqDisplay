"""ColabSeqDisplay — a no-code Colab facade over a protein-language-model workflow.

`colabsd.engine` owns the science: LoRA injection, the training loop, the mean of
the embeddings at the mutated sites, metrics and splits. It is vendored from the
SequenceDisplay-Workflow-Optimization research code (see ATTRIBUTION.md and the
header of each engine module). The rest of this package owns everything that makes
it runnable from a notebook: HuggingFace backbone loading, building sequences from
a user's CSV, WT 3Di construction, best-config lookup, orchestration, reporting and
model bundles.

The package is standalone: importing it pulls in no external research checkout
and touches no network.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def _data_root() -> Path:
    """Where `config/` and `examples/` sit.

    Two layouts, one answer. In a source tree (and in the `pip install -e` the notebooks
    do) they sit *beside* the package, at the repository root. In a built wheel they are
    packaged *inside* it — `pyproject.toml` maps them to `colabsd.config` and
    `colabsd.examples` — because a wheel cannot carry files from outside a package
    directory, and an install that leaves them behind gives you a `colabsd` whose config
    registry is empty and whose bundled example is missing.
    """
    return PACKAGE_ROOT if (PACKAGE_ROOT / "config" / "best").is_dir() else REPO_ROOT


DATA_ROOT = _data_root()
CONFIG_ROOT = DATA_ROOT / "config"
BEST_CONFIG_ROOT = CONFIG_ROOT / "best"
EXAMPLES_ROOT = DATA_ROOT / "examples"
