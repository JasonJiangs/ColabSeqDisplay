"""Vendored fine-tuning engine — the science `colabsd` runs on.

Every module in this subpackage is derived from the research project
**SequenceDisplay-Workflow-Optimization** (Python package `seqdisplay_opt`),
which is not published on PyPI. The code lives here so that `colabsd` is a
standalone, installable package: the notebooks name one repository and run for
anyone. Each module names the upstream file it came from in its own header, and
`ATTRIBUTION.md` at the repository root collects those attributions.

The rule for this subpackage is fidelity: these modules must compute exactly
what upstream computes, because the tuned configurations in `config/best/` were
selected against upstream's code. Departures are deliberate, documented in the
module header, and covered by a side-by-side equivalence test.

Nothing is imported eagerly here — importing `colabsd.engine` must stay free of
`torch`. Import the module you need:

    from colabsd.engine.lora import inject_lora
    from colabsd.engine.splits import create_all_splits
    from colabsd.engine.training import train_epoch
"""

from __future__ import annotations

__all__: list[str] = []
