# `colabsd/engine/` — vendored, not authored here

**This directory is a copy of someone else's research code.** It derives from
**SequenceDisplay-Workflow-Optimization** (Python package `seqdisplay_opt`), which owns
the method `colabsd` runs: LoRA injection, the training loop, pooling, the metrics, the
head registry and the split protocol. Every module here opens with a header naming the
upstream file it came from and what was changed on the way in.
[`ATTRIBUTION.md`](../../ATTRIBUTION.md) collects those headers, and records that the
upstream project ships no licence file.

It lives here so that ColabSeqDisplay is one installable package. Before this, `colabsd`
imported a second, unpublished repository, the notebooks asked for two locations, and the
tool could not run for anyone outside the group that had both.

## The rule for this directory: fidelity

The ten tuned configurations in `config/best/` were selected by running *upstream's* code.
If this copy drifts, those hyperparameters — and every benchmark number in this repository,
all of which come from that study — quietly stop describing what runs here. Nothing
about that failure is visible in the output.

So:

- **Do not redesign, refactor for taste, or "clean up" a module here.** Awkwardness that
  is upstream's is preserved on purpose. Several quirks are kept deliberately and say so
  in their own module — `create_split` caching on the output directory alone, an
  unreachable guard in `train_epoch`, a coverage-check message that cannot be reached from
  `inject_lora`.
- **A deliberate departure needs three things**: a note in the module header, a
  side-by-side test against the original, and an entry in `ATTRIBUTION.md`.
- **Fix bugs upstream-first where you can.** What is fixed here (the `micro_batch_size:
  auto` crash in the config loader) is listed in `ATTRIBUTION.md` and is worth sending back.
- **New behaviour does not belong here.** Everything that makes this runnable from a
  notebook — HuggingFace loading, CSV handling, region discovery, best-config lookup,
  orchestration, reporting, bundles — lives outside this directory and is ours to change
  freely.

## How the copy is held to the original

Equality was established by running the two side by side, not by reading them. For the
modules that carry a number — `lora.py`, `pooling.py`, `heads.py`, `train_config.py` —
the vendored copy and the original were driven on real input (the bundled
16,424-variant SlugCas9 library, all 28 shipped configs, a full LoRA fine-tune) and their
outputs compared, rather than their source.

Three further checks close the gaps that leaves. `create_split` is pinned to the
*recorded* index lists the study selected `config/best/` on, so it holds even if the
original changes later. `inject_lora` is exercised against real HuggingFace attention
naming (`EsmModel`, `T5EncoderModel`, built from a small config with no weights and no
network) rather than a toy module tree. And every shared definition is compared at AST
level, with docstrings, annotations, formatting and the deliberate renames normalised
away, so an undocumented edit to a body here is caught even where no behavioural check
reaches it — with the departures `ATTRIBUTION.md` records as the allow-list.

Anything that changes a module in this directory has to be put back through those
comparisons, which need a copy of the original to compare against.

## Layout

Nothing is imported eagerly: `import colabsd.engine` must stay free of `torch`. Import the
module you need.

| module | what it holds |
|---|---|
| `lora.py` | `LoRALinear`, `inject_lora`, `target_roles`, `validate_qkvo_coverage`, `lora_parameters` |
| `training.py` | `train_epoch` and its batching helpers — the loop the tuned configs were selected on |
| `train_config.py` | `train_eval_config` (one fine-tune + evaluation) and `load_lora_best_config` |
| `heads.py` | the head registry, `MLPHead`, `RidgeHead`, `create_head`, `select_safe_device` |
| `metrics.py` | `evaluate_predictions` and the validation-log helpers |
| `pooling.py` | the pooling registry and the five named mean-pooling strategies |
| `splits.py` | the reproducible 8:1:1 split protocol |
| `protein_db.py` | named residue-region lookup |
| `one_hot.py` | the amino-acid vocabulary and per-site one-hot encoder behind the floor |
| `formats.py` | per-family input formatting (SaProt 3Di interleaving, ProtT5, Ankh, SeqDance) |
| `adapters.py`, `schema.py`, `config_space.py`, `loader.py`, `sequences.py` | the protocol, config objects and small helpers the above depend on |
