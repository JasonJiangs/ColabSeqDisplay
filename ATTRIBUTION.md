# Attribution

**The science in this repository is not ours.** The LoRA implementation, the training
loop, the pooling, the metrics, the head registry, the split protocol and every tuned
hyperparameter in `config/best/` come from a separate research project:

> **SequenceDisplay-Workflow-Optimization** — Python package `seqdisplay_opt`.
> A research pipeline that compares protein language model representations for predicting
> sequence-display activity, evaluates frozen backbones from cached embeddings, and adapts
> selected models with LoRA on raw sequences.
> Repository: <https://github.com/JasonJiangs/SequenceDisplay-Workflow-Optimization>

`ColabSeqDisplay` is the notebook layer over that work: it loads backbones from
HuggingFace, builds variant sequences from a user's CSV, looks up a pre-tuned
configuration, orchestrates a run, reports it against a one-hot floor, and packs the
result into a `.zip`. It performs no method development. If you
publish a result produced with this tool, the method you are using is theirs — see
[Citation](#citation).

## Why the code is copied in

Until this release, `colabsd` imported `seqdisplay_opt` from a second checkout. That
package is not on PyPI and is not published, so the notebooks asked for two repository
locations and could not run for anyone outside the group that had both. The code the
notebooks actually reach is now vendored into [`colabsd/engine/`](colabsd/engine/), and
the notebooks name one repository.

Vendoring is a copy, not a rewrite. Roughly 2,100 lines across 15 modules were taken from
about 4,900 lines spanning 27 upstream modules; what came across is the part a notebook
reaches, and it is intended to compute exactly what the original computes. **Every module
in `colabsd/engine/` opens with a header naming the upstream module it derives from and
recording what was changed.** This file collects those headers; the headers are
authoritative for the details.

## What came from where

| Vendored module | Upstream origin | What was taken |
|---|---|---|
| `colabsd/engine/lora.py` | `seqdisplay_opt/finetuning/lora.py` | The whole module, verbatim: `LoRALinear`, `inject_lora`, `target_roles`, `validate_qkvo_coverage`, `lora_parameters`, `LORA_TARGET_MODULES` and the alias/matching helpers |
| `colabsd/engine/training.py` | `seqdisplay_opt/finetuning/training.py` | `train_epoch` — the loop that produced every tuned configuration, including its sample-weighted gradient accumulation — plus `batch_indices` and `as_float_tensor` |
| `colabsd/engine/adapters.py` | `seqdisplay_opt/finetuning/adapters.py` | The `SequenceAdapter` `Protocol` only. Upstream's concrete adapters load `fair-esm` checkpoints from a local model catalogue; `colabsd/backbones/` replaces them |
| `colabsd/engine/train_config.py` | `seqdisplay_opt/finetuning/lora_reevaluate.py`, `seqdisplay_opt/finetuning/lora_optuna.py` | `train_eval_config`, `load_lora_best_config`, `SourceTrial`, `LabelScaler`, the model-configuration, state-dict, prediction and split-evaluation helpers |
| `colabsd/engine/heads.py` | `seqdisplay_opt/models/heads.py`, `models/factory.py`, `utils/device.py` | `HEAD_REGISTRY`, `register_head`, `BaseHead`, `TorchHeadBase`, `MLPHead`, `RidgeHead`, `create_head`, `select_safe_device` |
| `colabsd/engine/metrics.py` | `seqdisplay_opt/metrics/evaluation.py`, `seqdisplay_opt/finetuning/validation.py` | `evaluate_predictions`, `precision_k`, `ndcg_k`, `mean_metric`, `TRACKED_METRICS`, `summarize_metrics`, `validation_log_row` |
| `colabsd/engine/pooling.py` | `seqdisplay_opt/pooling/{reducers,regions}.py` | `mean_reduce` and `select_positions`, verbatim. Upstream's strategy base class, registry, factory and five named poolings are not here: this pipeline pools one way, so `pool` composes the two reducers directly |
| `colabsd/engine/splits.py` | `seqdisplay_opt/data/splits.py`, `seqdisplay_opt/utils/torch_io.py` | `create_split`, `create_all_splits`, `load_split`, `create_nested_selection_split`; `load_torch` inlined as the private `_load_torch` |
| `colabsd/engine/protein_db.py` | `seqdisplay_opt/data/protein_database.py` | The record loader: `load_database` and `load_protein_record`. Upstream's `POOLING_REGIONS` table and its named-region resolvers are not here — a record lists the sites the library mutates, and there is no region to name |
| `colabsd/engine/config_space.py` | `seqdisplay_opt/config/optuna_space.py` | Three resolution helpers and `OBJECTIVE_METRICS`. The Optuna search space itself stayed upstream |
| `colabsd/engine/schema.py` | `seqdisplay_opt/config/schema.py` | `TrainingConfig` |
| `colabsd/engine/one_hot.py` | `seqdisplay_opt/baselines/one_hot.py` | The `AA3` vocabulary, `AA3_TO_INDEX`, and the per-site one-hot encoder |
| `colabsd/engine/loader.py` | `seqdisplay_opt/data/loader.py` | The FASTA reader, `load_single_fasta`, `load_foldseek_sequence` |
| `colabsd/engine/formats.py` | `seqdisplay_opt/models/backbones/{saprot,prott5,ankh,seqdance}.py` | Per-family input formatting only: the Foldseek vocabularies, SaProt AA/3Di interleaving, the ProtT5 and Ankh sequence formats, and the SeqDance model definition |
| `colabsd/engine/sequences.py` | `seqdisplay_opt/data/sequences.py` | `require_uniform_sequence_length` |

### What was deliberately left behind

Only what the notebooks reach was copied. Not here, and still upstream's: the Optuna study
machinery (study databases, trial pruning, the `fcntl` locking that lets two HPC workers
share a study), every `argparse` command line and console-script entry point, the
`fair-esm` and local-catalogue checkpoint loaders, the cached-embedding selection CLI, the
one-hot baseline's run harness and predictor registry, nine of upstream's eleven
regression heads — dropping those also drops a hard `xgboost` dependency — and the whole
pooling-strategy layer: the registry, the factory, the five named poolings and the
protein-region tables they resolve through.

## What we changed while copying, and why

Fidelity is the rule for `colabsd/engine/`: the tuned configurations in `config/best/`
were selected against upstream's exact code, so a silent behavioural drift would
invalidate them. Every departure below is deliberate, recorded in the module header, and
covered by a test.

1. **`micro_batch_size: auto` no longer raises in the config loader.** Upstream's
   `load_lora_best_config` defaults `training.micro_batch_size` to the string `"auto"` and
   then calls `int()` on it, so a best-config that omits the field dies with
   `ValueError: invalid literal for int() with base 10: 'auto'` — while `train_eval_config`
   resolves the very same `"auto"` correctly. Both now go through one helper,
   `resolve_micro_batch_size`, which applies the trainer's rule. This is a **bug fix**, not
   a behaviour change for anything that already worked: all 28 shipped configs set the
   field explicitly, and a 28-file field-for-field comparison shows none of them parses
   differently than it did upstream.
2. **Names we depend on are public.** They are this package's API now rather than another
   package's internals: `_predict` → `predict`, `_configure_model` → `configure_model`,
   `_device` → `default_device`, `_set_reproducible_seed` → `set_reproducible_seed`, the
   state-dict and split-evaluation helpers lost their underscores, `ankh._format_sequence`
   → `format_ankh_sequence`, `seqdance._make_seqdance_class` → `make_seqdance_class`,
   `_read_fasta_records` → `read_fasta_records`, and the `lru_cache`d protein database is
   reached through `load_database()` and cleared through `clear_database_cache()` instead
   of `_load_database.cache_clear`. Bodies are unchanged.
3. **`encode_five_site_one_hot` → `encode_site_one_hot`.** The upstream body already
   looped over its `columns` argument; only the name promised five sites. The five-site
   result is identical.
4. **The default target names are explicit.** `evaluate_predictions` names the first four
   unnamed targets after SlugCas9's four PAMs through a module-private table, which
   surprises anyone whose conditions are not PAMs. Same names, same order, now the public
   `DEFAULT_TARGET_NAMES` and `default_target_names()`.
5. **`RidgeHead`'s alpha grid is a constant, not an environment variable.** Upstream reads
   `PLM_RIDGE_N_ALPHAS`, an HPC sweep knob no notebook sets; it is now
   `RIDGE_N_ALPHAS = 17`, upstream's own default, so the fitted models are unchanged.
6. **The protein database has no packaged fallback.** Upstream falls back to a
   `proteins.yaml` shipped in its repository — SlugCas9's record. `colabsd` ships no such
   file and synthesizes one per run, so `database_path` is required rather than silently
   pooling a different protein, and a relative path resolves against the working directory
   rather than against upstream's checkout root.
7. **Imports are lazy where they were eager**, so that importing `colabsd.engine.pooling`
   or `colabsd.engine.formats` does not pull in `torch`. No behaviour depends on it.
8. **There is one pooling, so nothing dispatches on a name.** Upstream registers five
   `PoolingStrategy` classes and `pool(name, ...)` looks one up, each resolving a named
   region of a protein through `POOLING_REGIONS`. This pipeline averages the embeddings at
   the residues the library mutates and offers no alternative, so `pool(token_embs,
   positions)` composes `select_positions` and `mean_reduce` directly, a protein record
   lists `mutated_positions_1based` instead of a `regions` mapping, and
   `config_space.pooling_positions_0based` reads those positions rather than the pooling's
   name. `mutation_site_mean` survives as a *label* — the `config/best/` filename suffix
   and the string every artefact records — not as a choice.
9. **`load_lora_best_config` defaults nothing to upstream's own study.** Its
   `training.pooling` default is `POOLING_NAME` and its `protein:` default is empty, where
   upstream's are the strategy its study selected and the protein record packaged in its
   checkout. All 28 shipped entries set the pooling explicitly and none declares a protein,
   so no file on disk parses differently; what changes is that a config which omits the
   block is told so instead of being resolved against somebody else's protein.

Three upstream quirks were **kept on purpose**, because changing them would move a
guard or a number: `create_split` still caches on the output directory alone (`colabsd.data`
is the layer that refuses a stale cache); `train_epoch` keeps its empty-partition guard even
though the DataLoader raises first when shuffling; and `validate_qkvo_coverage` keeps its
existing message for a name list with no attention role. Each is noted in the module that
holds it.

## How we know the copy still computes the original

Equality is asserted, not assumed. `tests/test_engine_lora.py`,
`tests/test_engine_pooling.py`, `tests/test_engine_heads.py` and
`tests/test_engine_train_config.py` import both the vendored module and the original and
run them side by side on real input — the bundled 124-variant MG8 PETase library, all 28
shipped configs, a full LoRA fine-tune — asserting equal outputs rather than equal source.

Those comparisons need the research checkout. They locate it by importing `seqdisplay_opt`
and then by the `SEQDISPLAY_OPT_ROOT` environment variable, and **skip cleanly when neither
is available**, so the suite is green for a user who has only this repository. On a bare
clone every test passes and the skips are exactly these comparisons plus four that want
HuggingFace weights; with a checkout reachable, around sixty more tests run and only the
four weight-dependent ones skip. To run them:

```bash
SEQDISPLAY_OPT_ROOT=/path/to/SequenceDisplay-Workflow-Optimization python -m pytest tests/ -q
```

### The one thing that still needs the checkout

`scripts/search_best_config.py` — the offline Optuna search that *produces* a new
`config/best/` entry — was left composing with the research package rather than vendoring
it: the search space (`load_lora_optuna_config`, `suggest_lora_params`,
`DEFAULT_LORA_CONFIG`) and the per-trial training entry point (`train_one_trial`) stayed
upstream. It reaches them through one guarded helper, `_import_research`, which names the
checkout and the environment variable when it is missing rather than raising a bare
`ModuleNotFoundError`. It also stages the protein record in *both* shapes, because
upstream's own validator reaches the pooled residues through a `regions` mapping.

It is not part of the product. The notebooks, `colabsd`, the 28 shipped configs and the
test suite are all standalone; this one developer script is the documented exception.

## Data and hyperparameters

**`examples/mg8_petases/`** is not upstream's. It is a 124-variant engineered-PETase
library, derived here from the spreadsheet that ships beside it, and it replaced upstream's
much larger dataset as the bundled example so that a complete run fits inside one Colab
session. [`examples/mg8_petases/README.md`](examples/mg8_petases/README.md) records how the
table was built and what is known about it.

**`config/best/`** is entirely derived from upstream's study. The ten entries marked
`status: tuned` are its selected LoRA configurations, taken from
`results/p90_lora/selected_lora_configs.csv`, each the product of a 40-trial Optuna study
whose top three trials were re-trained across 3 split seeds × 3 model seeds. The other 18
are placeholders computed as the median of those ten, and say so at load time. The
`_meta` block of every file records which it is.

**Every performance number in this repository comes from that study.** The test Spearman
figures in `README.md`, in `config/best/README.md`, in each `_meta` block and in the
notebook's own dropdown — 0.5478 to 0.5636 on SlugCas9 5NNK — are upstream's
re-evaluation results, not measurements made here. Exactly one training run has ever gone
through `colabsd`'s own path end to end, and it is labelled as an illustration where it
appears. The one-hot floor numbers, and the single timed forward pass
`colabsd.ui.core.TIMED_PASS` records, are this repository's own.

## Licence

**The upstream project carries no licence file.** There is no `LICENSE`, `COPYING` or
licence declaration in its repository or its `pyproject.toml`, so the terms under which
its code may be copied, modified or redistributed are not stated anywhere. This is a
factual note about the state of that repository, not a legal opinion.

Anyone redistributing ColabSeqDisplay — publishing a fork, uploading it to PyPI, or
shipping it inside another product — should confirm terms with the authors of
SequenceDisplay-Workflow-Optimization first, because `colabsd/engine/` and `config/best/`
carry their work. This repository's own `LICENSE` covers the code written here; it cannot
grant permissions over code it did not originate.

## What is ours

So the boundary is clear, everything outside `colabsd/engine/` was written for this
project: the HuggingFace backbone adapters and registry (`colabsd/backbones/`), the
library specification and CSV handling (`colabsd/spec.py`, `colabsd/data.py`),
wild-type 3Di construction (`colabsd/structure.py`),
best-config lookup (`colabsd/bestconfig.py`), the run orchestration and locked-test
protocol (`colabsd/train.py`), the one-hot floor's harness (`colabsd/baseline.py`),
reporting (`colabsd/report.py`), the model bundle (`colabsd/bundle.py`), scoring
(`colabsd/predict.py`), the notebook wizards (`colabsd/ui/`), the notebook generator, the
tests and the documentation.

## Citation

Cite both. If you use ColabSeqDisplay, cite this software using
[`CITATION.cff`](CITATION.cff) — and cite SequenceDisplay-Workflow-Optimization, the study
whose method you ran and whose hyperparameters you used. It is listed as a reference in
that file, so a citation manager picks up both.
