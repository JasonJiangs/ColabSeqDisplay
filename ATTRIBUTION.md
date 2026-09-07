# Attribution

**The science in this repository is not ours.** The LoRA implementation, the training
loop, the pooling strategies, the metrics, the head registry, the split protocol and every
tuned hyperparameter in `config/best/` come from a separate research project:

> **SequenceDisplay-Workflow-Optimization** — Python package `seqdisplay_opt`.
> A research pipeline that compares protein language model representations for predicting
> sequence-display activity, evaluates frozen backbones from cached embeddings, and adapts
> selected models with LoRA on raw sequences.
> Repository: <https://github.com/JasonJiangs/SequenceDisplay-Workflow-Optimization>

`ColabSeqDisplay` is the notebook layer over that work: it loads backbones from
HuggingFace, builds variant sequences from a user's CSV, discovers pooling regions,
looks up a pre-tuned configuration, orchestrates a run, reports it against a one-hot
floor, and packs the result into a `.zip`. It performs no method development. If you
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
| `colabsd/engine/pooling.py` | `seqdisplay_opt/pooling/{base,factory,reducers,regions,named_mean}.py` | The five modules flattened into one: `PoolingStrategy`, the registry, `pool`, `mean_reduce`, position/region selection and the five named mean-pooling strategies |
| `colabsd/engine/splits.py` | `seqdisplay_opt/data/splits.py`, `seqdisplay_opt/utils/torch_io.py` | `create_split`, `create_all_splits`, `load_split`, `create_nested_selection_split`; `load_torch` inlined as the private `_load_torch` |
| `colabsd/engine/protein_db.py` | `seqdisplay_opt/data/protein_database.py` | The region-resolution half: `POOLING_REGIONS`, `load_protein_record`, `resolve_region_positions_0based`, `resolve_pooling_positions_0based` |
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
one-hot baseline's run harness and predictor registry, and nine of upstream's eleven
regression heads — dropping those also drops a hard `xgboost` dependency.

## What we changed while copying, and why

Fidelity is the rule for `colabsd/engine/`: the tuned configurations in `config/best/`
were selected against upstream's exact code, so a silent behavioural drift would
invalidate them. Every departure below is deliberate, recorded in the module header, and
checked against the original.

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

Three upstream quirks were **kept on purpose**, because changing them would move a
guard or a number: `create_split` still caches on the output directory alone (`colabsd.data`
is the layer that refuses a stale cache); `train_epoch` keeps its empty-partition guard even
though the DataLoader raises first when shuffling; and `validate_qkvo_coverage` keeps its
existing message for a name list with no attention role. Each is noted in the module that
holds it.

## How we know the copy still computes the original

Equality was asserted by execution, not assumed by reading. For every module that carries
a number — `lora.py`, `pooling.py`, `heads.py`, `train_config.py` — the vendored copy and
the original were driven side by side on real input (the bundled 16,424-variant SlugCas9
library, all 28 shipped configs, a full LoRA fine-tune) and their outputs compared, rather
than their source.

Three further checks close the gaps that argument leaves:

- `create_split` is pinned to the study's *recorded* index lists — the artefacts
  `config/best/` was selected on — so the check holds even if the original's source later
  changes.
- `inject_lora` is run against real HuggingFace attention naming (`EsmModel`,
  `T5EncoderModel`, built from a small config with no weights and no network) rather than
  a toy module tree.
- Every shared definition is compared at AST level, with docstrings, annotations,
  formatting and the deliberate renames normalised away, so an undocumented edit to a body
  here is caught even where no behavioural check reaches it. The departures listed above
  are its allow-list: one that is not recorded fails, and so does one left listed after it
  has been reverted.

Those comparisons need a copy of the original alongside this repository. Nothing a user
does requires one: `colabsd`, the notebooks, the 28 shipped configs and the bundled
example are standalone, and this repository installs and runs with no second checkout
anywhere on the machine.

### What is not here

The machinery that *produces* a new `config/best/` entry stayed with the study. Filling
one of the 18 placeholder pairs means running that Optuna search offline, against the
research code, and copying the winning trial's values in — `config/best/README.md`
describes the shape of the file that results. This repository consumes tuned
configurations; it does not search for them.

## Data and hyperparameters

**`examples/slugcas9_5nnk/`** is upstream's SlugCas9 5NNK dataset, rearranged into the
shape `colabsd` expects. `library.csv`, `wt.fasta` and `wt_3di.txt` are byte-identical to
upstream's `data/processed/5nnk_avg_mut_num.csv`, `data/protein/slugcas9_wt.fasta` and
`data/protein/slugcas9_wt_3di.txt`. The two `region_*.json` files are the exception: they
are recomputed here and differ from upstream's published region. The two agree on 69 of
106 positions at p90 and 48 of 53 at p95; what that costs, and what it does not, is set
out in [`README.md`](README.md#limitations) and
[`examples/slugcas9_5nnk/README.md`](examples/slugcas9_5nnk/README.md#the-two-region-files).

**`config/best/`** is entirely derived from upstream's study. The ten entries marked
`status: tuned` are its selected LoRA configurations, taken from that project's own
`results/p90_lora/selected_lora_configs.csv`, each the product of a 40-trial Optuna study
whose top three trials were re-trained across 3 split seeds × 3 model seeds. The other 18
are placeholders computed as the median of those ten, and say so at load time. The
`_meta` block of every file records which it is.

**Every performance number in this repository comes from that study.** The test Spearman
figures in `README.md`, in `config/best/README.md`, in each `_meta` block and in the
notebook's own dropdown — 0.5478 to 0.5636 on SlugCas9 5NNK — are upstream's
re-evaluation results, not measurements made here. Exactly one training run has ever gone
through `colabsd`'s own path end to end, and it is labelled as an illustration where it
appears. The one-hot floor numbers quoted in `TUTORIAL.md` are the exception: they are
this repository's own, produced by running `colabsd.baseline` over the bundled library.

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
library specification and CSV handling (`colabsd/spec.py`, `colabsd/data.py`), region
discovery (`colabsd/region.py`), wild-type 3Di construction (`colabsd/structure.py`),
best-config lookup (`colabsd/bestconfig.py`), the run orchestration and locked-test
protocol (`colabsd/train.py`), the one-hot floor's harness (`colabsd/baseline.py`),
reporting (`colabsd/report.py`), the model bundle (`colabsd/bundle.py`), scoring
(`colabsd/predict.py`), the notebook wizards (`colabsd/ui/`), the three notebooks and
the documentation.

## Citation

Cite both. If you use ColabSeqDisplay, cite this software using
[`CITATION.cff`](CITATION.cff) — and cite SequenceDisplay-Workflow-Optimization, the study
whose method you ran and whose hyperparameters you used. It is listed as a reference in
that file, so a citation manager picks up both.
