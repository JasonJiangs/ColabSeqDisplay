# The hyperparameter registry

The committed LoRA hyperparameters. `colab/ColabSeqDisplay.ipynb` looks up the backbone you
picked in the form, reads the seven values from here, and offers no way to change them —
**no hyperparameter search ever runs in Colab.**

The values are selected once, offline, and committed.

## What you are looking at

28 files, two per backbone, named `<model>_<pooling>.yaml`. The two halves are the two
pooling strategies the hyperparameter study compared, and they are why the directory is
twice the size of the list of backbones:

| filename half | count | status | reachable |
|---|---|---|---|
| `<model>_mutation_site_mean.yaml` | 14 | 10 `tuned`, 4 `provisional` | **yes**, this is the half `colabsd.bestconfig` reads |
| `<model>_cosine_p90_mean.yaml` | 14 | 10 `tuned`, 4 `provisional` | no |

`colabsd.bestconfig.load_best_config(model)` takes a backbone name and nothing else. It
always opens the `mutation_site_mean` entry, because the pipeline pools the embeddings at
the mutated sites and offers no alternative. Nothing in the package or the notebooks can
ask for a `cosine_p90_mean` entry, and `training.pooling` in a file is a label recording
which half of the study it belongs to, not a setting: an entry whose field and filename
disagree is refused rather than run under the wrong name.

**Ten of the 14 reachable entries are tuned**, and four are placeholders: `Ankh-large`,
`ESM2-8M`, `ESMC-600M` and `SeqDance`, for which the study ran no `mutation_site_mean` search.
For those four `_meta.status` is `provisional`, `BestConfig.is_provisional` is True, `describe()`
prints `PROVISIONAL` in capitals, and the same flag follows the run into the training log,
`model_bundle.zip`, `performance_report.zip` (as `hyperparameters.status` in `performance.json`
and as a paragraph in that archive's `README.txt`), the unlock cell and the Predict notebook. A
placeholder is a real, runnable configuration — built from the ten tuned entries — but no study
selected it, so nothing it produces is a result.

**The 14 `cosine_p90_mean` files stay on disk.** They are the written record of the study's other
half — the only place its selected trials, seeds and test scores appear — and `ATTRIBUTION.md`
points at them. Nothing reads them; they are here to be read by a person.

## Where the tuned numbers came from

Not from this package. The 10 `tuned` entries are the seqdisplay-opt study's selected LoRA
configurations for **this pooling**, measured on **its own SlugCas9 5NNK library, which is not
redistributed here** (see `../../ATTRIBUTION.md`). Each came from a 40-trial Optuna study on one
split seed; the selected trial was re-trained over three split seeds × three model seeds. ρ below
is that entry's mean ± sd **test** Spearman over those nine runs, as stored in its `_meta` block,
and `_meta.source_file` names the upstream file each parameter block was copied from.

| backbone | offered | ρ, study measurement | Optuna trial |
|---|---|---|---|
| ESM2-150M | yes | 0.5636 ± 0.0113 | 27 |
| ESM2-650M | yes | 0.5618 ± 0.0125 | 39 |
| ESMC-300M | yes | 0.5611 ± 0.0107 | 35 |
| ESM2-35M | yes | 0.5608 ± 0.0120 | 37 |
| ProtT5-XL | yes | 0.5581 ± 0.0162 | 21 |
| SaProt-650M | yes | 0.5580 ± 0.0085 | 2 |
| ESMDance | yes | 0.5569 ± 0.0128 | 28 |
| SaProt-1.3B | yes | 0.5559 ± 0.0162 | 12 |
| SaProt-35M | yes | 0.5555 ± 0.0126 | 2 |
| METL | no | 0.5482 ± 0.0158 | 12 |

**Two things those ρ are not.** They are one protein and one assay, not yours. And they are a
measurement of the *study's* library, not a prediction about the transfer to yours. What they are
no longer is a number from a different read-out: this half of the study pooled the mutated sites,
which is exactly what a run here does.

Whole spread across the ten: 0.5482 to 0.5636, a range of 0.0154 — comparable to a single
backbone's own ±0.0162 across nine runs. LoRA is fairly forgiving, which is why the median of
these is a sensible starting point for a backbone the study never searched. What it is not is a
measured one.

## File format

```yaml
_meta:                          # provenance; not read by the training code
  status: provisional           # or: tuned
  source: the 10 tuned mutation_site_mean configs
  replace_with: output of seqdisplay-lora-optuna for this model/pooling
  test_spearman_mean: unknown   # tuned entries carry the number, its sd,
                                # optuna_trial and n_reevaluation_runs instead

model: ESM2-650M                # must match the filename

parameters:                     # the seven searched hyperparameters
  adapter_lr: 0.00017
  head_lr: 5.5e-05
  adapter_weight_decay: 0.001
  lora_rank: 32
  lora_alpha: 128
  lora_dropout: 0.1
  effective_batch_size: 16

training:                       # held fixed across the study
  tuning_method: lora
  head: mlp
  pooling: mutation_site_mean   # must match the filename
  micro_batch_size: 8
  max_epochs: 20
  early_stopping_patience: 3
  max_grad_norm: 1.0
  head_weight_decay: 0.01
  loss: mse
  gradient_accumulation: derive_from_effective_batch_size
  label_preprocessing: train_split_zscore
  evaluate_test_during_optimization: false

evaluation:
  selection_objective: mean_validation_spearman
  split_seeds: [1, 2, 3]
  model_seeds: [11, 22, 33]
```

The `evaluation` block is not decoration: the notebook reads `split_seeds` and `model_seeds`
from it to set the ceilings on its own seed sliders, so the protocol an entry was selected
under is the protocol the form lets you repeat. Both lists are copied into
`performance_report.zip`.

Three rules the loader enforces: `model:` and `training.pooling:` must agree with the
filename; `training.micro_batch_size` must be an integer or the string `auto`, which the
trainer resolves from `effective_batch_size` (all 28 shipped files set it explicitly); and
all seven `parameters:` keys must be present. The registry is read by filename, so a
replacement entry is picked up with no code change.

**What the form lets you move.** All seven `parameters:` keys — the two learning rates, the
three LoRA settings, the weight decay and `effective_batch_size` — are shown read-only. The
panel prefills three editable boxes instead (`colabsd.ui.main_workflow.BUDGET_KEYS`):
`training.max_epochs`, `training.early_stopping_patience` and `training.micro_batch_size`.
A run records the values it actually used and which of them the user typed, in the training
log, `model_bundle.zip` and `performance_report.zip`; untouched fields are recorded as
looked up. `colabsd.bestconfig.BudgetOverrides` rejects any key outside
`BUDGET_FIELDS` by name.

## Filling a gap

What is missing is a tuned `mutation_site_mean` entry for `Ankh-large`, `ESM2-8M`, `ESMC-600M`
and `SeqDance`, the four backbones the study did not search. **None of the nine the notebook
offers is in that list** — a placeholder entry is what keeps a backbone off the form, and
`ESM2-8M` was the last exception to that rule. Closing a gap means running the
search offline against your own library with the seqdisplay-opt command line
(`seqdisplay-lora-optuna` to run the study, `seqdisplay-lora-reevaluate` to retrain its best
candidates across split and model seeds), then copying the winning trial's values in with
`status: tuned` and a `source` that says where they came from. Keep the test partition out
of the search: select on validation only, and read the test set exactly once, at the end.
