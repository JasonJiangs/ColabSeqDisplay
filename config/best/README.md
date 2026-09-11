# The hyperparameter registry

The committed LoRA hyperparameters. `colab/ColabSeqDisplay.ipynb` looks up the backbone you
picked in the form, reads the seven values from here, and offers no way to change them —
**no hyperparameter search ever runs in Colab.**

Two reasons, one practical and one scientific. A 40-trial Optuna study per backbone, plus a
re-evaluation of its best candidates over nine runs each, is dozens of full fine-tuning runs;
a Colab session is measured in hours and can be reclaimed mid-run. And a hyperparameter you
re-tune while watching your own validation score is a hyperparameter that has quietly eaten
your test set. The values are selected once, offline, and committed; what you get back from
your own library is then a measurement rather than a search result.

## What you are looking at

28 files, two per backbone, named `<model>_<pooling>.yaml`. The two halves are the two
pooling strategies the hyperparameter study compared, and they are why the directory is
twice the size of the list of backbones:

| filename half | count | status | reachable |
|---|---|---|---|
| `<model>_mutation_site_mean.yaml` | 14 | all `provisional` — placeholders | **yes**, this is the half `colabsd.bestconfig` reads |
| `<model>_cosine_p90_mean.yaml` | 14 | 10 `tuned`, 4 `provisional` | no |

`colabsd.bestconfig.load_best_config(model)` takes a backbone name and nothing else. It
always opens the `mutation_site_mean` entry, because the pipeline pools the embeddings at
the mutated sites and offers no alternative. Nothing in the package or the notebooks can
ask for a `cosine_p90_mean` entry, and `training.pooling` in a file is a label recording
which half of the study it belongs to, not a setting: an entry whose field and filename
disagree is refused rather than run under the wrong name.

**So every entry a run can reach today is a placeholder**, and the workflow says so before
training. `_meta.status` is `provisional` in all 14, `BestConfig.is_provisional` is True,
`describe()` prints `PROVISIONAL` in capitals, and the same flag follows the run into the
training log, `model_bundle.zip`, `performance_report.zip` (as `hyperparameters.status` in
`performance.json` and as a paragraph in that archive's `README.txt`), the unlock cell and
the Predict notebook. A placeholder is a real, runnable configuration — the median of the
10 tuned entries — but no study selected it, so nothing it produces is a result. The
one-hot floor drawn beside your model in the results step is what tells you whether a run
worked; a large model that ties a ridge regression on one-hot residues has told you the
landscape is additive.

**The 14 unreachable files stay on disk.** They are the written record of a real
hyperparameter study — the only place its selected trials, seeds and test scores appear —
and `ATTRIBUTION.md` points at them. Deleting them would delete the provenance of the
numbers the placeholders are the median of. Nothing reads them; they are here to be read by
a person.

## Where the tuned numbers came from

Not from this package. The 10 `tuned` entries are the seqdisplay-opt study's selected LoRA
configurations, measured on **its own SlugCas9 5NNK library, which is not redistributed
here** (see `../../ATTRIBUTION.md`). Each came from a 40-trial Optuna study on one split
seed; the three best validation configurations were re-trained over three split seeds ×
three model seeds, and the one with the highest mean validation Spearman across those nine
runs was kept. ρ below is that entry's mean ± sd **test** Spearman over the same nine runs,
as stored in its `_meta` block.

| backbone | offered | ρ, study measurement | Optuna trial |
|---|---|---|---|
| ESM2-650M | yes | 0.5636 ± 0.0128 | 21 |
| SaProt-650M | yes | 0.5592 ± 0.0110 | 38 |
| ProtT5-XL | no | 0.5583 ± 0.0105 | 32 |
| SaProt-1.3B | yes | 0.5578 ± 0.0142 | 33 |
| SaProt-35M | yes | 0.5576 ± 0.0162 | 18 |
| METL | no | 0.5574 ± 0.0187 | 22 |
| ESM2-150M | yes | 0.5546 ± 0.0105 | 30 |
| ESMC-300M | no | 0.5487 ± 0.0166 | 16 |
| ESM2-35M | yes | 0.5486 ± 0.0145 | 10 |
| ESMDance | no | 0.5478 ± 0.0092 | 18 |

The trial number is the one selected *after* re-evaluating the study's best candidates,
which is not always its single best validation trial.

**Two things those ρ are not.** They are one protein and one assay, not yours. And they were
selected with a read-out that averaged a discovered region of that protein — a region that
had to be recomputed for every new protein, and which this package no longer computes at
all, because pooling the mutated sites needs nothing computed first. How the values transfer
to the read-out a run actually uses has not been measured. Read the table as the provenance
of the placeholders, not as a prediction about your library.

Whole spread across the ten: 0.5478 to 0.5636, a range of 0.0158 — smaller than it looks
next to a single backbone's own ±0.0128 across nine runs. LoRA is fairly forgiving, which is
why a placeholder is a sensible starting point. What it is not is a measured one.

`ESM2-8M`, `Ankh-large`, `ESMC-600M` and `SeqDance` are provisional in both halves: the
study ran no LoRA search for them. ESM2-3B and ESM2-15B have no entries at all — they do not
fit the hardware this tool targets.

## File format

```yaml
_meta:                          # provenance; not read by the training code
  status: provisional           # or: tuned
  source: median of the 10 tuned cosine_p90_mean configs
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

## Filling a gap

What is missing is a tuned `mutation_site_mean` entry for any backbone — that is the whole
gap, and it is why all 14 reachable files are placeholders. Closing it means running the
search offline against your own library with the seqdisplay-opt command line
(`seqdisplay-lora-optuna` to run the study, `seqdisplay-lora-reevaluate` to retrain its best
candidates across split and model seeds), then copying the winning trial's values in with
`status: tuned` and a `source` that says where they came from. Keep the test partition out
of the search: select on validation only, and read the test set exactly once, at the end.
