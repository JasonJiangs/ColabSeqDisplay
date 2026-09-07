# The hyperparameter registry

One YAML file per `(backbone, pooling)` pair — 28 files, 14 backbones × 2 pooling
strategies. `colab/ColabSeqDisplay.ipynb` reads the seven LoRA hyperparameters for the pair you
picked from here, and offers no way to change them.

## Why a lookup instead of a search

Two reasons, one practical and one scientific.

A hyperparameter search is not something a Colab session can hold. The study behind the
tuned entries is 40 trials per backbone, plus a re-evaluation of its best candidates over
nine runs each — dozens of full fine-tuning runs. A free Colab session is measured in
hours and can be reclaimed mid-run.

More importantly, a hyperparameter you re-tune while watching your own validation score is
a hyperparameter that has quietly eaten your test set. The values are selected once,
offline, and committed; what you get back from your own library is then a measurement
rather than a search result.

## How the notebook chooses a file

Step 3 of `colab/ColabSeqDisplay.ipynb` asks for a **backbone** and a **pooling**. The pair maps
straight onto a filename in this directory:

```
<backbone>_<pooling>.yaml        e.g. ESM2-650M_cosine_p90_mean.yaml
```

Step 4 loads that file, prints its hyperparameters and its provenance, and passes it to
training unchanged. The two poolings the notebook offers are `cosine_p90_mean` (needs a
region file) and `mutation_site_mean` (needs none). If a pair has no file, the error names
the pairs that do exist, so nothing fails silently.

## `tuned` and `provisional` — and what the difference costs you

Every file declares its own status in its `_meta` block, and that status follows the run
all the way through: it is printed in step 3 and step 4 in capitals, recorded in the run
result, and written into the exported model bundle.

**`tuned` — 10 files.** The values come from a 40-trial Optuna search on the SlugCas9 5NNK
library. The study's top three trials were each retrained across nine runs (3 split seeds
× 3 model seeds) and the best of the three was kept. `_meta` records which trial won and
the test Spearman it reached.

**`provisional` — 18 files.** A placeholder: the median of the ten tuned configurations.
No search has been run for these pairs and no performance has been measured, so `_meta`
says `test_spearman_mean: unknown` and carries a `replace_with` line naming the command
that would fix it.

For your run, the practical difference is smaller than it sounds but real. LoRA is fairly
forgiving, and across the ten tuned pairs the whole spread of test Spearman is
0.5478–0.5636 — 0.0158 between best and worst. A provisional configuration is a sensible
starting point in that light. What it is not is a measured one: nothing stands behind it,
so read a result obtained with one as a lower bound rather than as this backbone's
ceiling. If a tuned pair will do for your question, prefer it.

Either way, the number that tells you whether the run worked is the one-hot baseline in
step 6, not the label on the configuration file. A large model that ties a ridge
regression on one-hot residues has told you the landscape is additive.

**One gap is worth planning around.** No `mutation_site_mean` pair has been tuned. A new
protein has no pooling region until you make one with `colab/ColabSeqDisplay_Prepare.ipynb`, and
without a region the workflow pools over the mutated sites — so a first run on your own
protein lands on a provisional configuration by default.

## Which pairs are which

Values are the mean and standard deviation of the test Spearman over the nine
re-evaluation runs, as stored in each file's `_meta`, rounded to four decimal places.

| backbone | `cosine_p90_mean` | `mutation_site_mean` |
|---|---|---|
| ESM2-8M | provisional | provisional |
| ESM2-35M | **tuned** — 0.5486 ± 0.0145 (trial 10) | provisional |
| ESM2-150M | **tuned** — 0.5546 ± 0.0105 (trial 30) | provisional |
| ESM2-650M | **tuned** — 0.5636 ± 0.0128 (trial 21) | provisional |
| SaProt-35M | **tuned** — 0.5576 ± 0.0162 (trial 18) | provisional |
| SaProt-650M | **tuned** — 0.5592 ± 0.0110 (trial 38) | provisional |
| SaProt-1.3B | **tuned** — 0.5578 ± 0.0142 (trial 33) | provisional |
| ProtT5-XL | **tuned** — 0.5583 ± 0.0105 (trial 32) | provisional |
| Ankh-large | provisional | provisional |
| ESMC-300M | **tuned** — 0.5487 ± 0.0166 (trial 16) | provisional |
| ESMC-600M | provisional | provisional |
| SeqDance | provisional | provisional |
| ESMDance | **tuned** — 0.5478 ± 0.0092 (trial 18) | provisional |
| METL | **tuned** — 0.5574 ± 0.0187 (trial 22) | provisional |

The trial number is the one selected *after* re-evaluating the study's best candidates,
which is not always the study's single best validation trial.

`METL` is the one tuned entry the notebook cannot offer. It is Rosetta-pretrained and
protein-specific, there are no public weights to download, and the backbone registry marks
it `local_only`; the file is kept because it is part of the published comparison. ESM2-3B
and ESM2-15B have no entries at all — they do not fit the hardware this tool targets.

## What every tuned entry assumes

**One protein, one assay.** Every number above is SlugCas9 5NNK: 16,424 variants, five
randomised sites, four PAM conditions, an 8:1:1 split. Treat a tuned configuration as a
good starting point for your library, not as a prediction about it.

**A slightly different p90 region.** The tuned `cosine_p90_mean` entries were selected
against a 106-position p90 list that shares 69 positions with the one shipped in
`examples/slugcas9_5nnk/region_p90.json` (see
[`../../examples/slugcas9_5nnk/README.md`](../../examples/slugcas9_5nnk/README.md)).
Whether the hyperparameters transfer exactly between the two has not been measured.

## File format

```yaml
_meta:                          # provenance; ignored by the training code
  status: tuned                 # or: provisional
  source: ...                   # where the values came from
  optuna_trial: 21              # tuned entries only
  test_spearman_mean: 0.5636    # `unknown` for a provisional entry
  test_spearman_sd: 0.0128
  n_reevaluation_runs: 9

model: ESM2-650M                # must match the filename

parameters:                     # the seven searched hyperparameters
  adapter_lr: 0.000193648444
  head_lr: 1.635795262e-05
  adapter_weight_decay: 0.001
  lora_rank: 32
  lora_alpha: 32
  lora_dropout: 0
  effective_batch_size: 16

training:                       # held fixed across the study
  tuning_method: lora
  head: mlp
  pooling: cosine_p90_mean      # must match the filename
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

## Adding or replacing an entry

The registry is read by filename, so a new file is picked up with no code change. Drop
`<backbone>_<pooling>.yaml` in this directory, with the blocks above.

Three rules the loader enforces:

- `model:` and `training.pooling:` must agree with the filename.
- `training.micro_batch_size` must be an integer, or the string `auto` — which the
  trainer resolves from `effective_batch_size`. All 28 shipped files set it explicitly.
- All seven `parameters:` keys must be present.

To fill one of the 18 gaps properly, run the search offline against your own library with
the `seqdisplay-opt` command line — `seqdisplay-lora-optuna` to run the study, then
`seqdisplay-lora-reevaluate` to retrain its best candidates across split and model seeds —
and copy the winning trial's values in, with `status: tuned` and a `source` that says
where they came from. Keep the test partition out of the search: the study should select
on validation only, and the test set should be read exactly once, at the end.
