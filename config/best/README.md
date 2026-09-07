# The hyperparameter registry

One YAML file per `(backbone, pooling)` pair — 28 files, 14 backbones × 2 pooling
strategies. `colab/ColabSeqDisplay.ipynb` reads the seven LoRA hyperparameters for the pair
you picked from here, and offers no way to change them.

**What is here and what the notebook can select are not the same list.** All 28 files ship
and all 28 are loadable from Python. The notebook offers two backbone families — ESM2 and
SaProt, seven backbones — so **14 of the 28 files are reachable from the form, and six of
those fourteen are tuned.** The other fourteen belong to the seven backbones that are not on
the dropdown — six of which still have a working adapter in the package, `METL` never having
had one; `README.md` in the repository root says why for each one. Nothing here was deleted
to shorten a list.

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

Step 2 of `colab/ColabSeqDisplay.ipynb` asks for a **backbone** and a **pooling** — the two
questions the whole notebook is derived from, asked once. The pair maps straight onto a
filename in this directory:

```
<backbone>_<pooling>.yaml        e.g. ESM2-650M_cosine_p90_mean.yaml
```

The hyperparameters step loads that file, prints its values and its provenance, and passes
it to training unchanged. It is step 4 when the pair calls for something to be prepared
first — a wild-type 3Di string for a SaProt backbone, a pooling region for
`cosine_p90_mean` — and step 3 when it does not, because the step numbers close up around a
step that is not needed.

The two poolings the notebook offers are `cosine_p90_mean` (needs a pooling region, which
the notebook discovers or reuses in the preparation step) and `mutation_site_mean` (needs
nothing). Those two are what it offers precisely because they are the two this directory has
entries for. If a pair has no file, the error names the pairs that do exist, so nothing
fails silently.

## `tuned` and `provisional` — and what the difference costs you

Every file declares its own status in its `_meta` block, and that status follows the run
all the way through: it is printed in capitals when you pick the pair and again beside the
hyperparameters, it is repeated after training, it is written into `model_bundle.zip`, and
it is written into `performance_report.zip` — in `performance.json` as
`hyperparameters.status` and in that archive's `README.txt` as a paragraph, so a reader six
months later does not have to remember.

**`tuned` — 10 files.** The values come from a 40-trial Optuna search on the SlugCas9 5NNK
library. The study's top three trials were each retrained across nine runs (3 split seeds
× 3 model seeds) and the best of the three was kept. `_meta` records which trial won and
the test Spearman it reached. Six of the ten are reachable from the notebook.

**`provisional` — 18 files.** A placeholder: the median of the ten tuned configurations.
No search has been run for these pairs and no performance has been measured, so `_meta`
says `test_spearman_mean: unknown` and carries a `replace_with` line naming the command
that would fix it. Eight of the eighteen are reachable from the notebook.

For your run, the practical difference is smaller than it sounds but real. LoRA is fairly
forgiving, and across the ten tuned pairs the whole spread of test Spearman is
0.5478–0.5636 — 0.0158 between best and worst; across the six the notebook can reach it is
0.5486–0.5636, a spread of 0.0150. A provisional configuration is a sensible starting point
in that light. What it is not is a measured one: nothing stands behind it, so read a result
obtained with one as a lower bound rather than as this backbone's ceiling. If a tuned pair
will do for your question, prefer it.

Either way, the number that tells you whether the run worked is the one-hot baseline the
results step draws beside your model, not the label on the configuration file. A large model
that ties a ridge regression on one-hot residues has told you the landscape is additive.

**One gap is worth planning around.** No `mutation_site_mean` pair has been tuned — not one
of the fourteen. A new protein has no pooling region until the notebook discovers one for
it, and without a region the workflow pools over the mutated sites, so a first run on your
own protein lands on a provisional configuration by default. The way out is inside the same
notebook: choosing `cosine_p90_mean` makes a preparation step appear that discovers a region
from your own library, with its runtime estimate on screen before you commit to it. That is
the real trade — a tuned configuration that costs a discovery run first, or a placeholder
that starts immediately.

## Which pairs are which

Values are the mean and standard deviation of the test Spearman over the nine
re-evaluation runs, as stored in each file's `_meta`, rounded to four decimal places.
**Offered** says whether the notebook puts that backbone on its dropdown.

| backbone | offered | `cosine_p90_mean` | `mutation_site_mean` |
|---|---|---|---|
| ESM2-8M | yes | provisional | provisional |
| ESM2-35M | yes | **tuned** — 0.5486 ± 0.0145 (trial 10) | provisional |
| ESM2-150M | yes | **tuned** — 0.5546 ± 0.0105 (trial 30) | provisional |
| ESM2-650M | yes | **tuned** — 0.5636 ± 0.0128 (trial 21) | provisional |
| SaProt-35M | yes | **tuned** — 0.5576 ± 0.0162 (trial 18) | provisional |
| SaProt-650M | yes | **tuned** — 0.5592 ± 0.0110 (trial 38) | provisional |
| SaProt-1.3B | yes | **tuned** — 0.5578 ± 0.0142 (trial 33) | provisional |
| ProtT5-XL | no | **tuned** — 0.5583 ± 0.0105 (trial 32) | provisional |
| Ankh-large | no | provisional | provisional |
| ESMC-300M | no | **tuned** — 0.5487 ± 0.0166 (trial 16) | provisional |
| ESMC-600M | no | provisional | provisional |
| SeqDance | no | provisional | provisional |
| ESMDance | no | **tuned** — 0.5478 ± 0.0092 (trial 18) | provisional |
| METL | no | **tuned** — 0.5574 ± 0.0187 (trial 22) | provisional |

That is: **six tuned entries the notebook can select** — `ESM2-35M`, `ESM2-150M`,
`ESM2-650M`, `SaProt-35M`, `SaProt-650M` and `SaProt-1.3B`, every one of them
`cosine_p90_mean` — and eight placeholders beside them.

The trial number is the one selected *after* re-evaluating the study's best candidates,
which is not always the study's single best validation trial.

**Four tuned entries are not on the dropdown, and they are kept anyway.** `ProtT5-XL`,
`ESMC-300M` and `ESMDance` have working adapters that `colabsd.backbones.registry.create_adapter`
still builds; they are withheld from the notebooks because they need hardware or an extra
install the notebooks do not assume, or because their pooled feature is not read like the
others. `METL` has no adapter at all: it is Rosetta-pretrained and protein-specific, with no
public weights to download. All four files are kept because they are part of the published
comparison, and because re-offering a family is one line in
`colabsd/backbones/registry.py` — at which point its files here are already in place.
ESM2-3B and ESM2-15B have no entries at all: they do not fit the hardware this tool targets.

## What every tuned entry assumes

**One protein, one assay.** Every number above is SlugCas9 5NNK: 16,424 variants, five
randomised sites, four PAM conditions, an 8:1:1 split. Treat a tuned configuration as a
good starting point for your library, not as a prediction about it.

**A slightly different p90 region.** The tuned `cosine_p90_mean` entries were selected
against a 106-position p90 list that shares 69 positions with the one shipped in
`examples/slugcas9_5nnk/region_p90.json` (see
[`../../examples/slugcas9_5nnk/README.md`](../../examples/slugcas9_5nnk/README.md)).
Whether the hyperparameters transfer exactly between the two has not been measured — and
they will transfer less exactly still to a region discovered from *your* library, which is
what the notebook will pool over when you run your own protein.

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

The `evaluation` block is not decoration: the notebook reads `split_seeds` and `model_seeds`
from it to set the ceilings on its own seed sliders, so the protocol a pair was selected
under is the protocol the form lets you repeat.

## Adding or replacing an entry

The registry is read by filename, so a new file is picked up with no code change. Drop
`<backbone>_<pooling>.yaml` in this directory, with the blocks above.

Three rules the loader enforces:

- `model:` and `training.pooling:` must agree with the filename.
- `training.micro_batch_size` must be an integer, or the string `auto` — which the
  trainer resolves from `effective_batch_size`. All 28 shipped files set it explicitly.
- All seven `parameters:` keys must be present.

A file for a backbone the notebook does not offer is still read by `colabsd.bestconfig` and
still usable from Python; putting it on the form as well takes one more edit, in
`OFFERED_FAMILIES` in `colabsd/backbones/registry.py`.

To fill one of the 18 gaps properly, run the search offline against your own library with
the `seqdisplay-opt` command line — `seqdisplay-lora-optuna` to run the study, then
`seqdisplay-lora-reevaluate` to retrain its best candidates across split and model seeds —
and copy the winning trial's values in, with `status: tuned` and a `source` that says
where they came from. Keep the test partition out of the search: the study should select
on validation only, and the test set should be read exactly once, at the end.
