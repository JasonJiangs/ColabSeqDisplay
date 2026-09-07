# ColabSeqDisplay tutorial

From opening a link to holding a ranked list of candidate variants.

This page walks every cell of the three notebooks. **Part 1** runs the bundled
SlugCas9 5NNK example end to end, with the real form fields, the real defaults and the
real output. **Part 2** points the same workflow at your own protein. **Part 3** scores
new variants from a model you exported months earlier. A troubleshooting section at the
end is built from the failures the code actually raises.

You never edit code. Every input is a form field on a Colab cell.

**Contents**

- [Before you open anything](#before-you-open-anything) — including [what has and has not been shown](#what-this-tool-has-been-shown-to-do--and-what-it-has-not)
- [Part 1 — the bundled example, end to end](#part-1--the-bundled-example-end-to-end)
  - [Cell 0 · Setup](#cell-0--setup) · [1 · Load](#cell-1--load-the-variant-library) ·
    [2 · 3Di](#cell-2--wild-type-3di) · [3 · Backbone and pooling](#cell-3--backbone-and-pooling) ·
    [4 · Hyperparameters](#cell-4--the-hyperparameters) · [5 · Fine-tune](#cell-5--fine-tune) ·
    [6 · One-hot floor](#cell-6--the-one-hot-floor) ·
    [**7 · Unlock the test set**](#cell-7--unlock-the-test-set) ·
    [8 · Report](#cell-8--report) · [9 · Export](#cell-9--export-the-model) ·
    [10 · Score](#cell-10--score-a-batch-of-variants)
- [Part 2 — your own protein](#part-2--your-own-protein)
- [Part 3 — scoring new variants](#part-3--scoring-new-variants-with-colabcolabseqdisplay_predictipynb)
- [Troubleshooting](#troubleshooting)
- [What ends up in the working directory](#what-ends-up-in-the-working-directory)
- [If you report a result](#if-you-report-a-result)

---

## Before you open anything

### What this tool has been shown to do — and what it has not

Worth knowing before you spend an hour of GPU time. None of it is hidden further down.

- **One benchmark protein.** Every performance number on this page and in
  [`README.md`](README.md) comes from the bundled SlugCas9 5NNK library. The workflow takes
  any library of the same shape, but no second protein has been run through it.
- **10 of the 28 hyperparameter entries are tuned; 18 are placeholders** — including every
  `mutation_site_mean` pair, which is where a new protein starts.
  [Reading the PROVISIONAL warning](#reading-the-provisional-warning) says what to do when
  you land on one.
- **`ESM2-8M` is the only backbone that has been fine-tuned end to end through this
  package.** For eleven of the thirteen selectable backbones the published checkpoint has
  been loaded and one pooled forward pass checked; for `ProtT5-XL` and `Ankh-large` only the
  tokenizers have been exercised. Any other training path can still meet a problem nobody
  has met yet.
- **Every T4 minute quoted here is the registry's own estimate**, not a timing of a Colab
  session. The only measured number on this page is the CPU one-hot floor in cell 6, and it
  says so where it appears.
- **The engine is a copy of somebody else's code.** The training loop, the LoRA
  injection, the pooling and the metrics live in `colabsd/engine/`, copied from the
  seqdisplay-opt study rather than written here, so this repository runs on its own — see
  [the install note](#one-thing-to-know-about-the-install) below. What that copy does and
  does not include is set out in [`ATTRIBUTION.md`](ATTRIBUTION.md).

`README.md` carries the full [Limitations](README.md#limitations) list and a
[Reproducibility](README.md#reproducibility) section with the seeds, the split sizes, the
dependency bounds and the hardware behind the benchmark numbers.

### The three notebooks

| notebook | when you run it | needs a GPU |
|---|---|---|
| `colab/ColabSeqDisplay.ipynb` | the main workflow: load → backbone → LoRA → report → export | yes, for the training step |
| `colab/ColabSeqDisplay_Prepare.ipynb` | once per **new** protein, and only if you need a wild-type 3Di string or a pooling region | yes, for region discovery and ESMFold |
| `colab/ColabSeqDisplay_Predict.ipynb` | any time after, to score variants with an exported bundle | faster with one, works without |

For the bundled example you need only the first. Open it from the Colab badge in
[`README.md`](README.md), or in Colab under **File ▸ Open notebook ▸ GitHub**.

### Set the runtime first

**Runtime ▸ Change runtime type ▸ T4 GPU**, before you run anything. A free T4 is enough
for the whole of Part 1.

Without a GPU the notebook still loads and checks your library (cells 1–4), runs the
one-hot floor (cell 6) and draws a report (cell 8). Cell 5 stops immediately with a
message telling you to switch the runtime — it does not fail halfway through a training
loop — and cells 7, 9 and 10 stop the same way, because there is nothing trained to
unlock, export or score with.

### One thing to know about the install

The setup cell installs one thing: this repository. The training code — LoRA injection,
the training loop, the pooling, the metrics and the data splits — ships inside it as
`colabsd/engine/`, so there is no second package to find and nothing else to configure.

The cell has a single field, **`colabsd_repository`**, and it already points at the public
repository. It takes either a git URL, which is cloned shallowly beside the notebook, or
the path of a folder you have already uploaded to the runtime — that is how you run a
branch or a local edit. Either way it is installed in place with `pip install -e`, which
keeps the repository's own `config/best/` registry and its bundled example readable and
editable in the runtime.

If the address cannot be reached, the cell raises there and quotes what `git` or `pip`
actually said, followed by a line naming the field to change. Nothing below it can run
until the install succeeds.

### Where files land

Everything the notebooks write goes to a `colabsd_work/` folder inside the runtime's
working directory — `/content/colabsd_work/` in Colab, and the setup cell prints the path it
actually used — and
every file is also pushed to your browser's downloads as it is written. **Colab runtimes
are temporary.** Keep `model_bundle.zip` — it is the only file the Predict notebook needs.

---

## Part 1 — the bundled example, end to end

The example is a real library, not a toy: 16,424 SlugCas9 variants of a 1054-residue
protein, 5 randomized positions, 4 measured PAM conditions. The library, the wild-type
sequence, its 3Di string and its pooling regions all ship with the repository, so there is
nothing to upload.

| cell | roughly how long |
|---|---|
| 0 Setup | a minute or two, dominated by `pip install` (estimate) |
| 1 Load the library | ~1 s |
| 2 Wild-type 3Di | skipped |
| 3 Backbone and pooling | seconds |
| 4 Hyperparameters | instant |
| 5 **Fine-tune** | **the long one** — registry estimate ~40 min for the default backbone on a T4 |
| 6 One-hot floor | ~11 s measured on a workstation CPU; allow a minute or two on a Colab CPU |
| 7 Unlock the test set | instant |
| 8 Report | ~1 s |
| 9 Export the bundle | seconds |
| 10 Score a batch | seconds for a few dozen variants on a GPU |

Every T4 minute figure on this page is the registry's own order-of-magnitude estimate,
not a measurement of your card. Cell 5 prints the real rate as it runs.

**How the notebook is laid out.** It has two code cells. The first installs the package
and draws a panel; that panel *is* the workflow, and it carries ten numbered sections that
appear and disappear as your answers make them relevant. The second cell, at the bottom,
is the one that reads the locked test set and writes the report.

The headings below are this tutorial's own numbering, in the order you actually work
through the notebook. They are not always the panel's section numbers: the panel puts
**Backbone and pooling** at 2 and **The wild-type structure** at 3, because the structure
section only appears once you have chosen a backbone that needs it.

---

### Cell 0 · Setup

Installs the package and draws the panel that is the rest of the notebook. The same cell
appears in all three notebooks; run it once per session.

| field | default | what it means |
|---|---|---|
| `colabsd_repository` | `https://github.com/JasonJiangs/ColabSeqDisplay.git` | the one thing that gets installed: a git URL, or the path of a folder you uploaded to this runtime |

**Leave it alone** unless you are deliberately running a branch or a copy you edited
yourself. There is no second source field: the fine-tuning engine ships inside this
repository.

On a fresh runtime it narrates the install — the `cloning` line is skipped if you gave it
a folder path — and then prints three lines about the session:

```
cloning https://github.com/JasonJiangs/ColabSeqDisplay.git ...
installing ColabSeqDisplay from /content/ColabSeqDisplay ...
colabsd 0.1.0   from /content/ColabSeqDisplay
GPU       Tesla T4  (<N> GB)
files     /content/colabsd_work
```

and then the panel appears underneath.

**Read those three lines before moving on.**

1. `colabsd 0.1.0` — the version that was installed, and the directory it was installed
   from. If you pointed the field at your own folder, this is where you confirm it took.
2. `GPU` — the card and its memory. Without one the line reads
   `none — Runtime > Change runtime type > T4 GPU, then run this cell again`; fix that now
   if you intend to train.
3. `files` — where everything this session writes will land.

Run the cell a second time in the same session and it skips the install and redraws the
panel. It counts the install as done only when `colabsd` imports **and** its hyperparameter
registry and bundled example came with it, so an install that carried only the Python
modules across is simply performed again rather than left to fail once you press train.

---

### Cell 1 · Load the variant library

One row per variant: one column per mutated site, one column per measured condition,
optionally a read count. Variant sequences are **built** by substituting each row's
residues into the wild type — never read from a FASTA — so every variant is full length
and every position is checked against the sequence you gave.

| field | default (the bundled example) |
|---|---|
| `data_source` | `bundled_example` — *Use the bundled SlugCas9 5NNK example* |
| `wt_sequence` (*Wild type*) | *(empty — only used for your own data)* |
| `positions_1based` | `984, 985, 990, 1012, 1016` |
| `mutation_columns` | `nnk1, nnk2, nnk3, nnk4, nnk5` |
| `condition_columns` | `NNGA, NNGT, NNGC, NNGG` |
| `three_letter_residues` | `True` |
| `count_column` | `count` |
| `min_count` | `0` |

Leave every field at its default and run. You get:

```
16424 variants from .../library.csv
wild type: 1054 residues
5 mutated sites at [984, 985, 990, 1012, 1016], wild-type residues NSMEK
4 conditions: NNGA, NNGT, NNGC, NNGG
targets (16424, 4), range 0.000 to 2.559
```

followed by the first five rows of the table.

**Those five lines are the entire data contract, and every later cell inherits them.**
Check the variant count, the site count, the positions and the condition names against
what you believe you loaded. `wild-type residues NSMEK` is the strongest check available:
those are the residues the wild type carries at the positions you listed, so if the
numbering is off by one — or off by a signal peptide — this string will not be the one you
expect.

A good `targets` line looks like your assay. `range 0.000 to 2.559` is a normalized
activity; a range of `0.000 to 4.7e+06` means you are about to fit raw read counts.

---

### Cell 2 · Wild-type 3Di

*Skip this cell for the bundled example.*

SaProt reads structure alongside sequence and needs one Foldseek 3Di letter per residue of
the wild type. Every other backbone reads sequence only.

| field | default |
|---|---|
| `three_di_source` | `none` — *Not chosen yet* |
| `chain` | *(empty)* |

On `none_needed` the cell simply prints which backbones would have needed a 3Di string:

```
No 3Di string attached. Every backbone works without one except: SaProt-1.3B, SaProt-35M, SaProt-650M.
Pick one of those in step 3 and it will send you back here.
Using the bundled example? Its 3Di string ships with the package — choose bundled_example_3di.
```

If you later pick a SaProt backbone in cell 3, that cell stops and sends you back here.
For the example, choose `bundled_example_3di` and re-run; for your own protein see
[Part 2](#part-2--your-own-protein).

---

### Cell 3 · Backbone and pooling

| field | default |
|---|---|
| `backbone` | `ESM2-35M` |
| `pooling` | `cosine_p90_mean` |
| `dtype` | `float32` |
| `region_source` | `bundled_example` |
| `region_filename` | `region_p90.json` |

**Leave the defaults.** They are a tuned, sequence-only pair that fits a free T4.

Pooling decides *which residues* the model is read out from, and matters more than the
backbone:

- `cosine_p90_mean` averages the residues whose embeddings vary most across your variants.
  It needs a region file. The example ships one.
- `mutation_site_mean` averages the mutated sites themselves, and needs no extra file.

The cell opens with the registry's own line for the pair you picked:

```
===================================================================
ESM2-35M · cosine_p90_mean · tuned (test ρ=0.5486 ± 0.0145, 9 runs)
===================================================================

tier native · family ESM2 · facebook/esm2_t12_35M_UR50D
pooled feature: 480-d · needs 3Di: no
runtime: the registry estimates 40 min per run on a T4 for a library the
  size of the bundled example (16,424 variants x 1054 residues).
  That is an estimate, not a measurement: step 5 rescales it for your 16,424 variants
  and prints the real rate while it runs.
notes: Good accuracy-per-minute trade-off on a T4.

region: 106 residues from .../region_p90.json
  1 - mean all-vs-all pairwise cosine similarity per residue across variants
  scored on 16424 variants with ESM2-650M, threshold 0.000337186

adapter ready: ESM2Adapter(model_name='ESM2-35M', pooling='cosine_p90_mean', hf_id='facebook/esm2_t12_35M_UR50D')
```

The region line is worth a second look: **106 of the protein's 1054 residues** are pooled,
and the file states which model scored it, on how many variants, and at what threshold.
The cell refuses a region file whose declared pooling or whose wild-type length does not
match this run, so a region computed for another protein cannot be averaged under this
one's name.

#### Choosing a different backbone

Do not agonise. Across the 10 tuned pairs the whole spread of test Spearman is
0.5478 to 0.5636 — 0.0158 between the best (`ESM2-650M · cosine_p90_mean`) and the worst
(`ESMDance · cosine_p90_mean`) on the benchmark library. That gap is smaller than the one
between a held-out number and a test set you have already looked at.

| backbone | tier | needs 3Di | `cosine_p90_mean` config | registry estimate, min/run on a T4 |
|---|---|---|---|---|
| `ESM2-8M` | native | no | provisional | 20 |
| `ESM2-35M` *(default)* | native | no | tuned, ρ 0.5486 | 40 |
| `ESM2-150M` | native | no | tuned, ρ 0.5546 | 90 |
| `ESM2-650M` | native | no | tuned, ρ 0.5636 | 240 |
| `SaProt-35M` | native | **yes** | tuned, ρ 0.5576 | 45 |
| `SaProt-650M` | native | **yes** | tuned, ρ 0.5592 | 260 |
| `SaProt-1.3B` | native | **yes** | tuned, ρ 0.5578 | needs an L4/A100 |
| `ProtT5-XL` | extra | no | tuned, ρ 0.5583 | needs an L4/A100 |
| `Ankh-large` | native | no | provisional | needs an L4/A100 |
| `ESMC-300M` | extra | no | tuned, ρ 0.5487 | 120 |
| `ESMC-600M` | extra | no | provisional | 220 |
| `SeqDance` | native | no | provisional | 45 |
| `ESMDance` | native | no | tuned, ρ 0.5478 | 45 |

Tier `extra` means one more `pip install`; the loader names it if it is missing
(`pip install esm` for ESM-C, `pip install sentencepiece` for ProtT5-XL). The registry
holds a fourteenth backbone, METL, which has no public weights and is not offered in the
notebook.

Every `mutation_site_mean` entry in the registry is provisional. If you pick one, the cell
adds:

```
These hyperparameters are a PLACEHOLDER. Nobody has tuned this pair, and nobody has
measured what it scores. Read every number it produces as a lower bound, and prefer a
pair whose line says 'tuned' if you can.
```

**In a hurry?** `ESM2-8M` roughly halves the wait and is the right way to check that a new
library flows through the whole notebook. It is a provisional pair, so it is the wrong way
to decide anything.

---

### Cell 4 · The hyperparameters

This section has no fields at all: it only displays what the registry holds for the pair
you picked. There are no sliders here, on purpose — a hyperparameter you re-tune while
watching your own validation score is a hyperparameter that has quietly eaten your test
set. For a given
(backbone, pooling) pair the seven LoRA hyperparameters are **looked up** from
`config/best/`, which records for each entry whether it is `tuned` or `provisional`, where
it came from, and — for a tuned entry — the test Spearman it was re-evaluated at.

For the default pair the cell prints the headline again, names the file, and tabulates:

| hyperparameter | value |
|---|---|
| `adapter_lr` | 0.0002114277988 |
| `head_lr` | 0.0001593959018 |
| `adapter_weight_decay` | 0.01 |
| `lora_rank` | 32 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0 |
| `effective_batch_size` | 32 |

then the fixed training block (`mlp` head, `mse` loss, `max_epochs` 20,
`early_stopping_patience` 3, `micro_batch_size` 8, `label_preprocessing`
`train_split_zscore`, and the rest), and finally:

```
evaluation protocol: split seeds [1, 2, 3] x model seeds [11, 22, 33] = 9 runs, selected on mean_validation_spearman
provenance: {"status": "tuned", "source": "seqdisplay-opt LoRA hyperparameter study on the SlugCas9 5NNK library", "optuna_trial": 10, "test_spearman_mean": 0.5486, "test_spearman_sd": 0.0145, "n_reevaluation_runs": 9}
```

For a provisional pair that last line is replaced by

```
status: PROVISIONAL — placeholder values, no measured performance behind them.
```

Of the 28 registry entries, **10 are tuned and 18 are placeholders.** Each file states its
own status; nothing is inferred.

---

### Cell 5 · Fine-tune

LoRA adapters on the attention projections plus a small MLP head; the backbone itself stays
frozen. Each run trains on the train split and early-stops on validation.

| field | default |
|---|---|
| `n_split_seeds` | `1` (slider, 1–3) |
| `n_model_seeds` | `1` (slider, 1–3) |
| `resume_finished_runs` | `True` |
| `run_name` | `run` |

Leave both sliders at 1 for a first pass. One split seed × one model seed is one run; the
registry's own protocol is 3 × 3, which is nine times the wait and is what you want for a
number you intend to report, not for a first look.

The library is split 8 : 1 : 1. For the example that is **13,139 train / 1,642 validation /
1,643 test** variants per split seed.

The cell prints its plan, then a progress counter:

```
split seeds [1] x model seeds [11] = 1 run(s)
very rough estimate: 40 min — the registry's 40 min/run for a 16,424-variant
library on a T4, scaled to your 16,424 variants and 1 run(s). Nobody has timed
your card: watch the counter below for the real rate. Keep this tab open.
The test partition is locked away as each run finishes; step 7 is the only way to read it.

  1/1  split1_seed11
finished 1 run(s) in <N> minutes
validation Spearman, averaged over conditions: <your number> ± nan  (1 runs)
One run shows no spread. Raise n_split_seeds / n_model_seeds before quoting a ± anywhere.
```

and then a table of validation Spearman, R² and NDCG@50 per condition.

Three things to notice.

- **Validation only.** As each run finishes, its test artefacts are moved into
  `run/locked_test/`, and the `test` block of the run's visible metrics is replaced by a
  sentence pointing at the unlock step. Nothing you can read here carries test information.
- **`± nan` is deliberate**, not a bug. One run cannot show reproducibility, so the
  standard deviation is undefined rather than zero.
- **Interrupted?** Re-run the cell. A finished run is reused only when its fingerprint —
  model, hyperparameters, pooled coordinates, split, and a hash of your data — is
  identical, so a resumed run can never mix a checkpoint with different numbers.

Keep the browser tab open: closing it disconnects the runtime, and a disconnected runtime
stops training.

---

### Cell 6 · The one-hot floor

No fields: both heads are always fitted, and the better of the two is the floor.

Ridge regression and a small MLP on the 20 × *k* one-hot encoding of the mutated residues —
for the example a 16,424 × 100 feature matrix — fitted on exactly the splits the pLM used.
No GPU needed.

**This is the line that makes a pLM result mean something.** A 650M-parameter model that
ties a ridge regression on one-hot residues has told you the landscape is additive, not
that the model is good.

Both heads over all 16,424 example variants, one split seed × one model seed, took
**10.7 seconds** measured once on a workstation CPU; allow a minute or two on a Colab CPU
runtime. In that run:

| head | validation Spearman (mean over conditions) |
|---|---|
| `ridge` | 0.4722 |
| `mlp` | **0.5368** ← the floor |

Per condition, the MLP reached 0.6157 (NNGA), 0.4437 (NNGT), 0.5083 (NNGC) and 0.5797
(NNGG). The splits are seeded, so the ridge number should reproduce exactly; the MLP may
move in the last digits with a different torch build.

The cell names the floor and, if cell 5 has run, says plainly whether the language model
cleared it:

```
floor: mlp reaches validation Spearman 0.5368 ± nan over 1 run(s)
ESM2-35M clears the floor by <gap> Spearman
```

If it did not, the line reads `FAILS to clear the floor, by <gap> Spearman`. That is a
result, not an error, and it is a real possibility with a small backbone on placeholder
hyperparameters: a one-hot MLP over five randomized sites is a strong baseline, not a
straw man.

---

### Cell 7 · Unlock the test set

**Stop and read this before you tick the box.** This is the one methodological decision the
notebook asks you to make, and the one a referee is most likely to probe.

| field | default |
|---|---|
| the confirmation checkbox — *I have stopped changing things. Read the test set once and count it.* | `False` |

#### What is actually locked

Everything above cell 7 reports **validation**. The training loop evaluates the test
partition at the end of each run, as training loops do — and the notebook immediately moves
every test artefact into `run/locked_test/` and replaces the test block of the visible
metrics with a sentence naming the one function that can read it back. Nothing your session
can touch, print or plot carries test information until you deliberately unlock it.

#### Why unlocking is a deliberate action

A test set is only held out while nobody has looked at it. The moment a test number
influences a choice — a different backbone, a different pooling, one more seed, a different
`min_count` — the partition has entered the model-selection loop, and the next number it
gives you is a validation score wearing a test label. Nothing in the number itself reveals
which of the two you are holding.

So the notebook makes the transition explicit: validation is free and repeatable, the test
set costs a tick-box, and the tick-box leaves a record. The confirmation resets itself
after every unlock, so a second one has to be given as deliberately as the first.

#### Why the count is recorded

Ticking the box increments a counter in `run/unlock.json`, which also keeps a small history
— when each unlock happened, which model, over how many runs. That count is then stamped
into everything downstream:

- the run result on disk,
- `report.csv`, `report.json`, and the footer of `report.png`
  (`test-set unlock count: N`), and the badge at the top right of the figure,
- the manifest inside `model_bundle.zip`, so a colleague opening the bundle a year later
  reads `test unlocked 1x` in its first printed line.

The counter is a property of the run directory, so it is honest about what it can see: a
new `run_name` starts a new directory with its own counter at zero, and deleting
`unlock.json` resets it. If the file is unreadable, the error says to delete it *and say so
in the report*. The record is an aid to honest reporting, not a lock on the filesystem.

One further guard: if a run directory is reused by a later training run with different
hyperparameters, data or splits, unlocking refuses rather than handing back the older
result's test numbers under the newer configuration's name.

#### How to run it

Unlock **once**, when you have stopped changing things. Tick the confirmation and press
**Unlock the test set**:

```
this run directory has been unlocked 0 time(s) so far

unlock #1 at <timestamp> over 1 run(s)
test Spearman, averaged over conditions: <your number> ± nan
```

followed by the test table. On provisional hyperparameters it adds:
`...on PROVISIONAL hyperparameters: this is a lower bound, not a benchmark number.`

#### What a reader should conclude from the count

| unlock count | what it means |
|---|---|
| **0** | no held-out number exists: the report shows validation and its badge says `test set LOCKED (0 unlocks)`. Predictions can still be worth ranking; the model has simply not been measured on unseen data. The notebook never writes a report at 0, because writing it is part of this cell and this cell only runs on a confirmed unlock — a 0 you see came from `colabsd.report.build_report` called directly. |
| **1** | a held-out result, in the usual sense. The test partition was read once, after the choices were made. |
| **2 or more** | the test partition was read more than once for this run directory. Each read after the first was available to inform a choice, so the last number is at best a second validation score. Quote it as such, and quote the count beside it. |

A report showing several unlocks is not a fabrication and not a disqualification — it is a
record of an exploratory session, and it should be described as one. What the count rules
out is presenting the *n*-th look at a test set as if it were the first.

---

### Cell 8 · Report

**This is the same cell as step 7.** The report is written by the unlock, with the metric
chosen beside the tick-box, so there is no separate cell to run and no report before the
test set has been opened.

| field | default |
|---|---|
| `metric` (*Report metric*) | `Spearman` (also R2, Pearson, P@10, P@50, NDCG@10, NDCG@50) |

It writes three files into `colabsd_work/report/` and offers each as a download.

| file | what it holds |
|---|---|
| `report.csv` | one row per source × partition × condition: `n_runs`, then `_mean` and `_sd` for R2, Pearson, Spearman, P@10, P@50, NDCG@10, NDCG@50 — plus a `mean` row across conditions |
| `report.png` | the figure below |
| `report.json` | the machine-readable version: `unlock_count`, `unlock_source`, `one_hot_floor`, `shown_partition`, `reported_partitions`, `n_runs`, `sources` |

The figure has two panels:

- **Top** — the chosen metric per condition, one bar group per condition plus a `mean`
  group, one bar per source (the pLM and each one-hot head), error bars ±1 sd across runs.
  A dashed horizontal line marks the one-hot floor and is labelled with its value. A badge
  in the top right states the partition and the unlock count.
- **Bottom** — all seven tracked metrics, averaged over conditions, same bars.

Above them, a one-line verdict: `<model> beats the one-hot floor: 0.xxx vs 0.xxx (+0.0xx)`,
or `does NOT beat`. Along the bottom, a footer: runs per source, what the error bars are,
the number of mutated sites and conditions, and the test-set unlock count with the file it
came from.

**Read it in this order.**

1. **Did the pLM beat the one-hot floor?** Same partition, the `mean` group. If it did not,
   nothing else on the page matters yet.
2. **Is the spread across conditions sensible?** In the example the four PAMs are not
   equally predictable — NNGT is the hardest and NNGA the easiest for both one-hot
   heads. A pattern like that is normal. *One* condition far below the others, when your
   assay says it should not be, is usually a data problem, not a model problem.
3. **How many runs?** `n_runs = 1` and a blank `sd` mean one draw, not an estimate. Do not
   quote a ± from it.
4. **The unlock badge.** It names the partition and the count. `1` is the number you want
   beside a reported result; anything higher is a record of how many looks it took.

A good result: the pLM bar clears the dashed line in the `mean` group and in most
conditions, the error bars over three or more runs are small compared with the gap, and the
unlock count is 1. A bad result: the pLM sits on or
below the dashed line, or clears it by less than the spread across runs.

---

### Cell 9 · Export the model

| field | default |
|---|---|
| `bundle_name` (*Bundle file name*) | `model_bundle.zip` |
| `notes` | *(empty — free text stored in the manifest)* |

One `.zip` holding the LoRA weights, the head, the library spec, the **frozen** pooling
coordinates, the hyperparameters and the provenance. It carries no backbone weights — those
are re-fetched by name — so it is small, and it carries no region file dependency: the
pooled positions are baked in, so scoring a year from now cannot silently pool a different
region.

```
ESM2-35M · cosine_p90_mean · tuned hyperparameters · 4 conditions · test unlocked 1x · written <timestamp>
pooling cosine_p90_mean over 106 frozen residue positions
conditions: NNGA, NNGT, NNGC, NNGG
```

Two things to know:

- The bundle holds **the single best-validation run**, not an ensemble of your runs. With
  3 × 3 seeds, the ± in your report describes the family of runs; the bundle is one member
  of it.
- It needs the training checkpoint on disk, so export before you clear
  `colabsd_work/`.

**Download it.** It is the only thing the Predict notebook needs.

---

### Cell 10 · Score a batch of variants

The minimal prediction outlet, so the notebook ends with something usable.

| field | default |
|---|---|
| `variant_source` | `library_head` — *The first rows of the library I loaded* |
| `how_many` | `32` |
| `random_seed` | `0` |
| `rank_by` | `pred_mean` |

Three sources: the first rows of the library you loaded, `random_combinations` (residues
drawn at random for each mutated site, then de-duplicated — the cheap way to sample a space
too large to enumerate; five NNK sites is 3.2 million combinations), or your own CSV.

Output is one row per variant: the mutation columns, one predicted column per condition,
plus two summaries.

- **`pred_mean`** — average predicted activity. Rank by this for overall activity.
- **`pred_min`** — the worst condition. **Rank by this when a hit has to work in every
  condition**, which for a PAM library is usually the interesting question.

The table is written to `colabsd_work/scored_variants.csv` and offered as a download; the
top 20 rows are shown.

**One caveat on the default source.** The first rows of the loaded library are mostly rows
the model trained on — 80 % of the library is the training split. Comparing those
predictions to the measurements you already have is a wiring check, not evidence. For
evidence, use the report; for candidates, use `random_combinations` or your own CSV.

---

### What you should have now

In `colabsd_work/`: `model_bundle.zip` and `scored_variants.csv`, plus
`report.csv`, `report.json` and `report.png` in `report/`. Each was pushed to your
browser's downloads as it was written.

And the one sentence that summarises it: **a validation Spearman above the one-hot floor
says the language model found structure that per-site additivity does not explain; a test
number is worth what the unlock count says it is worth.**

---

## Part 2 — your own protein

Same notebook, same order. The differences are all in cells 1–3.

### What you need

| file | required | notes |
|---|---|---|
| `library.csv` | always | one row per variant; one column per mutated site; one numeric column per measured condition; optionally a read-count column |
| the wild-type sequence | always | pasted as text, or the URL of a FASTA (a UniProt or AlphaFold link works) |
| `wt_3di.txt` | only for a SaProt backbone | made by `colab/ColabSeqDisplay_Prepare.ipynb`, section A |
| `region_p90.json` | only for `cosine_p90_mean` pooling | made by `colab/ColabSeqDisplay_Prepare.ipynb`, section B |

Your CSV must satisfy a few rules the loader enforces:

- **Unique column names.** A header that names `nnk1` twice is refused, because the second
  copy would be silently renamed and ignored.
- **Residues in one alphabet.** Either three-letter (`Asn`) with
  `three_letter_residues = True`, or one-letter (`N`) with it `False`. The error names the
  offending cell and tells you which way to flip the switch.
- **Numeric, finite condition values.** Missing or non-numeric measurements are refused
  with the first offending column and row; values too large for float32 are refused with a
  suggestion to rescale (for example, take a log).
- **Positions strictly increasing, and one mutation column per position, in the same
  order.** `positions_1based` and `mutation_columns` are read as parallel lists.

### Filling in the load form

Set `data_source` to `upload_my_csv` and run the cell — Colab will ask for the file. Then:

| field | what to put |
|---|---|
| `wt_sequence_or_fasta_url` | the full-length wild-type amino-acid sequence, pasted; or a URL beginning `http` pointing at its FASTA. Whitespace is stripped and the sequence is upper-cased |
| `positions_1based` | the mutated positions, **1-based, in the full-length wild-type numbering**, increasing: `984, 985, 990, 1012, 1016` |
| `mutation_columns` | the CSV columns holding the residues, **in the same order as the positions** |
| `condition_columns` | the CSV columns holding the measurements, one per condition |
| `three_letter_residues` | `True` for `Asn`, `False` for `N` |
| `count_column` | the read-count column, or blank if there is none |
| `min_count` | a read-count threshold; `0` keeps everything |

Then check the printed `wild-type residues` string. If those are not the residues your
construct carries at those positions, your numbering is wrong — stop here, because every
later cell inherits it. Two frequent causes: the FASTA includes a tag or signal peptide the
numbering does not, or the positions are 0-based.

If you set `min_count` but the table has no such column, the cell warns that the filter was
ignored and how many variants were kept, rather than silently keeping them.

### When you must run `colab/ColabSeqDisplay_Prepare.ipynb` first

| you plan to use | you need Prepare | why |
|---|---|---|
| an ESM2 / SeqDance / ESMDance / ESM-C / ProtT5 / Ankh backbone **and** `mutation_site_mean` pooling | **no** | go straight to the main notebook |
| any backbone with `cosine_p90_mean` pooling | **yes — section B** | pooling needs a region file computed on *your* variants |
| any SaProt backbone | **yes — section A** | SaProt needs a wild-type 3Di string |
| SaProt with `cosine_p90_mean` | **yes — both sections** | |

Note the tension, because it is real: `cosine_p90_mean` is the pooling every tuned
hyperparameter set uses, but for a new protein it costs a region-discovery run. Every
`mutation_site_mean` entry in the registry is provisional. You are choosing between a
tuned configuration that costs GPU time up front, and a placeholder configuration that
starts immediately.

### The Prepare notebook

One panel, two tick-boxes: **Make a wild-type 3Di string** and **Discover a pooling
region**. Tick either, both or neither, press **Load the protein**, then **Run the ticked
jobs**. The load form above them is the main notebook's, field for field — fill it in
exactly the same way, because region discovery scores *your variants*, not just the wild
type.

#### Section A · Structure → 3Di

| field | default |
|---|---|
| `three_di_source` | `upload_structure_pdb_or_cif` |
| `chain` | *(empty)* |
| `three_di_name` | `wt_3di.txt` |
| `esmfold_risk_accepted` (*I accept the ESMFold memory risk*) | `False` |

Three routes, cheapest first.

1. **Upload a structure** (`.pdb`, `.cif`, `.mmcif`, optionally gzipped) — the default.
   Foldseek is downloaded once and run locally. The structure's own sequence is checked
   against your wild type, so a mismatched chain is refused rather than silently shifting
   every position. Download the AlphaFold model of your protein from
   <https://alphafold.ebi.ac.uk> and upload the `.cif`; it is free, fast and matches your
   sequence exactly.
   If the file holds more than one chain the cell lists them with their lengths and asks
   you to name one in `chain`.
2. **Upload a 3Di text file** you already have — plain text or a one-record FASTA.
3. **Fold the wild type with ESMFold** — the last resort. Its memory grows with the
   *square* of the sequence length, and `colabsd.structure` puts the ceiling for a free
   16 GB T4 at **700 residues**. You must tick *I accept the ESMFold memory risk* to run
   it, and past 700 residues the panel refuses outright and tells you to download a real
   structure instead — which is why the bundled 1054-residue SlugCas9 cannot be folded
   this way.

It prints the 3Di length beside the wild-type length and shows the first 60 characters of
each, then offers `wt_3di.txt` as a download. Seconds, from a structure.

#### Section B · Region discovery

| field | default |
|---|---|
| `region_model` | `facebook/esm2_t33_650M_UR50D` (also 150M, 35M) |
| `n_sample` | `2000` |
| `percentiles` | `90, 95` |
| `batch_size` | `2` (slider, 1–16) |
| `seed` | `0` |
| `run_on_cpu_anyway` | `False` |

`cosine_p90_mean` pooling averages the residues whose embeddings disagree most across your
variants — the model's own answer to *which part of this protein does the assay move?* The
score is `1 − mean pairwise cosine similarity` per residue, and the region is every residue
at or above the 90th (or 95th) percentile of that score.

**The subsampling is not hidden.** Scoring long sequences through a 650M-parameter model is
the expensive part, so `n_sample` variants are drawn at random; the cell prints how many
were scored, with which seed, and the first sampled row indices, before it starts. A
different seed gives a different subset and therefore a slightly different region.
`n_sample = 0` scores every variant and removes the question — that is how the regions
shipped with the example were made (they record `n_variants_scored: 16424`).

Cost is linear in the number of variants scored. On a T4, expect tens of minutes for a
couple of thousand variants and a couple of hours for a library the size of the example —
an order of magnitude, not a measurement. The progress counter tells you your own rate
within the first minute, and a smaller `region_model` is the quickest way to cut it. Region
discovery needs a GPU; the cell refuses without one unless you tick `run_on_cpu_anyway`.

For each percentile it writes a region JSON and prints:

```
cosine_p90: 106 of 1054 residues  (threshold 0.000337186)
  5 of the 5 mutated positions are inside it: [984, 985, 990, 1012, 1016]
  first selected positions: [912, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949]
```

**What a good region looks like:** the mutated positions are inside it, and the selected
residues form a few contiguous blocks rather than scattered singletons. The example's
shipped p90 region is 106 positions in three blocks — 912, 939–1040 and 1043–1045 — and it
contains all five mutated positions. A region that excludes the mutated sites, or one that
selects almost the whole protein, means the percentile is doing nothing useful for this
library.

Section B also writes `region_p95.json`. Keep it for comparison — the main notebook offers
only `cosine_p90_mean` and `mutation_site_mean`, and it refuses a region file whose
declared pooling does not match the pooling you chose.

#### Carrying the files across

Each notebook runs in its own Colab runtime, so download `wt_3di.txt` and `region_p90.json`
from Prepare (the cells offer both), then in the main notebook upload them:
`wt_3di_source = upload_3di_text_file` in cell 2, and
`region_source = upload_region_json` in cell 3.

Each region file records its own provenance — which model, which score, which threshold,
how many variants — so two regions can always be told apart.

### Reading the PROVISIONAL warning

If the (backbone, pooling) pair you chose has never been tuned, the notebook says so in
capitals, in five places, and never stops you:

| where | what it says |
|---|---|
| cell 3, headline | `<backbone> · <pooling> · PROVISIONAL — hyperparameters are a placeholder, performance unknown`, followed by three lines telling you to read every number as a lower bound |
| cell 4, last line | `status: PROVISIONAL — placeholder values, no measured performance behind them.` |
| cell 5, after training | `WARNING: <backbone> hyperparameters are PROVISIONAL: they are a placeholder, not a tuned configuration. Treat the numbers as a lower bound.` |
| cell 7, after unlocking | `...on PROVISIONAL hyperparameters: this is a lower bound, not a benchmark number.` |
| the bundle, cell 9 and in Predict | `PROVISIONAL hyperparameters` in its description line |

What it means concretely: the seven LoRA values were carried over from the tuned pairs —
all 18 placeholder files record `source: median of the 10 tuned cosine_p90_mean configs` — nobody
has searched them for this pair, and no performance number stands behind them. A model
trained on them can still rank variants perfectly usefully. It cannot support a claim about
how well this backbone performs.

**And one honesty note that applies even to a `tuned` line.** Those hyperparameters were
selected once by an Optuna search on the benchmark sequence-display library, then
re-evaluated over nine runs. On *your* protein, "tuned" means *transferred from a tuned
search on another library* — a better starting point than a placeholder, not a
configuration fitted to your data. The notebook does not re-tune per run, and that is the
design: re-tuning while watching your own validation score is how a screen overfits.

### A sensible first pass on a new library

1. Load your CSV, check the `wild-type residues` line, and stop if it surprises you.
2. `backbone = ESM2-8M`, `pooling = mutation_site_mean` — no region file, no 3Di, roughly
   20 min per run on a T4 for a library the size of the example. This checks that your data
   flows through training, the floor and the report. Do not unlock the test set.
3. Run `colab/ColabSeqDisplay_Prepare.ipynb` section B while that is going, if you want
   `cosine_p90_mean`.
4. The real run: a tuned pair, `n_split_seeds = 3` and `n_model_seeds = 3`, one unlock at
   the end.

---

## Part 3 — scoring new variants with `colab/ColabSeqDisplay_Predict.ipynb`

Upload a `model_bundle.zip` and a table of variants; get a ranked prediction table back.
No training happens here. The bundle is self-contained — backbone name, LoRA weights, head,
library spec, frozen pooling coordinates, hyperparameters and provenance — so nothing else
has to match.

A GPU makes this fast. Without one it still runs, at the backbone's price: a small ESM2
scores a few hundred variants on a CPU in minutes, a 650M-parameter model takes hours.

The bundle carries the LoRA weights and the head, not the backbone, so the first scoring
run in a session downloads the backbone by name — this runtime needs internet access.

### The setup cell

Identical to the main notebook's. Same field, same output block. It draws one panel, and
the three stages below are that panel's three buttons, in order.

**Scoring uses the spec inside the bundle, and nothing else.** The bundle carries its own
library specification, so this notebook never asks you to describe your library: it asks
for the bundle, then for the variants.

### Stage 1 · Load the model bundle

| field | default |
|---|---|
| `bundle_source` (*Bundle*) | `upload_model_bundle_zip` |
| `bundle_filename` (*File name*) | `model_bundle.zip` (used only for `file_in_the_working_folder`) |

Run it and upload the `.zip`. It prints the bundle's own description and then what it
requires of your variants:

```
ESM2-35M · cosine_p90_mean · tuned hyperparameters · 4 conditions · test unlocked 1x · written <timestamp>

backbone ESM2-35M · pooling cosine_p90_mean over 106 frozen residue positions
mutated sites [984, 985, 990, 1012, 1016] · columns ['nnk1', 'nnk2', 'nnk3', 'nnk4', 'nnk5']
conditions NNGA, NNGT, NNGC, NNGG
```

**Read the first line as the model's provenance.** `PROVISIONAL hyperparameters` earns an
extra line — *rank with it, but do not quote its numbers*. `test unlocked 0x` earns another
— *its test set was never unlocked, so no held-out number stands behind these predictions*.
Both are statements about what the model is, and both travel with the file rather than with
your memory of the session that made it.

### Stage 2 · Choose the variants to score

| field | default |
|---|---|
| `variants_source` (*Variants*) | `upload_variants_csv` |
| `how_many` | `200` |
| `random_seed` (*Seed*) | `0` |

Your CSV needs one column per mutated site, **named exactly as stage 1 printed**. Any
other column is ignored, so a table that also carries measurements is fine. If a column is
missing, the panel names the missing ones and the ones the bundle was trained on.

The other two sources: `rows_of_a_library_csv` (rows of a library table you upload — a
wiring check, not evidence; see the caveat in Part 1 cell 10), and `random_combinations`,
which draws `how_many` rows of residues at the mutated sites and de-duplicates them.

It prints how many variants are ready and how many are distinct, then shows the first ten.
A large gap between those two numbers means your CSV repeats variants.

### Stage 3 · Score and rank

| field | default |
|---|---|
| `rank_by` | `pred_mean` (or `pred_min`) |
| `top_n_to_show` (*Rows to show*) | `20` |
| `score_batch_size` (*Batch size*) | `8` (1–32) |
| `output_name` (*Output file*) | `ranked_variants.csv` |

Adds a `rank` column, sorts, writes the CSV and offers it as a download.

```
ranked 200 variants by pred_mean
scale: the assay's own units — this bundle carries the label scaler it was trained with
top prediction: <value> · median <value>
```

**Check the scale line.** Training z-scores the labels on the train split, and the bundle
normally carries that scaler, so predictions come back in the assay's own units. If it does
not, the cell says `scale: z-scored training units` — the ranking is still meaningful, the
absolute values are not comparable to your measurements, and you should not plot them on
the same axis.

`pred_mean` ranks by average predicted activity; `pred_min` ranks by the worst condition,
which is what you want when a hit has to work everywhere. Both columns are always written,
so you can re-sort the CSV without re-scoring.

**A caveat worth repeating.** These are predictions from a model fitted to one library.
They rank variants; they do not measure them. Treat the top of the table as the list to
assay next, and check what the bundle's own line said about how it was validated.

---

## Troubleshooting

Every failure you can cause raises an error whose message says what to do. The common ones:

| what you see | what it means, and the fix |
|---|---|
| `ColabSeqDisplay could not be downloaded from <address>` (cell 0) | `git clone` failed — the address in `colabsd_repository` is not reachable from this runtime, or is not a repository. The message quotes git's own output above it, and names the field to change. Fix the address, or upload a folder and put its path in the field instead. |
| `This command failed:` followed by a `pip install -e` line (cell 0) | The clone worked and the install did not. The message carries the tail of pip's own output; read that before anything else. A truncated or partial copy of the repository is the usual cause — re-run the cell to fetch it again. |
| cell 0 reinstalls every time you run it | It counts the install as done only when `colabsd` imports **and** its hyperparameter registry and bundled example came with it, so a copy missing either is fetched again. If it keeps repeating, the folder in `colabsd_repository` is not a complete checkout. |
| `GPU       none — Runtime > Change runtime type > T4 GPU, then run this cell again` (cell 0) | **Runtime ▸ Change runtime type ▸ T4 GPU**, then re-run cell 0. Cells 1–4, 6 and 8 work without one; cell 5 refuses up front with `LoRA fine-tuning needs a GPU`, and cells 7, 9 and 10 refuse because there is no trained run. |
| `runtime: does NOT fit a T4 — this backbone needs an L4 or an A100` (cell 3) | A **warning**, not an exception: cell 3 still builds the adapter and cell 5 will then die with a CUDA out-of-memory error from torch. Pick a smaller backbone, or a Colab runtime with an L4/A100. `ProtT5-XL`, `Ankh-large` and `SaProt-1.3B` are the three; `SaProt-1.3B` additionally wants `dtype = float16`. |
| a CUDA out-of-memory error part-way through cell 5 | The backbone does not fit alongside your sequence length. In order of effect: choose a smaller backbone; set `dtype = float16`; shorten nothing else — `micro_batch_size` comes from the config and is not a form field. After an OOM the failed run's tensors usually still hold GPU memory, so **Runtime ▸ Restart** is the reliable way to retry. |
| `<file> has N residues but the WT sequence has M` | The 3Di string does not cover the same chain as your wild type. The usual cause is chain selection: pass `chain = "A"` (or the right id). The other is a structure with missing or extra residues — an AlphaFold or ESMFold model of the exact wild-type sequence always matches. |
| `wt_3di is N characters but the wild-type sequence is M residues` | The same mismatch, caught when the spec is validated. Regenerate the 3Di from a structure of this exact sequence. |
| `<file> contains 2 chains: A (N residues), B (M residues). Pass chain="A"` | Your structure holds more than one chain, so foldseek produced one 3Di string per chain. Put the chain id of *your* protein in the `chain` field and re-run. `<file> has no chain 'X'. Chains found: ...` means the id you gave is not in the file. |
| `<file> is a different protein from the WT: N of M residues disagree` | The chain you selected is not the protein your library mutates. Pick the right chain, or fold the wild-type sequence itself. A handful of disagreements is only reported, not refused — read that line, it names the positions. |
| `contains characters that are not foldseek 3Di states` | The file is an amino-acid FASTA, not a 3Di string. |
| `The variants table has no ['nnk1'] column(s). This bundle was trained on [...]` | Your CSV column names do not match the bundle's. Rename the columns to match, or use the library those names came from. The same check in the main notebook names the library's columns instead. |
| `The header of <file> names ['nnk1'] more than once` | Your CSV repeats a column name. pandas would rename the second copy to `nnk1.1` and quietly ignore it, so the file is refused. Fix the header. |
| a column is reported missing, and the message names a sibling like `nnk1.1` | The same duplicate-header problem, reaching a later cell through a table rather than a file. |
| `Unknown residue 'X' in column 'nnk1', row 0` | A residue outside the 20 standard ones, or three-letter data read as one-letter. The message says which way to flip `three_letter_residues`. |
| `N condition values are missing or not numeric, for example column 'NNGT' at row 12` | Drop or fill those rows before loading. A sibling message catches values too large for float32 and suggests rescaling. |
| `min_count=5 was ignored: ... has no 'count' column` | A warning, not an error — every variant was kept. Set `count_column` to the real column, or `min_count = 0` to say you meant to keep them all. |
| `<file> holds the cosine_p95_mean region, but this run pools cosine_p90_mean` | Averaging one region under another region's name is silently wrong, so it is refused. Pick the matching file. |
| `<file> was computed for a 1054-residue protein, but this wild type is N residues` | The region file belongs to another protein — very often the bundled SlugCas9 one, left on `bundled_example` while loading your own library. Run region discovery for your protein. |
| `The cached split in .../seed_1 covers N rows, not the M rows of this library` | You changed the library (a different CSV, or a different `min_count`) while reusing the working directory. Change `run_name`, or start a fresh runtime. |
| `No locked test results under .../locked_test` | Cell 5 has not produced a run in this directory. Train first, and keep `run_name` the same. |
| `holds locked results, but none of them belong to this RunResult` | The run directory belongs to a different set of seeds. Point cell 7 at the directory cell 5 actually wrote. |
| `holds the test metrics of a different training configuration` | The run directory was reused by a later training run with different hyperparameters, data or splits. Give each configuration its own `run_name`. |
| `unlock.json is not readable as an unlock record` | The counter file is corrupt. Delete it to reset the count to zero — **and say so in the report**, because the record of previous unlocks goes with it. |
| `<file> is not a readable .zip bundle`, `is missing ['manifest.json']`, `does not match its manifest checksum` | The bundle was truncated in transfer or was not written by this tool. Re-download or re-export it. |
| `No checkpoint at .../best_checkpoint.pt` | Cell 9 needs the training run still on disk. Export before clearing the working directory or restarting the runtime. |
| `ESM-C needs the EvolutionaryScale SDK` | `pip install esm`, then **Runtime ▸ Restart**. Beware the older `fair-esm` package: it installs a different module under the same name and has no ESM-C. |
| `'ProtT5-XL' reads a T5 sentencepiece vocabulary` | `pip install sentencepiece`, then restart the runtime. |
| `Could not download foldseek from <url>` | The runtime has no internet access to that host. Upload a ready-made 3Di text file instead, or install foldseek yourself and set `FOLDSEEK_BIN`. |
| `foldseek found no 3Di descriptor in <file>` | The file has no protein backbone atoms (nucleic-acid-only or ligand-only), or it is a minimal mmCIF foldseek cannot parse — converting it to `.pdb` fixes the second case. |
| `ESMFold ran out of GPU memory on N residues` | Fetch the AlphaFold model instead (<https://alphafold.ebi.ac.uk>), or fold it once at <https://esmatlas.com/resources?action=fold>, and upload the `.pdb`. |
| `The backbone holds its rotary position index in <dtype>, which represents only N of this protein's M residue positions distinctly` | `bfloat16` cannot represent every residue index of a long protein distinctly, so scores would be computed on a corrupted positional encoding and the loader refuses. Use `float32` (the default) or `float16`. |
| the runtime disconnects during cell 5 | Re-run the cell with `resume_finished_runs = True`. A finished run is reused only when its fingerprint — model, hyperparameters, pooled coordinates, split and a hash of your data — is identical, so a resume can never mix a checkpoint with different numbers. Keep the tab open and the laptop awake next time. |

---

## What ends up in the working directory

```
colabsd_work/                             (/content/colabsd_work/ in Colab)
├── splits/n16424/split/seed_1/          the cached 8:1:1 split, keyed by library size
├── run/                                 (named by run_name)
│   ├── runs/split1_seed11/              per-run checkpoints and visible metrics
│   ├── locked_test/split1_seed11/       the test partition, until you unlock it
│   ├── unlock.json                      the unlock counter and its history
│   ├── run_result.json                  the run, as the report reads it
│   ├── validation_runs.csv              one row per run
│   ├── validation_summary.csv           mean ± sd across runs
│   ├── test_metrics.json                written by an unlock
│   └── test_summary.csv                 written by an unlock
├── report/report.csv · report.json · report.png
├── model_bundle.zip
└── scored_variants.csv
```

Of these, keep **`model_bundle.zip`** (the model), **`report.png` / `report.csv`** (the
evidence) and **`unlock.json`** (the record of how many times the test set was read). The
rest can be regenerated from them and your library.

---

## If you report a result

Everything below is printed by the notebook or written into `report.json` and the bundle
manifest; none of it has to be reconstructed afterwards.

| state this | where it comes from |
|---|---|
| the tool, its version and its licence | `colabsd <version>` on the first line of cell 0; version, licence and the archive DOI are in `CITATION.cff` (see [`README.md`](README.md#code-availability)) |
| backbone, pooling, and the size of the pooling region | the headline of cell 3 and the bundle description in cell 9 |
| whether the hyperparameters were `tuned` or `provisional` | cell 4's last line; `report.json` and the bundle manifest carry the same flag |
| split seeds and model seeds, and how many runs | cell 5's first line, and `n_runs` in `report.csv` |
| the split ratio and sizes | 8 : 1 : 1; cell 5 prints the counts for your library |
| the one-hot floor you compared against | `one_hot_floor` in `report.json`, drawn as the dashed line in `report.png` |
| whether the number is validation or test, and how many times the test set was read | `shown_partition` and `unlock_count` in `report.json`, the badge in `report.png`, and `unlock.json` |
| that the numbers came from a Colab GPU of a given type | the GPU line printed by cell 0 |

Two sentences that are worth writing down verbatim, because a referee will otherwise ask:

- **`n_runs = 1` means one draw.** Quote the number without a ±, and say it is a single
  split seed and a single model seed.
- **A provisional configuration is a lower bound.** Say which of the two it was; the
  notebook prints it in capitals at five separate points precisely so it cannot be lost.

Cite ColabSeqDisplay, and cite the seqdisplay-opt study whose method it runs — both are
named in [`README.md`](README.md#acknowledgement), and both are in
[`CITATION.cff`](CITATION.cff), the second as a related reference, so a citation manager
picks up the pair. [`ATTRIBUTION.md`](ATTRIBUTION.md) says which parts of the workflow are
that study's work.
