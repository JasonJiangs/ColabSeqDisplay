# ColabSeqDisplay tutorial

**Part 1** runs the bundled example end to end. **Part 2** points the same page at your own
protein. **Part 3** scores new variants from a bundle you exported months ago.

The panel explains every field as you reach it. This page covers what it cannot: what to check,
what a number means, and what to do when something refuses. What the tool is, and what its numbers
are worth, is [README § Status](README.md#status).

[Before you start](#before-you-start) ·
[Part 1: the example](#part-1--the-bundled-example) ·
[Part 2: your protein](#part-2--your-own-protein) ·
[Part 3: scoring](#part-3--scoring-new-variants) ·
[A stopped run](#a-run-that-was-stopped-or-a-session-that-died) ·
[The working directory](#what-ends-up-in-the-working-directory) ·
[Reporting a result](#if-you-report-a-result)

---

## Before you start

**Set the runtime first: `Runtime ▸ Change runtime type ▸ T4 GPU`.** Without a GPU the page still
loads a library and writes both exports once something has been trained, but **Train** refuses up
front.

**There are two code cells.** The first installs the package and draws the panel that *is* this
workflow; its numbered steps appear and disappear as your answers make them relevant. The second,
at the bottom, is the only thing that can read the locked test set.

**Colab runtimes are temporary.** Everything goes to `colabsd_work/`, and both exports also land in
your browser's downloads. Keep `model_bundle.zip` (the only file Predict needs) and
`performance_report.zip` (the only file that says what your numbers were); tick **Save to Google
Drive** for the rest.

---

## Part 1 — the bundled example

Change nothing and press the buttons in order. The defaults load a 124-variant MG8 PETase library
and `ESM2-35M`, and the panel estimates about a minute.

### Step 1 · Your variant library

Press **Check my library**. Four lines come back:

> **124 variants** from `examples/mg8_petases/library.csv`.
>
> - Wild type: 287 residues.
> - 19 mutated sites at 22, 42, 49, 72, 78, 79, 80, 130, 131, 132, 150, 198, 199, 205, 206, 215, 216, 225, 279, wild-type residues `YVSIYVSTRTWVVHAGSNT`.
> - 1 condition: `activity` from 0 to 6.281.

Two of them catch errors nothing else catches:

- **`wild-type residues`** are what *your* sequence has at the positions *you* listed. Not your
  construct's? The numbering is wrong — positions counted from 0, or a tag in the FASTA. Stop here;
  every later step inherits it.
- **the range after each condition.** `from 0 to 6.281` is an activity; `from 3 to 900` is a
  read-count column you pointed at by mistake.

**Show the advanced settings** lives in this step and reveals advanced fields across the whole
page, including **Numeric precision:** in step 2.

### Step 2 · The backbone

The only modeling choice on the page. How the model reads a sequence is fixed: the embeddings at
your mutated sites, averaged. Nine backbones, cheapest first; widths, checkpoints and estimates are
in [README § Backbones](README.md#backbones). Two of them do not fit a free T4 at all, and the
panel blocks **Train** until the runtime is an L4 or A100.

| If you pick | Do this first |
|---|---|
| `ESM2-35M`, `ESM2-150M`, `ESM2-650M`, `ESMDance` | Nothing. Sequence only, free T4, no extra install. |
| any `SaProt` | A new **step 3 · The wild-type shape, as a 3Di string** appears and pushes everything below it down one ([how](#the-3di-step-for-a-saprot-backbone)). Every other backbone reads sequence alone and that step never appears. |
| `ESMC-300M` | `pip install 'esm>=3.1,<3.4'` in a cell of your own — quotes and upper pin included — then restart the runtime. From esm 3.4.0 LoRA can reach only part of the attention block and the adapter refuses the run, after a 1.3 GB download. |
| `ProtT5-XL` | An L4 or A100, and `pip install sentencepiece protobuf` — **both** — then restart. An 11 GB download. |
| `SaProt-1.3B` | An L4 or A100, **and the precision**: tick **Show the advanced settings** in step 1, then set **Numeric precision:** to `bfloat16` here. The default is `float32`, and in `float32` this backbone runs out of memory on every card Colab offers — after the download. |

**`float16` is on the precision dropdown and cannot train anything.** The optimizer's epsilon
rounds to zero in it, so every weight becomes `NaN` — and the run *finishes*, reporting a score of
`0.0000`. Picking it blocks **Train**. Use `bfloat16` when you want half precision: the same two
bytes a weight, so the same memory.

### Step 3 · The hyperparameters

*(Step 4 with a SaProt backbone.)*

The LoRA values are looked up from `config/best/` and shown read-only, with their provenance:
`ESM2-35M · tuned (test ρ=0.5608 ± 0.0120, 9 runs)`. All nine backbones on the form carry tuned
values.

**"Tuned" means tuned on another protein** — the seqdisplay-opt study's own library, with the same
mutated-site read-out a run here uses. Whether they transfer to your protein has not been measured.
A better starting point than a guess; not a configuration fitted to your data.

Three boxes are yours, prefilled from the same entry:

| Box | Prefilled | What it is |
|---|---|---|
| **Epochs, at most:** | 20 | The ceiling. Early stopping usually ends the run first. |
| **Give up after this many epochs with no gain:** | 3 | Early stopping, on the validation score. May equal the ceiling, never exceed it. |
| **Sequences on the GPU at once (memory):** | 8 | The **only** setting here that changes how much memory a run needs. Lower it after an out-of-memory error; it costs speed, not score. |

A box you move is tagged `changed · 20 looked up`, re-prices the estimate, and travels into both
exports marked as yours. A budget the loop could not honor is refused before **Train**, naming the
box.

### Steps 4 and 5 · How many runs, where they are kept

One run is one split seed × one model seed. Each slider caps at 3, so nine runs at most, and
training takes the first *n* of each — **the default single run is always split seed 1 with model
seed 11**. Leave both at 1 for a first pass; 3 × 3 is nine times the wait, and what a reportable
number needs. A single run cannot show reproducibility: the report writes its standard deviation as
`NaN`, not `0`.

Step 5 is storage. **Do not change the working directory between exporting and unlocking**, or you
end up with two archives instead of one updated file.

### Step 6 · Train

*(Step 7 with a SaProt backbone.)*

The line under the button reports as it goes:
`0/1 runs finished · run 1/1 · epoch 3/20 · batch 412/1643 · loss 0.183 · 2m14s`. The fraction on
the left counts *finished* runs, so it holds still through a single run. Expect one quiet stretch
before the first batch line, while the backbone downloads. **Keep the tab open** — closing it
disconnects the runtime, and that stops training.

The figure is three panels: training loss, validation loss, and validation Spearman, with the kept
epoch circled. Only the third decides anything; a validation loss turning up while the training
loss keeps falling is over-fitting.

**To stop a run, use Colab's own ▪** — the panel has no stop button. Your best epoch is kept,
scored and written out: see [a stopped run](#a-run-that-was-stopped-or-a-session-that-died).

### Step 7 · What came back

*(Step 8 with a SaProt backbone.)*

Validation only: the average over conditions, then Spearman, R2 and NDCG@50 per condition. Test
artifacts move into `run/locked_test/` as each run finishes.

**Read this example as a shape, not a target.** Its validation partition is 12 rows, and a rank
correlation over 12 rows moves a long way for a small change.

**Quote `Spearman` and `R2`.** Four of the archive's seven metrics are top-*k* ranking cuts, and
on a partition shorter than *k* they come out high rather than good: on 12 validation rows, `P@50`
is exactly 1.0000 for any ranking at all. The archive names the truncated metrics and the rows
behind them, so you need not note the count by hand.

### Step 8 · The two things you take away

*(Step 9 with a SaProt backbone.)*

Both buttons refuse for exactly two reasons: **nothing has been trained in this session**, or **the
settings have changed since this model was trained**.

**`model_bundle.zip`** holds `manifest.json`, `lora.pt` and `head.pt`, and **no backbone weights**,
so Predict re-fetches the backbone by name and needs internet. It is written, read straight back,
and described from the file:

```
ESM2-35M · 19 mutated sites · tuned hyperparameters · 20 epochs, micro-batch 8 (as looked up) · 1 condition · test unlocked 0x · written <utc>
```

It holds **the single best run**, not an ensemble, and is written from the training checkpoint —
export before you clear `colabsd_work/`.

**Write `performance_report.zip` now, before you unlock anything.** It publishes validation numbers
with the test partition untouched: a reportable artifact that costs nothing. Seven members — the
metrics, the figure, the training curve and its data, and a `README.txt` and `performance.json`
that say in plain words which partition the numbers describe, how many times the test set has been
read, and whether the hyperparameters were tuned. Every member carries a sha256, verified on
read-back.

One thing to know when you read them back: **`training_budget` is what a run was *given*, not what
it spent** — a headline saying 20 epochs can sit beside a two-row training curve.

### Step 9 · Score some variants

*(Step 10 with a SaProt backbone.)*

Three sources: the first rows of the library you loaded, random combinations of residues, or a
variants CSV of your own. **Score them** writes `colabsd_work/scored_variants.csv`: the mutation
columns, one predicted column per condition named `pred_<condition>`, then `pred_mean` and
`pred_min`. Every predicted column carries the `pred_` prefix, so a prediction is never written
under the name of something you measured. **Rank by `pred_min` when a hit has to work in every
condition.**

The first rows of the loaded library are mostly rows the model trained on. Comparing those
predictions with measurements you already have is a wiring check, not evidence.

### The last cell · Unlock the test set

**Stop and read this before you tick the box.** This is the one methodological decision the
notebook asks of you, and the one a referee is most likely to probe.

Everything in the panel reports validation. The test partition is evaluated at the end of each run,
but the numbers go straight into `run/locked_test/` and only this cell reads them back. Ticking the
box and pressing **Unlock the test set** increments a counter, and that count is stamped into
everything this cell writes.

| count | what the archive says, verbatim |
|---|---|
| **0** | *The test partition has not been read in this run directory.* — perfectly reportable, and the right state while you are still choosing. |
| **1** | *Read once, after the choices were made. This number means what a held-out number is supposed to mean.* |
| **2+** | *This is read number N of the same test partition. Choices made after the first read were informed by it, so treat this as an optimistic estimate, not a held-out one.* |

Several unlocks is neither a fabrication nor a disqualification; it is the record of an exploratory
session, and should be described as one.

**`model_bundle.zip` is not rewritten.** It was written in step 8, before this cell ran, so its
manifest records the count at export time. Press **Write model_bundle.zip** again afterwards if you
want the new count in it too.

**The counter belongs to the run directory.** A new run name starts a fresh counter at zero, and
deleting `unlock.json` resets it — say so in the report if you do. A run directory reused by a
later run with different settings refuses to unlock, rather than handing back the older result's
test numbers under the newer name.

---

## Part 2 — your own protein

Same page, same order; the differences are in steps 1 and 2. You need the wild-type sequence —
pasted, or as a FASTA URL — and one CSV:

| Column | What it holds | Required |
|---|---|---|
| `p22` … `p279` | One column per mutated site, in the order of the positions you give. Three-letter (`Asn`) or one-letter (`N`) residue codes — a form field says which. | yes |
| `activity` | One column per measured condition. Numbers. All conditions are predicted together. | yes |
| `count` | Sequencing reads behind each variant, if you have them. Only used to drop low-count rows. | no |

The names and the number of columns are yours, and any column you do not name is carried through
untouched. Three fields on the form connect the CSV to the protein: the wild-type sequence, the
positions of the mutated sites (counting from 1), and the mutation column names in the same order.

**Variant sequences are built from the wild type, never read from a FASTA**, so every position is
checked against the sequence you gave. The loader wants unique column names; residues all in one
alphabet (three-letter `Asn` with **Residues are written as Asn, not N** ticked, one-letter `N`
with it clear); numeric, finite condition values; and positions strictly increasing and **counted
from 1 in the full-length wild type**, with one mutation column per position in the same order.
Every refusal names the offending column and row. Then read the `wild-type residues` line, as in
[step 1](#step-1--your-variant-library).

**At least 20 rows**, counted after low-count rows have been dropped. The 8:1:1 split is 80/10/10
rounded down, so 19 variants hand validation exactly one row — nothing for a rank correlation to
rank. 20 gives 16/2/2 and goes through. That is a floor, not a library worth training on.

### The 3Di step, for a SaProt backbone

Only the SaProt family needs one, and **nothing of the kind is bundled**. Three sources:

- **Upload a structure of my wild type (.pdb / .cif) and read the shape off it** — use this.
  Download your protein from <https://alphafold.ebi.ac.uk> and upload the `.cif`: free, seconds on
  any runtime, and it matches your sequence exactly.
- **Upload a 3Di text file I already have**
- **Paste a 3Di string I already have**

The notebook does not predict structures — bring one, or bring the 3Di string.

The string is written to `colabsd_work/wt_3di.txt` and offered back for the rest of the session. If
your structure is a slightly different construct, a handful of disagreeing residues is reported
rather than refused — read that line, it names the positions.

### A sensible first pass on a new library

1. Load your CSV, read the `wild-type residues` line, and stop if it surprises you.
2. `ESM2-35M` — no 3Di step, cheapest run on the form. This checks that your data flows through
   training and the report. **Do not unlock the test set.**
3. Write the performance archive from that run anyway. It costs nothing.
4. The real run: the backbone you mean to use, 3 split seeds × 3 model seeds, **Save to Google
   Drive** if the estimate is long, the archive again, then one unlock at the end.

---

## Part 3 — scoring new variants

`colab/ColabSeqDisplay_Predict.ipynb`. Upload a `model_bundle.zip` and a table of variants; get a
ranked table back. No training, and the bundle is self-contained, so you are never asked to
describe your library — but it needs internet, because the backbone is fetched by name on the first
scoring run. Three buttons:

- **Load the bundle** prints what is in it: backbone, mutated sites, hyperparameter status, epochs
  trained, unlock count, conditions, and the variant columns it needs. Read those as provenance —
  placeholder hyperparameters, or an unlock count of zero, mean you can rank with it but should not
  quote its numbers. A bundle built on a backbone the training form no longer offers still loads
  and scores — except `METL`, which has no adapter here and is refused as it loads.
- **Prepare the variants** — your own CSV, random combinations of residues, or the first rows of a
  library CSV. The last two add a **How many:** box; random combinations also add a **Seed:** and
  drop duplicate draws, so you can get fewer rows than you asked for. Your own CSV needs one column
  per mutated site, named exactly as the bundle printed.
- **Score and rank** by average over conditions or by worst condition, into `ranked_variants.csv`.
  Both columns are always written, so you can re-sort without re-scoring.

**Check the `prediction scale` row of the bundle table.** With a label scaler the predictions are in
the assay's own units. Without one they are in z-scored training units: the ranking is meaningful,
the numbers are not comparable to your measurements, so do not plot them on the same axis.

These are predictions from a model fitted to one library. **They rank variants; they do not measure
them.**

---

## A run that was stopped, or a session that died

**The weights are already on disk.** `best_checkpoint.pt` under
`colabsd_work/run/runs/<split>_<seed>/` is rewritten every time the validation score improves, so
whatever epoch was best when the run ended is there. The next **Train** press renames it to
`unfinished_checkpoint.pt` *before* the retry writes anything, and prints where it went. Nothing in
this package deletes a checkpoint.

**See what survived**, in a cell of your own:

```python
from colabsd.train import recover_runs

for run in recover_runs("colabsd_work/run"):
    print(run.describe())
# split1_seed11 · interrupted after 2 epochs · best epoch 1 · validation Spearman 0.2857 · colabsd_work/run/runs/split1_seed11/best_checkpoint.pt
```

**Turn one into a model you can score with**, with no finished run behind it:

```python
from colabsd.bundle import save_bundle_from_checkpoint

save_bundle_from_checkpoint(
    "colabsd_work/rescued_bundle.zip",
    checkpoint="colabsd_work/run/runs/split1_seed11/unfinished_checkpoint.pt",
    spec=wizard.spec, best=wizard.best,      # the panel's own, after step 1 has run
)
```

It loads in Predict like any other bundle, and its manifest records that it came from a run that
was cut short.

**Or just press Train again.** A short run is retrained rather than re-reported. A run that *did*
finish is reused only when its fingerprint is identical: backbone, adapter, hyperparameters, the
three budget boxes, mutated sites, split, and a hash of your data. So if you fetch a better
structure and re-derive the 3Di, you get a new run, not the old structure's numbers.

No artifact ever publishes a 20-epoch budget as a 20-epoch run: `report.png` carries a footer, the
bundle manifest a `training_status` line, `README.txt` a matching sentence, and both JSON files a
`runs_stopped_early` list.

---

## What ends up in the working directory

```
colabsd_work/                            (/content/colabsd_work/ in Colab)
├── splits/n124/split/seed_1/            the cached 8:1:1 split, keyed by library size
├── wt_3di.txt                           written by the 3Di step, if that step existed
├── run/                                 (named in step 5)
│   ├── runs/split1_seed11/
│   │   ├── best_checkpoint.pt          the best epoch so far, rewritten as the run improves
│   │   ├── unfinished_checkpoint.pt    a cut-short run's weights, moved here before a retrain
│   │   ├── metrics.json                this run's metrics, test block removed
│   │   └── colabsd_run.json            written last; its absence means the run never finished
│   ├── locked_test/split1_seed11/       the test partition, until you unlock it
│   ├── unlock.json                      the unlock counter and its history
│   ├── run_result.json                  the run, as the report reads it
│   ├── validation_runs.csv              one row per run
│   ├── validation_summary.csv           mean ± sd across runs
│   ├── test_metrics.json                written by an unlock
│   └── test_summary.csv                 written by an unlock
├── report/report.csv · report.json · report.png · training_curve.csv · training_curve.png
├── model_bundle.zip
├── performance_report.zip
└── scored_variants.csv
```

Keep **`model_bundle.zip`** and **`performance_report.zip`**; the rest regenerates from them and
your library. A 3Di string is worth keeping only if you will train the same protein again in a
fresh runtime — tick **Save to Google Drive** before it is written.

---

## If you report a result

All of it is printed by the notebook or written into `performance_report.zip` and the bundle
manifest. Nothing has to be reconstructed afterwards. State:

- **the tool, its version and its license** — the setup cell's first line, and
  [`CITATION.cff`](CITATION.cff).
- **the backbone and the mutated sites it averages** — the bundle's description line names both.
- **whether the hyperparameters were tuned — on another protein — or a placeholder** — the
  `hyperparameters` status in `performance.json`, and the same flag in the manifest.
- **the budget the run was given, and which of it you set** — `training_budget` in both exports.
- **the seeds and how many runs.** `n_runs = 1` is one draw, quoted without a ±.
- **the split and the rows in each partition** — 8:1:1, and `partition_rows` in `report.json`.
- **validation or test, and how many times the test set was read** — `partition`, `unlock.count`
  and `unlock.verdict` in `performance.json`, and the badge on `report.png`. A validation-only
  archive is reportable, and says for itself that the test partition was never read.
- **the GPU** — the line the setup cell prints.

**Predictions rank variants; they do not measure them.**

Cite ColabSeqDisplay and the seqdisplay-opt study whose method it runs. Both are in
[`CITATION.cff`](CITATION.cff), the second as a related reference, so a citation manager picks up
the pair; [`ATTRIBUTION.md`](ATTRIBUTION.md) says which parts of the workflow are that study's.
