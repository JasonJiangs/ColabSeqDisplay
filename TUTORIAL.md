# ColabSeqDisplay tutorial

**Part 1** runs the bundled example end to end, **Part 2** points the same page at your own
protein, **Part 3** scores new variants from a bundle you exported months ago.

The panel explains every field as you reach it. This page covers what it cannot: what to check,
what a number means, what to do when something refuses. What the tool is and what its numbers are
worth is [README § Status](README.md#status).

**Contents** — [Before you start](#before-you-start) ·
[Part 1: the bundled example](#part-1--the-bundled-example) ·
[Part 2: your own protein](#part-2--your-own-protein) ·
[Part 3: scoring new variants](#part-3--scoring-new-variants) ·
[A run that was stopped](#a-run-that-was-stopped-or-a-session-that-died) ·
[Troubleshooting](#troubleshooting) ·
[The working directory](#what-ends-up-in-the-working-directory) ·
[If you report a result](#if-you-report-a-result)

---

## Before you start

**Set the runtime first: `Runtime ▸ Change runtime type ▸ T4 GPU`.** Without a GPU the page still
loads a library and writes both exports once something has been trained; **Train** and ESMFold
refuse up front.

**Two code cells.** The first installs the package and draws the panel that *is* this workflow; its
numbered steps appear and disappear as your answers make them relevant. Read the three lines it
prints: version, GPU, and the directory this session writes to. The second cell, at the bottom, is
the only thing that can read the locked test set.

**Colab runtimes are temporary.** Everything goes to `colabsd_work/`, and both exports also land in
your browser's downloads. Keep `model_bundle.zip` (the only file Predict needs) and
`performance_report.zip` (the only file that says what your numbers were); tick **Save to Google
Drive** for the rest.

---

## Part 1 — the bundled example

Change nothing and press the buttons in order. The defaults load a 124-variant MG8 PETase library
and `ESM2-35M`; the panel estimates **about 1 min for 1 run**.

### Step 1 · Your variant library

Press **Check my library**. Four lines come back:

> **124 variants** from `examples/mg8_petases/library.csv`.
>
> - Wild type: 287 residues.
> - 19 mutated sites at 22, 42, 49, 72, 78, 79, 80, 130, 131, 132, 150, 198, 199, 205, 206, 215, 216, 225, 279, wild-type residues `YVSIYVSTRTWVVHAGSNT`.
> - 1 condition: `activity` from 0 to 6.281.

Read all four. Two catch errors nothing else catches:

- **`wild-type residues`** are the residues *your* sequence carries at the positions *you* listed.
  Not your construct's? The numbering is wrong — 0-based positions, or a tag or signal peptide in
  the FASTA. Stop here; every later step inherits it.
- **the range after each condition.** `from 0 to 6.281` is an activity; `from 3 to 900` is a
  read-count column you pointed at by mistake.

**Show the advanced settings** lives here in step 1 and reveals advanced fields across the whole
page, including **Numeric precision:** in step 2.

### Step 2 · The backbone

The only modeling choice on the page: how the model reads a sequence is fixed — the embeddings at
your mutated sites, averaged. Nine backbones, cheapest first. Each label reads *name — whether it
needs a 3Di string and, for the seven that fit a T4, roughly how many minutes one run costs there*.
The two with no minutes (`ProtT5-XL`, `SaProt-1.3B`) do not fit a free T4 at all, and the panel
blocks **Train** until the runtime is an L4 or A100. Widths and checkpoints:
[README § Backbones](README.md#backbones).

| If you pick | Do this first |
|---|---|
| `ESM2-35M`, `ESM2-150M`, `ESM2-650M` | Nothing. Sequence only, free T4, no extra install. |
| `ESMDance` | Nothing — but read its score as a different read-out, not as a much smaller model doing as well: its feature is its own 50-dimensional `res_pred` dynamics prediction, where every other backbone's is a trunk embedding 480 to 1280 wide. |
| any `SaProt` | A new **step 3 · The wild-type shape, as a 3Di string** appears and pushes everything below down one: Train becomes step 7, Score step 10 ([how](#the-3di-step-for-a-saprot-backbone)). **Every other backbone reads sequence alone and that step never appears on the page.** |
| `ESMC-300M` | `pip install 'esm>=3.1,<3.4'` in a cell of your own — quotes and upper pin included — then restart. From esm 3.4.0 the attention block is fused, so LoRA reaches only `out_proj` and the adapter refuses the run by name — after a 1.3 GB download and a press of **Train**. The install also drags `transformers` below 4.48.2, so the restart is not optional. |
| `ProtT5-XL` | An L4 or A100, and `pip install sentencepiece protobuf` — **both** — then restart. `sentencepiece` reads the vocabulary, `protobuf` parses it once `transformers` falls back to the slow T5 tokenizer. An 11 GB download. |
| `SaProt-1.3B` | An L4 or A100, **and the precision**: tick **Show the advanced settings** (step 1), then set **Numeric precision:** to `bfloat16` here in step 2. It defaults to `float32`, and in `float32` this backbone runs out of memory on every card Colab offers — after the download. Not `float16`: see below. |

**`float16` is on the precision dropdown and cannot train anything.** It is two bytes a weight
exactly like `bfloat16`, so it saves the same memory, but the optimizer's epsilon — 1e-8 — is
smaller than the smallest number `float16` holds and rounds to zero in it: the first update divides
by zero and every weight becomes `NaN`. The run would not crash; it would *finish*, and the
results step would report **Validation Spearman, averaged over conditions: 0.0000**, which is the dead model and
not a score. Picking it therefore blocks **Train** with a red box saying so, and any run whose loss
curve or predictions come back non-finite for some other reason is refused rather than reported.
Use `bfloat16` when you want half precision. `float16` stays on the dropdown so that a reader who
followed an older instruction to it is told why rather than finding it gone, and a
`model_bundle.zip` that already *records* `float16` still scores — a frozen forward pass in it is
finite. You never pick that: the Predict notebook has no precision control and rebuilds the model
in the precision the bundle recorded.

Four of the nine publish weights only as `pytorch_model.bin` — the three SaProts and `ProtT5-XL` —
and cannot load below torch 2.6 at all. Colab clears that floor.

### Step 3 · The hyperparameters for this backbone

*(Step 4 with a SaProt backbone.)*

Seven LoRA values are looked up from `config/best/` and shown read-only, under their provenance:
`ESM2-35M · tuned (test ρ=0.5608 ± 0.0120, 9 runs)`. **All nine backbones on the form carry tuned
values**, so no choice here produces a placeholder.

**"Tuned" means tuned on another protein.** Those ρ, and the values behind them, come from an
Optuna search the seqdisplay-opt study ran on its own 16,424-variant SlugCas9 5NNK library, with
the same mutated-site read-out a run here uses. **How well they transfer to your protein has not
been measured.** A better starting point than a guess; not a configuration fitted to your data. The
nine ρ: [README § Configurations](README.md#configurations).

Three boxes are yours, prefilled from the same entry:

| Box | Prefilled | What it is |
|---|---|---|
| **Epochs, at most:** | 20 | The ceiling. Early stopping usually ends the run first. |
| **Give up after this many epochs with no gain:** | 3 | Early stopping, on the validation score. May equal the ceiling, never exceed it. |
| **Sequences on the GPU at once (memory):** | 8 | The **only** setting on this form that changes how much memory a run needs. Lower it after an out-of-memory error. |

A box you move is tagged `changed · 20 looked up`, re-prices step 4's estimate, and travels into
both exports as `training_budget` with `chosen_by_user` naming your fields. A budget the loop could
not honor is refused before **Train**, naming the box: no epochs; a patience longer than the run; a
micro-batch that does not divide the effective batch, exceeds it, or exceeds your smallest training
split.

### Steps 4 and 5 · How many runs, where they are kept

One run is one split seed × one model seed. The entry offers split seeds `[1, 2, 3]` and model
seeds `[11, 22, 33]`, so each slider caps at 3 — nine runs at most — and training takes the first
*n* of each, so **the default single run is always split seed 1 with model seed 11**. Leave both at
1 for a first pass; 3 × 3 is nine times the wait, and what a reportable number needs. A single run
cannot show reproducibility: the report writes its standard deviation as `NaN`, not `0`, and the
archive renders it *undefined (a single run)*.

Step 5 is storage. **Do not change the working directory between exporting and unlocking**, or you
end up with two archives instead of one updated file.

### Step 6 · Train

*(Step 7 with a SaProt backbone.)*

The line under the button reports as it goes:
`0/1 runs finished · run 1/1 · epoch 3/20 · batch 412/1643 · loss 0.183 · 2m14s`. The fraction on
the left counts **finished** runs, so it holds still through a single run; everything after it
belongs to the run in flight, and an epoch closes on `epoch 3/20 · val 0.7412 · loss 0.171`. Expect
one quiet stretch before the first batch line — `0/1 runs finished · starting`, while the backbone
downloads. **Keep the tab open:** closing it disconnects the runtime, and that stops training.

**The figure** is three panels side by side: training loss (MSE), the same MSE on validation, and
validation Spearman, one line per run with the kept epoch circled. Only the third decides anything;
a validation loss turning up while the training loss keeps falling is over-fitting. It is the same
figure as `training_curve.png` in the archive, and one image whose bytes are replaced, so the
output never fills with pictures.

**To stop a run, use Colab's own ▪** — the panel has no stop button. Your best epoch is kept,
scored and written out: see [A run that was stopped](#a-run-that-was-stopped-or-a-session-that-died).

The estimate above the button is arithmetic: a registry figure rescaled by your runs, epochs and
variant count, floored at one minute. **The elapsed time printed at the end is the only timing here
measured on your hardware** ([README § Status](README.md#status)).

### Step 7 · What came back

*(Step 8 with a SaProt backbone.)*

Validation only: the macro average over conditions, then Spearman, R2 and NDCG@50 per condition.
Test artifacts move into `run/locked_test/` as each run finishes.

**Read this example as a shape, not a target.** Its validation partition is 12 rows, and a rank
correlation over 12 rows moves a long way for a small change.

**Four of the archive's seven metrics are ranking cuts** — `P@10`, `P@50`, `NDCG@10`, `NDCG@50` —
each scoring the top `min(k, n)` of an `n`-row partition. The panel shows only Spearman, R2 and
NDCG@50, and drops even NDCG@50 with a note on a partition this short. What those four are worth on
12 rows:

| metric | what a *random* ranking of those rows scores |
|---|---|
| `P@50` | **exactly 1.0000, always.** Two twelve-element subsets of twelve are the same set. |
| `P@10` | **never below 0.8000** — never below 0.7000 on the 13 test rows. |
| `NDCG@10` | median **0.58**; 72% of random rankings clear 0.50. |
| `NDCG@50` | median **0.63**; only 43% clear 0.65. |

*(20,000 random rankings of the real split-seed-1 validation labels, which is the default run.
Split seeds 2 and 3 give visibly different medians — that is the point.)*

**Quote `Spearman` and `R2`,** and do not note the row count by hand. Both files carry
`partition_rows`, but they answer different questions with it: in `report.json` it is a mapping
over every partition — `{"train": ..., "validation": ..., "test": ...}` — while in
`performance.json` it is a single integer, the row count of the one partition that archive
reports on. Both carry `truncated_metrics`, and `README.txt` prints a `library size` line and a
`not a real cut` line naming the columns and the rows behind them.

### Step 8 · The two things you take away

*(Step 9 with a SaProt backbone.)*

Both buttons refuse for exactly two reasons and no others: **nothing has been trained in this
session**, or **the settings have changed since this model was trained**.

**`model_bundle.zip`** holds three members — `manifest.json`, `lora.pt`, `head.pt` — and **no
backbone weights**, so Predict re-fetches the backbone by name and needs internet. It is written,
read straight back, and described from the file:

```
ESM2-35M · 19 mutated sites · tuned hyperparameters · 20 epochs, micro-batch 8 (as looked up) · 1 condition · test unlocked 0x · written <utc>
```

`tuned` becomes `PROVISIONAL` for a placeholder entry, and the budget clause reads
`(budget set by hand)` if you moved a box. It holds **the single best-validation run**, not an
ensemble, and is written from the training checkpoint — export before you clear `colabsd_work/`.

**Write `performance_report.zip` now, before you unlock anything.** It publishes validation numbers
with the test partition untouched: a reportable artifact that costs nothing. Seven members:

| member | what it holds |
|---|---|
| `report.csv` | one row per partition × condition plus a `mean` row: `n_runs`, then a mean, sd and count for each of R2, Pearson, Spearman, P@10, P@50, NDCG@10, NDCG@50 |
| `report.json` | the headline numbers, plus `unlock_count`, `partition_rows`, `truncated_metrics` and `runs_stopped_early` |
| `report.png` | the metrics per condition and averaged over them, error bars ±1 sd, an unlock badge, and a footer line if a run was stopped by hand |
| `performance.json` | one manifest: partition, unlock count and verdict, partition rows, model, hyperparameter status, the budget the run was *given* and which of it was yours, conditions, seeds, version, notes |
| `README.txt` | the same answers in prose, for whoever will not read JSON |
| `training_curve.png` | the figure step 6 drew while it trained |
| `training_curve.csv` | its points: `split_seed`, `model_seed`, `run`, `epoch`, `train_loss_mse`, `val_loss_mse`, `val_spearman`, `objective`, `is_best` |

Every member carries a sha256, verified on read-back; the five report files also land loose in
`colabsd_work/report/` — everything in the table above except `README.txt` and `performance.json`,
because the report writer and the curve writer are handed the same output directory. **`training_budget` is what a run was *given*, not what it spent** — a
headline saying 20 epochs can sit in the same zip as a two-row `training_curve.csv`. If you stopped
a run, see [what the artifacts say](#a-run-that-was-stopped-or-a-session-that-died).

### Step 9 · Score some variants

*(Step 10 with a SaProt backbone.)*

Three sources: **The first rows of the library I loaded**, **Random combinations of residues**, or
**A variants CSV of my own**. **Score them** writes `colabsd_work/scored_variants.csv`: the
mutation columns, then one predicted column per condition named `pred_<condition>` —
`pred_activity` for the bundled example — then two summaries, `pred_mean` and `pred_min`, the worst
condition. Every predicted column carries the `pred_` prefix and nothing but the mutation columns
comes through from the table you handed in, so a prediction is never written under the name of
something you measured. **Rank by `pred_min` when a hit has to work in every condition.**

The first rows of the loaded library are mostly rows the model trained on, since 80% of the library
is the training split. Comparing those predictions with measurements you already have is a wiring
check, not evidence.

### The last cell · Unlock the test set

**Stop and read this before you tick the box.** This is the one methodological decision the
notebook asks of you, and the one a referee is most likely to probe.

Everything in the panel reports validation. The training loop does evaluate the test partition at
the end of each run, but every test artifact is moved straight into `run/locked_test/`, and only
`colabsd.train.unlock_test` reads it back. Ticking *I have stopped changing things. Read the test
set once and count it.* and pressing **Unlock the test set** increments a counter in
`run/unlock.json`. That count is stamped into the run result, `report.csv`, `report.json`, the
badge and footer of `report.png`, and `performance.json` inside the archive — every artifact this
cell writes or rewrites.

**`model_bundle.zip` is not one of them.** It was written back in step 8, before this cell ran, so
its manifest records the count *at export time* — `test unlocked 0x` in the first printed line of a
bundle exported before the first unlock, which is the order this page recommends. The panel says so
after an unlock. Press **Write model_bundle.zip** again afterwards if you want the new count in the
bundle too; a colleague opening that copy a year later reads `test unlocked 1x` in its first
printed line.

| count | the verdict the archive writes, verbatim (*N* is the count) |
|---|---|
| **0** | *The test partition has not been read in this run directory.* — perfectly reportable, and the right state while you are still choosing. |
| **1** | *Read once, after the choices were made. This number means what a held-out number is supposed to mean.* |
| **2+** | *This is read number N of the same test partition. Choices made after the first read were informed by it, so treat this as an optimistic estimate, not a held-out one.* |

A validation archive written after an unlock says so too: *These are validation numbers, but the
test partition of this run directory has already been read 1x.* Several unlocks is neither a
fabrication nor a disqualification; it is the record of an exploratory session, and should be
described as one.

**The counter belongs to the run directory.** A new run name starts a fresh counter at zero, and
deleting `unlock.json` resets it — say so in the report if you do. One guard: a run directory
reused by a later training run with different hyperparameters, data or splits **refuses to
unlock**, rather than handing back the older result's test numbers under the newer name.

The cell then prints the test Spearman and a boxed final report, and rewrites
`performance_report.zip` in place with the test numbers and the count.

---

## Part 2 — your own protein

Same page, same order; the differences are in steps 1 and 2. You need `library.csv` (schema and a
worked row: [README § Your own library](README.md#your-own-library)) and the wild-type sequence,
pasted or as a FASTA URL.

**Variant sequences are built from the wild type, never read from a FASTA**, so every position is
checked against the sequence you gave. The loader wants unique column names; residues all in one
alphabet (three-letter `Asn` with **Residues are written as Asn, not N** ticked, one-letter `N`
with it clear); numeric, finite condition values; and positions strictly increasing and **1-based
in full-length wild-type numbering**, with one mutation column per position in the same order — the
two fields are read as parallel lists. Every refusal names the offending column and row. Then read
the `wild-type residues` line, as in [step 1](#step-1--your-variant-library).

**At least 20 rows**, counted after `min_count` has dropped what it drops. The 8:1:1 split is
80/10/10 rounded down and its partition sizes come from the row count alone — the seed chooses
which rows go where, never how many — so 19 variants hand validation exactly one row, for every
seed, and one row is a partition NDCG is undefined on and Spearman has nothing to rank in. 20 gives
16/2/2 and goes through; anything smaller is refused by name before a run starts. That is a floor,
not a library worth training on.

### The 3Di step, for a SaProt backbone

Only the SaProt family needs one, and **nothing of the kind is bundled**. Four sources:

- **Upload a structure of my wild type (.pdb / .cif) and read the shape off it** — use this.
  Download your protein from <https://alphafold.ebi.ac.uk> and upload the `.cif`: free, seconds on
  any runtime, and it matches your sequence exactly.
- **Upload a 3Di text file I already have**
- **Paste a 3Di string I already have**
- **Fold the wild type here with ESMFold (slow, memory-hungry, last resort)** — a 2.6 GB checkpoint
  and ~14–16 GB of GPU memory, the whole of a free T4. The step refuses a wild type longer than
  **700 residues**, and refuses any wild type until you tick **I accept the ESMFold memory risk**.

The string is written to `colabsd_work/wt_3di.txt` and offered back for the rest of the session. If
your structure is a slightly different construct, a handful of disagreeing residues is reported
rather than refused — read that line, it names the positions.

### The PROVISIONAL warning

No backbone on the form is a placeholder, so the panel cannot show you one; only an older
`model_bundle.zip` loaded in Predict, or a run started from Python, still can. It reads, verbatim:

> `ESM2-8M · PROVISIONAL — hyperparameters are a placeholder, performance unknown. Read every number it produces as a lower bound, not a result.`

A backbone reaches the form only when `config/best/` holds tuned values for it and its adapter
builds; the five withheld names and their reasons are in
[README § Backbones](README.md#backbones).

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

- **Load the bundle** prints *What is in this bundle* and a table: backbone, mutated sites,
  hyperparameter status, epochs trained, unlock count, conditions, and the variant columns it
  needs. Read those rows as provenance — `PROVISIONAL` hyperparameters earn *rank with it, do not
  quote its numbers*, and `unlocked 0x during training` earns *no held-out number stands behind
  these predictions*, which the page also says in its own words above the table. Handing it a `performance_report.zip` by mistake is diagnosed, not crashed on. A
  bundle built on a backbone the training notebook no longer offers still loads and scores — except
  `METL`, which has no adapter here and is refused as the bundle loads.
- **Prepare the variants** — *A variants CSV of my own* (all of its rows), *Random combinations of
  residues*, or *The first rows of a library CSV*. The last two add a **How many:** box prefilled
  at 200; random combinations also add a **Seed:** (0) and drop duplicate draws, so you can
  get fewer rows than you asked for. Your own CSV needs one column per mutated site, named exactly
  as the bundle table printed; other columns are ignored. The panel prints how many are ready.
- **Score and rank** by *Average over conditions (pred_mean)* or *Worst condition (pred_min)*, into
  `ranked_variants.csv`. Both columns are always written, so you can re-sort without re-scoring, and
  the per-condition predictions sit beside them as `pred_<condition>` — the `pred_` prefix is there
  so a predicted number can never be mistaken for a measured one.

**Check the `prediction scale` row of the bundle table.** With a label scaler it reads *the assay's
own units — this bundle carries the label scaler it was trained with*. Without one: *z-scored
training units, because this bundle carries no label scaler. The ranking is meaningful; the numbers
are not comparable to your measurements* — so do not plot them on the same axis as your assay.

These are predictions from a model fitted to one library. **They rank variants; they do not measure
them.**

---

## A run that was stopped, or a session that died

**The weights are already on disk.** `best_checkpoint.pt` under
`colabsd_work/run/runs/<split>_<seed>/` is rewritten every time the validation score improves, so
whatever epoch was best when the run ended is there. The next **Train** press renames it to
`unfinished_checkpoint.pt` *before* the retry writes anything — numbered if one is already there —
and prints where it went. Nothing in this package deletes a checkpoint.

**See what survived,** in a cell of your own (importing by name, because `import colabsd` alone
does not bring its submodules):

```python
from colabsd.train import recover_runs

for run in recover_runs("colabsd_work/run"):
    print(run.describe())
# split1_seed11 · interrupted after 2 epochs · best epoch 1 · validation Spearman 0.2857 · colabsd_work/run/runs/split1_seed11/best_checkpoint.pt
```

`status` is `finished`, `interrupted`, or `unfinished` (a run that vanished without writing its
bookkeeping); each record also carries `epochs_run`, `best_epoch`, `best_validation_score` and the
path. **Turn one into a model you can score with**, with no finished run behind it:

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

**Or just press Train again.** *Reuse finished runs if this is re-run* counts a run directory as
finished only when it holds a `colabsd_run.json` whose `stopped_early` is not `interrupted`, so a
short run is retrained rather than re-reported. A run that *did* finish is reused only when its
fingerprint is identical: model, the adapter actually built from it (HuggingFace id, dtype, and for
SaProt the wild-type 3Di string), hyperparameters, the three budget boxes, mutated sites, split and
a hash of your data. Fetch a better structure, re-derive the 3Di, press Train, and you get a new
run rather than the old structure's numbers.

**What the artifacts say about a shortened run**, so a 20-epoch budget is never published as a
20-epoch run:

| where | what it reads |
|---|---|
| footer of `report.png` | `STOPPED EARLY BY HAND: split1_seed11 (2 of 20 epochs) — these are that shortened run's numbers` |
| bundle manifest, `training_status` | `STOPPED EARLY BY HAND: trained 2 of 20 epochs` |
| archive `README.txt` | `split 1 · seed 11 (2 of 20 epochs) — these numbers include a run that was stopped by hand` |
| `report.json`, `performance.json` | `runs_stopped_early`: one entry per stopped run, with the epochs it ran and the epochs it was given. An empty list is what makes a non-empty one worth reading |
| `validation_runs.csv` | `epochs_run`, `epochs_budget`, `stopped_early` columns |

---

## Troubleshooting

Every failure raises an error whose message says what to do. These are the ones where the fix is
somewhere the message cannot point:

| what you see | the fix |
|---|---|
| **the panel freezes and nothing responds** | A file picker is open somewhere on the page and Colab's upload blocks the kernel. **Cancel upload** releases it. |
| a backbone you used before is missing from the dropdown | It reaches the form only with tuned values in `config/best/` and an adapter that builds; the dropdown note names all five withheld and why ([README § Backbones](README.md#backbones)). Four of the five still score in Predict; `METL` does not. |
| `does not fit a free T4 … Switch to an L4 or A100` | `ProtT5-XL` and `SaProt-1.3B`. Pick a smaller backbone or change the runtime. Nothing is downloaded first — the block lands before the adapter is built. |
| `SaProt-1.3B` loads only in `bfloat16` | Tick **Show the advanced settings** (step 1), then set **Numeric precision:** to `bfloat16` in **step 2**, the backbone section — step 2 whether or not a 3Di step exists. The default is `float32`, and in `float32` this backbone runs out of memory on every card Colab offers, *after* the download. |
| **Numeric precision `float16` cannot train a model** | Set it to `bfloat16` — the same two bytes a weight, so the same memory — or to `float32`. The optimizer's epsilon (1e-8) is smaller than the smallest number `float16` holds and rounds to zero in it, so the first update divides by zero and every weight becomes `NaN`: the run *finishes* and reports Spearman `0.0000`, which is the dead model rather than a score. It is still on the dropdown so that a reader who followed an older instruction to it is told why rather than finding it gone, and a `model_bundle.zip` that already records `float16` still scores — the Predict notebook rebuilds the model in the precision the bundle recorded and never asks for one. |
| `trained a model whose numbers are not finite: train_loss is nan at epoch 0` | The run produced a dead model and is refused instead of published: nothing on the page shows those numbers, no bundle is written, and pressing **Train** again retrains it rather than reusing it. The usual cause is the precision above; a diverging learning rate or a non-finite condition value reaches the same place. The message names the run directory, and `training_log.json` inside it holds the loss curve the refusal read. |
| `Split seed 1 gives val_idx only 1 of this library's 14 variants` | The library is too small. **20 variants** is the floor, for every split seed, because 8:1:1 partition sizes come from the row count alone. Add rows, or lower **Drop variants seen fewer times than:** (tick **Show the advanced settings** to see it) if that is what dropped them. |
| `is the amino-acid sequence, not a 3Di string: it agrees with the wild type at N of M positions` | The 3Di box or file was given the protein sequence. The 3Di alphabet is the same twenty letters as the amino-acid alphabet, lower-cased, so the letters alone cannot tell them apart and the check compares the string with your wild type instead. Upload an AlphaFold `.pdb`/`.cif` and let foldseek read the shape off it, or fold the wild type with ESMFold. |
| `reads a T5 sentencepiece vocabulary …` | `ProtT5-XL` needs **two** packages: `pip install sentencepiece protobuf`, then restart the runtime. `sentencepiece` alone leaves the tokenizer unbuildable. |
| `ESMC-300M` refuses the run after the download | An unpinned `pip install esm` gets 3.4 or newer, where LoRA reaches only `out_proj`. Install `'esm>=3.1,<3.4'` with the quotes and restart — it also moves `transformers` below 4.48.2. |
| a SaProt or `ProtT5-XL` checkpoint will not load at all | Those four publish no safetensors, and `transformers` will not `torch.load` a `pytorch_model.bin` below torch 2.6 — the first cause the "could not load" message names. Colab clears that floor; an older install may not. |
| CUDA out of memory part-way through training | Halve **Sequences on the GPU at once (memory):** in the hyperparameters section — the only setting that changes how much memory a run needs. Gradient accumulation keeps the effective batch where the entry set it, so this costs speed, not score. If it is already 1, the backbone does not fit this card at all. `Runtime ▸ Restart` afterwards. |
| `The cached split in .../seed_1 covers N rows, not the M rows of this library` | You changed the library while reusing the working directory. Change the run name, or start a fresh runtime. |
| `holds the test metrics of a different training configuration` | The run directory was reused by a later run with different hyperparameters, data or splits. Give each configuration its own run name. |
| `No checkpoint at .../best_checkpoint.pt` | Exporting needs the training run on disk. Write both `.zip` files before clearing the working directory or restarting. |
| ``Your variants table is missing 1 column: `p22` …`` (Predict) | Your CSV's column names are not the bundle's. Rename them, or press **Download an example variants CSV** and fill that in. |
| `<file> is a schema-1 bundle` (Predict) | Written when the pooling was still a choice, so this version cannot tell which residues it was fitted on. Schema 2 and 3 still load. Re-export it from its training run, or use the version the message names. |
| the runtime disconnects during training | Press Train again: a cut-short run is retrained, its weights moved to `unfinished_checkpoint.pt` first. To read them instead, see [A run that was stopped](#a-run-that-was-stopped-or-a-session-that-died). Next time tick **Save to Google Drive** and keep the tab open. |
| `GPU  none — Runtime > Change runtime type > T4 GPU` | Switch the runtime and re-run the setup cell. Loading, hyperparameters and reports work without a GPU; training and ESMFold refuse up front. |
| `<name> could not be downloaded from <address>`, or a failing `pip install -e` (setup) | The clone or the install failed, and the tool's own output is quoted above the message. Check the address in `colabsd_repository`; a truncated copy is the usual install failure — re-run the cell. |
| the setup cell reinstalls every time | It counts the install as done only when `colabsd` imports **and** its hyperparameter registry and bundled example came with it. A partial copy repeats forever. |
| `<file> has N residues but the WT sequence has M`, or `wt_3di is N characters but the wild-type sequence is M residues` | The 3Di string does not cover the same chain as your wild type. Usually chain selection: put the right id in **Chain to read:**. |
| `<file> contains 2 chains: A (N residues), B (M residues). Pass chain="A"` | Foldseek made one 3Di string per chain. Name yours in **Chain to read:**; `has no chain 'X'` means that id is not in the file. |
| `<file> is a different protein from the WT: N of M residues disagree` | The chain you picked is not the protein your library mutates. Pick another, or fold the wild-type sequence itself. |
| `contains characters that are not foldseek 3Di states` | The file holds a character the 3Di alphabet has no state for — a FASTA header line, a gap character, or one of `B J O U X Z`. This is *not* the check that catches a pasted sequence: the other twenty 3Di letters are the twenty amino-acid letters lower-cased, so a clean sequence passes it. The row above is that check. |
| `Could not download foldseek from <url>` | The runtime cannot reach that host. Upload a ready-made 3Di text file instead. |
| `foldseek found no 3Di descriptor in <file>` | No protein backbone atoms, or a minimal mmCIF foldseek cannot parse — converting it to `.pdb` fixes the second case. |
| `ESMFold ran out of GPU memory on N residues` | Fetch the AlphaFold model (<https://alphafold.ebi.ac.uk>) or fold it once at <https://esmatlas.com/resources?action=fold>, and upload the `.pdb`. |
| `The header of <file> names ['p22'] more than once` | pandas would rename the second copy to `p22.1` and quietly ignore it, so the file is refused. Fix the header. |
| `Unknown residue 'X' in column 'p22', row 0` | A residue outside the 20 standard ones, or three-letter data read as one-letter. The message says which way to flip **Residues are written as Asn, not N**. |
| `N condition values are missing or not numeric, for example column 'activity' at row 12` | Drop or fill those rows before loading. |
| `The library that loaded has no <count> column, so the threshold of 5 filtered nothing` | A yellow note, not an error — every variant was kept, because the read-count column you named is not in the CSV. The note lists the columns the table does have. Correct the column name and press **Check my library** again, or set the threshold to 0 if you meant to keep every variant. |
| `No locked test results under .../locked_test`, or `holds locked results, but none of them belong to this RunResult` | Train first and keep the run name the same, or point the unlock cell at the directory the training step actually wrote. |
| `unlock.json is not readable as an unlock record` | The counter file is corrupt. Deleting it resets the count to zero — **say so in the report**, because the record of earlier unlocks goes with it. |
| two `performance_report.zip` files | The working directory changed between writing the archive and unlocking. The older one is still where it was written; keep the directory fixed once you have trained. |
| a corrupt archive, or a bundle that `does not match its manifest checksum` | A member does not match its recorded sha256 — a truncated download or an edited zip. Re-export or re-download. |

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
├── report/report.csv · report.json · report.png
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
manifest. Nothing has to be reconstructed afterwards.

| state this | where it comes from |
|---|---|
| the tool, its version, its license | the setup cell's first line; `colabsd_version` in `performance.json`; [`CITATION.cff`](CITATION.cff) |
| the backbone and the mutated sites it averages | the bundle's description line, which names both; `model` in `performance.json` |
| that the hyperparameters were `tuned` — on another protein — or `PROVISIONAL` | `hyperparameters.status` in `performance.json`, and the same flag in the manifest |
| epochs, patience and the two batch sizes the run was *given*, and which you set | `training_budget` in both exports; the *WHAT THE RUN WAS GIVEN* table in `README.txt` |
| the seeds and how many runs — `n_runs = 1` is one draw, quoted without a ± | step 4's plan line; `n_runs` in `report.csv`; `run` in `performance.json` |
| the split ratio and the rows in each partition | 8 : 1 : 1; `partition_rows` in `report.json`, which is a mapping over all three partitions — `performance.json`'s `partition_rows` is one integer, for the partition that archive reports on — and a `library size` line in `README.txt` |
| validation or test, and how many times the test set was read | `partition` and `unlock.count` with `unlock.verdict` in `performance.json`; the badge in `report.png` |
| that a validation-only archive is reportable — it states the test partition was never read | `verdict` in `performance.json`; the sentence under the headline in `README.txt` |
| the GPU the numbers came from | the GPU line the setup cell prints |

**Predictions rank variants; they do not measure them.**

Cite ColabSeqDisplay and the seqdisplay-opt study whose method it runs. Both are in
[`CITATION.cff`](CITATION.cff), the second as a related reference, so a citation manager picks up
the pair; [`ATTRIBUTION.md`](ATTRIBUTION.md) says which parts of the workflow are that study's, and
[README § Acknowledgement](README.md#acknowledgement) names them.
