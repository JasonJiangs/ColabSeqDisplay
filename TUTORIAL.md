# ColabSeqDisplay tutorial

From opening a link to holding a ranked list of candidate variants. **Part 1** runs the bundled
SlugCas9 5NNK example end to end; **Part 2** points the same workflow at your own protein;
**Part 3** scores new variants from a bundle you exported months earlier.

Every input is a form field, and the panel explains each one as you reach it — the note under a
control, the estimate on a button, the warning beside a choice. This page covers what the panel
cannot: what to check, what a number means, and what to do when something refuses. What the tool
is and what its numbers are worth is in [`README.md`](README.md), in particular
[Status](README.md#status).

**Contents** — [Before you start](#before-you-start) · [Part 1](#part-1--the-bundled-example) ·
[Part 2](#part-2--your-own-protein) · [Part 3](#part-3--scoring-new-variants) ·
[Troubleshooting](#troubleshooting) · [Working directory](#what-ends-up-in-the-working-directory) ·
[If you report a result](#if-you-report-a-result)

---

## Before you start

**Set the runtime first: Runtime ▸ Change runtime type ▸ T4 GPU.** Without a GPU the notebook
still loads and checks your library, shows the hyperparameters, and — once something has been
trained — draws the report and writes both exports; the train button, region discovery and ESMFold
each refuse up front.

**Two code cells.** The first installs the package — its one field, `colabsd_repository`, already
points at the right source — and draws a panel; that panel *is* the workflow, and its numbered
steps appear and disappear as your answers make them relevant. The second, at the bottom, is the
only thing that can read the locked test set. Read the three lines the setup cell prints before
moving on: the version installed and where from, the GPU, and the directory this session writes to.

**Colab runtimes are temporary.** Everything goes to `colabsd_work/`, and the two exports are also
pushed to your browser's downloads. Keep `model_bundle.zip` (the only file Predict needs) and
`performance_report.zip` (the only file that says what your numbers were). Tick **Save to Google
Drive** if you want the rest to survive a disconnect.

---

## Part 1 — the bundled example

16,424 SlugCas9 variants of a 1054-residue protein, 5 randomised positions, 4 measured PAM
conditions. The library, wild-type sequence, 3Di string and pooling regions all ship with the
repository, so with the defaults there is nothing to upload and nothing to compute before
training. Leave every field alone: the bundled example, `ESM2-35M`, `cosine_p90_mean`, `float32`,
one split seed × one model seed, the shipped region, Drive off, `colabsd_work`, run name `run`.

Everything is seconds except the setup cell (a minute or two), the one-hot floor in step 8
(seconds to a couple of minutes on a CPU) and **step 7, training — an estimated 40 min on a T4 for
the default backbone.** Step 7 counts runs off as they finish and reports the measured elapsed
minutes at the end; that number is yours.

### Step 1 — check the four lines it prints

Press **Check my library** with the defaults and you get:

> **16,424 variants** from `.../library.csv`.
> - Wild type: 1054 residues.
> - 5 mutated sites at [984, 985, 990, 1012, 1016], wild-type residues `NSMEK`.
> - 4 conditions: NNGA, NNGT, NNGC, NNGG.

**Those four lines are the entire data contract, and every later step inherits them.** Check the
variant count, the site count, the positions and the condition names against what you believe you
loaded. `wild-type residues NSMEK` is the strongest check available: those are the residues your
wild type carries at the positions you listed, so an off-by-one or a signal peptide in the FASTA
shows up here and nowhere else. Nothing below runs until this step succeeds.

### Steps 2 and 3 — the two questions, and what they need prepared

The dropdown lists seven backbones cheapest-first, each naming whether it needs a 3Di string and
its estimated minutes per run. Two poolings are offered — the two `config/best/` has entries for.
Leave the defaults for Part 1; the [configuration table](README.md#configurations) is what to read
when you do choose.

**Step 3 is derived, not asked**: nothing for an ESM2 backbone with `mutation_site_mean` (the step
is absent), a pooling region for `cosine_p90_mean`, a wild-type 3Di string for a SaProt backbone,
or both. For the bundled example it is one button already on the right answer. Two things the
panel does not tell you:

- **A 3Di string from AlphaFold beats one from ESMFold.** Download the model of your protein from
  <https://alphafold.ebi.ac.uk> and upload the `.cif`: free, fast, and it matches your sequence
  exactly. ESMFold is the last resort, and its free-T4 ceiling is 700 residues.
- **What a good region looks like:** the mutated positions are inside it, and the selected residues
  form a few contiguous blocks rather than scattered singletons. The example's shipped p90 region is
  106 of 1054 positions in three blocks — 912, 939–1040 and 1043–1045 — and contains all five
  mutated positions. A region that excludes the mutated sites, or selects almost the whole protein,
  means the percentile is doing nothing useful for this library.

Region discovery is the one step that can outlast a Colab session. The button carries its own
estimated span — *Discover the region (3 min to 19 min)* for the defaults (`ESM2-650M`, 2,000
sampled variants, seed 0, batch size 2), rising to `27 min to 2 h 39 min` if you set the sample to
0 and score all 16,424. Train never starts one for you; leave the radio on *discover* without
running it and Train refuses, naming the button to press.

### Steps 4 to 7 — hyperparameters, runs, storage, train

**Step 4 has no fields.** It displays the seven LoRA values looked up from `config/best/`, the
fixed training block and the provenance — for the default pair, `ESM2-35M · cosine_p90_mean · tuned
(test ρ=0.5486 ± 0.0145, 9 runs)`; for a placeholder pair, a red `PROVISIONAL` banner (see
[Reading the PROVISIONAL warning](#reading-the-provisional-warning)).

**Step 5** asks how many runs — one run is one split seed × one model seed, capped at 3 of each.
Leave both at 1 for a first pass; 3 × 3 is nine times the wait and what you want for a number you
intend to report. **Step 6** is storage: **do not change the working directory between exporting
and unlocking**, or you end up with two archives instead of one updated file.

**Step 7 trains.** The counter under the button advances once per **finished run**, not per epoch,
so on a single run it goes from `0/1 starting` to `1/1` and little else happens on screen for the
better part of an hour. That is normal. Keep the tab open — closing it disconnects the runtime, and
a disconnected runtime stops training. If it is interrupted, press again: a finished run is reused
only when its fingerprint (model, hyperparameters, pooled coordinates, split, and a hash of your
data) is identical.

### Step 8 — what came back

The language model and the one-hot floor side by side, per condition and averaged over conditions,
on validation. Test artefacts move to `run/locked_test/` as each run finishes; nothing you can read
here carries test information.

**`± nan` is deliberate**, not a bug: one run cannot show reproducibility, so the standard
deviation is undefined rather than zero. Raise the seed counts in step 5 before quoting a ±.

**The floor is the line that makes a pLM result mean something.** Ridge regression and a small MLP
on the 20 × *k* one-hot encoding of the mutated residues — a 16,424 × 100 feature matrix for the
example — fitted on exactly the splits the pLM used; the better of the two is the floor, and it
needs no GPU. On split seed 1 with model seed 11 the ridge head reaches validation Spearman
**0.4722** and the MLP **0.5368**, so the MLP is the floor. The splits are seeded, so the ridge
number reproduces exactly; the MLP moves in the last digits with a different torch build. A
650M-parameter model that ties a ridge regression on one-hot residues has told you the landscape is
additive, not that the model is good. If the language model does not clear the floor the panel says
so plainly — a result, not an error, and what happened in the one end-to-end fine-tune this
repository has run; see [Status](README.md#status).

### Step 9 — the two exports

Both buttons refuse for the same two reasons and no others: **nothing has been trained in this
session**, or **the settings have changed since this model was trained**.

**`model_bundle.zip`** is written, loaded straight back, and described from the file rather than
from the form — `ESM2-35M · cosine_p90_mean · tuned hyperparameters · 4 conditions · test unlocked
0x · written <timestamp>`. It holds **the single best-validation run**, not an ensemble, so with
3 × 3 seeds the ± in your report describes a family of runs and the bundle is one member of it; and
it needs the training checkpoint on disk, so export before you clear `colabsd_work/`.

**Write `performance_report.zip` now, before you unlock anything.** Inside:

| member | what it holds |
|---|---|
| `report.csv` | one row per source × partition × condition: `n_runs`, then `_mean` and `_sd` for R2, Pearson, Spearman, P@10, P@50, NDCG@10, NDCG@50 — plus a `mean` row across conditions |
| `report.png` | the chosen metric per condition and per source, error bars ±1 sd across runs, the one-hot floor as a dashed line, and a badge naming the partition and the unlock count |
| `report.json` | `unlock_count`, `unlock_source`, `one_hot_floor`, `shown_partition`, `reported_partitions`, `n_runs`, `sources` |
| `performance.json` | the manifest: partition, unlock count and verdict, model, hyperparameter status, floor, conditions, runs, seeds, `colabsd` version, your notes |
| `README.txt` | the same answers in prose, for whoever opens the zip and will not read JSON |

Every member carries a sha256 and reading the archive back verifies it. Report files land in
`colabsd_work/report/`, the archive in `colabsd_work/` itself.

### Step 10 — score some variants

The minimal prediction outlet. Three sources: the first rows of the library you loaded, *Random
combinations* (residues drawn at random per mutated site, then de-duplicated — five NNK sites is
3.2 million combinations), or your own CSV. Output is one row per variant with the mutation
columns, one predicted column per condition, and two summaries: **`pred_mean`**, and **`pred_min`**,
the worst condition — **rank by this when a hit has to work in every condition**.

The first rows of the loaded library are mostly rows the model trained on: 80 % of the library is
the training split. Comparing those predictions to measurements you already have is a wiring check,
not evidence.

### Cell 2 — unlock the test set

**Stop and read this before you tick the box.** This is the one methodological decision the
notebook asks you to make, and the one a referee is most likely to probe.

**What is locked.** Everything in the panel above reports validation. The training loop evaluates
the test partition at the end of each run — and the notebook immediately moves every test artefact
into `run/locked_test/` and replaces the test block of the visible metrics with a sentence naming
the one function that can read it back. That includes the performance archive from step 9, which is
why you can have it first.

**Ticking the box** increments a counter in `run/unlock.json`, which keeps a short history, and
that count is stamped into the run result on disk, `report.csv`, `report.json`, the footer and
badge of `report.png`, `performance.json` inside the archive, and the manifest inside
`model_bundle.zip` — so a colleague opening the bundle a year later reads `test unlocked 1x` in its
first printed line. The counter belongs to the run directory: a new `run_name` starts a fresh
counter at zero, and deleting `unlock.json` resets it. One guard: if a run directory is reused by a
later training run with different hyperparameters, data or splits, unlocking refuses rather than
handing back the older result's test numbers under the newer configuration's name.

**Unlock once, when you have stopped changing things.** The panel then gives the test Spearman and
a boxed final report — model, unlock count with a sentence saying what it is worth, per-condition
table, macro average, one-hot floor, files written — rewrites `performance_report.zip` in place
with the test numbers and the count, and offers the archive and `report.png` as downloads.

**Read the figure in this order:** did the pLM beat the one-hot floor, on the same partition, in
the `mean` group — if not, nothing else matters yet; is the spread across conditions sensible (in
the example NNGT is hardest and NNGA easiest for both one-hot heads, and a pattern like that is
normal, while *one* condition far below the others when your assay says otherwise is usually a data
problem); how many runs; and what the unlock badge says.

| unlock count | what a reader should conclude |
|---|---|
| **0** | no held-out number exists. The archive says `partition: validation` and records that the test set has never been read — a perfectly reportable state, and the right one while you are still choosing. Predictions can still be worth ranking; the model has simply not been measured on unseen data. |
| **1** | a held-out result in the usual sense: read once, after the choices were made. |
| **2 or more** | each read after the first was available to inform a choice, so the last number is at best a second validation score. Quote it as such, and quote the count beside it — the archive does. |

A report showing several unlocks is not a fabrication and not a disqualification; it is the record
of an exploratory session and should be described as one.

---

## Part 2 — your own protein

Same notebook, same order; the differences are all in steps 1–3. You need `library.csv` (schema and
a worked row: [README](README.md#your-own-library)) and the wild-type sequence, pasted or as the URL
of a FASTA. A SaProt backbone also needs a structure of your wild type — a `.pdb` or `.cif`. You do
**not** bring a pooling region; it is discovered in step 3 from the library you just loaded.

**Rules the loader enforces.** Unique column names. Residues all in one alphabet — three-letter
`Asn` with the *Asn, not N* box ticked, or one-letter `N` with it clear. Numeric, finite condition
values, and none too large for float32. Positions strictly increasing, **1-based in the full-length
wild-type numbering**, with one mutation column per position in the same order: *Mutated positions*
and *Mutation columns* are read as parallel lists. Every refusal names the offending column and row.

**Then check the printed `wild-type residues` string.** If those are not the residues your construct
carries at those positions, your numbering is wrong — **stop here**, because every later step
inherits it. Two frequent causes: the FASTA includes a tag or signal peptide the numbering does not,
or the positions are 0-based.

**What step 3 will cost you.** Loading your own library takes the bundled example's artefacts off
the menu — they describe SlugCas9 — so the cost follows entirely from step 2: nothing for an ESM2
backbone with `mutation_site_mean`; one discovery run, minutes to hours, for `cosine_p90_mean`;
seconds for a 3Di string from a structure you download; both for a SaProt backbone with
`cosine_p90_mean`. Once a region (or a 3Di string) has been discovered in a session it is offered
back for the rest of it.

**Note the tension.** `cosine_p90_mean` is the pooling every tuned hyperparameter set uses, but for
a new protein it costs a region-discovery run up front, and every `mutation_site_mean` entry is a
placeholder. A discovery plus a fine-tune in one free session is exactly the case the long-run
warning tells you to mount Drive for.

### Reading the PROVISIONAL warning

If the (backbone, pooling) pair you chose has never been tuned, the workflow says so in capitals
wherever that pair appears, and never stops you: the step 4 banner and its message board, the line
printed after training, the performance archive, the bundle's description line, and the unlock cell
and Predict notebook beside the numbers themselves.

All 18 placeholder files record `source: median of the 10 tuned cosine_p90_mean configs`: nobody
searched those values for this pair, and no performance number stands behind them. A model trained
on them can still rank variants usefully; it cannot support a claim about how well this backbone
performs.

**This applies to a `tuned` line too.** Those hyperparameters were selected once by an Optuna search
on the benchmark library, then re-evaluated over nine runs. On *your* protein, "tuned" means
*transferred from a tuned search on another library* — a better starting point than a placeholder,
not a configuration fitted to your data.

### A sensible first pass on a new library

1. Load your CSV, check the `wild-type residues` line, and stop if it surprises you.
2. `ESM2-8M` with `mutation_site_mean` — no region, no 3Di, no step 3 at all, an estimated 20 min
   per run on a T4 for a library the size of the example. This checks that your data flows through
   training, the floor and the report. Do not unlock the test set.
3. Write the performance archive from that run anyway. It costs nothing.
4. Switch to `cosine_p90_mean`, tick **Save to Google Drive**, and discover the region — reading its
   estimate before you press the button.
5. The real run: a tuned pair, 3 split seeds × 3 model seeds, the performance archive again, then
   one unlock at the end.

---

## Part 3 — scoring new variants

`colab/ColabSeqDisplay_Predict.ipynb`. Upload a `model_bundle.zip` and a table of variants; get a
ranked prediction table back. No training happens here, and the bundle is self-contained, so the
notebook never asks you to describe your library. It does need internet access: the backbone itself
is fetched by name on the first scoring run of a session. A GPU makes this fast; without one the
panel's estimate runs from minutes for a few hundred variants on a small ESM2 to hours on a 650M
model.

**Stage 1 loads the bundle** and prints its description line plus a table of what is inside and
what it requires of your variants: the backbone, the pooling with the number of frozen residue
positions and the region it came from, the hyperparameters with `PROVISIONAL` in red if that is
what they were, the unlock count in red if it is zero, the conditions, and the variant columns with
the positions they map to. **Read the description line as the model's provenance** —
`PROVISIONAL hyperparameters` earns *rank with it, but do not quote its numbers*, and `test
unlocked 0x` earns *no held-out number stands behind these predictions*. A bundle built on a
backbone the training notebook no longer offers still loads, describes itself and scores; and
handing this stage `performance_report.zip` by mistake is diagnosed, not crashed on.

**Stage 2 chooses the variants** — an uploaded CSV (200 rows by default, seed 0), rows of a library
table, or `random_combinations`. Your CSV needs one column per mutated site **named exactly as
stage 1 printed**; any other column is ignored. It prints how many variants are ready and how many
are distinct: a large gap means your CSV repeats variants.

**Stage 3 scores and ranks** by `pred_mean` or `pred_min`, writing `ranked_variants.csv`. Both
columns are always written, so you can re-sort without re-scoring. **Check the scale line.**
Training z-scores the labels on the train split and the bundle normally carries that scaler, so
predictions come back in the assay's own units; if it does not, the cell says `scale: z-scored
training units` — the ranking is still meaningful, the absolute values are not comparable to your
measurements, and you should not plot them on the same axis.

These are predictions from a model fitted to one library. They rank variants; they do not measure
them.

---

## Troubleshooting

Every failure raises an error whose message says what to do. The ones worth knowing in advance:

| what you see | what it means, and the fix |
|---|---|
| `... could not be downloaded from <address>` (setup) | `git clone` failed: the address in `colabsd_repository` is not reachable or is not a repository. Git's own output is quoted above it. Fix the address, or upload a folder and give its path instead. |
| a failing `pip install -e` line (setup) | The clone worked, the install did not; the tail of pip's output is in the message. A truncated copy of the repository is the usual cause — re-run the cell. |
| the setup cell reinstalls every time | It counts the install as done only when `colabsd` imports **and** its hyperparameter registry and bundled example came with it. If it keeps repeating, the folder in `colabsd_repository` is not a complete checkout. |
| `GPU  none — Runtime > Change runtime type > T4 GPU` | Switch the runtime and re-run the setup cell. Loading, hyperparameters and reports work without a GPU; training, region discovery and ESMFold refuse up front. |
| **the panel freezes and nothing responds** | A `files.upload()` picker is open somewhere on the page; Colab's upload blocks the kernel. **Cancel upload** releases it. |
| a backbone you used before is missing from the dropdown | Only ESM2 and SaProt are offered. The note under the dropdown names the seven that are not and why; their adapters are still in the package, and a bundle built from one still loads and scores in Predict. |
| `does not fit a free T4 … Switch to an L4 or A100` | `SaProt-1.3B` is the only offered backbone that does not fit a free T4. Pick a smaller one, or an L4/A100 runtime with `dtype = float16`. |
| CUDA out of memory part-way through training | The backbone does not fit alongside your sequence length. In order of effect: a smaller backbone, then `dtype = float16` (`micro_batch_size` comes from the config and is not a form field). After an OOM the failed run's tensors usually still hold GPU memory, so **Runtime ▸ Restart** is the reliable retry. |
| `No pooling region has been discovered in this session yet` (train) | You left step 3 on *discover* without running it. Press **Discover the region** — Train will not start an hour of GPU work on your behalf — or switch to a region you can reuse or upload. |
| `<file> has N residues but the WT sequence has M`, or `wt_3di is N characters but the wild-type sequence is M residues` | The 3Di string does not cover the same chain as your wild type. Usually chain selection: put the right id in *Chain to read*. Otherwise a structure with missing or extra residues — an AlphaFold or ESMFold model of the exact wild-type sequence always matches. |
| `<file> contains 2 chains: A (N residues), B (M residues). Pass chain="A"` | Foldseek produced one 3Di string per chain. Name your protein's chain in *Chain to read*. `has no chain 'X'` means the id you gave is not in the file. |
| `<file> is a different protein from the WT: N of M residues disagree` | The chain you selected is not the protein your library mutates. Pick the right chain, or fold the wild-type sequence itself. A handful of disagreements is reported, not refused — read that line, it names the positions. |
| `contains characters that are not foldseek 3Di states` | The file is an amino-acid FASTA, not a 3Di string. |
| ``Your variants table has no `nnk1` column(s). This bundle was trained on [...]`` | Your CSV column names do not match the bundle's. Rename them, or use the library those names came from. |
| `The header of <file> names ['nnk1'] more than once` | pandas would rename the second copy to `nnk1.1` and quietly ignore it, so the file is refused. Fix the header. A column reported missing next to a sibling like `nnk1.1` is the same problem reaching a later step. |
| `Unknown residue 'X' in column 'nnk1', row 0` | A residue outside the 20 standard ones, or three-letter data read as one-letter. The message says which way to flip the *Asn, not N* box. |
| `N condition values are missing or not numeric, for example column 'NNGT' at row 12` | Drop or fill those rows before loading. A sibling message catches values too large for float32 and suggests rescaling. |
| `min_count=5 was ignored: ... has no 'count' column` | A warning, not an error — every variant was kept. Name the real column, or set the threshold to 0 to say you meant to. |
| `<file> holds the cosine_p95_mean region, but this run pools cosine_p90_mean` | Averaging one region under another region's name is silently wrong, so it is refused. The percentile comes from the pooling chosen in step 2, so this only happens with an uploaded file. |
| `<file> was computed for a 1054-residue protein, but this wild type is N residues` | The region file belongs to another protein. Uploaded files are the only way to hit this: the example's region is offered for reuse only while the example library is loaded. |
| `The cached split in .../seed_1 covers N rows, not the M rows of this library` | You changed the library (a different CSV, or a different threshold) while reusing the working directory. Change the run name, or start a fresh runtime. |
| `No locked test results under .../locked_test` | No run in this directory. Train first, and keep the run name the same. |
| `holds locked results, but none of them belong to this RunResult` | The run directory belongs to a different set of seeds. Point the unlock cell at the directory the training step actually wrote. |
| `holds the test metrics of a different training configuration` | The run directory was reused by a later training run with different hyperparameters, data or splits. Give each configuration its own run name. |
| `unlock.json is not readable as an unlock record` | The counter file is corrupt. Delete it to reset the count to zero — **and say so in the report**, because the record of previous unlocks goes with it. |
| two `performance_report.zip` files | The working directory was changed between writing the archive and unlocking. The older archive is still where it was written. Keep the directory fixed once you have exported. |
| a corrupt performance archive, or a bundle that `does not match its manifest checksum` | A member does not match its recorded sha256 — a truncated download or an edited zip. Re-export or re-download rather than trusting the numbers. |
| `No checkpoint at .../best_checkpoint.pt` | The export step needs the training run still on disk. Export before clearing the working directory or restarting the runtime. |
| `Could not download foldseek from <url>` | The runtime cannot reach that host. Upload a ready-made 3Di text file instead, or install foldseek yourself and set `FOLDSEEK_BIN`. |
| `foldseek found no 3Di descriptor in <file>` | The file has no protein backbone atoms, or is a minimal mmCIF foldseek cannot parse — converting it to `.pdb` fixes the second case. |
| `ESMFold ran out of GPU memory on N residues` | Fetch the AlphaFold model (<https://alphafold.ebi.ac.uk>) or fold it once at <https://esmatlas.com/resources?action=fold>, and upload the `.pdb`. |
| `The backbone holds its rotary position index in ...` (region discovery) | `bfloat16` cannot represent every residue index of a long protein distinctly, so scores would be computed on a corrupted positional encoding. Use `float32` (the default) or `float16`. |
| the runtime disconnects during training | Press Train again with *Reuse finished runs* ticked; a finished run is reused only when its fingerprint is identical. Tick **Save to Google Drive** and keep the tab open next time. |

---

## What ends up in the working directory

```
colabsd_work/                            (/content/colabsd_work/ in Colab)
├── splits/n16424/split/seed_1/          the cached 8:1:1 split, keyed by library size
├── wt_3di.txt                           written by step 3, if that step existed
├── region_p90.json                      written by step 3, if it discovered one
├── run/                                 (named in step 5)
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
├── performance_report.zip               the three report files, plus performance.json and README.txt
└── scored_variants.csv
```

Keep **`model_bundle.zip`** and **`performance_report.zip`**. The rest can be regenerated from them
and your library. The prepared artefacts are worth keeping only if you plan to train the same
protein again in a fresh runtime — tick **Save to Google Drive** before they are written.

## If you report a result

Everything below is printed by the notebook or written into `performance_report.zip` and the bundle
manifest; none of it has to be reconstructed afterwards.

| state this | where it comes from |
|---|---|
| the tool, its version and its licence | `colabsd <version>` on the first line of the setup cell, and `colabsd_version` in `performance.json`; version, licence and archive DOI in `CITATION.cff` (see [Code availability](README.md#code-availability)) |
| backbone, pooling, and the size of the pooling region | step 2's summary, the `model` block of `performance.json`, and the bundle's description line |
| whether the hyperparameters were tuned or a placeholder | step 4's banner; `hyperparameters.status` in `performance.json` and the same flag in the bundle manifest |
| split seeds, model seeds and how many runs | step 5's plan line, `n_runs` in `report.csv`, and the `run` block of `performance.json` |
| the split ratio and sizes | 8 : 1 : 1; step 5 prints the counts for your library |
| the one-hot floor you compared against | `one_hot_floor` in `report.json` and `performance.json`, drawn as the dashed line in `report.png` |
| whether the number is validation or test, and how many times the test set was read | `partition` and `unlock.count` in `performance.json`, with `unlock.verdict`; plus the badge in `report.png` and `unlock.json` itself |
| that the numbers came from a Colab GPU of a given type | the GPU line printed by the setup cell |

Three sentences worth writing down verbatim:

- **`n_runs = 1` means one draw.** Quote the number without a ±, and say it is a single split seed
  and a single model seed.
- **A provisional configuration is a lower bound.** Say which of the two it was.
- **A validation-only performance archive is a reportable artefact.** It states that the test
  partition was never read, and the archive is what lets a reader tell that apart from a test number
  obtained after five looks.

Cite ColabSeqDisplay, and cite the seqdisplay-opt study whose method it runs — both are named in
[README's Acknowledgement](README.md#acknowledgement) and both are in
[`CITATION.cff`](CITATION.cff), the second as a related reference, so a citation manager picks up
the pair. [`ATTRIBUTION.md`](ATTRIBUTION.md) says which parts of the workflow are that study's work.
