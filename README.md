# ColabSeqDisplay

Fine-tune a protein language model on your own sequence-display variant library, in a
browser, without writing code. You bring a CSV with one row per variant, one column per
mutated site and one column per measured condition, plus the wild-type sequence. You get a
trained model, a report that compares it against a one-hot baseline on a test partition
that stays locked until you deliberately open it, and a portable `.zip` you can reload
months later to rank variants you have not made yet. Every input is a field you fill in,
not a line of code you write.

**Version 0.1.0 · MIT licence · [step-by-step tutorial](TUTORIAL.md)**

**One repository, and it is the whole program.** The fine-tuning engine — LoRA injection,
the training loop, the pooling strategies, the metrics and the split protocol — ships
inside this package as `colabsd/engine/`, copied from the study that developed it, each
module opening with a header naming where it came from and
[`ATTRIBUTION.md`](ATTRIBUTION.md) collecting them all. Clone this repository, or let the
setup cell clone it for you, and there is nothing else to fetch.

**Read this before you rely on a number it produces.** ColabSeqDisplay is a front end for a
fine-tuning protocol, not an independently benchmarked method, and three things about its
current state matter more than anything else on this page:

- Every performance number quoted here comes from **one protein** — the SlugCas9 5NNK
  library bundled with the repository. Nothing here shows that the results transfer.
- **10 of the 28 hyperparameter entries are tuned; 18 are placeholders.** Every
  `mutation_site_mean` pair is a placeholder, and that is the pooling a new protein starts
  from. The notebook prints `PROVISIONAL` in capitals whenever you land on one.
- **One of the 13 selectable backbones has been fine-tuned end to end** through this
  package. Two have never had their weights loaded at all.

[Reproducibility](#reproducibility) and [Limitations](#limitations) state each of these in
full, with the numbers behind them.

---

## The three notebooks

| | Open it | When |
|---|---|---|
| **ColabSeqDisplay** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay.ipynb) | The main workflow. Load a library, pick a backbone and pooling, fine-tune, read the report, export a model bundle. Start here. |
| **Prepare** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay_Prepare.ipynb) | Once per new protein, and only if you need it: a wild-type 3Di string (SaProt backbones only) or a pooling region (`cosine_p90_mean` pooling only). Skip it for the bundled example, and skip it if you use an ESM2 backbone with `mutation_site_mean` pooling. |
| **Predict** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay_Predict.ipynb) | Later. Upload a `model_bundle.zip` and a table of variants, get a ranked prediction table back. No training, and nothing else needed — the bundle is self-contained. |

> **There is one field to set, and its default is already right.** The setup cell of all
> three notebooks carries a single `colabsd_repository` field, pointing at this repository.
> It also accepts the path of a folder you have uploaded to the runtime, which is how you
> run a branch or a local edit. If the source cannot be reached the cell stops there,
> quotes what `git` or `pip` said, and names the field to change. See
> [Requirements](#requirements).

---

## Try it now

The bundled example is a real SlugCas9 5NNK library: 16,424 variants, five mutated
positions, four measured PAM conditions. Nothing to upload.

1. Open **ColabSeqDisplay** with the badge above, then `Runtime ▸ Change runtime type ▸ T4 GPU`.
2. Run the setup cell. It installs the package and prints the version it installed, the
   path it came from, the GPU it found and the directory your files will be written to.
   Then it draws the panel that is the rest of the notebook.
3. Leave every field in the panel alone. The defaults are the bundled library, the
   `ESM2-35M` backbone, `cosine_p90_mean` pooling over the shipped region, `float32`, one
   split seed × one model seed, and the tuned hyperparameters looked up for that pair.
4. Press the train button. One LoRA fine-tune runs, the one-hot floor the language model
   has to beat is fitted beside it, and the panel reports both — on validation only.
5. When you have stopped changing things, run the second cell at the bottom of the
   notebook: it is the one thing that can read the locked test partition, and it counts
   every read.

[`TUTORIAL.md`](TUTORIAL.md) walks the same path cell by cell, with the fields, the
defaults and the output of each one, and does the same for the other two notebooks.

You end with `report.csv`, `report.json` and `report.png` — every metric per condition,
mean ± s.d. across runs, the one-hot floor drawn as a line, and the number of test-set
unlocks in the footer — plus `model_bundle.zip` for the Predict notebook. Budget about an
hour on a free T4, nearly all of it in the training run; that is the registry's estimate
for a library this size, not a measured time, and the panel prints your own rate within
the first minute of training.

---

## Your own library

One CSV describes a library. Below is the schema, with the values of the second data row of
`examples/slugcas9_5nnk/library.csv`, shown exactly as they appear in the file.

| Column | What it holds | Required | Example row |
|---|---|---|---|
| `nnk1` … `nnk5` | One column per mutated site, in the order of the positions you give. Three-letter or one-letter residue codes (a form field says which). | yes | `Asn`, `Gln`, `Leu`, `Ala`, `Glu` |
| `count` | Reads supporting the variant. Used only to drop low-count rows via `min_count`. | no | `1725` |
| `NNGA`, `NNGT`, `NNGC`, `NNGG` | One column per measured condition. Numeric activity. All are predicted jointly by one multi-condition head. | yes | `0.7455072402954102`, `0.14260870218276978`, `0.4046376943588257`, `0.6857970952987671` |

The column names are yours: `nnk1…nnk5` and `NNGA…NNGG` are what this example happens to
use, and you type your own names into the form. So is how many of each — **any number of
mutated sites and any number of measured conditions are supported.** The head is built
with one output per condition column you name, so five sites and two conditions, or two
sites and twelve, work the same way.

Three form fields tie the CSV to the protein: the wild-type sequence (pasted, or the URL
of a FASTA), the 1-based positions of the mutated sites, and the mutation column names in
the same order. In the example those positions are 984, 985, 990, 1012 and 1016 of the
1054-residue wild type, where the wild-type residues are N, S, M, E and K.

Variant sequences are **built**, never read from a FASTA: each row's residues are
substituted into the wild type, so every variant is full length and every position is
checked against the sequence you supplied. A position that does not exist, or a residue
code that is not an amino acid, is refused at load time rather than at training time.

Two optional files, both made by the **Prepare** notebook and both already shipped for the
example: `wt_3di.txt` (one Foldseek 3Di state per residue; needed only by a SaProt
backbone) and `region_p90.json` (needed only by `cosine_p90_mean` pooling).

---

## How the workflow decides things

**Hyperparameters are looked up, not tuned.** For a given (backbone, pooling) pair the
seven LoRA hyperparameters come from `config/best/`, and step 4 shows them without
offering a way to change them. There is no tuning knob: hyperparameters re-tuned against
your own validation score consume the held-out set they were meant to be judged against.
Each file records its own status: `tuned` entries name the Optuna
trial they came from and the test Spearman they were re-evaluated at; `provisional`
entries are placeholders and say `unknown` there. Pick one and the notebook prints
**PROVISIONAL** in capitals wherever that pair appears — when you choose it, beside the
hyperparameters, with the trained result, and with the test number if you unlock one — and
stamps the same label into the exported bundle.

**Pooling decides which residues the model is read from.** `mutation_site_mean` averages
the mutated positions themselves and needs no extra file — the sensible default for a new
protein. `cosine_p90_mean` averages the residues whose embeddings disagree most across
your variants, scored as `1 − mean pairwise cosine similarity` per residue and cut at the
90th percentile; it needs a region file from the Prepare notebook. The two shipped for
SlugCas9 were computed with ESM2-650M in float32 over all 16,424 variants and hold 106
positions (p90, threshold 3.371860449376874 × 10⁻⁴) and 53 positions (p95). Every region
file records how it was made, and the notebook refuses one whose declared pooling or
sequence length does not match the run; `examples/slugcas9_5nnk/README.md` documents the format.

**The test set is locked, and unlocks are counted.** The panel reports validation only;
the test partition sits on disk untouched while training runs. The second cell of the
notebook opens it, increments a counter, and stamps that count into every report and
bundle written afterwards. Unlock once, at the end. Unlock five times and the footer of your report will say so, because a
test set read repeatedly while you change things is a validation set with extra steps.

The notebooks explain each step as you reach it, and [`TUTORIAL.md`](TUTORIAL.md) walks all
three cell by cell, field by field. This section is only the summary.

---

## Backbones

Fourteen entries in three tiers: `native` needs nothing beyond `transformers`, `extra`
needs one more pip install, `local_only` has no HuggingFace weights and no Colab adapter.
Thirteen are selectable in the notebook.

| Backbone | Needs 3Di? | Tier | Hardware | Real weights run? |
|---|---|---|---|---|
| `ESM2-8M` | no | native | free T4, ~20 min/run | yes — and the only backbone fine-tuned end to end here |
| `ESM2-35M` | no | native | free T4, ~40 min/run | yes |
| `ESM2-150M` | no | native | free T4, ~90 min/run | yes |
| `ESM2-650M` | no | native | free T4, ~240 min/run | yes |
| `SaProt-35M` | **yes** | native | free T4, ~45 min/run | yes |
| `SaProt-650M` | **yes** | native | free T4, ~260 min/run | yes |
| `SaProt-1.3B` | **yes** | native | **L4/A100**, `dtype="float16"`; a T4 runs out of memory | yes |
| `Ankh-large` | no | native | **L4/A100** (~1.2B-parameter encoder) | **no — tokenizer only** |
| `ProtT5-XL` | no | extra (`sentencepiece`) | **L4/A100** (~1.2B-parameter encoder) | **no — tokenizer only** |
| `ESMC-300M` | no | extra (`esm` SDK) | free T4, ~120 min/run | yes |
| `ESMC-600M` | no | extra (`esm` SDK) | free T4, ~220 min/run | yes |
| `SeqDance` | no | native | free T4, ~45 min/run | yes |
| `ESMDance` | no | native | free T4, ~45 min/run | yes |
| `METL` | yes | local_only | **not available in Colab** — Rosetta-pretrained, protein-specific, no HuggingFace weights | n/a |

*Minutes* are order-of-magnitude estimates recorded in the registry for one split seed ×
one model seed on a library the size of the bundled example (16,424 variants × 1054
residues) on a Colab T4. They are estimates, not measurements; the panel rescales them for
your library and then prints the real rate as the run proceeds.

*Real weights run?* — **yes** means the published checkpoint has been downloaded and
loaded, one pooled forward pass run, the pooled vector asserted to be exactly the width
the registry promises, and LoRA injection validated to cover the attention projections on
the real module tree. It does **not** mean the backbone has been fine-tuned: only
`ESM2-8M` has been trained end to end through this package. For `Ankh-large` and
`ProtT5-XL` only the tokenizers have been exercised; their 1.2B-parameter weights have
never been downloaded or run. Treat those two as untested.

`ESMDance` is worth one note: its pooled feature is the 50-dimensional `res_pred` head
output, not the 480-dimensional trunk, so heads are built 50 wide. That is deliberate and
checked against the real checkpoint.

**Which to pick.** On a free T4, `ESM2-35M` (≈40 min, sequence only) or `SaProt-35M`
(≈45 min, needs a 3Di string). Both are tuned pairs, and both land within 0.016 test
Spearman of the best tuned model on SlugCas9. Use `ESM2-8M` first as a sanity check: run
your library through it, confirm the numbers are not nonsense, then pay for a bigger run.

---

## Requirements

**Python ≥ 3.10**, the floor declared by `requires-python` in `pyproject.toml`, and the
version the linter targets (`target-version = "py310"`) — the two agree, so the floor is
the version the code is actually written against.

**One install, no second source.** Since the fine-tuning engine moved into
`colabsd/engine/`, the whole workflow is this package and its published dependencies:
there is no research checkout to obtain, nothing that is missing from PyPI, and a clone
of this repository imports and runs on its own.

Declared dependencies, exactly as `pyproject.toml` lists them:

| package | bound | what it is for |
|---|---|---|
| `numpy` | `>=1.26,<3` | arrays, splits, pooled coordinates |
| `pandas` | `>=2.1,<4` | the variant table and every report table |
| `scipy` | `>=1.11,<2` | the Pearson and Spearman correlations |
| `scikit-learn` | `>=1.3,<2` | the ridge head of the one-hot floor, and NDCG |
| `matplotlib` | `>=3.7,<4` | `report.png` |
| `tqdm` | `>=4.66,<5` | the progress counters |
| `torch` | `>=2.2,<3` | backbones, LoRA training, scoring |
| `transformers` | `>=4.40` | loading the HuggingFace checkpoints |
| `huggingface_hub` | `>=0.23` | fetching them |
| `pyyaml` | `>=6,<7` | reading `config/best/*.yaml` |

Eight of the ten are bounded above as well as below, so a future major release cannot
silently change the numbers. The two exceptions are the HuggingFace pair, and they are
deliberate: each backbone adapter discovers its LoRA targets from the module tree of the
checkpoint it actually loaded, and refuses a model whose attention blocks are not fully
targetable, rather than assuming a fixed layout — which is what lets a newer
`transformers` be used, and what makes it the dependency most worth recording. Every
exported `model_bundle.zip` stores the resolved `colabsd` and `torch` versions of the
session that produced it.

Optional extras: `[esmc]` pulls `esm>=3.1` for the two ESM-C backbones and `[prott5]`
pulls `sentencepiece>=0.1.99,<1` for the ProtT5-XL tokenizer. `pyproject.toml` also
declares `[optuna]` and `[dev]`, neither of which any notebook needs.

**Installing outside Colab:**

```bash
pip install .          # or: pip install -e .
```

`config/best/` and `examples/` are runtime data rather than documentation: the
hyperparameter lookup and the bundled example both read from them. In a clone they sit
beside the package, and the built distribution carries a copy of each inside it, so both
are found either way. The notebooks install in place (`pip install -e`) because that is
what keeps a cloned repository's own copies live — and because the setup cell counts an
install as complete only when the hyperparameter registry and the bundled example came
with it, so an install that carried only the Python modules across is installed again
rather than left to fail later.

**GPU.** A free Colab T4 (16 GB) fits ESM2 up to 650M, SaProt up to 650M, both ESM-C
models, SeqDance and ESMDance. `ProtT5-XL`, `Ankh-large` and `SaProt-1.3B` need an L4 or
an A100. ESMFold, the last-resort structure predictor in the Prepare notebook, is the
tightest fit of all: `colabsd.structure` puts its ceiling on a free T4 at about **700 residues**, and
the panel refuses to fold a wild type longer than that — so the 1054-residue bundled
example cannot be folded this way, which is why its 3Di string ships instead.

**Without a GPU** the notebooks still load and check a library, run the one-hot baseline
(both heads over all 16,424 example variants take about ten seconds on a workstation CPU),
draw the report, and score a few dozen variants from an existing bundle. Region discovery,
ESMFold and LoRA fine-tuning need a GPU, and each of those steps checks before it starts
rather than failing halfway through a training loop.

---

## Reproducibility

What a referee needs in order to repeat a run, and what repeating it can and cannot be
expected to give back.

**Randomness comes from two named seed families, and both are written down.** Every file
in `config/best/` ends with the same evaluation block:

```yaml
evaluation:
  selection_objective: mean_validation_spearman
  split_seeds: [1, 2, 3]
  model_seeds: [11, 22, 33]
```

`split_seeds` choose the partition of rows; `model_seeds` choose the initialisation of the
LoRA adapters and the head. Training takes the first `n_split_seeds` and the first
`n_model_seeds` of those lists, so the default single run is always split seed 1 with model
seed 11 — not a fresh random draw, and identical between two people who leave the sliders
alone. The one-hot floor is fitted on the same split objects with the same model seeds, so
the floor and the fine-tuned model are never compared across different partitions.
Step 10's `random_seed` field seeds only the sampler that invents random variant
combinations, and touches nothing that is trained.

**Splits are 8:1:1, cached, and checked against the library they were built for.** For the
bundled 16,424-variant example each of the three seeds gives 13,139 training, 1,642
validation and 1,643 test rows. Splits are cached under `colabsd_work/splits/n<rows>/`; a
cached split that does not cover exactly the rows of the library now loaded is refused
rather than reused, so changing `min_count` or the CSV cannot silently recycle the previous
partition. The test rows are moved to `run/locked_test/` as each run finishes and are read
only by the unlock cell, which counts every read.

**Fixed training settings** come from the same files and are shown in step 4: 20 maximum
epochs, early stopping with patience 3, MSE loss, targets z-scored on the training split,
gradient accumulation derived from the effective batch size, and no test evaluation during
optimisation.

**What is recorded with a result.** `report.csv` and `report.json` carry the number of runs,
the mean and s.d. per condition, the one-hot floor and the test-set unlock count with the
file it came from. `model_bundle.zip` carries the manifest — the library spec, the backbone
and pooling, the frozen pooled coordinates, the hyperparameters and their provisional
status, the unlock count at the time of writing, and the resolved `colabsd` and `torch`
versions of the session that produced it. A bundle therefore names its own software
environment; the dependency table above names the range that environment is allowed to come
from.

**Hardware and runtime.** The target is a free Colab T4 (16 GB); [Requirements](#requirements)
lists which backbones fit it. The per-backbone minutes in the [backbone
table](#backbones) are the registry's order-of-magnitude estimates for a library the size of
the bundled example, rescaled by the panel for your row count and then replaced by a
measured rate as the run proceeds. **No Colab session has been timed**, so no runtime on this page is
a measurement.

**The benchmark numbers were not produced on Colab hardware.** The 0.5478–0.5636 test
Spearman range, the tuned hyperparameters and the 40-trial Optuna budget behind them come
from the seqdisplay-opt study, whose environments pin a CUDA 12.8 PyTorch build for NVIDIA
B200-class GPUs. The code that produced them is in this repository — `colabsd/engine/`
holds the training loop and the LoRA injection they were measured with — but reproducing
the values themselves needs that hardware and the study's full run protocol, nine runs per
model over three splits × three seeds, not a single default Colab run, which reports `nan`
for the standard deviation precisely because one run cannot show reproducibility.

**Determinism, honestly.** Fixing a seed fixes the split and the initialisation. It does not
make GPU training bit-reproducible across different cards, driver versions or library
versions, and this package makes no such claim; the protocol's answer to run-to-run variance
is to repeat over seeds and report a spread, which is why the sliders go to three of each.

### What has been validated, and what has not

| | status |
|---|---|
| Library loading, sequence construction, position and residue checking | exercised on the bundled 16,424-variant example |
| The one-hot floor (ridge and MLP) over the full example library | run end to end, CPU only |
| Split creation, caching and the stale-cache refusal | run end to end |
| Test-set locking, the unlock counter and its stamping into report and bundle | run end to end |
| Report tables and figure, bundle export and reload, variant scoring | run end to end |
| Backbone loading, pooled-vector width, LoRA injection coverage | checked against the real published checkpoints for 11 of the 13 selectable backbones |
| `ProtT5-XL` and `Ankh-large` | **tokenizers only**; their ~1.2B-parameter weights have never been downloaded or run |
| LoRA fine-tuning end to end | **`ESM2-8M` only**; every other training path is untested here |
| Any protein other than SlugCas9 | **not attempted** |
| Any Colab runtime timing | **not measured** |

---

## Data availability

The example variant library distributed with this software is included in this repository
under `examples/slugcas9_5nnk/`: 16,424 SlugCas9 5NNK variants with read counts and measured activity
in four PAM conditions (NNGA, NNGT, NNGC, NNGG); the 1054-residue wild-type amino-acid
sequence; its Foldseek 3Di string; and two derived pooling regions of 106 and 53 residue
positions. `examples/slugcas9_5nnk/README.md` documents each file, its exact schema and its
provenance.

`library.csv`, `wt.fasta` and `wt_3di.txt` are redistributed unchanged — byte-identical to
their counterparts — from the seqdisplay-opt study (see [Acknowledgement](#acknowledgement));
`region_p90.json` and `region_p95.json` were computed by this package from `library.csv`
itself, and each records the model, score definition, percentile, threshold and number of
variants scored that produced it. The tuned hyperparameters in `config/best/` and every
benchmark value quoted in this README derive from the re-evaluation tables of that same
study.

**Terms.** All files under `examples/` are distributed as part of this repository and
are covered by its MIT licence (see [`LICENSE`](LICENSE)), which permits redistribution and
reuse with attribution. The primary source of the measurements is the seqdisplay-opt study,
which should be cited as such by anyone reusing them:
*[reference to be added on publication — see the Acknowledgement below]*.

This software generates no experimental data of its own. All outputs of a run — splits,
checkpoints, `report.csv` / `report.json` / `report.png`, `model_bundle.zip` and
`scored_variants.csv` — are derived from the library the user supplies and are written to
the runtime's `colabsd_work/` directory, from where the notebook also offers each one as a
download.

*For a manuscript that used ColabSeqDisplay, with the bracketed values filled in:*

> **Data availability.** The example variant library analysed by ColabSeqDisplay — 16,424
> SlugCas9 5NNK variants with read counts and measured activity in four PAM conditions, the
> 1054-residue wild-type sequence, its Foldseek 3Di string and two derived pooling regions —
> is included in the ColabSeqDisplay repository (https://github.com/JasonJiangs/ColabSeqDisplay)
> under `examples/slugcas9_5nnk/` and is archived with it at [DOI]. These measurements originate from
> [seqdisplay-opt reference]. No new experimental data were generated by this software.

## Code availability

ColabSeqDisplay **version 0.1.0** is openly available at
<https://github.com/JasonJiangs/ColabSeqDisplay> under the **MIT licence** (OSI-approved;
full terms in [`LICENSE`](LICENSE)), and runs in Google Colab from the badges at the top of
this page. Citation metadata — title, version, authors, licence and archive DOI — are in
[`CITATION.cff`](CITATION.cff), which GitHub renders as a ready-made reference.

The archived release carries the DOI recorded in `CITATION.cff`. That file holds a
placeholder DOI until the archive is deposited, because the DOI is minted at the moment of
deposit; `CITATION.cff` names every value that has to be filled in at that point.

**The repository is self-contained.** A reader can obtain, inspect, install, execute and
cite ColabSeqDisplay from this repository alone: the fine-tuning engine it runs is
`colabsd/engine/`, inside the package, and the setup cell of each notebook names this one
repository. [`ATTRIBUTION.md`](ATTRIBUTION.md) records where that engine came from, what
was taken and what was changed — module by module — and states one thing anyone
redistributing this software should read before doing so: the upstream project carries no
licence file of its own, so the terms covering the code in `colabsd/engine/` and the
values in `config/best/` are not stated anywhere and should be confirmed with its authors.
This repository's own MIT licence covers the code written here.

*And for the same manuscript's code statement:*

> **Code availability.** ColabSeqDisplay v0.1.0, the no-code Colab workflow used in this
> study, is openly available under the MIT licence at
> https://github.com/JasonJiangs/ColabSeqDisplay and archived at [DOI]. It is a front end for
> the fine-tuning protocol of [seqdisplay-opt reference]; the parts of that protocol the
> workflow runs are included in the repository under `colabsd/engine/` and attributed
> module by module in its `ATTRIBUTION.md`. Citation metadata are in the repository's
> `CITATION.cff`.

---

## Limitations

Stated plainly, so nobody has to discover them.

- **The fine-tuning engine is a copy, not a shared library.** `colabsd/engine/` holds the
  part of the seqdisplay-opt code the notebooks reach, copied across so that this
  repository runs on its own. A copy drifts: a correction made to the original does not
  reach this one by itself, and only what a notebook reaches was taken — the search
  machinery that *produces* a tuned configuration stayed with the original study.
  [`ATTRIBUTION.md`](ATTRIBUTION.md) lists what was copied, what was deliberately left
  behind, and the handful of deliberate departures from the original.
- **18 of the 28 hyperparameter files are placeholders.** They include *every*
  `mutation_site_mean` pair — which is the pooling a new protein starts from, since it
  needs no region file — plus the `cosine_p90_mean` entries for `ESM2-8M`, `ESMC-600M`,
  `SeqDance` and `Ankh-large`. Provisional values are the median of the 10 tuned configs;
  no measured performance stands behind them, and the notebook says so in capitals at
  five separate points. The 10 tuned files come from a 40-trial Optuna study per model on split seed 1
  and training seed 11, after which the three best validation configurations were re-trained
  over three data splits × three training seeds and the one with the highest mean validation
  Spearman across those nine runs was kept. One of the ten is for `METL`, which has no Colab
  adapter — so nine tuned pairs are reachable from the notebook.
- **Two backbones have never had real weights loaded, and only one has ever been
  fine-tuned.** `ProtT5-XL` and `Ankh-large` — the two 1.2B-parameter encoders — have had
  only their tokenizers exercised; their weights have never been downloaded or run. For
  the other eleven with an adapter, a loaded checkpoint and one pooled forward pass is not
  a training run: `ESM2-8M` is the only backbone fine-tuned end to end through this
  package, so every other entry in the table is an untested training path.
- **All reported numbers come from one protein.** The 10 tuned pairs span 0.5478 to 0.5636
  test Spearman (LoRA + MLP, mean over nine runs), the best being `ESM2-650M` with
  `cosine_p90_mean` at 0.5636 and the worst `ESMDance` at 0.5478 — a spread of 0.0158
  across the whole backbone range. Every one of those numbers is SlugCas9 5NNK. There is
  no second dataset, and nothing here establishes that the ranking transfers to another
  protein or another assay.
- **The T4 runtimes are estimates**, recorded as such in the registry. No Colab session has
  been timed.
- **The tuned hyperparameters were selected with a pooling region computed slightly
  differently from the one shipped here.** The two regions agree on 69 of 106 positions at
  p90 and 48 of 53 at p95, and they agree on what matters most: the five mutated positions
  rank first by variability in every configuration tested, and a long contiguous
  C-terminal block around them is selected either way. Whether the shipped region trains
  better, worse or the same has not been measured. Note also that a percentile cut keeps
  106 positions whatever the scores look like; do not read the p90 list as 106
  mechanistically meaningful residues.
- **Predictions rank variants; they do not measure them.** Read the top of a ranked table
  as the list to assay next, and read a test number for exactly what the unlock counter in
  the report footer says it is worth.

---

## Acknowledgement

The method — LoRA fine-tuning of protein language models on sequence-display libraries,
the pooling definitions, the multi-condition head, the evaluation protocol — and the tuned
hyperparameters in `config/best/` come from the **seqdisplay-opt** work. This repository is
the no-code Colab front end for it. It develops no method of its own: the code that runs
that method is copied into `colabsd/engine/`, every module opening with a header naming
the upstream module it derives from, and [`ATTRIBUTION.md`](ATTRIBUTION.md) collects those
headers and sets out exactly what was taken, what was changed and what was not.
Every performance number on this page is that study's measurement, not ours.

If you use ColabSeqDisplay, please cite both this software (see
[`CITATION.cff`](CITATION.cff)) and the seqdisplay-opt study:

> *[Citation for the seqdisplay-opt study to be added on publication.]*
