# ColabSeqDisplay

Fine-tune a protein language model on your own sequence-display variant library, in a browser,
without writing code. Bring one CSV — a row per variant, a column per mutated site, a column per
measured condition — and the wild-type sequence. Press buttons. Take away two `.zip` files: the
trained model, and a report that says which partition its numbers came from.

**It ranks variants you have not made yet.** Nothing on the form has been fine-tuned end to end in
this repository, and nothing here has been benchmarked on any protein but the one its
hyperparameters were tuned on. [Status](#status) is the full account.

**Version 0.1.0 · MIT license · [tutorial](TUTORIAL.md) · [status](#status)**

| | Open it | What it does |
|---|---|---|
| **ColabSeqDisplay** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay.ipynb) | The whole workflow: load a library, pick a backbone, fine-tune, read the report, export. Start here. |
| **Predict** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay_Predict.ipynb) | Upload a `model_bundle.zip` and a table of variants, get a ranked prediction table back. |

The fine-tuning engine — LoRA injection, training loop, pooling, metrics, split protocol — ships
inside this package as `colabsd/engine/`, attributed module by module in
[`ATTRIBUTION.md`](ATTRIBUTION.md). Nothing else to fetch.

**Try it.** The bundled example is a 124-variant MG8 PETase library: 19 mutated positions, one
measured activity, a 287-residue wild type. Open **ColabSeqDisplay**, set
`Runtime ▸ Change runtime type ▸ T4 GPU`, run the setup cell, change nothing, press **Check my library**,
then **Train**, then **Write performance_report.zip**. The panel estimates *about 1 min for 1 run*. Its 13 test variants
are a demonstration, not a benchmark.

---

## Your own library

One CSV describes a library. The schema, with the second data row of
`examples/mg8_petases/library.csv` as the example:

| Column | What it holds | Required | Example row |
|---|---|---|---|
| `p22` … `p279` | One column per mutated site, in the order of the positions you give. Three-letter or one-letter residue codes (a form field says which). | yes | `Y`, `V`, `S`, `I`, `F`, … `T` — 19 codes, one per column |
| `count` | Reads supporting the variant. Used only to drop low-count rows via `min_count`. | no | — (this example has none) |
| `activity` | One column per measured condition. Numeric, all predicted jointly by one multi-condition head. | yes | `1.57318039171052` |

The names and the number of columns are yours. A column you do not name — the example's `variant`
labels — is carried through untouched. Three form fields tie the CSV to the protein: the wild-type
sequence (pasted, or a FASTA URL), the 1-based positions of the mutated sites, and the mutation
column names in the same order.

Variant sequences are **built** from the wild type, never read from a FASTA, so every position is
checked against the sequence you supplied. A position that does not exist, or a residue code that
is not one of the 20 amino acids, is refused at load time, naming the column and the row.

**The library needs at least 20 rows** once `min_count` has dropped what it drops. The 8:1:1 split
is 80/10/10 rounded down and its partition sizes come from the row count alone — the seed chooses
membership, not sizes — so 19 variants give a one-row validation partition for every seed, which
NDCG is undefined on and Spearman has nothing to rank in. 20 is the smallest library where all
three partitions reach two rows, and anything smaller is refused by name before a run starts. It is
a floor, not a size worth training on.

Press **Check my library** and read the four lines it prints:

> **124 variants** from `examples/mg8_petases/library.csv`.
>
> - Wild type: 287 residues.
> - 19 mutated sites at 22, 42, 49, 72, 78, 79, 80, 130, 131, 132, 150, 198, 199, 205, 206, 215, 216, 225, 279, wild-type residues `YVSIYVSTRTWVVHAGSNT`.
> - 1 condition: `activity` from 0 to 6.281.

The wild-type residues catch 0-based numbering, or a sequence carrying a tag the positions were not
counted from. The range catches a condition column pointing at read counts (`from 3 to 900`)
instead of activity.

## The one question: which backbone

Everything after it follows from the answer.

- **A SaProt backbone needs a wild-type 3Di string.** The panel builds one in the same session from
  an AlphaFold `.cif`, your own `.pdb`, a 3Di file or a 3Di string you already have, or ESMFold on
  the runtime. Nothing of the kind is bundled. Every other backbone reads sequence alone, and the
  step is not on the page at all.
- **Modeling hyperparameters are looked up, not tuned.** The seven LoRA values come from
  `config/best/` for the backbone you picked, read-only.

**The budget is yours** — three boxes, prefilled from the same entry and editable:

| Box | Prefilled | What it does |
|---|---|---|
| **Epochs, at most:** | 20 | The ceiling. Early stopping usually ends the run before it. |
| **Give up after this many epochs with no gain:** | 3 | Early stopping, judged on the validation score. It may equal the ceiling but not exceed it. |
| **Sequences on the GPU at once (memory):** | 8 | The only setting that changes how much memory a run needs. Lower it after an out-of-memory error: `effective_batch_size` is looked up and differs by entry (32 for `ESM2-35M` and the three SaProts, 16 for the other five), and gradient accumulation holds the batch there whatever you pick, so this costs speed, not score. |

A budget the loop cannot honor is refused before **Train**, with a sentence naming the box. What
you change is recorded, and marked as yours, in both files you take away.

**The test set is locked, and unlocks are counted.** The panel reports validation only; each run's
test artifacts go straight into `<run>/locked_test/`. The notebook's fourth cell reads them,
increments a counter, and stamps that count — with a sentence saying what the number is worth —
into every report, archive and bundle written afterwards. The [tutorial](TUTORIAL.md) explains what
each count means.

## Backbones

Nine on the form, in the order the dropdown offers them. Seven need nothing beyond `transformers`;
two need one extra install, named in the row and again by the panel when you pick them.

| Backbone | Checkpoint | Pooled width | 3Di | GPU, and any extra install | Est. min/run, T4 |
|---|---|---|---|---|---|
| `ESM2-35M` *(default)* | `facebook/esm2_t12_35M_UR50D` | 480 | — | free T4 | 40 |
| `ESMDance` | `ChaoHou/ESMDance` | **50** | — | free T4 | 45 |
| `SaProt-35M` | `westlake-repl/SaProt_35M_AF2` | 480 | **yes** | free T4 | 45 |
| `ESM2-150M` | `facebook/esm2_t30_150M_UR50D` | 640 | — | free T4 | 90 |
| `ESMC-300M` | `EvolutionaryScale/esmc-300m-2024-12` | 960 | — | free T4, `pip install 'esm>=3.1,<3.4'`, then restart the runtime | 120 |
| `ESM2-650M` | `facebook/esm2_t33_650M_UR50D` | 1280 | — | free T4 | 240 |
| `SaProt-650M` | `westlake-repl/SaProt_650M_AF2` | 1280 | **yes** | free T4 | 260 |
| `ProtT5-XL` | `Rostlab/prot_t5_xl_uniref50` | 1024 | — | **L4/A100**, `pip install sentencepiece protobuf`, then restart the runtime | — |
| `SaProt-1.3B` | `westlake-repl/SaProt_1.3B_AF2` | 1280 | **yes** | **L4/A100**, `bfloat16` only | — |

*Pooled width* is the width of the vector the head is built on. `ESMDance` is the one row where it
is not a trunk embedding: its feature is its own 50-dimensional `res_pred` prediction of
per-residue dynamics, so read its ρ as a different read-out, not as a much smaller model doing as
well.

*Est. min/run* is an order-of-magnitude figure for one split seed × one model seed, **sized
against** the study's 16,424-variant library — never measured on a Colab runtime. The panel
rescales it by your variants, epochs and runs (floor: one minute) and reports measured elapsed time
at the end. The two rows with no figure cannot train on a free T4 at all, and the panel blocks
**Train** until the runtime is an L4 or A100.

Every entry is *load checked*: checkpoint loaded, one pooled forward pass, the pooled vector
asserted to be exactly the width above, LoRA injection validated against the real module tree. None
has been fine-tuned here. Four need torch ≥ 2.6 even to load (see [Requirements](#requirements)),
and `ProtT5-XL` is an 11 GB download — the published file is the full encoder-decoder and only the
encoder survives it.

**Numeric precision.** The dropdown offers `float32` (the default), `bfloat16` and `float16`, and
**`float16` cannot train any model here** — the optimizer's epsilon, 1e-8, rounds to zero in it, so
the first update divides by zero and every weight becomes `NaN`, and the run *finishes* and reports
a validation Spearman of `0.0000`. Picking it blocks **Train** with a message saying so, and a run
whose loss curve or predictions come back non-finite for any other reason is refused rather than
reported. `bfloat16` is the same two bytes a weight, so the same memory, and trains; it is what
`SaProt-1.3B` needs. `float16` remains on the dropdown — a reader who followed an
older instruction to it has to be told why rather than find it gone — and a `model_bundle.zip`
that already records `float16` still scores: the Predict notebook has no precision control and
rebuilds the model in whatever precision the bundle recorded.

**Which to pick.** On a free T4, `ESM2-35M` (sequence only) or `SaProt-35M` (needs a 3Di string).
`ESM2-35M` is the cheapest run on the form: put a new library through it once before paying for a
long run.

**What reaches the form.** A backbone appears only when `config/best/` holds **tuned**
hyperparameters for it and its adapter builds; all nine above are tuned. The registry holds
fourteen entries, and the panel names the five it withholds and why. Thirteen have a working
adapter, so Predict loads a bundle built on any of those; the fourteenth, `METL`, has none, and a
bundle naming it is refused as it loads.

### Five ways to lose an hour

| What costs the hour | What to do instead |
|---|---|
| `ProtT5-XL` or `SaProt-1.3B` on a free T4 — a multi-GB download for a model that cannot train on the card. | Switch the runtime to an L4 or A100 first. The panel blocks **Train** until you do. |
| `SaProt-1.3B` at the default `float32` — out of memory on every card Colab offers, after the download. | Tick **Show the advanced settings** in step 1 and set **Numeric precision:** to `bfloat16` in step 2, the backbone section. |
| **Numeric precision:** `float16` — an hour of training that ends in `NaN` and is reported as Spearman `0.0000`. | Use `bfloat16`: two bytes a weight either way, so the same memory, and the optimizer's epsilon does not round to zero in it. The panel refuses `float16` for training before anything downloads, and refuses any run whose loss curve comes back non-finite whatever caused it, so you find this out at the press of **Train** rather than at the end of the hour. |
| An unpinned `pip install esm` — from 3.4.0 LoRA reaches only `out_proj`, and the adapter refuses the run by name after a 1.3 GB download and a press of **Train**. | Install `'esm>=3.1,<3.4'`, quotes included, and restart the runtime: it also moves `transformers` to below 4.48.2. |
| Clearing `colabsd_work/` before exporting — the bundle is written from the training checkpoint on disk. | Write both `.zip` files first. |

## Configurations

`config/best/` holds 28 YAML files, two per backbone. `colabsd.bestconfig` always reads the 14
`mutation_site_mean` half, of which **10 are tuned and 4 are placeholders** (`Ankh-large`,
`ESM2-8M`, `ESMC-600M`, `SeqDance` — none of them on the form). A placeholder prints `PROVISIONAL`
wherever a tuned entry prints its ρ, and that flag follows a run into every artifact.
[`config/best/README.md`](config/best/README.md) has the full accounting and the file format.

**ρ is the mean ± sd test Spearman over nine re-evaluation runs, measured by the seqdisplay-opt
study on its own 16,424-variant SlugCas9 5NNK library** (see [Acknowledgement](#acknowledgement)) —
not by this package, and not on your protein. Each entry came from a 40-trial Optuna study on split
seed 1 with model seed 11, and the selected trial was re-trained over three split seeds × three
model seeds.

| Backbone | ρ, study measurement | Optuna trial |
|---|---|---|
| `ESM2-150M` | 0.5636 ± 0.0113 | 27 |
| `ESM2-650M` | 0.5618 ± 0.0125 | 39 |
| `ESMC-300M` | 0.5611 ± 0.0107 | 35 |
| `ESM2-35M` | 0.5608 ± 0.0120 | 37 |
| `ProtT5-XL` | 0.5581 ± 0.0162 | 21 |
| `SaProt-650M` | 0.5580 ± 0.0085 | 2 |
| `ESMDance` | 0.5569 ± 0.0128 | 28 |
| `SaProt-1.3B` | 0.5559 ± 0.0162 | 12 |
| `SaProt-35M` | 0.5555 ± 0.0126 | 2 |

The nine span 0.0081, less than one backbone's own ± 0.0162 across its nine runs. Pick on cost, not
on this table.

**Identical in every entry**: 20 maximum epochs, early stopping with patience 3, MSE loss, targets
z-scored on the training split, gradient accumulation derived from the effective batch size, no
test evaluation during optimization. The first two, and the micro-batch that accumulation is built
from, are the three editable boxes above; the rest are fixed for everyone.

### Seeds and splits

Every entry ends with the same evaluation block, and both lists are copied into
`performance_report.zip`:

```yaml
evaluation:
  selection_objective: mean_validation_spearman
  split_seeds: [1, 2, 3]
  model_seeds: [11, 22, 33]
```

They set the ceilings on the two seed sliders — at most three of each, so at most nine runs — and
training takes the first *n* of each, so the default single run is always split seed 1 with model
seed 11. **A single run cannot show reproducibility**: the report writes its standard deviation as
`NaN`, not `0`, and the panel says so before you press **Train**.

Splits are **8:1:1** — 99 training / 12 validation / 13 test rows per seed for the bundled
example — cached under `colabsd_work/splits/n<rows>/`, and a cached split that does not cover
exactly the rows of the library now loaded is refused rather than reused. Test rows move into
`<run>/locked_test/` as each run finishes.

## What you take away

**`model_bundle.zip` — the model.** Three members: `manifest.json`, `lora.pt`, `head.pt`. No
backbone weights: the backbone is re-fetched by name on the first scoring run, so Predict needs
internet. The bundle holds the single best-validation run, not an ensemble, and is written from the
training checkpoint, so export it before you clear `colabsd_work/`. Its manifest carries your library
specification (which is also the residues the model averages), the hyperparameters, the budget the
run was *given* with your own numbers marked as yours, the unlock count and the provenance.
Predictions come back in the assay's own units when the bundle carries the label scaler it was
trained with, and in z-scored training units when it does not — Predict says which, and in the
second case only the ranking is meaningful. Bundles are schema 3; the only ones this version
refuses are schema 1, written when the pooling was still a choice.

**`performance_report.zip` — the numbers, and what they are.** Seven members: `README.txt`,
`performance.json`, `report.csv`, `report.png`, `report.json`, `training_curve.csv`,
`training_curve.png`. `report.csv` is one row per partition × condition plus a mean row across
conditions, with a mean, a standard deviation and a count for each of seven metrics.
`training_curve.png` is the same figure the panel draws while the run is going. Every member
carries a sha256, verified on read-back. `README.txt` and `performance.json` come from one object,
so they cannot disagree. Both name the partition, the unlock count and what it is worth, the
hyperparameter status (`tuned` or `PROVISIONAL`), the backbone, the budget, the conditions, the
runs and the seeds. `performance.json` adds `partition_rows` — **the row count of the one
partition it reports on**, a single integer, so you need not note it by hand; `README.txt` quotes
that same number only in its `not a real cut` line, where a metric ranked more rows than the
partition holds. The counts for *every* partition are in `report.json`, one of the seven members
of the same zip, where `partition_rows` is a mapping keyed by partition rather than one integer.

**Write this archive before you unlock anything.** It publishes validation numbers with the test
partition untouched. Its only two blockers are *nothing has been trained in this session* and *the
settings have changed since this model was trained*. Press the same button after an unlock and the
file is rewritten in place, with the test numbers and the count.

Four of the seven metrics — `P@10`, `P@50`, `NDCG@10` and `NDCG@50` — are ranking cuts, and on a
partition shorter than the cut they come out high rather than good: on the example's 12 validation
rows, `P@50` is exactly 1.0000 for every possible model. Quote `Spearman` and `R2` instead. The
archive's `README.txt` prints that warning with the row count, and the [tutorial](TUTORIAL.md) has
the arithmetic.

**If the session dies, or you stop a run**, the best epoch is already on disk as
`colabsd_work/<run>/runs/<split>_<seed>/best_checkpoint.pt`, rewritten every time the validation
score improved. Nothing deletes it: the next **Train** press renames it out of the way first and
says where it went. A run that was stopped is never reused as a finished one. The
[tutorial](TUTORIAL.md) shows the two readers that turn a checkpoint back into a loadable bundle.

## Requirements

**Python ≥ 3.10**; [`pyproject.toml`](pyproject.toml) lists the rest and is the authority. Outside
Colab, `pip install .`. Two of its bounds are load-bearing:

| Pin | Why it is there |
|---|---|
| `torch>=2.6` | Since CVE-2025-32434 `transformers` refuses to `torch.load` a `pytorch_model.bin`, and four of the nine offered backbones publish only `.bin`: `SaProt-35M`, `SaProt-650M`, `SaProt-1.3B` and `ProtT5-XL`. Below the floor those four cannot load at all, cache or no cache, and it is the first cause the panel's "could not load" message names. Colab is already above it. |
| `esm>=3.1,<3.4` | The `[esmc]` extra, for `ESMC-300M`. From 3.4.0 the SDK's fused attention projection is no longer a layer LoRA can reach, and the adapter refuses the run rather than let a quarter-adapted one start. |

`[prott5]` is `sentencepiece` and `protobuf`, the two packages `ProtT5-XL` needs before its
tokenizer will build. `[optuna]` and `[dev]` are declared too.

**GPU.** A free Colab T4 (16 GB) fits every offered backbone except `SaProt-1.3B` and `ProtT5-XL`.
ESMFold, if you use it for the 3Di step, is a 2.6 GB download that then takes almost the whole
card: `colabsd.structure` puts its free-T4 ceiling at **700 residues**, well above the 287-residue
example. Without a GPU the notebooks still check a library, draw the report, write both archives
and score from an existing bundle.

## Status

| item | status |
|---|---|
| The workflow — library loading, sequence construction, splits and the stale-cache refusal, test locking and the unlock counter, report tables and figure, both exports and their read-back, bundle reload, variant scoring | run end to end on the bundled 124-variant MG8 PETase example |
| Backbone loading, pooled-vector width, LoRA injection coverage | load checked against the published checkpoints for all nine offered backbones and three of the withheld ones — `ESM2-8M`, `ESMC-600M` and `SeqDance`. `Ankh-large` has an adapter and has never had its weights loaded; the row below says so. The four `.bin`-only backbones' checks self-skip below torch 2.6. **None fine-tuned end to end** |
| `Ankh-large` (withheld) | **tokenizer only** — its ~1.2B-parameter weights have never been downloaded or run |
| LoRA fine-tuning end to end | **`ESM2-8M` only**, and it is not on the form, so no training path the panel offers has been exercised here. That run used the seqdisplay-opt study's 16,424-variant SlugCas9 5NNK library, which is not bundled here, and that study's region read-out, which this package no longer computes (placeholder hyperparameters, one split seed × one model seed): it reached test Spearman 0.5392 |
| How well any backbone predicts activity | **not measured here.** Every benchmark number on this page is the seqdisplay-opt study's, on SlugCas9 5NNK. The bundled example shows the workflow runs; its 13 test variants measure nothing |
| Any Colab runtime timing | **not measured** — every minute quoted here is an estimate, scaled from one timed forward pass: `ESM2-650M` in float16 over 16,424 sequences of 1054 residues, 319 s, on a B200 |
| Benchmark ρ in the [configuration table](#configurations) | produced by the seqdisplay-opt study on NVIDIA B200-class GPUs, not here |
| The hyperparameters those ρ selected | selected on that study's protein, with the same mutated-site read-out a run here uses. **How they transfer to your protein has not been measured.** The four entries with no study behind them stay provisional, and the form offers none of them |
| `colabsd/engine/` | a copy of the seqdisplay-opt code the notebooks reach, not a shared library; the search machinery that *produces* a tuned configuration stayed with that study. [`ATTRIBUTION.md`](ATTRIBUTION.md) lists what was copied, what was left behind and the deliberate departures |
| Determinism | fixing a seed fixes the split and the initialization; it does not make GPU training bit-reproducible across cards, drivers or library versions |
| Predictions | rank variants; they do not measure them. The first rows of your library are mostly rows the model trained on, so scoring them and seeing agreement is a wiring check, not evidence |

---

## Data availability

The example library is under `examples/mg8_petases/`: the 124 MG8 PETase variants described above,
whose activity is `ln(1 + raw)` of the value assayed relative to the wild type, plus the 287-residue
wild-type sequence. `examples/mg8_petases/README.md` documents each file, its derivation from the
source spreadsheet kept beside it, and the two pairs of rows that are the same protein twice. No
3Di string is bundled. Everything under `examples/` is covered by this repository's MIT license
([`LICENSE`](LICENSE)).

The tuned hyperparameters in `config/best/` and every ρ on this page come from the seqdisplay-opt
study's re-evaluation tables, measured on a 16,424-variant SlugCas9 5NNK library. **That library is
not redistributed here**, and anyone reusing those values should cite the study: *[reference to be
added on publication]*. The MG8 PETase measurements are from *[reference or accession to be added
on publication]*.

*For a manuscript that used ColabSeqDisplay, with the bracketed values filled in:*

> **Data availability.** The example variant library analyzed by ColabSeqDisplay — 124 MG8 PETase
> variants with a measured activity at 19 mutated positions, and the 287-residue wild-type
> sequence — is included in the ColabSeqDisplay repository
> (https://github.com/JasonJiangs/ColabSeqDisplay) under `examples/mg8_petases/` and is archived
> with it at [DOI]. These measurements originate from [reference or accession for the MG8 PETase
> activity data]; the hyperparameters the software ships derive from [seqdisplay-opt reference],
> whose variant library is not redistributed here. No new experimental data were generated by this
> software.

## Code availability

ColabSeqDisplay **version 0.1.0** is openly available at
<https://github.com/JasonJiangs/ColabSeqDisplay> under the **MIT license** (full terms in
[`LICENSE`](LICENSE)), and runs in Colab from the badges above. Citation metadata are in
[`CITATION.cff`](CITATION.cff), which holds a placeholder DOI until the archive is deposited.

The repository is self-contained: the engine it runs is `colabsd/engine/` inside the package, and
each notebook's setup cell names this one repository. One caveat for anyone redistributing this
software: the upstream project carries no license file of its own, so the terms covering
`colabsd/engine/` and `config/best/` are not stated anywhere and should be confirmed with its
authors. This repository's MIT license covers the code written here.

*And for the same manuscript's code statement:*

> **Code availability.** ColabSeqDisplay v0.1.0, the no-code Colab workflow used in this
> study, is openly available under the MIT license at
> https://github.com/JasonJiangs/ColabSeqDisplay and archived at [DOI]. It is a front end for
> the fine-tuning protocol of [seqdisplay-opt reference]; the parts of that protocol the
> workflow runs are included in the repository under `colabsd/engine/` and attributed
> module by module in its `ATTRIBUTION.md`. Citation metadata are in the repository's
> `CITATION.cff`.

---

## Acknowledgement

The method — LoRA fine-tuning of protein language models on sequence-display libraries — and the
tuned hyperparameters in `config/best/` come from the **seqdisplay-opt** work. This repository is
its no-code Colab front end and develops no method of its own; every benchmark number on this page
is that study's measurement, not ours. [`ATTRIBUTION.md`](ATTRIBUTION.md) sets out, module by
module, what was taken, what was changed and what was not.

If you use ColabSeqDisplay, please cite both this software ([`CITATION.cff`](CITATION.cff)) and the
seqdisplay-opt study:

> *[Citation for the seqdisplay-opt study to be added on publication.]*
