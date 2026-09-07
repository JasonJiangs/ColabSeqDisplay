# ColabSeqDisplay

Fine-tune a protein language model on your own sequence-display variant library, in a browser,
without writing code. Bring a CSV — one row per variant, one column per mutated site, one column
per measured condition — plus the wild-type sequence. Get back a trained model, a report
comparing it against a one-hot baseline, a test partition that stays locked until you
deliberately open it, and two `.zip` files: the model, and the performance analysis that says
which partition its numbers came from.

**Version 0.1.0 · MIT licence · [tutorial](TUTORIAL.md) · [status](#status)**

| | Open it | What it does |
|---|---|---|
| **ColabSeqDisplay** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay.ipynb) | The whole workflow: load a library, pick a backbone and a pooling, let it prepare whatever those two answers need, fine-tune, read the report, export. Start here. |
| **Predict** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay_Predict.ipynb) | Upload a `model_bundle.zip` and a table of variants, get a ranked prediction table back. The bundle is self-contained. |

The fine-tuning engine — LoRA injection, training loop, pooling, metrics, split protocol — ships
inside this package as `colabsd/engine/`, copied from the study that developed it and attributed
module by module in [`ATTRIBUTION.md`](ATTRIBUTION.md). Nothing else to fetch.

**Try it.** The bundled example is a real SlugCas9 5NNK library: 16,424 variants, five mutated
positions, four measured PAM conditions. Open **ColabSeqDisplay**, set `Runtime ▸ Change runtime
type ▸ T4 GPU`, run the setup cell, change nothing, press train, then press **Write
performance_report.zip** — estimated at about an hour on a free T4.

---

## Your own library

One CSV describes a library. The schema, with the second data row of
`examples/slugcas9_5nnk/library.csv` exactly as it appears in the file:

| Column | What it holds | Required | Example row |
|---|---|---|---|
| `nnk1` … `nnk5` | One column per mutated site, in the order of the positions you give. Three-letter or one-letter residue codes (a form field says which). | yes | `Asn`, `Gln`, `Leu`, `Ala`, `Glu` |
| `count` | Reads supporting the variant. Used only to drop low-count rows via `min_count`. | no | `1725` |
| `NNGA`, `NNGT`, `NNGC`, `NNGG` | One column per measured condition. Numeric activity, all predicted jointly by one multi-condition head. | yes | `0.7455072402954102`, `0.14260870218276978`, `0.4046376943588257`, `0.6857970952987671` |

The names and their number are yours: any number of mutated sites, any number of measured
conditions. Three form fields tie the CSV to the protein — the wild-type sequence (pasted, or the
URL of a FASTA), the 1-based positions of the mutated sites, and the mutation column names in the
same order. In the example: positions 984, 985, 990, 1012 and 1016 of the 1054-residue wild type,
whose residues there are N, S, M, E and K. Variant sequences are **built**, never read from a
FASTA — each row's residues are substituted into the wild type — so every variant is full length
and every position is checked against the sequence you supplied. A position that does not exist,
or a residue code that is not an amino acid, is refused at load time.

## What the two answers decide

Backbone and pooling are asked once; everything after follows, including what must be prepared
first. The panel makes whichever your answers call for, in the same session, and ships both for
the bundled example. Region discovery can outlast a Colab session, so its estimated cost is shown
before you commit, and train never starts one for you.

| Pooling | Reads out from | Prepared first |
|---|---|---|
| `mutation_site_mean` | the mutated positions | nothing |
| `cosine_p90_mean` | the residues whose embeddings disagree most across your variants — `1 − mean pairwise cosine similarity` per residue, cut at the percentile named in the pooling | a pooling region |

A SaProt backbone additionally needs a wild-type 3Di string. The regions shipped for SlugCas9 were
computed with ESM2-650M in float32 over all 16,424 variants: 106 positions (p90, threshold
3.371860449376874 × 10⁻⁴) and 53 (p95). Each records how it was made, and one whose declared
pooling or sequence length does not match the run is refused.

**Hyperparameters are looked up, not tuned** — the seven LoRA values for a pair come from
`config/best/`, with no way to change them. **The test set is locked and unlocks are counted:**
the panel reports validation only; the notebook's second cell opens the test partition, increments
a counter, and stamps that count into every report, archive and bundle written afterwards.

## What you take away

**`model_bundle.zip` — the model.** LoRA weights, the head, your library specification, the
**frozen** pooled residue coordinates, the hyperparameters and the provenance. No backbone weights
(re-fetched by name) and no dependency on a region file. The only file Predict needs.

**`performance_report.zip` — the numbers, and what they are.** `report.csv`, `report.png` and
`report.json` as `colabsd.report.build_report` writes them, plus `performance.json` for a machine
and `README.txt` for a person, generated from one object so they cannot disagree. Both name the
partition (`partition`, `describes_test_partition`, a badge in `report.png`), the unlock count with
a sentence saying what it is worth (`unlock.count`, `unlock.source`, `unlock.verdict`), the
hyperparameter status spelled `tuned` or `PROVISIONAL`, the backbone, adapter and pooling, the
one-hot floor, the conditions, the number of runs and the seeds. Every member carries a sha256,
verified on read-back.

**This archive is obtainable without unlocking the test partition** — `build_report` publishes
validation numbers while the test partition is untouched. Its only blockers are *nothing has been
trained yet* and *the settings have changed since the run*. Press the same button after an unlock
and the file is rewritten with the test numbers and the count.

---

## Backbones

Seven entries, all `native` tier — nothing beyond `transformers` to install.

| Backbone | Checkpoint | Pooled width | 3Di | GPU | Est. min/run, T4 | Exercised here |
|---|---|---|---|---|---|---|
| `ESM2-8M` | `facebook/esm2_t6_8M_UR50D` | 320 | no | free T4 | 20 | **fine-tuned** |
| `ESM2-35M` *(default)* | `facebook/esm2_t12_35M_UR50D` | 480 | no | free T4 | 40 | load checked |
| `ESM2-150M` | `facebook/esm2_t30_150M_UR50D` | 640 | no | free T4 | 90 | load checked |
| `ESM2-650M` | `facebook/esm2_t33_650M_UR50D` | 1280 | no | free T4 | 240 | load checked |
| `SaProt-35M` | `westlake-repl/SaProt_35M_AF2` | 480 | **yes** | free T4 | 45 | load checked |
| `SaProt-650M` | `westlake-repl/SaProt_650M_AF2` | 1280 | **yes** | free T4 | 260 | load checked |
| `SaProt-1.3B` | `westlake-repl/SaProt_1.3B_AF2` | 1280 | **yes** | **L4/A100**, `dtype="float16"` | — (a T4 runs out of memory) | load checked |

*Pooled width* is the width of the vector the head is built on, not always the encoder's hidden
size. *Est. min/run* are order-of-magnitude registry estimates for one split seed × one model seed
on a library the size of the bundled example; the panel rescales them for your library and reports
measured elapsed time afterwards. *load checked*: checkpoint downloaded and loaded, one pooled
forward pass run, the pooled vector asserted to be exactly the width the registry promises, LoRA
injection validated against the real module tree — not fine-tuned. *fine-tuned*: trained end to
end here; see [Status](#status) for what that run produced.

**Which to pick.** On a free T4, `ESM2-35M` (sequence only) or `SaProt-35M` (needs a 3Di string);
both are tuned pairs. Run `ESM2-8M` first as a sanity check on a new library before paying for a
long run.

`colabsd.backbones.registry` holds fourteen entries, thirteen with a working adapter, and the seven
above are on screen. Every other adapter is still in the package, buildable as
`create_adapter("ESMC-300M", pooling=...)`, and carries its `config/best/` entry; the panel lists
all seven withheld backbones by name, each with the `withheld_reason` recorded for its family, and
Predict loads a bundle from any of them.

## Configurations

`config/best/` holds one file per (backbone, pooling) pair; below, the fourteen the notebooks can
reach. **ρ is the mean test Spearman over the nine re-evaluation runs recorded in that file,
measured on the SlugCas9 5NNK benchmark library by the seqdisplay-opt study** (see
[Acknowledgement](#acknowledgement)) — not by this package, and not on your protein.

| Backbone | `cosine_p90_mean` | `mutation_site_mean` |
|---|---|---|
| `ESM2-8M` | placeholder | placeholder |
| `ESM2-35M` | **tuned**, ρ 0.5486 ± 0.0145 | placeholder |
| `ESM2-150M` | **tuned**, ρ 0.5546 ± 0.0105 | placeholder |
| `ESM2-650M` | **tuned**, ρ 0.5636 ± 0.0128 | placeholder |
| `SaProt-35M` | **tuned**, ρ 0.5576 ± 0.0162 | placeholder |
| `SaProt-650M` | **tuned**, ρ 0.5592 ± 0.0110 | placeholder |
| `SaProt-1.3B` | **tuned**, ρ 0.5578 ± 0.0142 | placeholder |

*tuned*: a 40-trial Optuna study per model on split seed 1 and training seed 11; the three best
validation configurations were re-trained over three splits × three training seeds and the highest
mean validation Spearman across those nine runs kept. *placeholder*: `_meta.status: provisional`,
the median of the 10 tuned configs, with no measured performance behind it — the panel, the
training log, both exports, the bundle manifest, the unlock cell and Predict all print
`PROVISIONAL` on such a pair. Across all 28 files: 10 tuned, 18 placeholder; the four tuned files
outside this table are `ESMC-300M`, `ProtT5-XL`, `ESMDance` and `METL`.

**Fixed training settings**, identical in every file: 20 maximum epochs, early stopping with
patience 3, MSE loss, targets z-scored on the training split, gradient accumulation derived from
the effective batch size, no test evaluation during optimisation.

### Seeds and splits

Every file also ends with the same evaluation block, and both of its lists go into
`performance_report.zip`:

```yaml
evaluation:
  selection_objective: mean_validation_spearman
  split_seeds: [1, 2, 3]
  model_seeds: [11, 22, 33]
```

`split_seeds` choose the partition of rows, `model_seeds` the initialisation of the LoRA adapters
and the head; training takes the first `n_split_seeds` and `n_model_seeds` of each, so the default
single run is always split seed 1 with model seed 11, and the one-hot floor is fitted on the same
split objects with the same model seeds. Splits are **8:1:1** — 13,139 training / 1,642 validation
/ 1,643 test rows per seed for the bundled example — cached under `colabsd_work/splits/n<rows>/`,
and a cached split that does not cover exactly the rows of the library now loaded is refused rather
than reused. Test rows move to `run/locked_test/` as each run finishes and are read only by the
unlock cell.

## Requirements

**Python ≥ 3.10**, and the dependencies `pyproject.toml` lists: `numpy>=1.26,<3`, `pandas>=2.1,<4`,
`scipy>=1.11,<2`, `scikit-learn>=1.3,<2`, `matplotlib>=3.7,<4`, `tqdm>=4.66,<5`, `torch>=2.2,<3`,
`transformers>=4.40`, `huggingface_hub>=0.23`, `pyyaml>=6,<7`, `ipywidgets>=8,<9`, `ipython>=8`.
Extras `[esmc]` and `[prott5]` serve withheld backbones; `[optuna]` and `[dev]` are declared too.
Outside Colab, `pip install .`.

**GPU.** A free Colab T4 (16 GB) fits every offered backbone except `SaProt-1.3B`.
`colabsd.structure` puts the ESMFold ceiling on a free T4 at **700 residues**; the 1054-residue
bundled example ships a 3Di string instead. Without a GPU the notebooks still load and check a
library, run the one-hot baseline, draw the report, write both archives, and score from an existing
bundle; region discovery, ESMFold and fine-tuning each check for a GPU first.

## Status

| item | status |
|---|---|
| Library loading and sequence construction; split creation, caching and the stale-cache refusal; test-set locking, the unlock counter and its stamping into report, archive and bundle; report tables and figure, both exports and their read-back, bundle reload, variant scoring | run end to end on the bundled 16,424-variant example |
| The one-hot floor (ridge and MLP) over the full example library | run end to end, CPU only |
| Backbone loading, pooled-vector width, LoRA injection coverage | load checked against the published checkpoints for all seven offered backbones, and for four of the six withheld ones that have an adapter |
| `ProtT5-XL` and `Ankh-large` (withheld) | **tokenizers only** — their ~1.2B-parameter weights have never been downloaded or run |
| LoRA fine-tuning end to end | **`ESM2-8M` only** — every other training path is untested here. In that run (full library, placeholder hyperparameters, one split seed × one model seed) the fine-tuned model reached test Spearman 0.5392 against a one-hot MLP floor of 0.5571 |
| Any protein other than SlugCas9 | **not attempted** — every performance number on this page is SlugCas9 5NNK |
| Any Colab runtime timing | **not measured** — every minute quoted here is an estimate. The region-discovery estimate is scaled from one timed forward pass: ESM2-650M over all 16,424 SlugCas9 variants of 1054 residues in float16, 319 s, on a B200 |
| Benchmark ρ in the [configuration table](#configurations) | produced by the seqdisplay-opt study on NVIDIA B200-class GPUs, nine runs per model over three splits × three seeds. A single default Colab run reports `nan` for the standard deviation |
| The pooling region behind the tuned hyperparameters | computed slightly differently from the one shipped here: the two agree on 69 of 106 positions at p90 and 48 of 53 at p95. Whether the shipped region trains better, worse or the same has not been measured |
| `colabsd/engine/` | a copy of the seqdisplay-opt code the notebooks reach, not a shared library; the search machinery that *produces* a tuned configuration stayed with the original study. [`ATTRIBUTION.md`](ATTRIBUTION.md) lists what was copied, what was left behind, and the deliberate departures |
| Determinism | fixing a seed fixes the split and the initialisation; it does not make GPU training bit-reproducible across cards, drivers or library versions |
| Predictions | rank variants; they do not measure them |

---

## Data availability

The example variant library distributed with this software is included under
`examples/slugcas9_5nnk/`: 16,424 SlugCas9 5NNK variants with read counts and measured activity in
four PAM conditions (NNGA, NNGT, NNGC, NNGG); the 1054-residue wild-type amino-acid sequence; its
Foldseek 3Di string; and two derived pooling regions of 106 and 53 residue positions.
`examples/slugcas9_5nnk/README.md` documents each file, its schema and its provenance.

`library.csv`, `wt.fasta` and `wt_3di.txt` are redistributed unchanged — byte-identical to their
counterparts — from the seqdisplay-opt study (see [Acknowledgement](#acknowledgement));
`region_p90.json` and `region_p95.json` were computed by this package from `library.csv` itself,
each recording the model, score definition, percentile, threshold and number of variants scored.
The tuned hyperparameters in `config/best/` and every benchmark value in this README derive from
the re-evaluation tables of that study. All files under `examples/` are covered by this
repository's MIT licence (see [`LICENSE`](LICENSE)), which permits redistribution and reuse with
attribution; the primary source of the measurements is the seqdisplay-opt study, which anyone
reusing them should cite as such:
*[reference to be added on publication — see the Acknowledgement below]*.

This software generates no experimental data of its own. All outputs of a run — splits,
checkpoints, the prepared 3Di string and pooling region, the three report files, both archives and
`scored_variants.csv` — are derived from the library the user supplies and written to the
runtime's `colabsd_work/` directory, from where the notebook also offers the exports as downloads.

*For a manuscript that used ColabSeqDisplay, with the bracketed values filled in:*

> **Data availability.** The example variant library analysed by ColabSeqDisplay — 16,424
> SlugCas9 5NNK variants with read counts and measured activity in four PAM conditions, the
> 1054-residue wild-type sequence, its Foldseek 3Di string and two derived pooling regions —
> is included in the ColabSeqDisplay repository (https://github.com/JasonJiangs/ColabSeqDisplay)
> under `examples/slugcas9_5nnk/` and is archived with it at [DOI]. These measurements originate from
> [seqdisplay-opt reference]. No new experimental data were generated by this software.

## Code availability

ColabSeqDisplay **version 0.1.0** is openly available at
<https://github.com/JasonJiangs/ColabSeqDisplay> under the **MIT licence** (OSI-approved; full
terms in [`LICENSE`](LICENSE)), and runs in Google Colab from the badges above. Citation metadata —
title, version, authors, licence and archive DOI — are in [`CITATION.cff`](CITATION.cff), which
GitHub renders as a ready-made reference; it holds a placeholder DOI until the archive is deposited
and names every value to be filled in then.

**The repository is self-contained**: a reader can obtain, inspect, install, execute and cite
ColabSeqDisplay from it alone, the engine it runs is `colabsd/engine/` inside the package, and the
setup cell of each notebook names this one repository. [`ATTRIBUTION.md`](ATTRIBUTION.md) records
where that engine came from, what was taken and what was changed, module by module, and states what
anyone redistributing this software should read first: the upstream project carries no licence file
of its own, so the terms covering the code in `colabsd/engine/` and the values in `config/best/`
are not stated anywhere and should be confirmed with its authors. This repository's own MIT licence
covers the code written here.

*And for the same manuscript's code statement:*

> **Code availability.** ColabSeqDisplay v0.1.0, the no-code Colab workflow used in this
> study, is openly available under the MIT licence at
> https://github.com/JasonJiangs/ColabSeqDisplay and archived at [DOI]. It is a front end for
> the fine-tuning protocol of [seqdisplay-opt reference]; the parts of that protocol the
> workflow runs are included in the repository under `colabsd/engine/` and attributed
> module by module in its `ATTRIBUTION.md`. Citation metadata are in the repository's
> `CITATION.cff`.

---

## Acknowledgement

The method — LoRA fine-tuning of protein language models on sequence-display libraries, the pooling
definitions, the multi-condition head, the evaluation protocol — and the tuned hyperparameters in
`config/best/` come from the **seqdisplay-opt** work. This repository is its no-code Colab front end
and develops no method of its own: that code is copied into `colabsd/engine/`, every module opening
with a header naming the upstream module it derives from, and [`ATTRIBUTION.md`](ATTRIBUTION.md)
sets out what was taken, what was changed and what was not. Every performance number on this page
is that study's measurement, not ours.

If you use ColabSeqDisplay, please cite both this software (see [`CITATION.cff`](CITATION.cff)) and
the seqdisplay-opt study:

> *[Citation for the seqdisplay-opt study to be added on publication.]*
