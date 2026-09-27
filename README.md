# ColabSeqDisplay

Train a protein language model on your own variant library, in a browser, without writing code.

You bring one CSV — a row per variant, a column per mutated site, a column per measured condition —
and the wild-type sequence. You press buttons. You leave with two `.zip` files: the trained model,
and a report of how well it did.

**It ranks variants you have not made yet.** It does not tell you how well it will work on your
protein. Nothing on the form has been trained end to end in this repository, and nothing here has
been tested on any protein but the one its settings were tuned on. [Status](#status) is the full
account.

**Version 0.1.0 · MIT license · [tutorial](TUTORIAL.md) · [status](#status)**

| | Open it | What it does |
|---|---|---|
| **ColabSeqDisplay** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay.ipynb) | The whole workflow: load a library, pick a backbone, train, read the report, export. Start here. |
| **Predict** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JasonJiangs/ColabSeqDisplay/blob/main/colab/ColabSeqDisplay_Predict.ipynb) | Upload a `model_bundle.zip` and a table of variants, get a ranked table back. |

**Try it first.** A 124-variant MG8 PETase library is bundled. Open **ColabSeqDisplay**, set
`Runtime ▸ Change runtime type ▸ T4 GPU`, run the setup cell, change nothing, and press **Check my
library**, **Train**, **Write performance_report.zip**. About a minute.

Nothing else to install: the training code ships inside the package as `colabsd/engine/`, credited
module by module in [`ATTRIBUTION.md`](ATTRIBUTION.md).

---

## Backbones

Nine on the form, cheapest first. Seven need nothing extra; two need one `pip install`, named in
the row and again by the panel when you pick them.

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

**Which one?** On a free T4, `ESM2-35M` if you have only the sequence, `SaProt-35M` if you also
have a structure. `ESM2-35M` is the cheapest run on the form — put a new library through it once
before paying for a long run. The nine score within 0.008 of each other on the library they were
tuned on ([`config/best/README.md`](config/best/README.md) has those numbers), so pick on cost.

**A SaProt model needs a 3Di string for your wild type.** The panel makes one from an AlphaFold
`.cif`, your own `.pdb`, or a 3Di file or string you already have. It does not predict structures,
and none is bundled. Every other model reads the sequence alone and the step never appears.

**The minutes are estimates**, not measurements — scaled from the study's 16,424-variant library
and never timed on Colab. The panel rescales them for your table and reports the real elapsed time
at the end. The two rows with no number cannot train on a free T4 at all, and the panel blocks
**Train** until the runtime is an L4 or A100.

**`ESMDance`'s width is not comparable to the others.** Its feature is its own 50-dimensional
prediction of per-residue dynamics; every other row uses a trunk embedding. Read its score as a
different kind of read-out, not as a much smaller model doing just as well.

**`float16` cannot train anything.** The optimizer's epsilon rounds to zero in it, so every weight
becomes `NaN` — and the run *finishes* and reports a score of `0.0000`. The panel blocks **Train**
if you pick it. Use `bfloat16` instead: the same memory, and it is what `SaProt-1.3B` needs.

Every model is *load checked* — checkpoint loaded, one forward pass, the pooled vector confirmed to
be the width above, LoRA injection verified against the real module tree. **None has been trained
end to end here.** The package registers fourteen models and offers nine; a model reaches the form
only when tuned settings exist for it and its adapter builds, and the panel says why each of the
other five is missing.

## What you take away

**`model_bundle.zip` — the model.** It carries the LoRA weights, the head, and a description of
your library, but no backbone weights: the backbone is downloaded again by name on the first
scoring run, so Predict needs internet. It holds the single best run, not an ensemble, and it is
written from the training checkpoint — so export it before you clear the working directory.

**`performance_report.zip` — the numbers.** Seven files: a CSV and a JSON of the metrics, the
figure, the training curve and its data, plus a `README.txt` and a `performance.json` that say in
plain words which partition the numbers describe, how many times the test set has been read, and
whether the settings were tuned or a placeholder. Every file carries a checksum, verified when it
is read back.

**Write the report before you unlock anything.** It publishes validation numbers with the test
partition untouched, which costs nothing. Press the same button after an unlock and the file is
rewritten with the test numbers and the count.

**Quote `Spearman` and `R2`.** Four of the seven metrics are top-*k* ranking cuts, and on a
partition shorter than *k* they come out high rather than good — on the example's 12 validation
rows, `P@50` is exactly 1.0000 for any model at all. The report prints that warning with the row
count.

**The test set is locked, and every unlock is counted.** The panel reports validation only. The
last cell of the notebook is the only thing that reads the test partition; it increments a counter
and stamps that count, with a sentence saying what the number is worth, into everything written
afterwards. The [tutorial](TUTORIAL.md) explains what each count means.

**If a run is stopped or the session dies**, the best epoch is already on disk and nothing deletes
it. The [tutorial](TUTORIAL.md) shows how to turn it back into a loadable bundle.

## Status

| item | status |
|---|---|
| The workflow — loading a library, building sequences, splits, test locking and the unlock counter, the report, both exports and their read-back, reloading a bundle, scoring variants | runs end to end on the bundled 124-variant example |
| Model loading, pooled width, LoRA injection | load checked against the published checkpoints for all nine offered models and three withheld ones. **None trained end to end** |
| Training end to end | **`ESM2-8M` only**, which is not on the form — so no path the panel offers has been exercised here. That run used the seqdisplay-opt study's library with placeholder settings and one seed pair, and reached test Spearman 0.5392 |
| How well any model predicts activity | **not measured here.** Every benchmark number on this page is the seqdisplay-opt study's, on its own protein. The bundled example shows the workflow runs; its 13 test variants measure nothing |
| Colab timings | **not measured** — every minute quoted here is an estimate |
| The settings in `config/best/` | chosen on the study's protein, with the same read-out a run here uses. **Whether they transfer to your protein has not been measured** |
| `colabsd/engine/` | a copy of the seqdisplay-opt code, not a shared library; the search machinery that *produces* tuned settings stayed with that study. [`ATTRIBUTION.md`](ATTRIBUTION.md) lists what was copied and what was not |
| Determinism | a fixed seed fixes the split and the initialization; it does not make GPU training bit-reproducible across cards, drivers or library versions |
| Predictions | they rank variants, they do not measure them. The first rows of your library are mostly rows the model trained on, so scoring those and seeing agreement is a wiring check, not evidence |

---

## Data availability

The example library is under `examples/mg8_petases/`: 124 MG8 PETase variants whose activity is
`ln(1 + raw)` of the value assayed relative to the wild type, plus the 287-residue wild-type
sequence. `examples/mg8_petases/README.md` documents each file and where it came from. No 3Di
string is bundled. Everything under `examples/` is covered by this repository's MIT license
([`LICENSE`](LICENSE)).

The settings in `config/best/` and every ρ on this page come from the seqdisplay-opt study's
re-evaluation tables, measured on a 16,424-variant SlugCas9 5NNK library. **That library is not
redistributed here**, and anyone reusing those values should cite the study: *[reference to be added
on publication]*. The MG8 PETase measurements are from *[reference or accession to be added on
publication]*.

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
tuned settings in `config/best/` come from the **seqdisplay-opt** work. This repository is its
no-code Colab front end and develops no method of its own; every benchmark number on this page is
that study's measurement, not ours. [`ATTRIBUTION.md`](ATTRIBUTION.md) sets out, module by module,
what was taken, what was changed and what was not.

If you use ColabSeqDisplay, please cite both this software ([`CITATION.cff`](CITATION.cff)) and the
seqdisplay-opt study:

> *[Citation for the seqdisplay-opt study to be added on publication.]*
