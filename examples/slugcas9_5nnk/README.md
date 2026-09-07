# The bundled example — SlugCas9 5NNK

Every default in the three notebook forms describes the files in this directory, so the
workflow runs end to end without uploading anything. It is also the reference shape for
your own data: if your library can be written the way `library.csv` is written, the
notebooks will read it.

| file | what it is |
|---|---|
| `library.csv` | the variant table: 16,424 variants, 5 randomised residues each, a read count, and 4 measured activities |
| `wt.fasta` | the 1054-residue SlugCas9 wild-type amino-acid sequence |
| `wt_3di.txt` | the Foldseek 3Di string for that wild type, 1054 letters — needed only by the SaProt backbones |
| `region_p90.json` | the pooling region used by `cosine_p90_mean`: 106 residue positions |
| `region_p95.json` | the same region at the tighter 95th-percentile cut: 53 residue positions. Reference only — no pooling the notebooks offer reads it |

## Provenance

`library.csv`, `wt.fasta` and `wt_3di.txt` are the SlugCas9 5NNK dataset of the
`seqdisplay-opt` research package, copied unchanged — each is byte-identical to its
counterpart there. `region_p90.json` and `region_p95.json` are computed by
`colabsd.region` from `library.csv` itself; see [The two region files](#the-two-region-files).

The assay behind the table is a sequence-display screen of a SlugCas9 variant library.
Five residues of the 1054-residue wild type — positions **984, 985, 990, 1012 and 1016**,
1-based — were randomised with NNK codons, and each surviving variant was measured
against four PAM sequences: **NNGA, NNGT, NNGC, NNGG**. One row is one variant; one
activity column is one PAM condition.

## `library.csv` — the exact schema

Ten columns, comma-separated, one header line, 16,424 data rows.

| column | type | meaning |
|---|---|---|
| `nnk1` … `nnk5` | text | the residue at each randomised position, in the same order as the positions above. Three-letter codes, initial capital: `Ala`, `Arg`, `Asn`, … Only the 20 canonical amino acids occur; stop-codon variants are not in the table. |
| `count` | integer | reads supporting this variant. Optional — the notebook can filter on it (`min_count`) or ignore it. |
| `NNGA`, `NNGT`, `NNGC`, `NNGG` | float | measured activity, one column per PAM condition. |

The header and the first data row, verbatim:

```csv
nnk1,nnk2,nnk3,nnk4,nnk5,count,NNGA,NNGT,NNGC,NNGG
Asn,Asn,Met,Glu,Lys,265,0.7849056720733643,0.09811320900917053,0.4415094256401062,0.9283018708229065
```

What the numbers look like across the whole table:

| column | min | median | max |
|---|---|---|---|
| `count` | 100 | 285 | 8,033,951 |
| `NNGA` | 0.176471 | 0.539334 | 2.059322 |
| `NNGT` | 0.000000 | 0.194030 | 2.408010 |
| `NNGC` | 0.061069 | 0.395895 | 2.559322 |
| `NNGG` | 0.224299 | 0.609918 | 2.510980 |

No variant appears twice, and all five sites take all 20 canonical residues. The activity
values are not bounded at 1 and are not on a common scale with each other; nothing in the
workflow assumes they are: targets are z-scored on the training split before training,
and models are selected on Spearman correlation, which depends only on rank.

## How the notebook reads it

Variant sequences are **built**, never read from a FASTA: each row's residues are
substituted into `wt.fasta`, so every sequence is exactly as long as the wild type and the
pooling coordinates stay meaningful. The 16,424 sequences built this way are identical to
the prebuilt variant FASTA distributed with `seqdisplay-opt`.

```python
from colabsd.data import load_library, load_wt_sequence
from colabsd.spec import LibrarySpec

spec = LibrarySpec(
    wt_sequence=load_wt_sequence("examples/slugcas9_5nnk/wt.fasta"),
    positions_1based=[984, 985, 990, 1012, 1016],
    mutation_columns=["nnk1", "nnk2", "nnk3", "nnk4", "nnk5"],
    condition_columns=["NNGA", "NNGT", "NNGC", "NNGG"],
    three_letter=True,
    count_column="count",
)
df, sequences, targets = load_library("examples/slugcas9_5nnk/library.csv", spec)
# 16424 rows, 16424 sequences of 1054 residues, targets of shape (16424, 4)
```

`positions_1based` must be strictly increasing, and `mutation_columns[i]` must name the
column holding the residue at `positions_1based[i]`.

## `wt.fasta`

One record, header `>slugcas9_wt`, the 1054-residue sequence on a single line. Its
wild-type residues at the five randomised positions are `N`, `S`, `M`, `E`, `K` (984, 985,
990, 1012, 1016) — worth checking against your own numbering before you trust a position
list.

## `wt_3di.txt`

The Foldseek 3Di structural alphabet string for the wild type: 1054 lowercase letters,
wrapped over 11 lines and read with the line breaks stripped. It must be exactly as long
as the amino-acid sequence and describe the same chain.

Of the backbones the notebook offers, only the three SaProt models — `SaProt-35M`,
`SaProt-650M`, `SaProt-1.3B` — read it; the rest are sequence-only and need nothing from
this file. To make one for your own
protein, use `colab/ColabSeqDisplay_Prepare.ipynb`, which will predict a structure with ESMFold
if you do not have one and convert it with Foldseek.

## The two region files

A *pooling region* is the set of residue positions whose embeddings get averaged into the
single vector the prediction head sees. `cosine_p90_mean` pools over `region_p90.json`.
The notebooks offer one other pooling, `mutation_site_mean`, which needs no file — it pools
over the mutated positions themselves. `region_p95.json` is the same computation at the
95th percentile, shipped for comparison: no pooling offered in the notebooks reads it, and
a region file is refused if its declared pooling is not the one the run uses.

Both files were produced by `colabsd.region` from this library:

1. Embed the variant sequences with **ESM2-650M** (`facebook/esm2_t33_650M_UR50D`), taking
   `last_hidden_state` in float32.
2. Score every one of the 1054 positions by how much the variants disagree there:
   `score = 1 − mean all-vs-all pairwise cosine similarity per residue across variants`.
3. Keep the positions whose score is above the requested percentile of that score
   distribution.

All 16,424 variants were scored, not a subsample.

| file | percentile | threshold | positions kept | where they are |
|---|---|---|---|---|
| `region_p90.json` | 90 | 0.0003371860449376874 | 106 | 912, 939–1040, 1043–1045 |
| `region_p95.json` | 95 | 0.0016604637069272882 | 53 | 971, 973–974, 976–1025 |

Both contain all five randomised positions. Each file records how it was made, so a region
can always be told apart from one computed some other way:

```json
{
  "region_name": "cosine_p90",
  "pooling": "cosine_p90_mean",
  "region_source_model": "ESM2-650M",
  "embedding_source": "HuggingFace transformers EsmModel.last_hidden_state, facebook/esm2_t33_650M_UR50D, float32",
  "score_definition": "1 - mean all-vs-all pairwise cosine similarity per residue across variants",
  "percentile": 90.0,
  "threshold": 0.0003371860449376874,
  "n_variants_scored": 16424,
  "seq_length": 1054,
  "mutation_positions_1based": [984, 985, 990, 1012, 1016],
  "selected_positions_1based": [912, 939, "…"],
  "n_selected_positions": 106
}
```

`colabsd` reads `selected_positions_1based` and checks it against the wild-type length;
the rest is provenance you can inspect.

**Two things to know before you read a region as biology.** First, a percentile keeps a
fixed number of positions whether or not the scores have a natural break there: 10% of
1054 residues is 106, 5% is 53, and that is exactly what comes out. The p90 list is a
pooling window, not a claim that 106 residues matter mechanistically; p95 is the tighter,
more conservative cut. Second, these scores are small — of order 1e-4 to 1e-3 — so the
numerical precision of the embedding pass is not negligible against them. The files record
the precision they were computed at, and a region computed under different settings can
select a partly different set of positions. One consequence is noted in
[`../../config/best/README.md`](../../config/best/README.md): the tuned `cosine_p90_mean`
hyperparameters were selected against a p90 list that shares 69 of these 106 positions.

## Building the equivalent for your own protein

Nothing here is magic — the notebook forms take paths, and the filenames are yours to
choose. You need:

1. **A variant table.** One row per variant. One column per randomised position (three-
   letter codes as above, or one-letter codes with `three_letter_residues` unticked), in
   position order. One column per measured condition — one is fine, four is fine. A read-
   count column is optional. Any activity scale works.
2. **The full-length wild-type sequence**, as a FASTA file, a URL, or pasted into the form.
   Your positions are 1-based against *this* sequence.
3. **A 3Di string** — only if you intend to use a SaProt backbone.
   `colab/ColabSeqDisplay_Prepare.ipynb` writes one.
4. **A region file** — only if you intend to use `cosine_p90_mean` pooling.
   `colab/ColabSeqDisplay_Prepare.ipynb` writes one, in the JSON shape above, from your own
   library. Skip it and use `mutation_site_mean` instead, which needs no file; read
   [`../../config/best/README.md`](../../config/best/README.md) first, because no
   `mutation_site_mean` pair has been tuned.
