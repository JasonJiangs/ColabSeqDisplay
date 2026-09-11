# MG8 PETases — the bundled example

A 124-variant engineered-PETase library in the shape `colabsd` expects. It is here to show
that the workflow runs end to end — load a table, build sequences, split, fine-tune, report,
export — on something small enough that a complete run finishes inside one free Colab
session. It is **not a benchmark**: an 8:1:1 split leaves 13 test variants, and no number
computed from 13 points settles which backbone is better.

| file | what it is |
|---|---|
| `library.csv` | 124 variants: 19 mutated-position columns, one `activity` column, one `variant` name column |
| `wt.fasta` | the 287-residue MG8 PETase wild type |
| `MG8_PETases_dataset.xlsx` | the source spreadsheet `library.csv` was derived from — provenance, not input (see below) |

There is no bundled 3Di string. SaProt needs a wild-type 3Di, and the preparation step builds
one from a structure **you** supply (`colabsd.structure`); with this example that path always
starts from your own PDB or mmCIF file.

## Schema

One row per variant. Column order is the order below; `colabsd` reads columns by name.

```
variant,p22,p42,p49,p72,p78,p79,p80,p130,p131,p132,p150,p198,p199,p205,p206,p215,p216,p225,p279,activity
Y78F,Y,V,S,I,F,V,S,T,R,T,W,V,V,H,A,G,S,N,T,1.57318039171052
```

* `variant` — the name the source spreadsheet gave the row. `colabsd` never parses it; it is
  carried through the run so a scored table is readable.
* `p22` … `p279` — one column per mutated position, named after its 1-based position in
  `wt.fasta`. Residues are **single-letter** (`three_letter=False`), so a row spells out the
  residue at every one of the 19 sites, not just the mutated ones. The first row, `MG8-WT`,
  therefore carries the wild-type residue in all 19 columns.
* `activity` — the single measured condition, `ln(1 + raw)` (see below).

There is no read-count column. Build the spec with `count_column=None`; `load_library` then
refuses a `min_count` threshold rather than quietly keeping every row.

### The 19 positions

```
22, 42, 49, 72, 78, 79, 80, 130, 131, 132, 150, 198, 199, 205, 206, 215, 216, 225, 279
```

Wild-type residues at those positions, in order: `Y V S I Y V S T R T W V V H A G S N T`.

Four of the 19 sites are saturated — 49, 78, 79 and 150 carry all 20 residues across the
library. Of the remaining 15, thirteen carry two residues, position 130 carries three and
position 132 carries five. By number of substituted sites the library is 1 wild type, 79
single mutants, 34 carrying two to seven substitutions, and 10 carrying eight at once.

### The activity column

The spreadsheet records a raw activity per variant, expressed relative to the wild type,
whose raw value is exactly 1.0. `activity` is `ln(1 + raw)` of that number: the wild type is
`ln 2 = 0.693147`, the four dead variants (raw 0) are exactly 0, and the best variant is
6.2811 (raw 533.4). 47 of the 124 variants score above the wild type.

The raw column is not shipped. It and `activity` are the same measurement under a fixed
monotone transform, so carrying both would put two copies of one condition in front of a
user who then has to pick one. The transform is invertible: `raw = expm1(activity)`.

## Loading it

```python
from colabsd.data import load_library, load_wt_sequence
from colabsd.spec import LibrarySpec

spec = LibrarySpec(
    wt_sequence=load_wt_sequence("wt.fasta"),
    positions_1based=[22, 42, 49, 72, 78, 79, 80, 130, 131, 132, 150, 198, 199, 205, 206, 215, 216, 225, 279],
    mutation_columns=["p22", "p42", "p49", "p72", "p78", "p79", "p80", "p130", "p131", "p132",
                      "p150", "p198", "p199", "p205", "p206", "p215", "p216", "p225", "p279"],
    condition_columns=["activity"],
    three_letter=False,
    count_column=None,
)
df, sequences, targets = load_library("library.csv", spec)
# 124 rows, 124 sequences of 287 residues each, targets of shape (124, 1)
```

Every sequence is built by substituting the row's 19 residues into `wt.fasta`, so all 124 are
287 residues long.

## What a run costs

124 variants split 8:1:1 give **99 train, 12 validation, 13 test**. The whole library is
124 x 287 = 35,588 residue tokens — 486 times fewer than the tuning study's own library
(16,424 variants of 1054 residues; see `config/best/README.md`), which the registry's
per-run estimates are sized against. That is the point: a complete run fits inside one Colab
session instead of spilling over several. Treat the resulting numbers as
evidence that the pipeline ran, not as a measurement of how well a backbone predicts activity.

## Data notes

**One variant is named for a mutation its sequence does not carry.** The row named
`V42I-Y78F` has V at position 42 — the wild-type residue — in the source sequence. Residues in
this table were read from the sequences, never parsed from the names, so the table reflects the
sequences: `p42` is `V` for that row. As a consequence it is residue-for-residue the same
protein as the row named `Y78F`, with a different measured activity (0.9209 against 1.5732).

**One more pair of rows is the same protein twice.**
`V79M-W150H-V199I-V198C-H205Y-G215D-R131Q-T130G` and
`V79M-W150H-V199I-V198C-H205Y-G215D-T130G-R131Q` name the same eight substitutions in a
different order and their sequences are identical (activities 6.0577 and 6.0462).

Both pairs are kept as they are. The rows are what the source file contains, and a split can
put one member of a pair in train and the other in test — a small, real leak worth knowing
about before reading anything into a 13-point test score.

## Provenance

`library.csv` was derived from `MG8_PETases_dataset.xlsx`, which holds, per variant, the full
protein sequence (with a trailing `*` stop), the variant name, the raw activity and its
`ln(1 + raw)` transform. Building the table meant: dropping the stop codon, taking the 19
positions at which any sequence differs from the wild type, reading the residue at each of them
straight out of the sequence, stripping stray whitespace from one name, and keeping the
transformed activity.

The spreadsheet is kept here because it is the only record of where these 124 numbers came
from; nothing in the pipeline reads it — a run needs only `library.csv` and `wt.fasta`. Three
checks were run against it, and `tests/test_data.py` re-runs all three whenever the spreadsheet
is present (they skip if it is not, or if pandas has no Excel reader installed):

1. All 124 rebuilt sequences match the spreadsheet's sequences exactly.
2. `activity` equals `ln(1 + raw)` for all 124 rows.
3. The variant names match the spreadsheet's, after whitespace stripping.
