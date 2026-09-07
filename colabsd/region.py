"""Unsupervised pooling-region discovery, streamed.

Upstream scores a residue by how much the variants disagree there:

    score(position) = 1 - mean all-vs-all pairwise cosine similarity per residue
                          across variants

and keeps the positions above a percentile of that score. It computed this from a
materialised `(N, L, D)` embedding tensor — 16,424 x 1054 x 1280 is ~44 GB in
float16, which no Colab runtime has.

The same number comes out of a single streaming pass. For L2-normalised residue
vectors `x_1..x_N` at one position, the mean over unordered pairs is

    mean_pairwise_cos = (||sum_i x_i||^2 - N) / (N * (N - 1))

so only `sum_i x_i` — one `(L, D)` accumulator, ~11 MB in float64 — ever exists.
The forward pass keeps one `(batch, L, D)` tensor at a time and drops it.

Precision matters here more than anywhere else in the package. Variants differ at a
handful of residues, so away from them the score is O(1e-5) and the arithmetic noise
of the forward pass is not negligible against it. Two rules follow, both established by
running this module at each dtype over the bundled library:

* Load the backbone with `from_pretrained(torch_dtype=...)`; never `model.to(dtype)`.
  ESM2 derives its rotary position index as `arange(L).type_as(inv_freq)`, and
  `.to(bfloat16)` casts that buffer, collapsing 1054 residue positions to 517
  distinct values. `_assert_exact_position_index` refuses to score in that state.
* float16 and float32 select the same positions; bfloat16 does not.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from colabsd.errors import BackboneError, ConfigError, DataError

SCORE_DEFINITION = "1 - mean all-vs-all pairwise cosine similarity per residue across variants"
EMBEDDING_SOURCE = "HuggingFace transformers EsmModel.last_hidden_state"
DEFAULT_REGION_MODEL = "facebook/esm2_t33_650M_UR50D"
STANDARD_RESIDUES = "ACDEFGHIKLMNPQRSTVWYBUZOX"
EPS = 1e-8

MODEL_LABELS = {
    "facebook/esm2_t6_8M_UR50D": "ESM2-8M",
    "facebook/esm2_t12_35M_UR50D": "ESM2-35M",
    "facebook/esm2_t30_150M_UR50D": "ESM2-150M",
    "facebook/esm2_t33_650M_UR50D": "ESM2-650M",
    "facebook/esm2_t36_3B_UR50D": "ESM2-3B",
}


class CosineVariabilityAccumulator:
    """Streaming accumulator for per-residue cosine variability.

    Feed batches of residue representations shaped `(batch, length, dim)`; only a
    `(length, dim)` float64 sum of L2-normalised vectors is retained.
    """

    def __init__(self, length: int, dim: int) -> None:
        self.length = int(length)
        self.dim = int(dim)
        self.count = 0
        self._sum_normed = np.zeros((self.length, self.dim), dtype=np.float64)

    def update(self, reps: Any) -> None:
        """Accumulate one `(batch, length, dim)` batch of residue representations."""
        batch = np.asarray(reps.detach().cpu().float().numpy() if hasattr(reps, "detach") else reps)
        if batch.ndim != 3:
            raise DataError(f"Expected residue representations shaped (batch, length, dim), got {batch.shape}.")
        if batch.shape[1:] != (self.length, self.dim):
            raise DataError(
                f"Batch shape {batch.shape} does not match the accumulator "
                f"(length {self.length}, dim {self.dim})."
            )
        batch = batch.astype(np.float32, copy=False)
        norms = np.linalg.norm(batch, axis=-1, keepdims=True)
        self._sum_normed += (batch / np.maximum(norms, EPS)).sum(axis=0, dtype=np.float64)
        self.count += batch.shape[0]

    def scores(self) -> np.ndarray:
        """Return the `(length,)` cosine-variability score."""
        if self.count < 2:
            raise DataError(
                f"Cosine variability needs at least 2 variants, got {self.count}. "
                "Increase n_sample or supply more rows in the library."
            )
        sum_sq = np.sum(self._sum_normed**2, axis=-1)
        mean_pairwise_cos = (sum_sq - self.count) / (self.count * (self.count - 1))
        return np.clip(1.0 - mean_pairwise_cos, 0.0, None)


def naive_cosine_variability(embeddings: np.ndarray) -> np.ndarray:
    """Reference implementation over a materialised `(N, L, D)` array. Tests only."""
    array = np.asarray(embeddings, dtype=np.float64)
    n = array.shape[0]
    if n < 2:
        raise DataError("Cosine variability needs at least 2 variants.")
    normed = array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), EPS)
    scores = np.empty(array.shape[1], dtype=np.float64)
    for position in range(array.shape[1]):
        gram = normed[:, position, :] @ normed[:, position, :].T
        pairs = gram[np.triu_indices(n, k=1)]
        scores[position] = 1.0 - float(pairs.mean())
    return np.clip(scores, 0.0, None)


def subsample(n_rows: int, n_sample: int, *, seed: int = 0) -> np.ndarray:
    """Return sorted row indices, all of them when `n_sample` covers the library."""
    if n_sample is None or n_sample >= n_rows or n_sample <= 0:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n_sample, replace=False))


def mutated_positions_1based(sequences: Sequence[str], wt_sequence: str) -> list[int]:
    """Positions where any variant differs from the WT, 1-based."""
    length = len(wt_sequence)
    ragged = [index for index, sequence in enumerate(sequences) if len(sequence) != length]
    if ragged:
        raise DataError(
            f"Variant sequences must be the same length as the WT ({length}); rows "
            f"{ragged[:5]} are not. Rebuild the sequences with colabsd.data.load_library."
        )
    joined = "".join(sequences).encode("ascii", errors="replace")
    matrix = np.frombuffer(joined, dtype="S1").reshape(len(sequences), length)
    wt = np.frombuffer(wt_sequence.encode("ascii", errors="replace"), dtype="S1")
    return [int(index) + 1 for index in np.flatnonzero((matrix != wt).any(axis=0))]


def _check_sequences(sequences: Sequence[str], wt_sequence: str) -> int:
    from colabsd.engine.sequences import require_uniform_sequence_length

    try:
        length = require_uniform_sequence_length(list(sequences))
    except ValueError as exc:
        raise DataError(str(exc)) from exc
    if length != len(wt_sequence):
        raise DataError(
            f"Variant sequences are {length} residues but the WT is {len(wt_sequence)}. "
            "Region discovery pools on WT coordinates, so both must be the same length — check "
            "that the WT FASTA is the one the library was built from."
        )
    return length


def _resolve_device(device: str):
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[colabsd] No GPU visible; running region discovery on CPU. This is slow for a large model.")
        return torch.device("cpu")
    return torch.device(device)


def _unexpected_residues(sequence: str) -> list[str]:
    return sorted(set(sequence) - set(STANDARD_RESIDUES))


def _assert_exact_position_index(model, length: int) -> None:
    """Refuse to score when the model dtype cannot represent every residue position.

    ESM2 builds its rotary index as `arange(L).type_as(inv_freq)`. In bfloat16 that
    index is exact only to 256, so residues beyond it share positional encodings.
    """
    import torch

    index = torch.arange(length, dtype=torch.float64)
    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if inv_freq is None:
            continue
        rounded = index.to(inv_freq.dtype).to(torch.float64)
        if torch.equal(rounded, index):
            continue
        distinct = int(torch.unique(rounded).numel())
        raise BackboneError(
            f"The backbone holds its rotary position index in {inv_freq.dtype}, which represents only "
            f"{distinct} of this protein's {length} residue positions distinctly. Every score would be "
            "computed on a corrupted positional encoding. Load the model with "
            "from_pretrained(torch_dtype=...) rather than model.to(dtype), and use "
            'dtype="float16" or "float32" for region discovery.'
        )


def _residue_representations(model, tokenizer, batch: list[str], device, expected_length: int, offset: int = 0):
    import torch

    encoded = tokenizer(batch, return_tensors="pt", padding=True, return_special_tokens_mask=True)
    special = encoded.pop("special_tokens_mask")
    if not bool((special == special[0]).all()):
        row = int((special != special[0]).any(dim=1).nonzero()[0][0])
        odd = _unexpected_residues(batch[row]) or _unexpected_residues(batch[0])
        raise DataError(
            f"Sequence {offset + row} tokenizes to a different number of tokens than the others, so the "
            f"residue axis cannot be shared across the batch. Characters the model does not know: {odd}. "
            "Variant sequences must be upper-case one-letter amino acids with no gaps or separators."
        )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
    reps = output.last_hidden_state
    keep = (special[0] == 0).nonzero(as_tuple=True)[0].to(reps.device)
    reps = reps.index_select(1, keep)
    if reps.shape[1] != expected_length:
        odd = _unexpected_residues(batch[0])
        raise DataError(
            f"After stripping special tokens the residue axis is {reps.shape[1]}, expected "
            f"{expected_length}, so positions would not line up with 1-based protein coordinates. "
            f"Characters the model does not tokenize one-per-residue: {odd}. Variant sequences must be "
            "upper-case one-letter amino acids."
        )
    return reps


def cosine_variability_scores(
    sequences: Sequence[str],
    wt_sequence: str,
    *,
    hf_id: str = DEFAULT_REGION_MODEL,
    device: str = "cuda",
    batch_size: int = 2,
    dtype: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    """Per-residue cosine variability over `sequences`, one streaming forward pass.

    `hf_id` must name an encoder-only backbone whose `last_hidden_state` is one
    vector per token (the ESM2 family; upstream scored regions with ESM2-650M).
    """
    length = _check_sequences(sequences, wt_sequence)

    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = _resolve_device(device)
    if dtype is None:
        torch_dtype = torch.float16 if resolved_device.type == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, str(dtype), None)
        if not isinstance(torch_dtype, torch.dtype):
            raise ConfigError(f"dtype={dtype!r} is not a torch dtype. Use \"float16\", \"bfloat16\" or \"float32\".")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        try:
            model = AutoModel.from_pretrained(hf_id, torch_dtype=torch_dtype, add_pooling_layer=False)
        except TypeError:
            model = AutoModel.from_pretrained(hf_id, torch_dtype=torch_dtype)
    except Exception as exc:  # noqa: BLE001 - a notebook must not see a HuggingFace stack trace
        raise BackboneError(
            f"Could not load {hf_id} for region discovery ({type(exc).__name__}: {exc}). The model is "
            "downloaded once from HuggingFace, so check the runtime's internet access, or pass a smaller "
            "hf_id= you already have (facebook/esm2_t12_35M_UR50D)."
        ) from exc
    model = model.eval().to(resolved_device)
    _assert_exact_position_index(model, length)

    accumulator: CosineVariabilityAccumulator | None = None
    total = len(sequences)
    label = MODEL_LABELS.get(hf_id, hf_id)
    try:
        for start in range(0, total, batch_size):
            batch = list(sequences[start : start + batch_size])
            reps = _residue_representations(model, tokenizer, batch, resolved_device, length, offset=start)
            if accumulator is None:
                accumulator = CosineVariabilityAccumulator(length, reps.shape[-1])
            accumulator.update(reps)
            del reps
            if progress is not None:
                progress(min(start + batch_size, total), total, f"{label} forward")
    finally:
        # Free the backbone before returning: the notebook loads a second one to train with.
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    if accumulator is None:
        raise DataError("No sequences were given to region discovery.")
    return accumulator.scores()


def build_region_record(
    scores: np.ndarray,
    percentile: float,
    *,
    seq_length: int,
    mutation_positions_1based: Iterable[int] = (),
    region_source_model: str = "ESM2-650M",
    embedding_source: str = EMBEDDING_SOURCE,
    n_variants_scored: int = 0,
) -> dict:
    """Select the positions at or above `percentile` and shape the region record.

    The record carries its own provenance — which embeddings, which score, which
    threshold — because a percentile cut is only meaningful next to them.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise DataError(f"Scores must be one score per residue, got shape {values.shape}.")
    if not 0 <= float(percentile) < 100:
        raise ConfigError(f"percentile={percentile!r} must be in [0, 100); 90 keeps the top 10% of residues.")
    threshold = float(np.percentile(values, percentile))
    selected = [int(index) + 1 for index in np.flatnonzero(values >= threshold)]
    name = f"cosine_p{int(percentile)}"
    return {
        "region_name": name,
        "pooling": f"{name}_mean",
        "region_source_model": region_source_model,
        "embedding_source": embedding_source,
        "score_definition": SCORE_DEFINITION,
        "percentile": float(percentile),
        "threshold": threshold,
        "n_variants_scored": int(n_variants_scored),
        "seq_length": int(seq_length),
        "mutation_positions_1based": [int(p) for p in mutation_positions_1based],
        "selected_positions_1based": selected,
        "n_selected_positions": len(selected),
    }


def discover_region(
    sequences: Sequence[str],
    wt_sequence: str,
    *,
    percentiles: Sequence[float] = (90, 95),
    n_sample: int = 2000,
    hf_id: str = DEFAULT_REGION_MODEL,
    device: str = "cuda",
    batch_size: int = 2,
    seed: int = 0,
    dtype: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Discover pooling regions from residue-embedding disagreement across variants.

    Subsamples `n_sample` variants, runs one frozen forward pass, and returns
    `{region_name: record}` — one record per percentile, each in the JSON shape of
    `examples/slugcas9_5nnk/region_p90.json`.
    """
    if len(sequences) == 0:
        raise DataError("Region discovery needs at least 2 variant sequences, got none.")
    if isinstance(percentiles, (int, float)):
        percentiles = (float(percentiles),)
    length = _check_sequences(sequences, wt_sequence)
    indices = subsample(len(sequences), n_sample, seed=seed)
    sampled = [sequences[int(i)] for i in indices]
    if len(sampled) < 2:
        raise DataError(f"Region discovery needs at least 2 variant sequences, got {len(sampled)}.")

    scores = cosine_variability_scores(
        sampled,
        wt_sequence,
        hf_id=hf_id,
        device=device,
        batch_size=batch_size,
        dtype=dtype,
        progress=progress,
    )
    mutated = mutated_positions_1based(sequences, wt_sequence)
    label = MODEL_LABELS.get(hf_id, hf_id)
    precision = dtype or "float16 (cuda) / float32 (cpu)"
    records = {}
    for percentile in percentiles:
        record = build_region_record(
            scores,
            percentile,
            seq_length=length,
            mutation_positions_1based=mutated,
            region_source_model=label,
            embedding_source=f"{EMBEDDING_SOURCE}, {hf_id}, {precision}",
            n_variants_scored=len(sampled),
        )
        records[record["region_name"]] = record
    return records


def save_region(record: dict, path: Path | str) -> None:
    """Write one region record as JSON."""
    if not isinstance(record, dict) or not (record.get("selected_positions_1based") or record.get("positions_1based")):
        raise ConfigError(
            "save_region writes a single region record. discover_region returns one record per "
            "percentile, so save them separately: save_region(records[\"cosine_p90\"], path)."
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n")


def load_region(path: Path | str) -> dict:
    """Read a region record, exposing its positions as `positions_1based`."""
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Region file not found: {source}. Run discover_region first, or fix the path.")
    try:
        record = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{source} is not valid JSON ({exc}).") from exc
    if not isinstance(record, dict):
        raise ConfigError(f"{source} must hold a region record object, got {type(record).__name__}.")
    positions = record.get("selected_positions_1based", record.get("positions_1based"))
    if not positions:
        raise ConfigError(
            f"{source} has no 'selected_positions_1based'. A region file is the JSON written by "
            "save_region; see examples/slugcas9_5nnk/region_p90.json."
        )
    return {**record, "positions_1based": [int(p) for p in positions]}
