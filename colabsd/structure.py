"""WT 3Di construction for structure-aware backbones (SaProt).

Three ways in, in order of preference:

1. `load_three_di` — the user already has a 3Di string (plain text or FASTA).
2. `three_di_from_structure` — a `.pdb`/`.cif`/`.mmcif` file plus foldseek.
3. `three_di_from_esmfold` — fold the WT first. Last resort: ESMFold is the
   single most memory-hungry step in the whole notebook and will OOM on a free
   Colab T4 for anything much past ~700 residues.

Every path returns a lowercase 3Di string whose length equals the WT length, and
every failure raises `StructureError` with the fix in the message.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from colabsd.errors import StructureError

FOLDSEEK_BASE_URL = "https://mmseqs.com/foldseek/"
STRUCTURE_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")
# The 20 foldseek 3Di states plus the mask state, mirroring upstream's
# colabsd.engine.formats.FOLDSEEK_STRUC_VOCAB. Anything outside
# this set becomes SaProt's <unk> token, so it must not pass validation.
THREE_DI_STATES = "pynwrqhgdlvtmfsaeikc#"
THREE_DI_ALPHABET = frozenset(THREE_DI_STATES)
ESMFOLD_MODEL = "facebook/esmfold_v1"

_ESMFOLD_SAFE_LENGTH_T4 = 700
_SEQUENCE_MISMATCH_TOLERANCE = 0.10


def default_cache_dir() -> Path:
    """Directory used to cache the foldseek binary."""
    root = os.environ.get("COLABSD_CACHE_DIR")
    return Path(root).expanduser() if root else Path.home() / ".cache" / "colabsd"


def _archive_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return "foldseek-osx-universal.tar.gz"
    if system != "Linux":
        raise StructureError(
            f"No prebuilt foldseek binary for {system}. Install foldseek yourself and pass "
            "foldseek_bin=..., or set the FOLDSEEK_BIN environment variable."
        )
    if machine in {"aarch64", "arm64"}:
        return "foldseek-linux-arm64.tar.gz"
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        flags = "avx2"
    return "foldseek-linux-avx2.tar.gz" if "avx2" in flags else "foldseek-linux-sse2.tar.gz"


def _candidate_paths(dest: Path) -> list[Path]:
    env_bin = os.environ.get("FOLDSEEK_BIN")
    candidates = [Path(env_bin)] if env_bin else []
    candidates += [dest / "foldseek" / "bin" / "foldseek", dest / "bin" / "foldseek", dest / "foldseek"]
    which = shutil.which("foldseek")
    if which:
        candidates.append(Path(which))
    return candidates


def find_foldseek(dest: Path | str | None = None) -> Path | None:
    """Return an existing foldseek binary, or None. Never downloads, never runs it.

    Looks at `$FOLDSEEK_BIN`, the cache directory and `$PATH`, so an environment
    that already ships foldseek never pays for a download.
    """
    root = Path(dest).expanduser() if dest is not None else default_cache_dir()
    for candidate in _candidate_paths(root):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def ensure_foldseek(dest: Path | str | None = None) -> Path:
    """Return a runnable foldseek binary, downloading a static build if needed."""
    root = Path(dest).expanduser() if dest is not None else default_cache_dir()
    found = find_foldseek(root)
    if found is not None:
        _check_runnable(found)
        return found

    root.mkdir(parents=True, exist_ok=True)
    url = FOLDSEEK_BASE_URL + _archive_name()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "foldseek.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as exc:
            raise StructureError(
                f"Could not download foldseek from {url} ({exc}). If this notebook has no internet "
                "access, upload a 3Di text file and use load_three_di, or install foldseek yourself "
                "and set the FOLDSEEK_BIN environment variable."
            ) from exc
        try:
            with tarfile.open(archive, "r:gz") as tar:
                try:
                    tar.extractall(root, filter="data")
                except TypeError:
                    tar.extractall(root)
        except (tarfile.TarError, OSError) as exc:
            raise StructureError(f"The foldseek archive from {url} could not be unpacked ({exc}).") from exc

    binary = root / "foldseek" / "bin" / "foldseek"
    if not binary.is_file():
        raise StructureError(
            f"foldseek was downloaded to {root} but no binary appeared at {binary}. "
            "Unpack it by hand and pass foldseek_bin=<path>."
        )
    binary.chmod(0o755)
    _check_runnable(binary)
    return binary.resolve()


def _check_runnable(binary: Path) -> None:
    try:
        subprocess.run([str(binary), "version"], capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StructureError(
            f"The foldseek binary at {binary} is not runnable here ({exc}). This usually means the "
            "wrong CPU build (avx2 vs sse2) or an old glibc. Install foldseek with conda/apt and set "
            "the FOLDSEEK_BIN environment variable."
        ) from exc


def _parse_descriptor(text: str) -> list[tuple[str, str, str]]:
    records = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[2]:
            records.append((parts[0], parts[1], parts[2]))
    return records


def _chain_id(record_name: str, path: Path | str | None = None) -> str:
    """The chain id out of one `structureto3didescriptor` record name.

    With `--chain-name-mode 1` foldseek names each record `<basename>_<chain>` — and for a
    PDB it appends the HEADER title after a space, so a file straight from RCSB yields
    `1ubq.pdb_A STRUCTURE OF UBIQUITIN REFINED AT 1.8 ANGSTROMS RESOLUTION`. Splitting on
    the last underscore alone therefore returned the whole title as the "chain", and
    `chain="A"` matched nothing on any real PDB.

    Stripping the known basename first is what makes this survive both a description
    containing an underscore and a file name containing a space; the trailing split is the
    fallback for a name that does not start with the basename.
    """
    if path is not None:
        prefix = f"{Path(path).name}_"
        if record_name.startswith(prefix):
            return record_name[len(prefix) :].split(" ", 1)[0]
    return record_name.rsplit("_", 1)[-1].split(" ", 1)[0]


def _select_record(records: list[tuple[str, str, str]], chain: str | None, path: Path) -> tuple[str, str, str]:
    if chain is not None:
        wanted = str(chain)
        matches = [r for r in records if _chain_id(r[0], path) == wanted]
        if not matches:
            matches = [r for r in records if _chain_id(r[0], path).lower() == wanted.lower()]
        if not matches:
            available = ", ".join(f"{_chain_id(r[0], path)} ({len(r[2])} residues)" for r in records)
            raise StructureError(f"{path} has no chain '{wanted}'. Chains found: {available}.")
        return matches[0]
    if len(records) > 1:
        available = ", ".join(f"{_chain_id(r[0], path)} ({len(r[2])} residues)" for r in records)
        raise StructureError(
            f"{path} contains {len(records)} chains: {available}. Pass chain=\"A\" (or whichever chain "
            "is the protein your library mutates) so the 3Di string matches the WT sequence."
        )
    return records[0]


def validate_three_di(three_di: str, expected_length: int | None = None, *, source: str = "3Di input") -> str:
    """Return the validated lowercase 3Di string, or raise an actionable error."""
    cleaned = "".join(three_di.split()).lower()
    if not cleaned:
        raise StructureError(f"{source} produced an empty 3Di string.")
    bad = sorted(set(cleaned) - THREE_DI_ALPHABET)
    if bad:
        raise StructureError(
            f"{source} contains characters that are not foldseek 3Di states: {bad}. The 3Di alphabet is "
            f"'{THREE_DI_STATES}', one state per residue — check that the file is a foldseek 3Di string "
            "and not an amino-acid FASTA."
        )
    if expected_length is not None and len(cleaned) != expected_length:
        raise StructureError(
            f"{source} has {len(cleaned)} residues but the WT sequence has {expected_length}. "
            "The usual cause is chain selection: a structure file holding several chains gives one 3Di "
            "string per chain, so pass chain=\"A\" (or the right chain id). The other common cause is a "
            "structure with missing/extra residues — an ESMFold or AlphaFold model of the exact WT "
            "sequence always matches, or trim the structure to the WT numbering."
        )
    return cleaned


def _check_matches_wt(structure_aa: str, expected_sequence: str, *, source: str) -> None:
    """Compare the structure's own residues with the WT sequence the library uses."""
    observed = structure_aa.upper()
    expected = "".join(expected_sequence.split()).upper()
    if len(observed) != len(expected):
        return  # validate_three_di owns the length message
    mismatched = [i + 1 for i, (a, b) in enumerate(zip(observed, expected, strict=True)) if a != b and a != "X"]
    if not mismatched:
        return
    preview = ", ".join(f"{expected[i - 1]}{i}->{observed[i - 1]}" for i in mismatched[:5])
    if len(mismatched) > max(5, int(_SEQUENCE_MISMATCH_TOLERANCE * len(expected))):
        raise StructureError(
            f"{source} is a different protein from the WT: {len(mismatched)} of {len(expected)} residues "
            f"disagree ({preview}...). The 3Di string would not line up with your library's coordinates. "
            "Pick the chain that holds your protein (chain=\"A\", ...), or fold the WT sequence itself."
        )
    print(
        f"[colabsd] {source} differs from the WT at {len(mismatched)} residue(s) ({preview}). "
        "Using its 3Di string anyway; check this is the intended structure."
    )


def three_di_from_structure(
    path: Path | str,
    *,
    chain: str | None = None,
    foldseek_bin: Path | str | None = None,
    expected_length: int | None = None,
    expected_sequence: str | None = None,
) -> str:
    """Run `foldseek structureto3didescriptor` on a structure and return its 3Di string."""
    structure = Path(path).expanduser()
    if not structure.is_file():
        raise StructureError(f"Structure file not found: {structure}. Upload it, or pass the right path.")
    suffixes = [s.lower() for s in structure.suffixes]
    if not any(s in STRUCTURE_SUFFIXES for s in suffixes):
        raise StructureError(
            f"{structure} is not a structure file. three_di_from_structure reads "
            f"{', '.join(STRUCTURE_SUFFIXES)} (optionally gzipped); for a ready-made 3Di string use "
            "load_three_di instead."
        )
    if expected_sequence is not None and expected_length is None:
        expected_length = len("".join(expected_sequence.split()))

    binary = Path(foldseek_bin) if foldseek_bin is not None else ensure_foldseek()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "descriptor"
        command = [
            str(binary),
            "structureto3didescriptor",
            "-v",
            "0",
            "--threads",
            "1",
            "--chain-name-mode",
            "1",
            str(structure),
            str(out),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        except (OSError, subprocess.SubprocessError) as exc:
            raise StructureError(f"Could not run foldseek at {binary} ({exc}).") from exc
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            raise StructureError(
                f"foldseek failed on {structure} (exit {result.returncode}): {' | '.join(tail)}. "
                "Check that the file is a real structure with backbone atoms."
            )
        if not out.is_file():
            raise StructureError(f"foldseek wrote no output for {structure}.")
        records = _parse_descriptor(out.read_text())

    if not records:
        raise StructureError(
            f"foldseek found no 3Di descriptor in {structure}. Either the file has no protein backbone "
            "atoms (N, CA, C) — nucleic-acid-only or ligand-only files produce nothing — or it is a "
            "minimal mmCIF that foldseek cannot parse. Converting it to .pdb fixes the second case."
        )
    name, amino_acids, three_di = _select_record(records, chain, structure)
    source = f"{structure} (chain {_chain_id(name, structure)})"
    if len(amino_acids) != len(three_di):
        raise StructureError(
            f"foldseek returned {len(amino_acids)} residues but {len(three_di)} 3Di states for {source}. "
            "This foldseek build is not producing the expected descriptor format; upgrade it."
        )
    cleaned = validate_three_di(three_di, expected_length, source=source)
    if expected_sequence is not None:
        _check_matches_wt(amino_acids, expected_sequence, source=source)
    return cleaned


def load_three_di(path: Path | str, *, expected_length: int | None = None) -> str:
    """Load a 3Di string from a plain-text or single-record FASTA file."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise StructureError(f"3Di file not found: {source}. Upload it, or pass the right path.")
    text = source.read_text()
    headers = [line for line in text.splitlines() if line.startswith(">")]
    if len(headers) > 1:
        raise StructureError(
            f"{source} holds {len(headers)} FASTA records ({', '.join(h[:30] for h in headers[:3])}...). "
            "Keep only the WT 3Di record."
        )

    from colabsd.engine.loader import load_foldseek_sequence

    try:
        # Upstream joins a relative path onto its data_root, so hand it an absolute one.
        three_di = load_foldseek_sequence(source.resolve(), source.parent)
    except (OSError, ValueError) as exc:
        raise StructureError(f"Could not read a 3Di string from {source} ({exc}).") from exc
    return validate_three_di(three_di, expected_length, source=str(source))


def _load_esmfold(use_cuda: bool):
    from transformers import EsmForProteinFolding

    try:
        model = EsmForProteinFolding.from_pretrained(ESMFOLD_MODEL)
    except Exception as exc:  # noqa: BLE001 - any load failure must stay actionable
        raise StructureError(
            f"Could not load {ESMFOLD_MODEL} ({type(exc).__name__}: {exc}). It is a 2.6 GB download that "
            "needs internet access and free disk. Instead, download the AlphaFold model of your protein "
            "(https://alphafold.ebi.ac.uk) and pass it to three_di_from_structure."
        ) from exc
    model = model.eval()
    if use_cuda:
        model = model.to("cuda")
        model.esm = model.esm.half()
    model.trunk.set_chunk_size(64)
    return model


def three_di_from_esmfold(
    wt_sequence: str,
    *,
    device: str = "cuda",
    foldseek_bin: Path | str | None = None,
    pdb_out: Path | str | None = None,
) -> str:
    """Fold the WT with ESMFold, then read its 3Di string. Memory-hungry fallback.

    ESMFold holds a 3B-parameter language model plus the folding trunk: expect
    ~14-16 GB of GPU memory, which is the whole of a free Colab T4. Sequences
    past ~700 residues usually OOM there. Prefer an AlphaFold DB model, the ESM
    Atlas, or any experimental structure and `three_di_from_structure`.
    """
    sequence = "".join(wt_sequence.split()).upper()
    if not sequence:
        raise StructureError("three_di_from_esmfold needs a non-empty WT sequence.")
    bad = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWYXBZUO"))
    if bad:
        raise StructureError(f"The WT sequence contains non-amino-acid characters: {bad}.")

    import torch

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        raise StructureError(
            "three_di_from_esmfold was asked for CUDA but no GPU is visible. In Colab: Runtime > "
            "Change runtime type > T4 GPU. On CPU, pass device=\"cpu\" — folding a 1000-residue "
            "protein then takes hours, so downloading an AlphaFold model is usually faster."
        )
    if not use_cuda:
        print("[colabsd] ESMFold on CPU: expect tens of minutes to hours for a long protein.")
    elif len(sequence) > _ESMFOLD_SAFE_LENGTH_T4:
        print(
            f"[colabsd] ESMFold on {len(sequence)} residues needs well over 16 GB of GPU memory; "
            "a free-tier T4 will most likely run out. An AlphaFold DB model is the cheap way out."
        )

    model = _load_esmfold(use_cuda)
    try:
        with torch.no_grad():
            pdb_text = model.infer_pdb(sequence)
    except torch.cuda.OutOfMemoryError as exc:
        raise StructureError(
            f"ESMFold ran out of GPU memory on {len(sequence)} residues. Fetch a model from the "
            "AlphaFold DB (https://alphafold.ebi.ac.uk) or fold it once at "
            "https://esmatlas.com/resources?action=fold, then pass the .pdb to three_di_from_structure."
        ) from exc
    finally:
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    if pdb_out is not None:
        target = Path(pdb_out).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pdb_text)
        return three_di_from_structure(target, foldseek_bin=foldseek_bin, expected_sequence=sequence)
    with tempfile.TemporaryDirectory(prefix="colabsd_esmfold_") as tmp:
        target = Path(tmp) / "wt.pdb"
        target.write_text(pdb_text)
        return three_di_from_structure(target, foldseek_bin=foldseek_bin, expected_sequence=sequence)


def get_wt_3di(
    wt_sequence: str,
    *,
    three_di_path: Path | str | None = None,
    structure_path: Path | str | None = None,
    chain: str | None = None,
    foldseek_bin: Path | str | None = None,
    device: str = "cuda",
) -> str:
    """Resolve the WT 3Di string from whatever the user has, cheapest source first."""
    wt = "".join(wt_sequence.split())
    if not wt:
        raise StructureError("get_wt_3di needs the WT amino-acid sequence to check the 3Di length against.")
    if three_di_path is not None:
        return load_three_di(three_di_path, expected_length=len(wt))
    if structure_path is not None:
        return three_di_from_structure(
            structure_path,
            chain=chain,
            foldseek_bin=foldseek_bin,
            expected_length=len(wt),
            expected_sequence=wt,
        )
    return three_di_from_esmfold(wt, device=device, foldseek_bin=foldseek_bin)
