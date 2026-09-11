"""The one preparation step of the main panel: the wild-type shape, as a 3Di string.

There used to be a second notebook here that asked which backbone, wrote `wt_3di.txt` to the
downloads folder, and handed it back to a main notebook asking the same question again. The
backbone is asked once now, in `colabsd.ui.main_workflow`, and this module is the section that
panel composes underneath it — on screen only when that backbone reads structure as well as
sequence (`colabsd.ui.core.needs_structure`).

Nothing ships a 3Di string, so every route starts from something the user provides — a
structure file, a 3Di file, a pasted string — or from ESMFold, offered last and at a stated
risk. The exception is a string this session already made: it stays in the session and is
offered back rather than recomputed (`session_artefact`).

Every decision is a pure function of a `colabsd.ui.core.WizardState`; `ThreeDiSection` is the
thin ipywidgets layer above them and owns no rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colabsd.ui import core, theme
from colabsd.ui.core import Message, esmfold_safe_length

#: The file this step writes, and the name it is written under.
THREE_DI_FILENAME = "wt_3di.txt"


# ----------------------------------------------------------------------------------------
# What is already here. Recomputing a string this session has made is an ESMFold run spent on
# a file that already exists.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Artefact:
    """The wild-type 3Di string this session made, and the protein it describes.

    `describes` is what decides whether it may be reused at all: one protein's structure
    states read against another protein's sequence are wrong at every position.
    """

    path: Path
    describes: str
    length: int = 0
    detail: str = ""

    @property
    def label(self) -> str:
        """The radio line offering it back."""
        tail = f" — {self.detail}" if self.detail else ""
        return f"Reuse the one made earlier in this session ({self.describes}{tail}) — nothing to compute"


def session_artefact(state: core.WizardState, artefact: Artefact | None) -> Artefact | None:
    """The session's 3Di string, when it still describes the wild type on screen.

    Length is what decides it, for the reason `Artefact` gives.
    """
    if artefact is None:
        return None
    if artefact.length and state.wt_length and artefact.length != int(state.wt_length):
        return None
    return artefact


def written_artefact(path: Path, three_di: str) -> Artefact:
    """Register the 3Di string this step just wrote, so re-reading the library is free.

    `on_check_library` drops the attached 3Di, because a re-read library may be a different
    protein. When it is the same one, this makes the answer a click rather than an ESMFold run.
    """
    return Artefact(
        path=Path(path),
        describes=f"your {len(three_di):,d}-residue wild type",
        length=len(three_di),
        detail="made by this step",
    )


# ----------------------------------------------------------------------------------------
# The choices the step offers, built from the state: a control that is irrelevant right now is
# not on screen at all.
# ----------------------------------------------------------------------------------------


def three_di_choices(state: core.WizardState, artefact: Artefact | None = None) -> list[tuple[str, str]]:
    """Where the wild-type 3Di string can come from, for this library.

    The reuse line is offered only when there is something to reuse; ESMFold comes last and
    says what it costs.
    """
    choices: list[tuple[str, str]] = [("Not chosen yet", "none")]
    if artefact is not None:
        choices.append((artefact.label, "session"))
    choices += [
        ("Upload a structure of my wild type (.pdb / .cif) and read the shape off it", "upload_structure"),
        ("Upload a 3Di text file I already have", "upload_3di"),
        ("Paste a 3Di string I already have", "paste"),
        ("Fold the wild type here with ESMFold (slow, memory-hungry, last resort)", "esmfold"),
    ]
    return choices


# ----------------------------------------------------------------------------------------
# The contextual messages. `colabsd.ui.core` owns everything true of a 3Di string whatever
# produced it; these are the ones this step itself earns.
# ----------------------------------------------------------------------------------------


def notices(
    state: core.WizardState,
    *,
    artefact: Artefact | None = None,
    has_gpu: bool | None = None,
) -> list[Message]:
    """Every message this step earns, in reading order.

    `has_gpu=None` means nobody has looked at the machine yet, which is not the same as
    "no GPU": the hardware refusal stays silent rather than firing backwards.
    """
    if not core.needs_structure(state):
        return []
    out: list[Message] = []
    if state.three_di_source == "session" and artefact is None:
        out.append(
            Message(
                "three_di_reuse_gone",
                "stop",
                "The 3Di string you were reusing describes another protein — the library changed under it. "
                "Upload a structure of your own wild type instead, or a 3Di file you already have.",
            )
        )
    computing = state.three_di_source in ("upload_structure", "esmfold")
    if artefact is not None and computing and not state.wt_3di_length:
        out.append(
            Message(
                "three_di_reuse_available",
                "info",
                f"A 3Di string for {artefact.describes} is already here ({artefact.path.name}): pick the "
                "reuse line above and this step is done. Compute one only if you want a different structure "
                "of the same wild type.",
            )
        )
    if state.three_di_source == "esmfold":
        if has_gpu is False:
            out.append(
                Message(
                    "esmfold_needs_gpu",
                    "stop",
                    "ESMFold needs a GPU and this runtime has none. **Runtime → Change runtime type → T4 "
                    "GPU**, or download a structure of your wild type from "
                    "[alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk) and upload that instead.",
                )
            )
        if not state.get("esmfold_risk_accepted"):
            out.append(
                Message(
                    "esmfold_risk_not_accepted",
                    "stop",
                    "On a free T4, ESMFold is the step most likely to end your session. Tick **I accept the "
                    "ESMFold memory risk** to run it anyway, or upload a structure instead.",
                )
            )
    # `core` already emits `esmfold_too_long` and `esmfold_expensive`. Repeating either here
    # would only bury the one that matters.
    return out


# ----------------------------------------------------------------------------------------
# What the step writes, and what stops its button.
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputFile:
    """A file this step writes, what needs it, and where it goes."""

    name: str
    needed_by: str
    where_it_goes: str


def output_files(state: core.WizardState, *, work_dir: str | Path = core.DEFAULT_WORK_DIR) -> list[OutputFile]:
    """What this step will write for this configuration, and what the file is for.

    The panel that makes the file is the panel that trains, so the "where it goes" column says
    where it is kept, not what to upload where. It is still written: a Drive mount survives a
    disconnect.
    """
    if not core.needs_structure(state):
        return []
    return [
        OutputFile(
            name=THREE_DI_FILENAME,
            needed_by=f"{state.backbone}, and nothing else on this page",
            where_it_goes=(
                f"kept in this session and written to `{Path(work_dir)}`, so a Drive mount survives a disconnect"
            ),
        )
    ]


def three_di_blockers(state: core.WizardState, artefact: Artefact | None = None) -> list[str]:
    """What stops the get-the-3Di button, each said as the thing to go and fix."""
    source = state.three_di_source
    problems: list[str] = []
    if source in {"none", ""}:
        problems.append("Choose where the 3Di string should come from first.")
    if source == "session" and artefact is None:
        problems.append(
            "There is no 3Di string here to reuse for this protein. Upload a structure of your own wild type, "
            "or a 3Di file you already have."
        )
    if source == "paste" and not str(state.get("three_di_text") or "").strip():
        problems.append("The 3Di box is empty. Paste the string, one lowercase letter per residue.")
    if source == "upload_3di" and not str(state.get("three_di_file") or "").strip():
        problems.append("No 3Di file yet. Upload one with the button above.")
    if source == "upload_structure" and not str(state.get("structure_file") or "").strip():
        problems.append("No structure yet. Upload a .pdb or .cif of your wild type.")
    if source == "esmfold":
        if not state.get("esmfold_risk_accepted"):
            problems.append("Tick the box accepting the ESMFold memory risk, or upload a structure instead.")
        if state.wt_length > esmfold_safe_length():
            problems.append(
                f"Your wild type is {state.wt_length:,d} residues and ESMFold runs out of memory on a free T4 "
                f"past about {esmfold_safe_length():,d}. Download a structure from the PDB or AlphaFold and "
                "upload it instead."
            )
    return problems


# ----------------------------------------------------------------------------------------
# Rendering. Pure string builders on top of `colabsd.ui.theme`, so the prose is testable.
# ----------------------------------------------------------------------------------------


def section_note(state: core.WizardState) -> str:
    """What this step is for, said beside its own controls.

    Nobody has to know what a 3Di string is to decide whether they need one: they chose a
    backbone, and this says what that choice means for the step below it.
    """
    if not core.needs_structure(state):
        return ""
    return (
        f"**{state.backbone}** reads your protein's *shape* as well as its sequence — choose an ESM2 "
        "backbone above and this step is not here at all. The shape is one letter per residue, a Foldseek "
        "3Di string. The usual answer is a structure file you already have or can download from the PDB or "
        "[AlphaFold](https://alphafold.ebi.ac.uk); folding it here is the last resort. The string stays in "
        "this session: there is nothing to download and upload back."
    )


def outputs_html(files: Sequence[OutputFile]) -> str:
    """What each written file is for and where it is kept."""
    if not files:
        return ""
    rows = "".join(
        "<tr>"
        f"<td style='padding:3px 10px 3px 0'><code>{item.name}</code></td>"
        f"<td style='padding:3px 10px 3px 0'>{item.needed_by}</td>"
        f"<td style='padding:3px 0'>{theme.render_markdown_inline(item.where_it_goes)}</td>"
        "</tr>"
        for item in files
    )
    return (
        "<div style='margin:6px 0;line-height:1.5'><b>This step will write:</b>"
        "<table style='border-collapse:collapse;margin-top:4px'>"
        "<tr style='text-align:left;border-bottom:1px solid rgba(128,128,128,0.35)'>"
        "<th style='padding-right:10px'>file</th><th style='padding-right:10px'>what needs it</th>"
        "<th>where it is kept</th></tr>"
        f"{rows}</table></div>"
    )


# ----------------------------------------------------------------------------------------
# The widget layer: one section the main panel composes. It owns no decision and only builds
# and reads widgets.
# ----------------------------------------------------------------------------------------

_LABEL = {"description_width": "initial"}


@dataclass(frozen=True)
class SectionTools:
    """The main panel's own widget factories, so a composed section looks like the rest of it.

    Passing them in rather than importing them keeps the upload row — and with it the "this is
    about to block the kernel" notice — in exactly one place.
    """

    text: Callable[..., Any]
    upload_row: Callable[..., Any]
    button: Callable[..., Any]


class ThreeDiSection:
    """Source a wild-type 3Di string. On screen only because a SaProt backbone was chosen."""

    def __init__(self, state: core.WizardState, tools: SectionTools) -> None:
        import ipywidgets

        self.fields: dict[str, Any] = {}
        choices = three_di_choices(state, None)
        values = [value for _label, value in choices]
        self.fields["three_di_source"] = ipywidgets.RadioButtons(
            options=choices,
            value=state.three_di_source if state.three_di_source in values else "none",
            layout={"width": "max-content"},
            style=_LABEL,
        )
        self.fields["structure_file"] = tools.upload_row(
            "structure_file", "Upload .pdb / .cif", "the wild-type structure", "structure"
        )
        self.fields["chain"] = tools.text("chain", "Chain to read:", "A — leave empty for the first chain")
        self.fields["three_di_file"] = tools.upload_row(
            "three_di_file", "Upload wt_3di.txt", "your 3Di file", "structure"
        )
        self.fields["three_di_text"] = ipywidgets.Textarea(
            value=str(state.get("three_di_text") or ""),
            placeholder="dpvqlvvcccd… one lowercase letter per residue",
            description="3Di string:",
            layout={"width": "560px", "height": "80px"},
            style=_LABEL,
        )
        self.fields["esmfold_note"] = theme.note(
            "ESMFold folds the wild type on this GPU. A model from the AlphaFold database "
            "([alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk)), uploaded above, is faster and at least "
            "as accurate."
        )
        self.fields["esmfold_risk_accepted"] = ipywidgets.Checkbox(
            value=bool(state.get("esmfold_risk_accepted")),
            description="I accept the ESMFold memory risk",
            indent=False,
            style=_LABEL,
        )
        self.note = theme.note("")
        self.button = tools.button("Get the 3Di string", "primary")

    def children(self) -> list[Any]:
        """The widgets, in reading order. The board, the button and the log are the panel's."""
        return [
            self.note,
            self.fields["three_di_source"],
            self.fields["structure_file"],
            self.fields["chain"],
            self.fields["three_di_file"],
            self.fields["three_di_text"],
            self.fields["esmfold_note"],
            self.fields["esmfold_risk_accepted"],
        ]

    def sync(self, state: core.WizardState, artefact: Artefact | None) -> str | None:
        """Rebuild the source list for what is actually available; return a forced value.

        Re-reading a different library has to take the reuse choice away rather than leave a
        dead option selected.
        """
        choices = three_di_choices(state, artefact)
        values = [value for _, value in choices]
        widget = self.fields["three_di_source"]
        if tuple(widget.options) == tuple(choices):
            return None
        wanted = state.three_di_source if state.three_di_source in values else "none"
        widget.options = choices
        widget.value = wanted
        return wanted

    def resolve(
        self,
        state: core.WizardState,
        backend: Any,
        *,
        wt_sequence: str,
        artefact: Artefact | None,
        work_dir: Path,
        has_gpu: bool,
    ) -> str:
        """Produce the 3Di string this run will be trained with. Raises with what to fix."""
        problems = three_di_blockers(state, artefact)
        if problems:
            raise ValueError(" ".join(problems))
        length = len(wt_sequence)
        source = state.three_di_source
        if source == "session":
            assert artefact is not None  # three_di_blockers refused this above
            return backend.load_three_di(artefact.path, expected_length=length)
        if source == "paste":
            return backend.validate_three_di(str(state.get("three_di_text") or ""), length)
        if source == "upload_3di":
            return backend.load_three_di(
                Path(str(state.get("three_di_file")).strip()), expected_length=length
            )
        if source == "upload_structure":
            return backend.three_di_from_structure(
                Path(str(state.get("structure_file")).strip()),
                chain=str(state.get("chain") or "").strip() or None,
                expected_length=length,
                expected_sequence=wt_sequence,
            )
        return backend.three_di_from_esmfold(
            wt_sequence,
            device="cuda" if has_gpu else "cpu",
            pdb_out=Path(work_dir) / "wt_esmfold.pdb",
        )
