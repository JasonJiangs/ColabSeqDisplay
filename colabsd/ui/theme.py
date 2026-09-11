"""Presentation vocabulary for the ColabSeqDisplay wizards.

Everything the user reads is a short HTML block built here: markdown rendered to
HTML and dropped into an `ipywidgets.HTML`. There are four registers and
deliberately no more.

* `heading` — where you are in the form.
* `note` — one or two sentences of plain language, aimed at a biologist.
* `message(text, severity)` — inline and contextual, coloured by what ignoring
  it costs:

  ``info``
      context. Nothing is wrong.
  ``warning``
      the run will finish, but the number it produces means less than you think.
  ``stop``
      the run will fail, or it will waste an hour. This is the red one.

The string builders (`*_html`) are pure and import no widget library, so the
wording and the severity of every message is testable without a display.
"""

from __future__ import annotations

import html as _html
import re
from typing import Any, Literal

Severity = Literal["info", "warning", "stop"]

#: Severity names, cheapest first.
SEVERITIES: tuple[Severity, ...] = ("info", "warning", "stop")

#: Sort key: the expensive ones go to the top of a message list.
SEVERITY_RANK: dict[str, int] = {"stop": 0, "warning": 1, "info": 2}

#: Readable on both the light and the dark Colab theme.
SEVERITY_COLOR: dict[str, str] = {"info": "#1f6feb", "warning": "#c26a00", "stop": "#d1242f"}
SEVERITY_LABEL: dict[str, str] = {"info": "Note", "warning": "Warning", "stop": "Stop"}
SEVERITY_ICON: dict[str, str] = {"info": "\N{INFORMATION SOURCE}", "warning": "\N{WARNING SIGN}", "stop": "⛔"}

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
_CODE = re.compile(r"`([^`\n]+)`")
_SINGLE_PARAGRAPH = re.compile(r"\A<p>(.*)</p>\Z", re.DOTALL)


def check_severity(severity: str) -> Severity:
    """Return `severity` if it is one of `SEVERITIES`, else raise `ValueError`."""
    if severity not in SEVERITY_RANK:
        raise ValueError(f"Unknown severity {severity!r}. Use one of: {', '.join(SEVERITIES)}.")
    return severity  # type: ignore[return-value]


def render_markdown(text: str) -> str:
    """Render markdown to HTML.

    Falls back to `_fallback_markdown` when the `markdown` package is absent, so the wizards
    still read correctly in a bare environment. Raw HTML in `text` passes through untouched
    either way, which is what lets a note carry a link.
    """
    try:
        import markdown as _markdown
    except ImportError:
        return _fallback_markdown(text)
    return _markdown.markdown(text)


def render_markdown_inline(text: str) -> str:
    """Render markdown and unwrap a lone `<p>`, so the result nests inside a line."""
    rendered = render_markdown(text).strip()
    match = _SINGLE_PARAGRAPH.fullmatch(rendered)
    if match and "<p>" not in match.group(1):
        return match.group(1)
    return rendered


def _fallback_markdown(text: str) -> str:
    """Enough markdown for a notebook form: headings, bullets, bold, italic, code, links."""
    blocks = [block for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    return "\n".join(_render_block(block) for block in blocks)


def _render_block(block: str) -> str:
    lines = [line for line in block.split("\n") if line.strip()]
    if not lines:
        return ""
    if lines[0].lstrip().startswith("<"):
        return block.strip()
    heading = _HEADING.match(lines[0].strip())
    if heading:
        level = len(heading.group(1))
        rest = _render_block("\n".join(lines[1:])) if len(lines) > 1 else ""
        return f"<h{level}>{_inline(heading.group(2))}</h{level}>{rest}"
    bullets = [_BULLET.match(line) for line in lines]
    if all(bullets):
        items = "".join(f"<li>{_inline(match.group(1))}</li>" for match in bullets if match)
        return f"<ul>{items}</ul>"
    return "<p>" + "<br>".join(_inline(line.strip()) for line in lines) + "</p>"


def _inline(text: str) -> str:
    codes: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        codes.append(match.group(1))
        return f"\x00{len(codes) - 1}\x00"

    out = _CODE.sub(_stash, text)
    out = _LINK.sub(lambda m: f'<a href="{_html.escape(m.group(2), quote=True)}" target="_blank">{m.group(1)}</a>', out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    for index, code in enumerate(codes):
        out = out.replace(f"\x00{index}\x00", f"<code>{_html.escape(code)}</code>")
    return out


def heading_html(text: str, level: int = 3) -> str:
    """A form section title."""
    level = max(1, min(6, int(level)))
    return f'<h{level} style="margin:14px 0 4px 0">{render_markdown_inline(text)}</h{level}>'


def note_html(text: str) -> str:
    """An explanation aimed at a biologist: quieter than the form, never grey-on-grey.

    `opacity` rather than a fixed grey, so it stays legible on the dark theme too.
    """
    return f'<div style="opacity:0.85;line-height:1.45;margin:2px 0 8px 0">{render_markdown(text)}</div>'


def message_html(text: str, severity: str = "info") -> str:
    """One contextual message, coloured and labelled by what ignoring it costs."""
    check_severity(severity)
    color = SEVERITY_COLOR[severity]
    label = f"{SEVERITY_ICON[severity]} {SEVERITY_LABEL[severity]}:"
    return (
        f'<div data-severity="{severity}" style="border-left:4px solid {color};padding:3px 10px;'
        f'margin:5px 0;line-height:1.45;color:{color}">'
        f"<b>{label}</b> {render_markdown_inline(text)}</div>"
    )


def rule_html() -> str:
    """A separator."""
    return '<hr style="border:none;border-top:1px solid rgba(128,128,128,0.35);margin:14px 0">'


def _widgets() -> Any:
    """Import ipywidgets on demand, so the string builders work without it."""
    import ipywidgets

    return ipywidgets


def html(text: str, *, visible: bool = True, width: str | None = None) -> Any:
    """An `ipywidgets.HTML` holding rendered markdown."""
    widget = _widgets().HTML(value=render_markdown(text))
    if not visible:
        widget.layout.display = "none"
    if width:
        widget.layout.width = width
    return widget


def heading(text: str, level: int = 3, *, visible: bool = True) -> Any:
    """A section title widget."""
    widget = _widgets().HTML(value=heading_html(text, level))
    if not visible:
        widget.layout.display = "none"
    return widget


def note(text: str, *, visible: bool = True) -> Any:
    """An explanatory-note widget."""
    widget = _widgets().HTML(value=note_html(text))
    if not visible:
        widget.layout.display = "none"
    return widget


def message(text: str, severity: str = "info", *, visible: bool = True) -> Any:
    """A contextual-message widget; set `widget.value = message_html(...)` to change it."""
    widget = _widgets().HTML(value=message_html(text, severity))
    if not visible:
        widget.layout.display = "none"
    return widget


def info(text: str, **kwargs: Any) -> Any:
    """`message(text, "info")`."""
    return message(text, "info", **kwargs)


def warning(text: str, **kwargs: Any) -> Any:
    """`message(text, "warning")`."""
    return message(text, "warning", **kwargs)


def stop(text: str, **kwargs: Any) -> Any:
    """`message(text, "stop")` — the red one."""
    return message(text, "stop", **kwargs)


def rule() -> Any:
    """A separator widget."""
    return _widgets().HTML(value=rule_html())
