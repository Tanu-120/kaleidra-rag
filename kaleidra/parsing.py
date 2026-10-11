"""Custom parsers for legacy and modern formats. Every parser returns a list of Sections."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Section:
    text: str
    heading: str = ""
    page: int | None = None


def _txt(path: Path) -> list[Section]:
    return [Section(path.read_text(errors="ignore"))]


def _md(path: Path) -> list[Section]:
    """Split markdown on headings so chunks inherit their heading as context."""
    sections, heading, buf = [], "", []
    for line in path.read_text(errors="ignore").splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if buf and "".join(buf).strip():
                sections.append(Section("\n".join(buf).strip(), heading))
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append(Section("\n".join(buf).strip(), heading))
    return sections or [Section(path.read_text(errors="ignore"))]


def _html(path: Path) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    for t in soup(["script", "style", "nav", "footer"]):
        t.decompose()
    out, heading, buf = [], "", []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "td"]):
        if el.name in ("h1", "h2", "h3"):
            if buf:
                out.append(Section(" ".join(buf), heading))
            heading, buf = el.get_text(" ", strip=True), []
        else:
            buf.append(el.get_text(" ", strip=True))
    if buf:
        out.append(Section(" ".join(buf), heading))
    return out


def _docx(path: Path) -> list[Section]:
    import docx

    d = docx.Document(str(path))
    out, heading, buf = [], "", []
    for p in d.paragraphs:
        if p.style.name.lower().startswith("heading"):
            if buf:
                out.append(Section("\n".join(buf), heading))
            heading, buf = p.text.strip(), []
        elif p.text.strip():
            buf.append(p.text.strip())
    if buf:
        out.append(Section("\n".join(buf), heading))
    return out


def _pdf(path: Path) -> list[Section]:
    # Layout-aware parsing with Docling when installed, otherwise pypdf per page.
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        md = DocumentConverter().convert(str(path)).document.export_to_markdown()
        tmp = path.with_suffix(".docling.md")
        tmp.write_text(md)
        try:
            return _md(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:
        pass
    from pypdf import PdfReader

    return [
        Section(p.extract_text() or "", page=i + 1)
        for i, p in enumerate(PdfReader(str(path)).pages)
        if (p.extract_text() or "").strip()
    ]


def _legacy_csv(path: Path) -> list[Section]:
    """Legacy delimited exports -> one natural-language line per row."""
    import csv

    with path.open(newline="", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    lines = ["; ".join(f"{k}: {v}" for k, v in r.items() if v) for r in rows]
    return [Section("\n".join(lines), heading=path.stem)]


PARSERS = {
    ".txt": _txt, ".md": _md, ".markdown": _md, ".html": _html, ".htm": _html,
    ".docx": _docx, ".pdf": _pdf, ".csv": _legacy_csv, ".log": _txt,
}


def parse(path: Path) -> list[Section]:
    fn = PARSERS.get(path.suffix.lower())
    if not fn:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return [s for s in fn(path) if s.text.strip()]
