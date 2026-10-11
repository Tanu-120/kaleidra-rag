"""Custom parsers for legacy and modern formats. Every parser returns a list of Sections."""
from __future__ import annotations

import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass
class Section:
    text: str
    heading: str = ""
    page: int | None = None


def clean_text(text: str) -> str:
    """Normalize messy PDF / HTML extract for readable answers."""
    if not text:
        return ""
    t = text.replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)  # IsA / ModelUnderstanding -> spaced
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" *\n+ *", "\n", t)
    t = re.sub(r"(?<![.!?:])\n(?![A-Z#\d•\-])", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


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


def _pdf_native_text(path: Path) -> list[Section]:
    """Extract embedded text via PyMuPDF (preferred) then pypdf."""
    try:
        import pymupdf

        doc = pymupdf.open(str(path))
        out = []
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            text = clean_text(raw)
            if text:
                # first non-empty line as a soft heading
                head = ""
                for line in text.splitlines():
                    if 3 <= len(line) <= 80:
                        head = line.strip()
                        break
                out.append(Section(text, heading=head, page=i + 1))
        if out:
            return out
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        return [
            Section(clean_text(p.extract_text() or ""), page=i + 1)
            for i, p in enumerate(PdfReader(str(path)).pages)
            if clean_text(p.extract_text() or "")
        ]
    except Exception:
        return []


def _ocr_cache_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".ocr.json")


def _load_ocr_cache(path: Path) -> list[Section] | None:
    cache = _ocr_cache_path(path)
    if not cache.exists():
        return None
    try:
        data = json.loads(cache.read_text())
        if data.get("source_mtime") != path.stat().st_mtime_ns:
            return None
        pages = data.get("pages") or []
        out = [
            Section(clean_text(p.get("text", "")), heading=p.get("heading", ""), page=p.get("page"))
            for p in pages
            if (p.get("text") or "").strip()
        ]
        return out or None
    except Exception:
        return None


def _save_ocr_cache(path: Path, sections: list[Section]) -> None:
    cache = _ocr_cache_path(path)
    payload = {
        "source_mtime": path.stat().st_mtime_ns,
        "pages": [{"page": s.page, "heading": s.heading, "text": s.text} for s in sections],
    }
    cache.write_text(json.dumps(payload))


def _ocr_one_page(path: str, page_index: int, scale: float) -> tuple[int, str]:
    import pymupdf
    import pytesseract
    from PIL import Image

    doc = pymupdf.open(path)
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img) or ""
        return page_index, text
    finally:
        doc.close()


def _pdf_ocr(path: Path) -> list[Section]:
    """OCR scanned/image PDFs. Results are cached beside the file."""
    if settings.ocr_max_pages <= 0:
        return []
    cached = _load_ocr_cache(path)
    if cached is not None:
        return cached

    try:
        import pymupdf
        import pytesseract  # noqa: F401 — require tesseract bindings
    except Exception as e:
        print(f"[kaleidra] OCR unavailable for {path.name} ({type(e).__name__})")
        return []

    try:
        doc = pymupdf.open(str(path))
        n_pages = doc.page_count
        doc.close()
    except Exception as e:
        print(f"[kaleidra] cannot open PDF for OCR {path.name}: {e}")
        return []

    limit = min(n_pages, settings.ocr_max_pages)
    scale = settings.ocr_dpi_scale
    print(f"[kaleidra] OCR {path.name}: {limit}/{n_pages} pages (scanned PDF)")

    by_idx: dict[int, str] = {}
    workers = min(4, max(1, limit))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_ocr_one_page, str(path), i, scale) for i in range(limit)]
        for fut in as_completed(futs):
            try:
                idx, text = fut.result()
                by_idx[idx] = text
            except Exception as e:
                print(f"[kaleidra] OCR page failed ({type(e).__name__}): {e}")

    out: list[Section] = []
    for i in range(limit):
        text = clean_text(by_idx.get(i, ""))
        if not text:
            continue
        head = ""
        for line in text.splitlines():
            if 3 <= len(line) <= 90 and not line.lower().startswith("page "):
                head = line.strip()
                break
        out.append(Section(text, heading=head, page=i + 1))

    if out:
        try:
            _save_ocr_cache(path, out)
        except Exception:
            pass
    return out


def _pdf(path: Path) -> list[Section]:
    # Layout-aware parsing with Docling when installed, otherwise text extract / OCR.
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        md = DocumentConverter().convert(str(path)).document.export_to_markdown()
        if md and len(md.strip()) > 80:
            tmp = path.with_suffix(".docling.md")
            tmp.write_text(md)
            try:
                return _md(tmp)
            finally:
                tmp.unlink(missing_ok=True)
    except Exception:
        pass

    sections = _pdf_native_text(path)
    total = sum(len(s.text) for s in sections)
    # Image-only / barely-text PDFs need OCR
    if total < 120:
        ocr = _pdf_ocr(path)
        if ocr:
            return ocr
    return sections


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
