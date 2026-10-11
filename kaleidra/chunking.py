"""Heading-aware recursive chunker with sentence-boundary snapping and overlap."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .parsing import Section

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    heading: str = ""
    page: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def contextual(self) -> str:
        """Text used for embedding/BM25: heading + source prepended (contextual chunk headers)."""
        head = f"{self.source} > {self.heading}\n" if self.heading else f"{self.source}\n"
        return head + self.text


def chunk_sections(sections: list[Section], source: str, size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in sections:
        sentences = [s.strip() for s in _SENT.split(sec.text.replace("\r", "")) if s.strip()]
        buf: list[str] = []
        length = 0
        for s in sentences:
            if length + len(s) > size and buf:
                chunks.append(_mk(" ".join(buf), source, sec))
                # carry trailing sentences as overlap
                carry, c = [], 0
                for prev in reversed(buf):
                    if c + len(prev) > overlap:
                        break
                    carry.insert(0, prev)
                    c += len(prev)
                buf, length = carry, c
            buf.append(s)
            length += len(s)
        if buf:
            chunks.append(_mk(" ".join(buf), source, sec))
    return chunks


def _mk(text: str, source: str, sec: Section) -> Chunk:
    cid = hashlib.sha1(f"{source}|{text}".encode()).hexdigest()[:12]
    return Chunk(id=cid, text=text, source=source, heading=sec.heading, page=sec.page)
