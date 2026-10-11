"""Context deduplication: drop near-duplicate passages (Jaccard on word shingles) and
cap per-source dominance so the LLM sees diverse evidence."""
from __future__ import annotations

import re

from .store import Hit


def _shingles(text: str, n: int = 3) -> set[str]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


def dedupe(hits: list[Hit], threshold: float = 0.88, max_per_source: int = 3) -> tuple[list[Hit], list[dict]]:
    kept: list[Hit] = []
    dropped: list[dict] = []
    sigs: list[set[str]] = []
    per_source: dict[str, int] = {}
    for h in hits:
        s = _shingles(h.chunk.text)
        dup = next((k for k, ks in zip(kept, sigs) if len(s & ks) / max(len(s | ks), 1) >= threshold * 0.6), None)
        if dup is not None:
            dropped.append({"id": h.chunk.id, "reason": f"near-duplicate of {dup.chunk.id}"})
            continue
        if per_source.get(h.chunk.source, 0) >= max_per_source:
            dropped.append({"id": h.chunk.id, "reason": f"source cap ({max_per_source}) reached"})
            continue
        kept.append(h)
        sigs.append(s)
        per_source[h.chunk.source] = per_source.get(h.chunk.source, 0) + 1
    return kept, dropped
