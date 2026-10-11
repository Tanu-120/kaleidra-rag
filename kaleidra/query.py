"""Query transformation: multi-query expansion + HyDE (hypothetical document embedding).
Works with a local Ollama model when configured; otherwise uses a deterministic heuristic."""
from __future__ import annotations

import json
import re
import urllib.request

from .config import settings

_Q_WORDS = r"^(what|who|when|where|why|how|which|does|do|is|are|can|should|tell me|explain)\b[\s,]*"


def _ollama(prompt: str) -> str | None:
    if not settings.ollama_url:
        return None
    try:
        req = urllib.request.Request(
            settings.ollama_url.rstrip("/") + "/api/generate",
            data=json.dumps({"model": settings.ollama_model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["response"].strip()
    except Exception:
        return None


def hyde(query: str) -> str:
    """Write a short hypothetical answer; its embedding often lands closer to real passages."""
    out = _ollama(f"Write a 2-sentence factual passage that answers: {query}\nPassage:")
    if out:
        return out
    stem = re.sub(_Q_WORDS, "", query.strip().rstrip("?"), flags=re.I)
    return f"{stem[:1].upper() + stem[1:]} is described as follows. The documentation states that {stem}."


def expand(query: str, n: int = 2) -> list[str]:
    out = _ollama(f"Rewrite this search query {n} different ways, one per line, no numbering:\n{query}")
    if out:
        alts = [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:n]
        if alts:
            return alts
    stem = re.sub(_Q_WORDS, "", query.strip().rstrip("?"), flags=re.I).strip()
    alts = [stem] if stem and stem.lower() != query.lower() else []
    keywords = [w for w in re.findall(r"[A-Za-z0-9\-]{4,}", stem)]
    if len(keywords) >= 2:
        alts.append(" ".join(keywords))
    return alts[:n]


def transform(query: str, use_hyde: bool = True, use_expand: bool = True) -> list[dict]:
    """Returns [{kind, text}] with the original query always first."""
    qs = [{"kind": "original", "text": query}]
    if use_expand:
        qs += [{"kind": "rewrite", "text": t} for t in expand(query)]
    if use_hyde:
        qs.append({"kind": "hyde", "text": hyde(query)})
    return qs
