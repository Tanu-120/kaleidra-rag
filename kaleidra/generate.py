"""Grounded answer synthesis. Default: extractive & fully local (every sentence carries a citation).
Optional: Ollama for abstractive answers, then claim-level grounding is verified either way."""
from __future__ import annotations

import json
import re
import urllib.request

from .config import settings
from .store import Hit, tokenize

_SENT = re.compile(r"(?<=[.!?])\s+")


def _score_sentence(q_terms: set[str], s: str) -> float:
    t = set(tokenize(s))
    return len(q_terms & t) / (len(q_terms) or 1) if t else 0.0


def extractive_answer(query: str, hits: list[Hit], max_sentences: int = 4) -> list[dict]:
    q = set(tokenize(query))
    cands = []
    for n, h in enumerate(hits, 1):
        for s in _SENT.split(h.chunk.text):
            s = s.strip()
            if len(s) < 25:
                continue
            cands.append((_score_sentence(q, s) + 0.15 * (1 / n), s, n))
    cands.sort(key=lambda x: -x[0])
    chosen, seen = [], set()
    for sc, s, n in cands:
        key = " ".join(tokenize(s)[:8])
        if sc < 0.15 or key in seen:
            continue
        seen.add(key)
        chosen.append({"text": s, "cite": n})
        if len(chosen) >= max_sentences:
            break
    return chosen


def llm_answer(query: str, hits: list[Hit]) -> list[dict] | None:
    if not settings.ollama_url:
        return None
    ctx = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
    prompt = (
        "Answer using ONLY the numbered context. Cite sources like [1]. If the context lacks the answer say 'I don't know'.\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        req = urllib.request.Request(
            settings.ollama_url.rstrip("/") + "/api/generate",
            data=json.dumps({"model": settings.ollama_model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            text = json.loads(r.read())["response"].strip()
    except Exception:
        return None
    out = []
    for s in _SENT.split(text):
        m = re.findall(r"\[(\d+)\]", s)
        out.append({"text": re.sub(r"\s*\[\d+\]", "", s).strip(), "cite": int(m[0]) if m else None})
    return [o for o in out if o["text"]]


def faithfulness(claims: list[dict], hits: list[Hit]) -> dict:
    """Claim-level grounding: share of a claim's content words supported by retrieved evidence."""
    ev = [set(tokenize(h.chunk.text)) for h in hits]
    details, total = [], 0.0
    for c in claims:
        words = set(tokenize(c["text"]))
        if not words:
            continue
        best, best_i = 0.0, None
        for i, e in enumerate(ev):
            sup = len(words & e) / len(words)
            if sup > best:
                best, best_i = sup, i + 1
        details.append({"text": c["text"], "support": round(best, 3), "best_source": best_i, "grounded": best >= 0.6})
        total += best
    score = total / len(details) if details else 0.0
    return {"score": round(score, 3), "claims": details}


def answer(query: str, hits: list[Hit]) -> dict:
    claims = llm_answer(query, hits)
    mode = "llm" if claims else "extractive"
    if not claims:
        claims = extractive_answer(query, hits)
    if not claims:
        claims = [{"text": "I don't know — the indexed documents don't contain enough evidence to answer that.", "cite": None}]
        mode = "abstain"
    faith = faithfulness(claims, hits) if mode != "abstain" else {"score": 1.0, "claims": []}
    return {"mode": mode, "claims": claims, "faithfulness": faith}
