"""Grounded answer synthesis. Default: extractive & fully local (every sentence carries a citation).
Optional: Ollama for abstractive answers, then claim-level grounding is verified either way."""
from __future__ import annotations

import json
import re
import urllib.request
from collections import OrderedDict

from .config import settings
from .parsing import clean_text
from .store import Hit, tokenize

_SENT = re.compile(r"(?<=[.!?])\s+|\n+(?=[A-Z0-9])")
_DEF = re.compile(
    r"\b(is|are|means|refers to|defined as|consists of|provides|enables|allows)\b",
    re.I,
)


def _split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    parts = [p.strip() for p in _SENT.split(text) if p and p.strip()]
    out = []
    for p in parts:
        # drop heading-only crumbs / page noise
        if len(p) < 28:
            continue
        if re.fullmatch(r"(SECTION|CHAPTER|PAGE)\s+\d+.*", p, re.I):
            continue
        out.append(p)
    return out


def _score_sentence(q_terms: set[str], s: str, rank: int, query: str, *, page: int | None, heading: str) -> float:
    t = set(tokenize(s))
    if not t:
        return 0.0
    overlap = len(q_terms & t) / (len(q_terms) or 1)
    score = overlap + 0.12 * (1 / rank)
    ql = query.lower()
    definitional = ql.startswith("what is") or ql.startswith("what are") or " define " in f" {ql} "
    subj = [w for w in tokenize(query) if w not in {"what", "is", "are", "a", "an", "the"}]
    sl = s.lower()
    hl = (heading or "").lower()

    if definitional:
        if _DEF.search(s):
            score += 0.35
        if subj and any(sl.startswith(w) or f" {w} " in f" {sl} " for w in subj[:2]):
            score += 0.2
        # prefer intro / overview material over billing / credit deep-dives
        if any(k in hl or k in sl[:80] for k in ("overview", "introduction", "intro", "about", "concept", "architecture")):
            score += 0.3
        if any(k in hl or k in sl[:60] for k in ("billing", "credit", "pricing", "cost optimization", "warehouse actually")):
            score -= 0.35
        if page and page <= 3:
            score += 0.25 * (4 - page)
        # classic definition shape: "X is a …"
        if subj and re.search(rf"\b{re.escape(subj[0])}\b.{{0,40}}\b(is|are)\b.{{0,40}}\b(a|an|the|cloud|platform|service|warehouse)\b", sl):
            score += 0.45

    n = len(s.split())
    if 12 <= n <= 55:
        score += 0.08
    if n > 80:
        score -= 0.1
    return score


def extractive_answer(query: str, hits: list[Hit], max_sentences: int = 4) -> list[dict]:
    q = set(tokenize(query))
    cands = []
    for n, h in enumerate(hits, 1):
        heading = h.chunk.heading or ""
        page = h.chunk.page
        for s in _split_sentences(h.chunk.text):
            sc = _score_sentence(q, s, n, query, page=page, heading=heading)
            cands.append((sc, s, n, h.chunk.source, heading))
    cands.sort(key=lambda x: -x[0])
    definitional = query.lower().startswith("what is") or query.lower().startswith("what are")
    chosen, seen = [], set()
    for sc, s, n, source, heading in cands:
        key = " ".join(tokenize(s)[:8])
        if sc < 0.12 or key in seen:
            continue
        # skip title/subtitle crumbs when we can do better
        if definitional and not _DEF.search(s) and len(s.split()) < 18:
            continue
        if re.search(r"study guide|worked sql examples|section \d+", s, re.I) and not _DEF.search(s):
            continue
        seen.add(key)
        chosen.append({
            "text": clean_text(s),
            "cite": n,
            "source": source,
            "heading": heading,
        })
        if len(chosen) >= max_sentences:
            break
    return chosen


def llm_answer(query: str, hits: list[Hit]) -> list[dict] | None:
    if not settings.ollama_url:
        return None
    ctx = "\n\n".join(
        f"[{i}] ({h.chunk.source}) {clean_text(h.chunk.text)}" for i, h in enumerate(hits, 1)
    )
    prompt = (
        "Answer using ONLY the numbered context. Write 2-4 clear sentences. "
        "Cite sources like [1]. If the context lacks the answer say 'I don't know'.\n"
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
        cite = int(m[0]) if m else None
        body = clean_text(re.sub(r"\s*\[\d+\]", "", s))
        if not body:
            continue
        src = hits[cite - 1].chunk.source if cite and 1 <= cite <= len(hits) else None
        head = hits[cite - 1].chunk.heading if cite and 1 <= cite <= len(hits) else ""
        out.append({"text": body, "cite": cite, "source": src, "heading": head or ""})
    return out or None


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


def _sources_used(claims: list[dict], hits: list[Hit]) -> list[dict]:
    order: OrderedDict[str, dict] = OrderedDict()
    for c in claims:
        src = c.get("source")
        if not src and c.get("cite") and 1 <= c["cite"] <= len(hits):
            src = hits[c["cite"] - 1].chunk.source
        if not src:
            continue
        slot = order.setdefault(src, {"name": src, "cites": [], "headings": []})
        if c.get("cite") and c["cite"] not in slot["cites"]:
            slot["cites"].append(c["cite"])
        if c.get("heading") and c["heading"] not in slot["headings"]:
            slot["headings"].append(c["heading"])
    # also include top hits even if not cited, so UI can show "consulted"
    for i, h in enumerate(hits[:5], 1):
        order.setdefault(h.chunk.source, {"name": h.chunk.source, "cites": [], "headings": []})
        if i not in order[h.chunk.source]["cites"] and any(c.get("cite") == i for c in claims):
            order[h.chunk.source]["cites"].append(i)
    return list(order.values())


def answer(query: str, hits: list[Hit]) -> dict:
    claims = llm_answer(query, hits)
    mode = "llm" if claims else "extractive"
    if not claims:
        claims = extractive_answer(query, hits)
    if not claims:
        claims = [{
            "text": "I don't know — the indexed documents don't contain enough evidence to answer that.",
            "cite": None, "source": None, "heading": "",
        }]
        mode = "abstain"
    # ensure source fields filled
    for c in claims:
        if c.get("cite") and not c.get("source") and 1 <= c["cite"] <= len(hits):
            c["source"] = hits[c["cite"] - 1].chunk.source
            c["heading"] = hits[c["cite"] - 1].chunk.heading or ""
        if c.get("text"):
            c["text"] = clean_text(c["text"])
    faith = faithfulness(claims, hits) if mode != "abstain" else {"score": 1.0, "claims": []}
    summary = " ".join(c["text"] for c in claims if c.get("text"))
    sources = _sources_used(claims, hits) if mode != "abstain" else []
    return {
        "mode": mode,
        "claims": claims,
        "summary": summary,
        "sources_used": sources,
        "faithfulness": faith,
    }
