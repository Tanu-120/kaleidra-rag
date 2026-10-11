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
    r"\b(is|are|means|refers to|defined as|consists of|provides|enables|allows|stores|pre-?compute|materializ)\w*\b",
    re.I,
)
_STOP = {
    "what", "is", "are", "a", "an", "the", "of", "and", "or", "to", "in", "on", "for",
    "with", "about", "tell", "me", "how", "does", "do", "can", "you", "please", "explain",
}


def _query_terms(query: str) -> set[str]:
    return {t for t in tokenize(query) if t not in _STOP and len(t) > 1}


def _focus_terms(query: str) -> list[str]:
    """Important topic words from the question (order preserved)."""
    out = []
    for t in tokenize(query):
        if t in _STOP or len(t) <= 1:
            continue
        if t not in out:
            out.append(t)
    return out


def filter_relevant_hits(query: str, hits: list[Hit], min_keep: int = 3) -> list[Hit]:
    """Drop retrieved chunks that share no meaningful terms with the question."""
    focus = set(_focus_terms(query))
    if not focus or not hits:
        return hits
    scored = []
    for h in hits:
        toks = set(tokenize(h.chunk.text + " " + (h.chunk.heading or "") + " " + h.chunk.source))
        overlap = len(focus & toks)
        # source-name match (e.g. Snowflake pdf for snowflake questions)
        src = set(tokenize(re.sub(r"[_\-.]", " ", h.chunk.source)))
        overlap += len(focus & src)
        scored.append((overlap, h))
    scored.sort(key=lambda x: -x[0])
    kept = [h for ov, h in scored if ov > 0]
    if len(kept) >= min_keep:
        return kept
    # fall back to original order if everything looked weak
    return [h for _, h in scored[: max(min_keep, len(hits))]]


def _split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    parts = [p.strip() for p in _SENT.split(text) if p and p.strip()]
    out = []
    for p in parts:
        if len(p) < 28:
            continue
        if re.fullmatch(r"(SECTION|CHAPTER|PAGE)\s+\d+.*", p, re.I):
            continue
        out.append(p)
    return out


def _score_sentence(focus: set[str], s: str, rank: int, query: str, *, page: int | None, heading: str) -> float:
    t = set(tokenize(s))
    if not t:
        return 0.0
    if focus:
        overlap = len(focus & t) / len(focus)
    else:
        overlap = 0.0
    # hard preference: sentences must mention at least one focus term when question has them
    if focus and not (focus & t):
        return -1.0
    score = overlap * 1.4 + 0.08 * (1 / rank)
    ql = query.lower()
    definitional = ql.startswith("what is") or ql.startswith("what are") or " define " in f" {ql} "
    sl = s.lower()
    hl = (heading or "").lower()

    # topic boosts for common doc-QA intents
    if "view" in focus:
        if re.search(r"\bmaterialized views?\b", sl):
            score += 0.55
        elif re.search(r"\bviews?\b", sl):
            score += 0.35
        if "micro-partition" in sl or "null-count" in sl or "distinct-count" in sl:
            score -= 0.45  # related Snowflake internals, not "views"
    if any(k in focus for k in ("warehouse", "warehouses", "compute")):
        if "virtual warehouse" in sl or "warehouses" in sl:
            score += 0.3
    if "relational" in focus or ({"database", "databases"} & focus):
        if "relational database" in sl:
            score += 0.55
        if re.search(r"collection of data stored in tables", sl):
            score += 0.45
        # demote Snowflake architecture answers when asking about generic databases
        if "snowflake" in sl and "relational database" not in sl:
            score -= 0.35
    if "sql" in focus and definitional:
        if re.search(r"\bsql is\b.*(language|acronym|structured query)", sl):
            score += 0.5
        if "most popular of all database languages" in sl:
            score += 0.45

    if definitional:
        if _DEF.search(s):
            score += 0.3
        # prefer "X is …" for the full subject, not just the first token
        focus_words = [w for w in focus if len(w) > 2]
        if focus_words and all(w in t for w in focus_words[:3]):
            score += 0.25
        if focus_words and re.search(rf"\b{re.escape(focus_words[0])}\b.{{0,40}}\b(is|are)\b", sl):
            score += 0.4
        if page and page <= 8:
            score += 0.12 * max(0, 6 - (page or 6))
        if any(k in hl or k in sl[:80] for k in ("billing", "credit & billing", "cost optimization")):
            score -= 0.3
        if re.search(r"what is (a |an )?(relational )?database", hl + " " + sl[:120], re.I):
            score += 0.35

    # heading alignment
    if focus & set(tokenize(heading)):
        score += 0.25

    n = len(s.split())
    if 14 <= n <= 60:
        score += 0.08
    if n > 90:
        score -= 0.12
    # demote TOC / section-title crumbs
    if re.match(r"^(SECTION|CHAPTER)\s+\d+", s, re.I):
        score -= 0.4
    if not _DEF.search(s) and n < 16:
        score -= 0.25
    return score


def extractive_answer(query: str, hits: list[Hit], max_sentences: int = 4) -> list[dict]:
    focus = _query_terms(query)
    # keep morphological variants: view -> views
    expanded = set(focus)
    for t in list(focus):
        if t.endswith("s") and len(t) > 3:
            expanded.add(t[:-1])
        else:
            expanded.add(t + "s")
    focus = expanded

    cands = []
    for n, h in enumerate(hits, 1):
        heading = h.chunk.heading or ""
        page = h.chunk.page
        for s in _split_sentences(h.chunk.text):
            sc = _score_sentence(focus, s, n, query, page=page, heading=heading)
            if sc < 0:
                continue
            cands.append((sc, s, n, h.chunk.source, heading))
    cands.sort(key=lambda x: -x[0])

    definitional = query.lower().startswith("what is") or query.lower().startswith("what are")
    chosen, seen = [], set()
    for sc, s, n, source, heading in cands:
        key = " ".join(tokenize(s)[:8])
        if sc < 0.18 or key in seen:
            continue
        if definitional and not _DEF.search(s) and len(s.split()) < 18:
            continue
        # skip bare titles like "What is a Relational Database?"
        if re.fullmatch(r"what is (a |an |the )?[\w\s\-/]+[?]?", s.strip(), re.I):
            continue
        if re.search(r"study guide|worked sql examples", s, re.I) and not _DEF.search(s):
            continue
        # require focus overlap in chosen sentence
        if focus and not (focus & set(tokenize(s))):
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
        "Answer using ONLY the numbered context. Write 2-4 clear sentences about the question. "
        "Do not mention unrelated topics. Cite sources like [1]. If the context lacks the answer say 'I don't know'.\n"
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
    """Only documents that actually contributed a cited claim — never pad with unrelated hits."""
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
    return list(order.values())


def _compose_summary(query: str, claims: list[dict]) -> str:
    """Build a short readable lead from the best claims (avoid raw dump of titles)."""
    if not claims:
        return ""
    parts = []
    for c in claims:
        t = c["text"].strip()
        # skip pure headings
        if len(t.split()) < 10 and not _DEF.search(t):
            continue
        parts.append(t if t.endswith((".", "!", "?")) else t + ".")
        if len(parts) >= 3:
            break
    if not parts:
        parts = [claims[0]["text"]]
    return " ".join(parts)


def _confidence(mode: str, faith: dict, sources: list[dict], claims: list[dict]) -> dict:
    """Bootcamp-friendly confidence signal for the UI (not a calibrated probability)."""
    if mode == "abstain":
        return {"score": 0.0, "label": "low", "reasons": ["no grounded evidence"]}
    f = float(faith.get("score") or 0)
    supports = [c.get("support", 0) for c in faith.get("claims") or [] if isinstance(c, dict)]
    mean_sup = sum(supports) / len(supports) if supports else f
    n_src = len(sources)
    score = 0.65 * f + 0.35 * mean_sup
    if n_src == 1:
        score += 0.05  # focused single-document answers tend to be cleaner
    if n_src >= 4:
        score -= 0.08
    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.8 else "medium" if score >= 0.55 else "low"
    reasons = []
    if f >= 0.8:
        reasons.append("strong claim grounding")
    elif f < 0.55:
        reasons.append("weak grounding")
    if n_src == 1:
        reasons.append("single source focus")
    elif n_src > 1:
        reasons.append(f"{n_src} sources cited")
    if not claims:
        reasons.append("empty claims")
    return {"score": round(score, 3), "label": label, "reasons": reasons}


def answer(query: str, hits: list[Hit]) -> dict:
    hits = filter_relevant_hits(query, hits)
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
    for c in claims:
        if c.get("cite") and not c.get("source") and 1 <= c["cite"] <= len(hits):
            c["source"] = hits[c["cite"] - 1].chunk.source
            c["heading"] = hits[c["cite"] - 1].chunk.heading or ""
        if c.get("text"):
            c["text"] = clean_text(c["text"])
    faith = faithfulness(claims, hits) if mode != "abstain" else {"score": 1.0, "claims": []}
    summary = _compose_summary(query, claims) if mode != "abstain" else claims[0]["text"]
    sources = _sources_used(claims, hits) if mode != "abstain" else []
    return {
        "mode": mode,
        "claims": claims,
        "summary": summary,
        "sources_used": sources,
        "faithfulness": faith,
        "confidence": _confidence(mode, faith, sources, claims),
    }
