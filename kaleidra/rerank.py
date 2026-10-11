"""Cross-encoder reranking (open model) with a lexical-overlap fallback."""
from __future__ import annotations

from .store import Hit, tokenize


class LiteReranker:
    name = "lexical-overlap-lite"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        out = []
        for t in texts:
            toks = tokenize(t)
            if not toks or not q:
                out.append(0.0)
                continue
            cover = len(q & set(toks)) / len(q)
            # reward query terms appearing close together
            pos = [i for i, tk in enumerate(toks) if tk in q]
            span = (pos[-1] - pos[0] + 1) if pos else len(toks)
            density = len(pos) / max(span, 1)
            out.append(cover + 0.3 * density)
        return out


class CrossEncoderReranker:
    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.name = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.model.predict([(query, t) for t in texts])]


def load_reranker(model: str, backend: str = "auto"):
    if backend != "lite":
        try:
            return CrossEncoderReranker(model)
        except Exception as e:
            if backend == "hf":
                raise
            print(f"[kaleidra] cross-encoder unavailable ({type(e).__name__}); using lexical reranker")
    return LiteReranker()


def rerank(reranker, query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return hits
    scores = reranker.score(query, [h.chunk.contextual for h in hits])
    for h, s in zip(hits, scores):
        h.rerank_score = s
    ordered = sorted(hits, key=lambda h: -(h.rerank_score or 0))
    for r, h in enumerate(ordered, 1):
        h.rerank_rank = r
    return ordered
