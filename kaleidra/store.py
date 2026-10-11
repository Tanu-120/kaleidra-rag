"""Hybrid store: dense vectors (ChromaDB, or numpy fallback) + BM25, fused with RRF."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of and or to in on for with is are was were be by as at from that this it its".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, corpus: list[list[str]]):
        self.docs = corpus
        self.df = Counter(t for d in corpus for t in set(d))
        self.avgdl = sum(len(d) for d in corpus) / max(len(corpus), 1)
        self.tf = [Counter(d) for d in corpus]

    def scores(self, query: list[str]) -> np.ndarray:
        n = len(self.docs)
        out = np.zeros(n, dtype=np.float32)
        for t in set(query):
            if t not in self.df:
                continue
            idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avgdl))
        return out


@dataclass
class Hit:
    chunk: Chunk
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf: float = 0.0
    rerank_score: float | None = None
    rerank_rank: int | None = None
    fused_rank: int | None = None


class HybridStore:
    def __init__(self, embedder, persist_dir: Path, use_chroma: bool = True):
        self.embedder = embedder
        self.persist_dir = persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self.bm25 = BM25()
        self._chroma = None
        self._chroma_client = None
        self._chroma_name = "kaleidra_" + re.sub(r"\W", "_", embedder.name)[:40]
        if use_chroma:
            try:
                import chromadb

                self._chroma_client = chromadb.PersistentClient(path=str(persist_dir / "chroma"))
                self._reset_chroma()
            except Exception as e:
                print(f"[kaleidra] ChromaDB unavailable ({type(e).__name__}); using in-memory numpy index")
                self._chroma_client = None

    def _reset_chroma(self):
        """Drop and recreate the collection so re-ingest never hits a stale handle."""
        if self._chroma_client is None:
            return
        try:
            self._chroma_client.delete_collection(self._chroma_name)
        except Exception:
            pass
        self._chroma = self._chroma_client.create_collection(
            self._chroma_name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def backend(self) -> str:
        return "chromadb" if self._chroma is not None else "numpy"

    def index(self, chunks: list[Chunk]):
        self.chunks = list(chunks)
        if not chunks:
            return
        self.matrix = self.embedder.encode([c.contextual for c in chunks])
        self.bm25.fit([tokenize(c.contextual) for c in chunks])
        if self._chroma_client is not None:
            self._reset_chroma()
            B = 256
            for i in range(0, len(chunks), B):
                part = chunks[i:i + B]
                self._chroma.add(
                    ids=[c.id for c in part],
                    embeddings=self.matrix[i:i + B].tolist(),
                    documents=[c.text for c in part],
                    metadatas=[{"source": c.source, "heading": c.heading, "page": c.page or 0} for c in part],
                )
        (self.persist_dir / "chunks.json").write_text(json.dumps([c.__dict__ for c in chunks]))

    def _dense(self, qvec: np.ndarray, n: int) -> list[tuple[int, float]]:
        if self._chroma is not None and len(self.chunks):
            try:
                res = self._chroma.query(
                    query_embeddings=[qvec.tolist()],
                    n_results=min(n, len(self.chunks)),
                )
                idx = {c.id: i for i, c in enumerate(self.chunks)}
                out = []
                for i, d in zip(res["ids"][0], res["distances"][0]):
                    if i in idx:
                        out.append((idx[i], 1 - d))
                if out:
                    return out
            except Exception as e:
                # Stale / wiped Chroma handle — fall back to in-memory vectors
                print(f"[kaleidra] Chroma query failed ({type(e).__name__}); using numpy dense index")
                self._chroma = None
        if self.matrix.size == 0:
            return []
        sims = self.matrix @ qvec
        order = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in order]

    def search(self, queries: list[str], n: int = 20, rrf_k: int = 60, mode: str = "hybrid") -> list[Hit]:
        """Multi-query hybrid retrieval. Rank lists from every (query x retriever) are fused with RRF."""
        hits: dict[int, Hit] = {}
        fused: Counter = Counter()

        def hit(i: int) -> Hit:
            return hits.setdefault(i, Hit(self.chunks[i]))

        for qi, q in enumerate(queries):
            weight = 1.0 if qi == 0 else 0.6  # original query outweighs rewrites / HyDE
            if mode in ("hybrid", "bm25"):
                s = self.bm25.scores(tokenize(q))
                for rank, i in enumerate(np.argsort(-s)[:n]):
                    if s[i] <= 0:
                        break
                    h = hit(int(i))
                    if h.bm25_rank is None or rank + 1 < h.bm25_rank:
                        h.bm25_rank, h.bm25_score = rank + 1, float(s[i])
                    fused[int(i)] += weight / (rrf_k + rank + 1)
            if mode in ("hybrid", "dense"):
                qv = self.embedder.encode([q], query=(qi == 0))[0]
                for rank, (i, sc) in enumerate(self._dense(qv, n)):
                    h = hit(i)
                    if h.dense_rank is None or rank + 1 < h.dense_rank:
                        h.dense_rank, h.dense_score = rank + 1, sc
                    fused[i] += weight / (rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:n]
        out = []
        for r, (i, sc) in enumerate(ranked, 1):
            h = hits[i]
            h.rrf, h.fused_rank = sc, r
            out.append(h)
        return out
