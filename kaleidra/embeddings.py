"""Local embeddings. Uses a Hugging Face model; falls back to a deterministic hashing
embedder (char n-grams + words) so the whole pipeline still runs fully offline."""
from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    name = "hashing-lite"
    dim = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            if len(t) > 4:
                feats += [t[i:i + 4] for i in range(len(t) - 3)]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class HFEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.name = model
        self.dim = self.model.get_sentence_embedding_dimension()
        self._bge = "bge" in model.lower()

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query and self._bge:
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def load_embedder(model: str, backend: str = "auto"):
    if backend != "lite":
        try:
            return HFEmbedder(model)
        except Exception as e:  # offline / not installed
            if backend == "hf":
                raise
            print(f"[kaleidra] HF embedder unavailable ({type(e).__name__}); using offline hashing embedder")
    return HashingEmbedder()
