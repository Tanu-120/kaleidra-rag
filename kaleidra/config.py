"""Central configuration. Override any value with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("KALEIDRA_DATA", str(ROOT / "data" / "corpus"))))
    store_dir: Path = field(default_factory=lambda: Path(_env("KALEIDRA_STORE", str(ROOT / ".kaleidra_store"))))
    embed_model: str = _env("KALEIDRA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    rerank_model: str = _env("KALEIDRA_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # "auto" -> try HF models, silently fall back to offline lexical backends
    backend: str = _env("KALEIDRA_BACKEND", "auto")  # auto | hf | lite
    chunk_size: int = int(_env("KALEIDRA_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(_env("KALEIDRA_CHUNK_OVERLAP", "120"))
    top_k_candidates: int = int(_env("KALEIDRA_CANDIDATES", "20"))
    top_k_final: int = int(_env("KALEIDRA_TOP_K", "5"))
    rrf_k: int = 60
    dedupe_threshold: float = float(_env("KALEIDRA_DEDUPE", "0.88"))
    ollama_url: str = _env("KALEIDRA_OLLAMA_URL", "")  # e.g. http://localhost:11434
    ollama_model: str = _env("KALEIDRA_OLLAMA_MODEL", "llama3.2")


settings = Settings()
