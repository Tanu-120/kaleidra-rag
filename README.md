# Kaleidra

**Hybrid Retrieval-Augmented Generation (RAG) with visible retrieval, guardrails, and evaluation.**

Kaleidra is a local, production-shaped RAG system: documents are parsed, PII-redacted, chunked, and indexed for hybrid search; questions are guarded, rewritten, retrieved, reranked, and answered with cited claims and a confidence score. A single-page dashboard exposes every stage so you can inspect *why* an answer was produced.

No cloud API keys are required. The default path runs fully offline in `lite` mode; optional Hugging Face and Ollama backends upgrade embedding quality and answer style without changing the pipeline contract.

---

## Features

| Area | Capability |
|---|---|
| **Ingestion** | Markdown, HTML, DOCX, PDF, CSV → heading-aware chunks with contextual headers |
| **Hybrid retrieval** | BM25 + dense vectors → Reciprocal Rank Fusion (RRF, `k=60`) |
| **Query enrichment** | Lexical rewrites + HyDE (hypothetical document) multi-query fusion |
| **Reranking** | Cross-encoder (or lexical fallback) over fused candidates |
| **Knowledge graph** | NetworkX entity co-occurrence with 1-hop chunk expansion |
| **Context hygiene** | Shingle Jaccard dedupe, per-source caps, poisoned-context quarantine |
| **Generation** | Extractive cited answers by default; optional Ollama abstractive answers with claim grounding |
| **Guardrails** | Input injection shield, India-aware + Presidio PII, output re-scan |
| **Evaluation** | Hit@k / MRR / nDCG / faithfulness, ablation configs, production pick, JSON export |
| **Product UI** | Sidebar dashboard: Ask, Compare, Eval Lab, Guards, Docs, Demo Mode, retrieval X-Ray |

---

## Architecture

```
INGESTION
  files → parsers → PII redaction → chunker
       → embeddings → ChromaDB
       → BM25 index
       → entity graph (NetworkX)

QUERY
  question → input guard (block / sanitize)
          → query forge (original + rewrites + HyDE)
          → BM25 ∥ dense → RRF
          → graph hop
          → rerank → dedupe
          → context guard
          → generate → claim faithfulness
          → output guard → response + trace
```

Design notes (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for contracts and rationale):

- **RRF** fuses sparse and dense ranks without score calibration.
- **PII is redacted before indexing** so vectors and chunks never store raw secrets.
- **Guards apply to retrieved text**, not only the user query (indirect injection).
- **Abstention** when evidence is weak — the system prefers “I don’t know” over hallucination.
- Every heavy dependency has a **fallback**, so CI and demos stay runnable offline.

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic v2 |
| Sparse search | `rank-bm25` (Okapi BM25) |
| Dense store | ChromaDB (cosine); in-memory NumPy fallback if needed |
| Embeddings | `sentence-transformers` → **BAAI/bge-small-en-v1.5** (default) |
| Reranker | `sentence-transformers` CrossEncoder → **cross-encoder/ms-marco-MiniLM-L-6-v2** |
| Graph | NetworkX |
| Parsing | pypdf, python-docx, BeautifulSoup |
| PII | Microsoft Presidio (+ spaCy NER optional), regex with Luhn / India patterns |
| Generation | Extractive synthesis (default); Ollama HTTP API (optional) |
| Frontend | Single-file dashboard (`static/index.html`) — no build step |
| Tests | pytest, httpx |

---

## Models & backends

Kaleidra separates **pipeline logic** from **model backends**. Switch quality with environment variables; the API and UI stay the same.

### Embedding

| Backend | Model / method | Notes |
|---|---|---|
| `hf` / `auto` (success) | `BAAI/bge-small-en-v1.5` | ~33M params; query prefix applied for BGE; normalized vectors |
| `lite` / `auto` (fallback) | Hashing embedder (512-d) | Deterministic char n-grams + word features; no downloads |

Override: `KALEIDRA_EMBED_MODEL=<huggingface-id>`

### Reranking

| Backend | Model / method | Notes |
|---|---|---|
| `hf` / `auto` (success) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pairwise relevance scoring |
| `lite` / `auto` (fallback) | Lexical-overlap reranker | Term coverage + proximity density |

Override: `KALEIDRA_RERANK_MODEL=<huggingface-id>`

### Answer generation

| Mode | When | Behavior |
|---|---|---|
| **Extractive** (default) | Always available | Selects grounded sentences from retrieved chunks; each claim carries a citation and faithfulness score |
| **LLM (Ollama)** | `KALEIDRA_OLLAMA_URL` set | Abstractive answer via local Ollama (`KALEIDRA_OLLAMA_MODEL`, default `llama3.2`), then claim-level grounding against evidence |

### Backend selection

| `KALEIDRA_BACKEND` | Behavior |
|---|---|
| `lite` | Offline hashing embedder + lexical rerank — instant demos and CI |
| `auto` (default) | Try Hugging Face models; fall back to lite if download/load fails |
| `hf` | Require Hugging Face models; fail loudly if unavailable |

Optional NER upgrade for Presidio:

```bash
python -m spacy download en_core_web_sm
```

---

## Quick start

```bash
cd kaleidra-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fast offline demo
KALEIDRA_BACKEND=lite uvicorn kaleidra.api:app --port 8000

# Stronger retrieval (downloads models once, then works offline)
# KALEIDRA_BACKEND=hf uvicorn kaleidra.api:app --port 8000

# Optional abstractive answers (local Ollama)
# KALEIDRA_OLLAMA_URL=http://localhost:11434 KALEIDRA_OLLAMA_MODEL=llama3.2 \
#   uvicorn kaleidra.api:app --port 8000
```

Open **http://127.0.0.1:8000** · interactive OpenAPI at **/docs**

```bash
pytest -q                     # unit + API contract tests
python scripts/run_eval.py    # CLI leaderboard → eval_report.json
bash scripts/demo.sh          # curl walkthrough
```

Demo script for judges: [docs/DEMO.md](docs/DEMO.md)

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `KALEIDRA_BACKEND` | `auto` | `auto` \| `hf` \| `lite` |
| `KALEIDRA_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model |
| `KALEIDRA_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `KALEIDRA_CHUNK_SIZE` | `700` | Target chunk size (chars) |
| `KALEIDRA_CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `KALEIDRA_CANDIDATES` | `20` | Pre-rerank candidate pool |
| `KALEIDRA_TOP_K` | `5` | Final evidence count |
| `KALEIDRA_DEDUPE` | `0.88` | Jaccard dedupe threshold |
| `KALEIDRA_DATA` | `data/corpus` | Corpus directory |
| `KALEIDRA_STORE` | `.kaleidra_store` | Persistent index directory |
| `KALEIDRA_OLLAMA_URL` | _(empty)_ | Enable Ollama generation |
| `KALEIDRA_OLLAMA_MODEL` | `llama3.2` | Ollama model name |

---

## Project layout

```
kaleidra/
  api.py          FastAPI routes
  pipeline.py     End-to-end ask / compare orchestration
  parsing.py      Document loaders
  chunking.py     Heading-aware chunker
  embeddings.py   HF + hashing embedders
  store.py        BM25 + dense hybrid store
  query.py        Rewrites / HyDE
  rerank.py       Cross-encoder + lexical rerank
  graph.py        Entity co-occurrence graph
  dedupe.py       Near-duplicate filtering
  generate.py     Extractive / Ollama answers + faithfulness
  guardrails.py   Injection, PII, content, output checks
  evals.py        Synthetic benchmarks + selection gates
  config.py       Environment-driven settings
static/index.html Dashboard UI
data/corpus/      Seed docs, fixtures, uploads
docs/             Architecture & demo scripts
scripts/          Eval CLI + curl demo
tests/            Contract and unit tests
```

---

## API overview

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/ask` | Full RAG ask (guards, retrieve, answer, trace) |
| `POST` | `/api/compare` | Ablation: same question, two configs, metric deltas |
| `POST` | `/api/guard` | Live PII / injection scan |
| `POST` | `/api/upload` | Multipart file ingest |
| `POST` | `/api/ingest` | Re-index corpus directory |
| `POST` | `/api/eval` | Run benchmark + production selection |
| `GET` | `/api/eval/export` | Download last eval JSON report |
| `GET` | `/api/status` | Health, ingest stats, active models |
| `GET` | `/api/graph` | Entity graph for visualization |

Full request/response schemas: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Evaluation

1. **Synthetic Q/A** — questions templated from chunk entities; gold passage = source chunk  
2. **Retrieval metrics** — Hit@k, MRR, Precision@k, Recall@k, nDCG@k  
3. **Faithfulness** — lexical support of answer claims against evidence  
4. **Ablations** — BM25, Dense, Hybrid, +HyDE, +Rerank, Full pipeline  
5. **Production pick** — gates (e.g. faithfulness ≥ 0.6, p95 ≤ 1500 ms), then maximize MRR + nDCG  
6. **Red-team** — injection precision/recall and PII detection accuracy on labelled fixtures  

---

## Guardrails

| Layer | Mechanism |
|---|---|
| Input injection | Weighted regex rule families (noisy-OR); block at risk ≥ 0.6 |
| Content filter | Topic rules for disallowed intents |
| PII | Presidio NER + validated regex (cards, Aadhaar, PAN, IFSC, phones) |
| Indirect injection | Sentence-level scan of retrieved chunks; quarantine markers |
| Output | PII re-scan and prompt-leak heuristics |

Extension seam: add a classifier or Llama Guard into `check_input` without changing the API shape.

---

## Limitations

- In `lite` mode, synthetic evals skew toward lexical match; use HF embeddings before trusting absolute leaderboard ranks.  
- Injection defense is rule-based (fast and explainable); production deployments should add a dedicated classifier on the same guard seam.  
- Default faithfulness is lexical support, not NLI or an LLM-as-judge.  
- Extractive answers prioritize grounding over fluency; enable Ollama when narrative style matters.

---

## License

Bootcamp exercise project — free to use and extend for learning and demonstration.
