# Kaleidra — Hybrid RAG with a Retrieval X-Ray

> Production-Grade AI Engineering Bootcamp · Week 3 · Advanced RAG & Guardrails

Kaleidra is a **fully local** hybrid-search RAG pipeline with an **Optical Bench** UI. A query enters as one beam of light and is *refracted*
into BM25, dense, HyDE and knowledge-graph retrievals, recombined with Reciprocal Rank Fusion, re-ordered by a
cross-encoder, de-duplicated, guarded, and answered with **claim-level grounding** — and the UI lets you *see* every
stage re-order the candidates.

## What makes it different

| Idea | What you see |
|---|---|
| **Retrieval X-Ray** | A slope chart showing each candidate's rank at BM25 → Dense → RRF → Cross-encoder. Hover a line, its evidence card lights up. |
| **Live Guard Shield** | As you type, PII is highlighted and an injection-risk meter moves in real time. Attack presets are one click away. |
| **Claim-level faithfulness** | Every answer sentence carries a citation chip and an evidence-support bar; the gauge is the mean support. The system abstains rather than guess. |
| **Poisoned-document defense** | The sample corpus contains a planted *indirect* prompt injection. It is detected in retrieved context and quarantined before generation. |
| **PII never indexed** | Redaction runs at ingestion, so emails/PAN/cards/phones never reach the vector store (India-aware: Aadhaar, PAN, IFSC, +91). |
| **Ablation toggles** | Turn HyDE / cross-encoder / graph hop / dedupe on and off and watch the trace and results change. |
| **Eval Lab** | One click: synthetic dataset from *your* corpus → 6 configs benchmarked → production pick under quality/latency gates → guardrail red-team. |

## Quick start

```bash
git clone <your-repo> kaleidra-rag && cd kaleidra-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # optional, enables Presidio NER (PERSON/LOCATION)

uvicorn kaleidra.api:app --port 8000               # open http://localhost:8000
```

First run downloads two small open models from Hugging Face (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`),
then everything is offline. No API keys.

**No internet / no GPU?** `KALEIDRA_BACKEND=lite uvicorn kaleidra.api:app` swaps in a hashing embedder and lexical reranker so the
entire pipeline, UI and evals still run (used by the test-suite).

Optional local LLM: `KALEIDRA_OLLAMA_URL=http://localhost:11434 KALEIDRA_OLLAMA_MODEL=llama3.2` enables abstractive answers,
LLM-written HyDE and query rewrites. Without it, answers are extractive (every sentence is a quote from a source).

## Commands

```bash
pytest -q                          # 13 tests: chunking, PII, injection, retrieval, evals, API contract
python scripts/run_eval.py         # CLI benchmark, prints a leaderboard + writes eval_report.json
bash scripts/demo.sh               # curl walkthrough used in the 2-minute demo
```

## Project layout

```
kaleidra/
  parsing.py     md/html/docx/pdf (Docling if installed, else pypdf)/legacy csv -> heading-aware Sections
  chunking.py    sentence-snapped chunks with overlap + contextual headers ("source > heading")
  embeddings.py  HF sentence-transformers (BGE) | offline hashing fallback
  store.py       ChromaDB (cosine) + in-house BM25, multi-query RRF fusion
  query.py       multi-query rewrite + HyDE (Ollama or heuristic)
  rerank.py      cross-encoder reranker | lexical fallback
  dedupe.py      shingle-Jaccard near-duplicate removal + per-source cap
  graph.py       NetworkX entity co-occurrence graph, 1-hop expansion retrieval
  guardrails.py  PII (Presidio + regex), injection rules, content filter, context sanitizer, output scan
  generate.py    extractive / Ollama answers + claim-level faithfulness
  evals.py       synthetic dataset, Hit/MRR/P/R/nDCG, config benchmark, selection rule, red-team
  pipeline.py    orchestrator that emits a per-stage trace
  api.py         FastAPI service
static/index.html  single-file UI (no build step)
data/corpus/       sample docs incl. PII and a poisoned page
docs/ARCHITECTURE.md  architecture, API contracts, design decisions
```

## API at a glance

`POST /api/ask` · `POST /api/guard` · `POST /api/upload` · `POST /api/ingest` · `GET /api/graph` · `POST /api/eval` · `GET /api/status`
— full request/response contracts in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Interactive docs at `/docs`.

## Honest limitations

- The synthetic eval questions are generated from the chunks themselves, which **favours lexical retrieval** (BM25 often wins in
  `lite` mode). With real BGE + cross-encoder models and paraphrased/human questions the gap changes — treat the benchmark as a
  regression harness, and add a hand-labelled set before trusting absolute numbers.
- Injection defense is rule-based (fast, explainable, ~no false positives on the red-team set) but not exhaustive. For production,
  add Llama Guard / a classifier as a second layer — the `guardrails.check_input` seam is designed for that.
- Faithfulness is lexical-support based, not an NLI model.

## 2-minute demo script

1. **0:00** Open the UI, point at the status chips (Chroma + local models + Presidio).
2. **0:15** Ask *"How does the retry mechanism work?"* — show the X-Ray re-ordering and grounded claims.
3. **0:45** Toggle cross-encoder off/on; hover an X-Ray line to light up its evidence.
4. **1:00** Type an email + card number — PII highlights live. Click the red *DAN mode* preset — pipeline halts at the guard.
5. **1:20** Ask about the security runbook — show the *injection quarantined* badge on the poisoned source.
6. **1:35** Eval Lab → Run Benchmark → leaderboard and the production pick.
7. **1:55** Close on the repo layout.
