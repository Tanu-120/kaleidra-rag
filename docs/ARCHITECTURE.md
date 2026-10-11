# Kaleidra — System Architecture

## 1. Dataflow

```
            ┌──────────────────────── INGESTION ────────────────────────┐
files ─▶ parsers ─▶ PII redaction ─▶ heading-aware chunker ─▶ ┬▶ embeddings ─▶ ChromaDB (cosine)
(md/html/docx/                                                 ├▶ BM25 index
 pdf/csv)                                                      └▶ entity graph (NetworkX)

            ┌──────────────────────── QUERY ────────────────────────────┐
question ─▶ INPUT GUARD ──block──▶ refusal (trace shows halt)
              │ allow (PII-sanitized query)
              ▼
        query forge: original + rewrites + HyDE passage
              ▼
   BM25 ranks ┐
   dense ranks┴▶ multi-query Reciprocal Rank Fusion (w=1.0 original, 0.6 variants)
              ▼
        graph hop: query entities → neighbours → extra chunks
              ▼
        cross-encoder rerank ─▶ dedupe (shingle Jaccard, per-source cap)
              ▼
        CONTEXT GUARD (indirect injection quarantine)
              ▼
        generate (extractive | Ollama) ─▶ claim-level faithfulness
              ▼
        OUTPUT GUARD (PII re-scan, leak check) ─▶ response + full trace
```

## 2. Key design decisions

| Decision | Rationale |
|---|---|
| RRF instead of score fusion | BM25 and cosine scores live on different scales; ranks are comparable and need no tuning. `k=60` per the original paper. |
| Contextual chunk headers | Prepending `source > heading` to the embedded/BM25 text restores context lost by chunking and lifts recall on short sections. |
| Redact **before** indexing | A leaked vector or retrieved chunk can't expose PII that was never stored. Output guard is a second layer. |
| Guard retrieved text, not only the query | Indirect injection (instructions hidden in documents) is the realistic RAG attack. Suspicious sentences are replaced with a marker. |
| Abstention | If no sentence overlaps the query enough, the system says *I don't know* rather than hallucinating. |
| Graceful degradation | Every heavy dependency (HF models, Chroma, Presidio, Docling, Ollama) has a fallback, so the pipeline is always runnable and testable. |
| Trace-first pipeline | Each stage returns timings and rank metadata; the UI X-Ray is a pure rendering of the trace. |

## 3. Guardrail layers

| Layer | Where | Mechanism |
|---|---|---|
| Input injection | `guardrails.scan_injection` | 8 weighted regex rule families combined by noisy-OR; block at risk ≥ 0.6 |
| Content filter | `guardrails.content_filter` | topic rules (weapons, self-harm methods, malware) |
| PII | `detect_pii` | Presidio NER (PERSON/LOCATION) + regex w/ validation (Luhn for cards); India: Aadhaar, PAN, IFSC, +91 phones |
| Indirect injection | `sanitize_context` | sentence-level scan of retrieved chunks |
| Output | `check_output` | PII re-scan and system-prompt leak heuristic |
| Extension seam | `check_input` | add Llama Guard / classifier verdict into the same dict |

## 4. API contracts

All JSON. Errors use FastAPI's `{"detail": ...}` (422 for validation).

### `POST /api/ask`
```jsonc
// request
{ "question": "string (2-2000)", "mode": "hybrid|bm25|dense", "use_rerank": true, "use_hyde": true,
  "use_graph": true, "use_dedupe": true, "top_k": 5 }
// response (abridged)
{ "blocked": false, "sanitized_query": "...",
  "guard":  { "allowed": true, "injection": {"risk": 0.0, "reasons": []}, "pii": [], "sanitized_query": "..." },
  "variants": [{"kind": "original|rewrite|hyde", "text": "..."}],
  "candidates": [{"id","source","heading","text","bm25_rank","dense_rank","fused_rank","rerank_rank","rrf","rerank_score"}],
  "hits": [ /* final evidence, same shape */ ],
  "dropped": [{"id","reason"}], "quarantined": [{"id","risk","reasons"}],
  "graph": {"entities": [], "added": ["chunk ids"]}, "graph_view": {"nodes": [], "edges": []},
  "answer": { "mode": "extractive|llm|abstain",
              "claims": [{"text","cite": 1}],
              "faithfulness": {"score": 0.93, "claims": [{"text","support","best_source","grounded"}]} },
  "output_guard": {"pii": [], "leak_suspected": false},
  "trace": {"stages": [{"name","ms", "...stage info"}], "total_ms": 29.1} }
// when blocked: {"blocked": true, "guard": {...}, "message": "...", "answer": null, "hits": [], "trace": {...}}
```

### `POST /api/guard`  `{ "text": "..." }` → the `guard` object above (used for live typing feedback).
### `POST /api/upload`  multipart `files[]` → ingest summary. `POST /api/ingest` re-indexes the corpus directory.
### `GET /api/status` → `{version, ingest:{documents,chunks,pii_redacted,vector_store,embedder,reranker,graph}, docs, presidio}`
### `GET /api/graph` → `{nodes:[{id,mentions}], edges:[{source,target,weight}]}`
### `POST /api/eval`  `{ "n": 25, "k": 5 }`
```jsonc
{ "dataset_size": 25, "k": 5,
  "results": [{"name","n","hit","mrr","precision","recall","ndcg","faithfulness","latency_ms_p50","latency_ms_p95"}],
  "selection": {"winner": "...", "gates": {"min_faithfulness": 0.6, "max_p95_ms": 1500}, "passed_gates": true},
  "guardrails": {"injection": {"precision","recall","false_positives","false_negatives","misses"}, "pii": {"accuracy","cases"}} }
```

## 5. Evaluation methodology

1. **Synthetic dataset** — for each sampled chunk, a question is templated around an extracted entity or key terms; the chunk is the gold passage.
2. **Retrieval metrics** at *k*: Hit@k, MRR, Precision@k, Recall@k, nDCG@k.
3. **Faithfulness** — mean over answer claims of the fraction of content words supported by the best evidence passage.
4. **Benchmark** — six configs (BM25, Dense, Hybrid, Hybrid+HyDE, Hybrid+Rerank, Full pipeline) on identical questions.
5. **Selection for production** — filter by gates (faithfulness ≥ 0.6, p95 ≤ 1500 ms), maximise `MRR + nDCG`, tie-break on p50 latency.
6. **Red-team** — labelled attack/benign prompts → injection precision/recall; labelled strings → PII accuracy.

## 6. Configuration (env vars)

`KALEIDRA_BACKEND` (auto|hf|lite) · `KALEIDRA_EMBED_MODEL` · `KALEIDRA_RERANK_MODEL` · `KALEIDRA_CHUNK_SIZE` · `KALEIDRA_CHUNK_OVERLAP` ·
`KALEIDRA_CANDIDATES` · `KALEIDRA_TOP_K` · `KALEIDRA_DATA` · `KALEIDRA_STORE` · `KALEIDRA_OLLAMA_URL` · `KALEIDRA_OLLAMA_MODEL`

## 7. Scaling path

Swap `HybridStore` for Qdrant (native sparse+dense) · move the graph to Neo4j · replace the lexical faithfulness check with an
NLI/LLM judge · add Llama Guard to `check_input` · stream responses over SSE.
