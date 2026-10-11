# Evaluation and API

## Automated evaluation

The Eval Lab builds a synthetic question set from indexed chunks, then benchmarks six pipeline configurations: BM25 only, dense only, hybrid, hybrid with HyDE, hybrid with rerank, and the full pipeline. Metrics include Hit@k, MRR, Precision@k, Recall@k, nDCG@k, claim faithfulness, and latency percentiles.

## Production selection rule

Configs must pass gates of faithfulness at least 0.6 and p95 latency at most 1500 milliseconds. Among configs that pass, Kaleidra maximises `MRR + nDCG` and breaks ties with lower p50 latency. Guardrail red-team results report injection precision/recall and PII detection accuracy on labelled cases.

## HTTP API contracts

`POST /api/ask` accepts a question plus toggles for mode, HyDE, rerank, graph, and dedupe. The response includes guard verdicts, query variants, candidate ranks for the X-Ray, final hits, quarantined ids, claim-level answers, faithfulness, and a stage trace.

`POST /api/guard` powers the live shield while typing. `POST /api/upload` accepts multipart files into the corpus directory and re-indexes. `POST /api/ingest` rebuilds from disk. `GET /api/status` returns version, ingest summary, per-document stats, and whether Presidio is active. `POST /api/eval` runs the benchmark.

## SLA note for local demos

For classroom demos on a laptop, treat interactive Ask latency under one second in lite mode as the target experience. Full Hugging Face embedding and rerank models are slower on first load because weights download once, then run offline.
