# Hybrid Retrieval in Kaleidra

Hybrid retrieval combines lexical and semantic search so both exact terms and paraphrases can surface the right passage.

## BM25 and dense search

BM25 scores keyword overlap and is strong on identifiers, error codes, and exact product phrases. Dense embeddings (default model BAAI/bge-small-en-v1.5 when available) score meaning and help when the user paraphrases the source wording. Kaleidra runs both retrievers and merges their ranked lists.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges ranks without forcing BM25 scores and cosine similarities onto one scale. For each candidate, Kaleidra adds `1 / (k + rank)` across lists, with `k = 60`. The original query is weighted higher than rewrite and HyDE variants.

## HyDE and query rewrites

Query forge produces the original question, lighter lexical rewrites, and an optional HyDE passage — a hypothetical answer-like paragraph used as an extra dense query. HyDE often improves recall on vague questions. Operators can disable HyDE, the graph hop, the cross-encoder, or dedupe from the Ask workspace toggles.

## Cross-encoder reranking and dedupe

After fusion, a cross-encoder (ms-marco-MiniLM-L-6-v2 when available) scores query–passage pairs jointly and reorders candidates. Near-duplicate passages are removed with shingle Jaccard similarity, and a per-source cap keeps one document from dominating the context window.

## Retry behaviour in the pipeline client path

When a retrieval or model call fails transiently during local development, operators should retry the request up to five times with exponential backoff starting at 100 milliseconds. The retry mechanism doubles the delay after every failed attempt and adds random jitter of up to twenty percent to avoid thundering herds against the local API.
