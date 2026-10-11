# Kaleidra Overview

Kaleidra is a local hybrid-search Retrieval-Augmented Generation (RAG) console built for the Production-Grade AI Engineering Bootcamp Week 3 exercise. It answers questions from indexed project documents instead of inventing facts from model memory.

## What runs locally

The service is a FastAPI application. Documents are parsed, PII-redacted, chunked, and indexed into three structures at once: a dense vector store (ChromaDB), a BM25 keyword index, and a NetworkX entity co-occurrence graph. Queries pass through input guards, query rewriting, hybrid retrieval, optional graph expansion, cross-encoder reranking, context sanitization, answer generation, and output guards.

## Design goals

Kaleidra is built to be inspectable. Every pipeline stage emits timings and rank metadata so the dashboard can show a retrieval X-Ray. The system prefers abstention over unsupported answers. Heavy dependencies degrade gracefully: if Hugging Face models are unavailable, hashing embeddings and a lexical reranker keep the pipeline runnable.

## Operators

Start the API with uvicorn on port 8000. The browser UI is served from `/`. Interactive OpenAPI docs are available at `/docs`. No cloud API keys are required for the default local path.
