"""Kaleidra pipeline orchestrator. Every stage emits a trace the UI renders as the 'retrieval X-ray'."""
from __future__ import annotations

import re
import time
from pathlib import Path

from . import dedupe as dd
from . import generate, guardrails, query as qt
from .chunking import Chunk, chunk_sections
from .config import settings
from .embeddings import load_embedder
from .graph import KnowledgeGraph
from .parsing import PARSERS, parse
from .rerank import load_reranker, rerank
from .store import Hit, HybridStore


def _hit_dict(h, order: int | None = None) -> dict:
    return {
        "id": h.chunk.id,
        "source": h.chunk.source,
        "heading": h.chunk.heading,
        "text": h.chunk.text,
        "bm25_rank": h.bm25_rank,
        "dense_rank": h.dense_rank,
        "fused_rank": h.fused_rank,
        "rerank_rank": h.rerank_rank,
        "bm25_score": round(h.bm25_score, 3),
        "dense_score": round(h.dense_score, 3),
        "rrf": round(h.rrf, 5),
        "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 3),
    }


class Kaleidra:
    def __init__(self, backend: str | None = None):
        self.backend = backend or settings.backend
        self.embedder = load_embedder(settings.embed_model, self.backend)
        self.reranker = load_reranker(settings.rerank_model, self.backend)
        self.store = HybridStore(self.embedder, settings.store_dir)
        self.graph = KnowledgeGraph()
        self.chunks: list[Chunk] = []
        self.doc_stats: dict[str, dict] = {}

    # ------------------------------------------------------------ ingestion
    def ingest_dir(self, path: Path | None = None, redact_pii: bool = True) -> dict:
        path = Path(path or settings.data_dir)
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in PARSERS)
        return self.ingest_files(files, redact_pii)

    def ingest_files(self, files: list[Path], redact_pii: bool = True) -> dict:
        t0 = time.time()
        chunks: list[Chunk] = []
        pii_total = 0
        for f in files:
            try:
                sections = parse(f)
            except Exception as e:
                self.doc_stats[f.name] = {"error": str(e)}
                continue
            n_pii = 0
            if redact_pii:  # PII never enters the index
                for s in sections:
                    s.text, found = guardrails.redact(s.text)
                    n_pii += len(found)
            cs = chunk_sections(sections, f.name, settings.chunk_size, settings.chunk_overlap)
            chunks += cs
            pii_total += n_pii
            self.doc_stats[f.name] = {"chunks": len(cs), "pii_redacted": n_pii}
        self.chunks = chunks
        self.store.index(chunks)
        self.graph.build(chunks)
        return {
            "documents": len(self.doc_stats), "chunks": len(chunks), "pii_redacted": pii_total,
            "seconds": round(time.time() - t0, 2), "vector_store": self.store.backend,
            "embedder": self.embedder.name, "reranker": self.reranker.name,
            "graph": {"nodes": self.graph.g.number_of_nodes(), "edges": self.graph.g.number_of_edges()},
        }

    # ------------------------------------------------------------ query
    def ask(self, question: str, *, mode: str = "hybrid", use_rerank: bool = True, use_hyde: bool = True,
            use_graph: bool = True, use_dedupe: bool = True, top_k: int | None = None) -> dict:
        top_k = top_k or settings.top_k_final
        trace: dict = {"stages": []}
        t0 = time.time()

        def stage(name, t, **info):
            trace["stages"].append({"name": name, "ms": round((time.time() - t) * 1000, 1), **info})

        # 1. input guardrails
        t = time.time()
        gi = guardrails.check_input(question)
        stage("input_guard", t, allowed=gi["allowed"], risk=gi["injection"]["risk"])
        if not gi["allowed"]:
            return {"blocked": True, "guard": gi, "trace": trace, "answer": None, "hits": [],
                    "message": "Blocked by input guardrails (" + gi["block_reason"] + ")."}
        q = gi["sanitized_query"]

        # 2. query transformation
        t = time.time()
        variants = qt.transform(q, use_hyde=use_hyde and mode != "bm25", use_expand=True)
        stage("query_transform", t, variants=len(variants))

        # 3. hybrid retrieval
        t = time.time()
        hits = self.store.search([v["text"] for v in variants], n=settings.top_k_candidates, rrf_k=settings.rrf_k, mode=mode)
        stage("hybrid_retrieval", t, candidates=len(hits), mode=mode)

        # 4. graph expansion
        graph_info = {"entities": [], "added": []}
        if use_graph and self.chunks:
            t = time.time()
            ents, cids = self.graph.expand(q)
            have = {h.chunk.id for h in hits}
            by_id = {c.id: c for c in self.chunks}
            for cid in cids[:4]:
                if cid not in have and cid in by_id:
                    h = Hit(by_id[cid])
                    h.fused_rank = len(hits) + 1
                    hits.append(h)
                    graph_info["added"].append(cid)
            graph_info["entities"] = ents
            stage("graph_expansion", t, entities=len(ents), added=len(graph_info["added"]))

        # 5. rerank
        if use_rerank:
            t = time.time()
            hits = rerank(self.reranker, q, hits)
            stage("rerank", t, model=self.reranker.name)
        pre = [_hit_dict(h) for h in hits]  # full candidate set with every stage's rank, for the X-ray

        # 6. dedupe
        dropped: list[dict] = []
        if use_dedupe:
            t = time.time()
            hits, dropped = dd.dedupe(hits, settings.dedupe_threshold)
            stage("dedupe", t, dropped=len(dropped))
        hits = hits[:top_k]

        # 6b. topic boost: pull chunks that literally discuss the asked concept
        topic = None
        ql = q.lower()
        if re.search(r"\bmaterialized views?\b|\bviews?\b", ql):
            topic = ["materialized views", "regular views", "view in Snowflake", "CREATE VIEW"]
        elif re.search(r"\bvirtual warehouses?\b|\bwarehouses?\b", ql):
            topic = ["virtual warehouse", "warehouse compute", "auto-suspend"]
        if topic:
            t = time.time()
            have = {h.chunk.id for h in hits}
            added = 0
            for h in self.store.search(topic + [q], n=12, mode="hybrid"):
                if h.chunk.id in have:
                    continue
                if not re.search(r"\bviews?\b|\bwarehouse", h.chunk.text, re.I):
                    continue
                hits.append(h)
                have.add(h.chunk.id)
                added += 1
                if added >= 5:
                    break
            stage("topic_boost", t, added=added)

        # 6c. definitional questions: pull early-page / overview chunks the main ranker may miss
        if re.match(r"^\s*what\s+(is|are)\b", q, re.I):
            subj = re.sub(r"^\s*what\s+(is|are)\s+", "", q, flags=re.I).strip(" ?.")
            if subj:
                t = time.time()
                have = {h.chunk.id for h in hits}
                subj_l = subj.lower()
                # 1) lexical/hybrid extras
                extra = self.store.search(
                    [subj, f"{subj} architecture", f"{subj} overview", f"what is {subj}", f"{subj} three-layer"],
                    n=15, mode="hybrid",
                )
                # 2) direct corpus scan: early pages that mention the subject (reliable for PDFs)
                scanned = []
                for c in self.chunks:
                    if (c.page or 99) > 4:
                        continue
                    blob = (c.text[:500] + " " + (c.heading or "") + " " + c.source).lower()
                    if subj_l not in blob:
                        continue
                    scanned.append(Hit(c))
                pool = extra + scanned
                pool = sorted(
                    pool,
                    key=lambda h: (
                        0 if re.search(
                            r"architecture|overview|introduction|independent layers|cloud data|data platform|is a\b",
                            h.chunk.text[:280],
                            re.I,
                        ) else 1,
                        h.chunk.page or 99,
                        0 if subj_l in h.chunk.text[:120].lower() else 1,
                    ),
                )
                added = 0
                for h in pool:
                    if h.chunk.id in have:
                        continue
                    hits.append(h)
                    have.add(h.chunk.id)
                    added += 1
                    if added >= 6:
                        break
                stage("definition_boost", t, added=added, subject=subj)

        # 7. indirect-injection sanitization of retrieved context
        t = time.time()
        quarantined = []
        for h in hits:
            clean, v = guardrails.sanitize_context(h.chunk.text)
            if v.risk > 0:
                h.chunk = type(h.chunk)(**{**h.chunk.__dict__, "text": clean})
                quarantined.append({"id": h.chunk.id, "risk": v.risk, "reasons": v.reasons})
        stage("context_guard", t, quarantined=len(quarantined))

        # 8. generation + faithfulness
        t = time.time()
        ans = generate.answer(q, hits)
        stage("generate", t, mode=ans["mode"])

        # 9. output guardrails
        t = time.time()
        full = " ".join(c["text"] for c in ans["claims"])
        go = guardrails.check_output(full)
        if go["pii"]:
            for c in ans["claims"]:
                c["text"], _ = guardrails.redact(c["text"])
        stage("output_guard", t, pii=len(go["pii"]))

        trace["total_ms"] = round((time.time() - t0) * 1000, 1)
        return {
            "blocked": False, "question": question, "sanitized_query": q, "guard": gi,
            "variants": variants, "candidates": pre, "hits": [_hit_dict(h) for h in hits],
            "dropped": dropped, "graph": graph_info, "quarantined": quarantined,
            "answer": ans, "output_guard": {"pii": go["pii"], "leak_suspected": go["leak_suspected"]},
            "trace": trace,
        }
