"""Knowledge graph built with NetworkX: entities co-occurring in chunks become weighted edges.
Used for (1) graph-expansion retrieval, (2) the interactive graph in the UI."""
from __future__ import annotations

import re
from collections import Counter

import networkx as nx

from .chunking import Chunk

_CAP = re.compile(r"\b(?:[A-Z][a-zA-Z0-9]+(?:[ -][A-Z][a-zA-Z0-9]+){0,3})\b")
_ACR = re.compile(r"\b[A-Z]{2,6}\b")
_NOISE = set("The This That These Those It In On At For And But Or If When While We You They He She A An As By Of To From With Each Every Some Any All Our Your Their If Section Table Figure Note Use Using Also However Because".split())


def extract_entities(text: str) -> list[str]:
    ents = []
    for m in _CAP.findall(text):
        words = [w for w in m.split() if w not in _NOISE]
        e = " ".join(words)
        if len(e) > 2 and not (len(words) == 1 and m == text[: len(m)]):
            ents.append(e)
    ents += [a for a in _ACR.findall(text) if a not in _NOISE and len(a) >= 2]
    return ents


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.Graph()
        self.entity_chunks: dict[str, set[str]] = {}

    def build(self, chunks: list[Chunk]):
        self.g.clear()
        self.entity_chunks = {}
        counts: Counter = Counter()
        per_chunk: dict[str, list[str]] = {}
        for c in chunks:
            ents = list(dict.fromkeys(extract_entities(c.text)))
            per_chunk[c.id] = ents
            counts.update(ents)
        keep = {e for e, n in counts.items() if n >= 2 or " " in e}
        for c in chunks:
            ents = [e for e in per_chunk[c.id] if e in keep][:12]
            for e in ents:
                self.entity_chunks.setdefault(e, set()).add(c.id)
                if e not in self.g:
                    self.g.add_node(e, mentions=0, sources=set())
                self.g.nodes[e]["mentions"] += 1
                self.g.nodes[e]["sources"].add(c.source)
            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    w = self.g.get_edge_data(a, b, {}).get("weight", 0)
                    self.g.add_edge(a, b, weight=w + 1)

    def expand(self, query: str, hops: int = 1, limit: int = 6) -> tuple[list[str], list[str]]:
        """Entities in the query -> neighbours -> chunk ids reachable through the graph."""
        q = query.lower()
        seeds = [e for e in self.g.nodes if e.lower() in q]
        reach: Counter = Counter()
        for s in seeds:
            for nb, d in nx.single_source_shortest_path_length(self.g, s, cutoff=hops).items():
                if nb != s:
                    reach[nb] += self.g[s][nb]["weight"] if d == 1 else 0.5
        neighbours = [e for e, _ in reach.most_common(limit)]
        chunk_ids: list[str] = []
        for e in seeds + neighbours:
            for cid in self.entity_chunks.get(e, ()):
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
        return seeds + neighbours, chunk_ids

    def to_json(self, focus: list[str] | None = None, max_nodes: int = 60) -> dict:
        if focus:
            nodes = set(focus)
            for f in list(focus):
                if f in self.g:
                    nodes |= set(self.g[f])
        else:
            nodes = {n for n, _ in sorted(self.g.degree, key=lambda x: -x[1])[:max_nodes]}
        nodes = set(list(nodes)[:max_nodes])
        sub = self.g.subgraph(nodes)
        return {
            "nodes": [{"id": n, "mentions": self.g.nodes[n]["mentions"], "focus": bool(focus and n in focus)} for n in sub.nodes],
            "edges": [{"source": a, "target": b, "weight": d["weight"]} for a, b, d in sub.edges(data=True)],
        }
