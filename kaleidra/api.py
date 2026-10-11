"""FastAPI service. Run: uvicorn kaleidra.api:app --reload"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, evals, guardrails
from .config import ROOT, settings
from .pipeline import Kaleidra

app = FastAPI(title="Kaleidra RAG", version=__version__, description="Hybrid-search RAG with guardrails, graph and evals")
_state: dict = {"kaleidra": None, "ingest": None, "eval": None}


def get_kaleidra() -> Kaleidra:
    if _state["kaleidra"] is None:
        p = Kaleidra()
        _state["ingest"] = p.ingest_dir()
        _state["kaleidra"] = p
    return _state["kaleidra"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    mode: str = Field("hybrid", pattern="^(hybrid|bm25|dense)$")
    use_rerank: bool = True
    use_hyde: bool = True
    use_graph: bool = True
    use_dedupe: bool = True
    top_k: int = Field(5, ge=1, le=15)


class GuardRequest(BaseModel):
    text: str = Field(..., max_length=5000)


class EvalRequest(BaseModel):
    n: int = Field(25, ge=5, le=100)
    k: int = Field(5, ge=1, le=10)


@app.get("/api/status")
def status():
    p = get_kaleidra()
    return {"version": __version__, "ingest": _state["ingest"], "docs": p.doc_stats,
            "presidio": guardrails._get_presidio() is not None}


@app.post("/api/ask")
def ask(req: AskRequest):
    p = get_kaleidra()
    res = p.ask(req.question, mode=req.mode, use_rerank=req.use_rerank, use_hyde=req.use_hyde,
                use_graph=req.use_graph, use_dedupe=req.use_dedupe, top_k=req.top_k)
    focus = res.get("graph", {}).get("entities") or None
    res["graph_view"] = p.graph.to_json(focus)
    return res


@app.post("/api/guard")
def guard(req: GuardRequest):
    """Live guardrail inspection used by the UI's shield panel."""
    g = guardrails.check_input(req.text)
    return g


@app.post("/api/ingest")
def reingest():
    p = get_kaleidra()
    _state["ingest"] = p.ingest_dir()
    return _state["ingest"]


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    p = get_kaleidra()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "upload.txt").name
        with (settings.data_dir / name).open("wb") as out:
            shutil.copyfileobj(f.file, out)
    _state["ingest"] = p.ingest_dir()
    return _state["ingest"]


@app.get("/api/graph")
def graph():
    return get_kaleidra().graph.to_json()


@app.post("/api/eval")
def run_eval(req: EvalRequest):
    p = get_kaleidra()
    if not p.chunks:
        raise HTTPException(400, "Index is empty - ingest documents first")
    _state["eval"] = evals.benchmark(p, req.n, req.k)
    return _state["eval"]


@app.get("/api/eval")
def last_eval():
    return _state["eval"] or JSONResponse({"detail": "no evaluation run yet"}, status_code=404)


@app.get("/api/health")
def health():
    return {"ok": True}


STATIC = ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
