"""FastAPI service. Run: uvicorn kaleidra.api:app --reload"""
from __future__ import annotations

import json
import shutil
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
EVAL_PATH = settings.store_dir / "last_eval.json"


def get_kaleidra() -> Kaleidra:
    if _state["kaleidra"] is None:
        p = Kaleidra()
        _state["ingest"] = p.ingest_dir()
        _state["kaleidra"] = p
        if EVAL_PATH.exists() and _state["eval"] is None:
            try:
                _state["eval"] = json.loads(EVAL_PATH.read_text())
            except Exception:
                pass
    return _state["kaleidra"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    mode: str = Field("hybrid", pattern="^(hybrid|bm25|dense)$")
    use_rerank: bool = True
    use_hyde: bool = True
    use_graph: bool = True
    use_dedupe: bool = True
    top_k: int = Field(5, ge=1, le=15)


class ConfigFlags(BaseModel):
    label: str = "config"
    mode: str = Field("hybrid", pattern="^(hybrid|bm25|dense)$")
    use_rerank: bool = True
    use_hyde: bool = True
    use_graph: bool = True
    use_dedupe: bool = True
    top_k: int = Field(5, ge=1, le=15)


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    a: ConfigFlags = Field(default_factory=lambda: ConfigFlags(label="Full pipeline"))
    b: ConfigFlags = Field(default_factory=lambda: ConfigFlags(
        label="BM25 only", mode="bm25", use_rerank=False, use_hyde=False, use_graph=False, use_dedupe=True
    ))


class GuardRequest(BaseModel):
    text: str = Field(..., max_length=5000)


class EvalRequest(BaseModel):
    n: int = Field(25, ge=5, le=100)
    k: int = Field(5, ge=1, le=10)


def _run_ask(p: Kaleidra, question: str, cfg: ConfigFlags | AskRequest) -> dict:
    res = p.ask(
        question,
        mode=cfg.mode,
        use_rerank=cfg.use_rerank,
        use_hyde=cfg.use_hyde,
        use_graph=cfg.use_graph,
        use_dedupe=cfg.use_dedupe,
        top_k=cfg.top_k,
    )
    focus = res.get("graph", {}).get("entities") or None
    res["graph_view"] = p.graph.to_json(focus)
    return res


@app.get("/api/status")
def status():
    p = get_kaleidra()
    return {
        "version": __version__,
        "ingest": _state["ingest"],
        "docs": p.doc_stats,
        "presidio": guardrails._get_presidio() is not None,
        "has_eval": _state["eval"] is not None,
        "features": ["hybrid", "xray", "guardrails", "graph", "evals", "compare", "upload-in-ask"],
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        p = get_kaleidra()
        if not p.chunks:
            raise HTTPException(503, "Corpus is empty or still indexing — open Documents and re-ingest, then retry.")
        return _run_ask(p, req.question, req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ask failed: {type(e).__name__}: {e}") from e


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Ablation compare: same question, two pipeline configs — ideal for demos/judges."""
    try:
        p = get_kaleidra()
        if not p.chunks:
            raise HTTPException(503, "Corpus is empty — ingest documents first.")
        left = _run_ask(p, req.question, req.a)
        right = _run_ask(p, req.question, req.b)
        return {
            "question": req.question,
            "a": {"label": req.a.label, "config": req.a.model_dump(), "result": left},
            "b": {"label": req.b.label, "config": req.b.model_dump(), "result": right},
            "delta": {
                "faithfulness": round(
                    (left.get("answer") or {}).get("faithfulness", {}).get("score", 0)
                    - (right.get("answer") or {}).get("faithfulness", {}).get("score", 0),
                    3,
                ),
                "latency_ms": round(
                    (left.get("trace") or {}).get("total_ms", 0) - (right.get("trace") or {}).get("total_ms", 0),
                    1,
                ),
                "confidence": round(
                    (left.get("answer") or {}).get("confidence", {}).get("score", 0)
                    - (right.get("answer") or {}).get("confidence", {}).get("score", 0),
                    3,
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Compare failed: {type(e).__name__}: {e}") from e


@app.post("/api/guard")
def guard(req: GuardRequest):
    """Live guardrail inspection used by the UI's shield panel."""
    return guardrails.check_input(req.text)


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
    settings.store_dir.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(_state["eval"], indent=2))
    return _state["eval"]


@app.get("/api/eval")
def last_eval():
    if _state["eval"] is None and EVAL_PATH.exists():
        try:
            _state["eval"] = json.loads(EVAL_PATH.read_text())
        except Exception:
            pass
    return _state["eval"] or JSONResponse({"detail": "no evaluation run yet"}, status_code=404)


@app.get("/api/eval/export")
def export_eval():
    data = _state["eval"]
    if data is None and EVAL_PATH.exists():
        data = json.loads(EVAL_PATH.read_text())
    if not data:
        raise HTTPException(404, "no evaluation run yet")
    path = settings.store_dir / "eval_report.json"
    path.write_text(json.dumps(data, indent=2))
    return FileResponse(path, filename="kaleidra_eval_report.json", media_type="application/json")


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__}


STATIC = ROOT / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    html = STATIC / "index.html"
    # Bust browser/proxy caches so UI motion updates always land after refresh
    etag = f'W/"{html.stat().st_mtime_ns}-{html.stat().st_size}"'
    return FileResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "ETag": etag,
        },
    )
