"""
HTTP API for rag_cyber_assistant — backend should call this instead of Ollama directly.

Run: uvicorn rag_api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager, redirect_stdout
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests.exceptions import RequestException

from rag_cyber_assistant import (
    AppConfig,
    build_bm25_index,
    chat_turn,
    load_config,
    load_local_chunks,
    model_label,
    warmup_ollama_gpu_probe,
    _validate_llm_config,
)

# Lifespan state (set after startup).
_cfg: AppConfig | None = None
_rows: list[dict[str, Any]] | None = None
_bm25: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _rows, _bm25
    cfg = load_config()
    _validate_llm_config(cfg)
    rows = load_local_chunks(cfg.chunks_path)
    if not rows:
        raise RuntimeError("No rows found in local chunks file.")
    idx = build_bm25_index(rows)

    if cfg.force_gpu and cfg.llm_backend == "ollama":
        try:
            warmup_ollama_gpu_probe(cfg)
        except (RequestException, RuntimeError) as exc:
            raise RuntimeError(f"FORCE_GPU startup check failed: {exc}") from exc

    _cfg, _rows, _bm25 = cfg, rows, idx
    yield
    _cfg, _rows, _bm25 = None, None, None


app = FastAPI(title="RAG Cyber Assistant API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Optional prior turns; roles user|assistant",
    )


class ChatResponse(BaseModel):
    response: str
    model: str


@app.get("/health")
async def health():
    ok = _cfg is not None and _rows is not None and _bm25 is not None
    return {"status": "ok" if ok else "starting", "chunks": len(_rows or [])}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    if _cfg is None or _rows is None or _bm25 is None:
        raise HTTPException(status_code=503, detail="RAG index not ready")

    cfg = _cfg
    rows = _rows
    bm25_index = _bm25

    history = [
        h for h in body.history[-16:] if h.get("role") in {"user", "assistant"} and h.get("content")
    ]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            answer = chat_turn(cfg, rows, bm25_index, body.message.strip(), history)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(response=answer, model=model_label(cfg))
