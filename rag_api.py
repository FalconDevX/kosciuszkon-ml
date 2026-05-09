"""
HTTP API for rag_cyber_assistant — backend should call this instead of Ollama directly.

Run: uvicorn rag_api:app --host 0.0.0.0 --port 8080

Chat — VirusTotal file scan runs **on this server** before the LLM (not OpenAI-style tool calling).
The LLM only sees JSON results in the prompt.

Ways to attach a file:
1) multipart/form-data: fields `message`, optional `history` (JSON string), optional `file` (binary).
2) application/json: same as ChatRequest plus optional `file_base64` (standard base64) and `file_name`.

If the UI sends only `{ "message": "check this file" }` without bytes, **no scan runs** — the model
will guess from RAG context only.
"""

from __future__ import annotations

import base64
import io
import json
import os
from contextlib import asynccontextmanager, redirect_stdout
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from requests.exceptions import RequestException
from starlette.datastructures import UploadFile

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
    file_base64: str | None = Field(
        default=None,
        description="Optional file as standard base64 (for JSON clients). Triggers VirusTotal file scan.",
    )
    file_name: str | None = Field(
        default=None,
        description="Original filename when using file_base64 (e.g. id_ed25519.pub).",
    )


class ChatResponse(BaseModel):
    response: str
    model: str


@app.get("/health")
async def health():
    ok = _cfg is not None and _rows is not None and _bm25 is not None
    return {"status": "ok" if ok else "starting", "chunks": len(_rows or [])}


def _normalize_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for h in raw[-16:]:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        content = h.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content.strip()})
    return out


def _decode_optional_base64_file(parsed: ChatRequest) -> tuple[bytes, str] | None:
    if not parsed.file_base64 or not parsed.file_base64.strip():
        return None
    cleaned = parsed.file_base64.strip()
    try:
        raw = base64.b64decode(cleaned, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid file_base64: {exc}") from exc
    if not raw:
        return None
    max_mb = max(1, int(os.getenv("VIRUSTOTAL_MAX_FILE_MB", "32")))
    max_bytes = max_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds VIRUSTOTAL_MAX_FILE_MB={max_mb}",
        )
    name = (parsed.file_name or "").strip() or "upload.bin"
    return (raw, name)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request):
    if _cfg is None or _rows is None or _bm25 is None:
        raise HTTPException(status_code=503, detail="RAG index not ready")

    cfg = _cfg
    rows = _rows
    bm25_index = _bm25

    uploaded: tuple[bytes, str] | None = None
    ct = (request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in ct:
        form = await request.form()
        msg_val = form.get("message")
        if msg_val is None or not str(msg_val).strip():
            raise HTTPException(status_code=422, detail="message is required")
        message = str(msg_val).strip()

        hist_val = form.get("history")
        hist_raw = hist_val if isinstance(hist_val, str) else "[]"
        try:
            history_payload = json.loads(hist_raw or "[]")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"invalid history JSON: {exc}") from exc
        history = _normalize_history(history_payload)

        up = form.get("file")
        if up is not None:
            if not isinstance(up, UploadFile):
                raise HTTPException(status_code=422, detail="file must be an upload")
            raw_bytes = await up.read()
            if raw_bytes:
                fname = up.filename or "upload.bin"
                uploaded = (raw_bytes, fname)
    else:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="invalid JSON body") from exc
        try:
            parsed = ChatRequest.model_validate(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        message = parsed.message.strip()
        history = _normalize_history(parsed.history)
        uploaded = _decode_optional_base64_file(parsed)

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            answer = chat_turn(cfg, rows, bm25_index, message, history, uploaded_file=uploaded)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(response=answer, model=model_label(cfg))
