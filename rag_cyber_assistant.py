#!/usr/bin/env python3
import os
import sys
import io
import json
import hashlib
import base64
import re
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from requests.exceptions import RequestException, ReadTimeout

import requests


def _configure_stdio_utf8() -> None:
    """Avoid Polish mojibake (narzÄdzie) when UTF-8 is printed but console expects CP1252."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    enc = getattr(sys.stdout, "encoding", None) or ""
    if enc.lower().replace("-", "") == "utf8":
        return
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    else:
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            pass


_configure_stdio_utf8()
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi


SYSTEM_PROMPT = """You are an advanced cybersecurity AI assistant.

Your goals:
- analyze threats carefully
- reason step-by-step internally
- provide intelligent cybersecurity analysis
- explain risks clearly
- avoid generic responses

Rules:
- prioritize defensive cybersecurity
- never assist offensive or illegal actions
- off-topic, abusive, or harassing input: refuse in one or two short sentences only — no long sermons or reused CONTEXT dumping
- if external evidence exists, use it critically
- do not blindly trust user claims
- think carefully before answering

Style:
- natural
- analytical
- concise but insightful
- avoid repetitive templates

Grounding: When CONTEXT or attached external scan results are provided, weigh them against the question. If evidence is missing or insufficient, say so instead of inventing facts.
"""


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


@dataclass
class AppConfig:
    chunks_path: str
    top_k: int
    llm_backend: str
    ollama_url: str
    ollama_model: str
    openai_base_url: str
    openai_model: str
    openai_api_key: str
    ollama_num_ctx: int
    ollama_num_predict: int
    ollama_temperature: float
    ollama_num_gpu: int
    ollama_num_thread: int
    ollama_timeout_secs: int
    ollama_retries: int
    ollama_stream: bool
    max_context_chars: int
    virustotal_api_key: str
    force_gpu: bool


def load_config() -> AppConfig:
    load_dotenv()

    chunks_path = os.getenv("CHUNKS_PATH", "data/ouch_dataset/processed/chunks.jsonl").strip()
    if not Path(chunks_path).exists():
        raise ValueError(
            f"CHUNKS_PATH does not exist: {chunks_path}. "
            "Point CHUNKS_PATH to your local chunks JSONL file."
        )

    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/")
    llm_backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    openai_model = os.getenv("OPENAI_MODEL", "").strip() or os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

    return AppConfig(
        chunks_path=chunks_path,
        top_k=int(os.getenv("TOP_K", "5")),
        llm_backend=llm_backend,
        ollama_url=ollama_url,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "4096")),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        ollama_num_gpu=int(os.getenv("OLLAMA_NUM_GPU", "999")),
        ollama_num_thread=int(os.getenv("OLLAMA_NUM_THREAD", "8")),
        ollama_timeout_secs=int(os.getenv("OLLAMA_TIMEOUT_SECS", "300")),
        ollama_retries=int(os.getenv("OLLAMA_RETRIES", "2")),
        ollama_stream=os.getenv("OLLAMA_STREAM", "true").lower() in {"1", "true", "yes"},
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "7000")),
        virustotal_api_key=os.getenv("VIRUSTOTAL_API_KEY", "").strip(),
        force_gpu=os.getenv("FORCE_GPU", "false").lower() in {"1", "true", "yes"},
    )


def _validate_llm_config(cfg: AppConfig) -> None:
    if cfg.llm_backend in {"openai", "hf_openai", "openai_compatible"}:
        if not cfg.openai_base_url:
            raise ValueError(
                "LLM_BACKEND is OpenAI-compatible but OPENAI_BASE_URL is empty. "
                "Example: OPENAI_BASE_URL=https://matnowa3-qwen.hf.space/v1"
            )
        if cfg.force_gpu:
            raise ValueError(
                "FORCE_GPU=1 only applies to LLM_BACKEND=ollama (local GPU). "
                "Remote OpenAI-compatible APIs use the provider's hardware — unset FORCE_GPU."
            )
    if cfg.force_gpu and cfg.llm_backend == "ollama" and cfg.ollama_num_gpu <= 0:
        raise ValueError("FORCE_GPU=1 requires OLLAMA_NUM_GPU > 0.")


def _build_chat_messages(
    question: str,
    context_block: str,
    chat_history: list[dict[str, str]],
    tool_results: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if tool_results:
        messages.append(
            {
                "role": "system",
                "content": (
                    "External scan results (JSON):\n"
                    f"{build_tool_evidence(tool_results)}\n\n"
                    "Integrate critically with the user's question; cite key counts/conclusions from this data. "
                    "Do not fabricate scan outcomes beyond what is shown."
                ),
            }
        )
    messages.extend(chat_history[-8:])
    if tool_results:
        user_prompt = (
            f"User question:\n{question}\n\n"
            "Use the external scan results in the system message for URLs and/or uploaded files."
        )
    else:
        user_prompt = (
            f"CONTEXT (retrieved excerpts):\n{context_block}\n\n"
            f"User question:\n{question}\n\n"
            "Answer defensively and accurately; if context is thin or irrelevant, acknowledge limits."
        )
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _openai_chat_url(cfg: AppConfig) -> str:
    return f"{cfg.openai_base_url}/chat/completions"


def _openai_headers(cfg: AppConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.openai_api_key:
        headers["Authorization"] = f"Bearer {cfg.openai_api_key}"
    return headers


def _openai_completion_budget(cfg: AppConfig) -> int:
    """Max output tokens per HF/OpenAI completion round."""
    override = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "0"))
    if override > 0:
        return override
    return max(cfg.ollama_num_predict, 512)


def _openai_single_completion_round(
    cfg: AppConfig,
    messages: list[dict[str, str]],
    *,
    stream: bool,
    max_tokens: int,
    print_header: bool,
) -> tuple[str, str | None]:
    """One chat/completions call; returns (assistant_text, finish_reason)."""
    url = _openai_chat_url(cfg)
    headers = _openai_headers(cfg)
    payload: dict[str, Any] = {
        "model": cfg.openai_model,
        "messages": messages,
        "temperature": cfg.ollama_temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=cfg.ollama_timeout_secs,
        stream=bool(stream),
    )
    response.raise_for_status()

    finish_reason: str | None = None
    full_text = ""

    if stream:
        if print_header:
            print("\nAssistant> ", end="", flush=True)
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            ch0 = choices[0] if isinstance(choices[0], dict) else {}
            fr = ch0.get("finish_reason")
            if fr:
                finish_reason = fr
            delta = ch0.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                full_text += piece
                print(piece, end="", flush=True)
        print("\n", flush=True)
    else:
        body = response.json()
        choices = body.get("choices") or []
        if choices:
            ch0 = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = ch0.get("finish_reason")
            msg = ch0.get("message") or {}
            full_text = (msg.get("content") or "").strip()
        if full_text:
            if print_header:
                print("\nAssistant> ", end="", flush=True)
                print(full_text, end="", flush=True)
                print("\n", flush=True)
            else:
                print(full_text, end="", flush=True)
                print("", flush=True)

    return full_text.strip(), finish_reason


def _openai_chat_with_continuation(cfg: AppConfig, messages: list[dict[str, str]]) -> str:
    """
    Avoid mid-sentence cuts when the provider stops with finish_reason=length (max_tokens hit).
    Optional env: OPENAI_CONTINUE_MAX_ROUNDS (default 8).
    """
    working: list[dict[str, str]] = [dict(m) for m in messages]
    budget = _openai_completion_budget(cfg)
    max_rounds = max(1, int(os.getenv("OPENAI_CONTINUE_MAX_ROUNDS", "8")))
    aggregated = ""

    for rnd in range(max_rounds):
        use_stream = bool(cfg.ollama_stream) and rnd == 0
        print_header = rnd == 0

        text, reason = _openai_single_completion_round(
            cfg,
            working,
            stream=use_stream,
            max_tokens=budget,
            print_header=print_header,
        )

        if not text and rnd == 0:
            raise RuntimeError("OpenAI-compatible API returned an empty response.")

        aggregated += text

        if reason != "length":
            break

        if not text.strip():
            break

        working.append({"role": "assistant", "content": text})
        working.append(
            {
                "role": "user",
                "content": (
                    "Your previous reply was cut off due to length. Continue exactly where "
                    "you stopped — same tone — without repeating what you already said."
                ),
            }
        )

    result = aggregated.strip()
    if not result:
        raise RuntimeError("OpenAI-compatible API returned an empty response.")
    return result


def ask_openai_compatible(
    cfg: AppConfig,
    messages: list[dict[str, str]],
) -> str:
    last_error = None
    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            return _openai_chat_with_continuation(cfg, messages)
        except ReadTimeout as exc:
            last_error = exc
            print(
                f"[warn] LLM timed out (attempt {attempt}/{cfg.ollama_retries}, "
                f"timeout={cfg.ollama_timeout_secs}s)."
            )
        except RequestException as exc:
            last_error = exc
            print(f"[warn] LLM request failed (attempt {attempt}/{cfg.ollama_retries}): {exc}")

    raise RuntimeError(
        "OpenAI-compatible LLM failed after retries. Check OPENAI_BASE_URL, OPENAI_MODEL, and network."
    ) from last_error


def load_local_chunks(chunks_path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(chunks_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_bm25_index(rows: list[dict[str, Any]]) -> BM25Okapi:
    corpus_tokens: list[list[str]] = []
    for row in rows:
        blob = f"{row.get('title', '')} {row.get('text', '')}"
        corpus_tokens.append(tokenize_bm25(blob))
    return BM25Okapi(corpus_tokens)


def retrieve_context_bm25(
    index_rows: list[dict[str, Any]],
    bm25: BM25Okapi,
    cfg: AppConfig,
    question: str,
) -> list[dict[str, Any]]:
    q_tokens = tokenize_bm25(question)
    if not q_tokens or not index_rows:
        return []

    scores = bm25.get_scores(q_tokens)
    k = min(cfg.top_k, len(index_rows))
    top_idx = heapq.nlargest(k, range(len(scores)), key=lambda i: scores[i])

    results = []
    for idx in top_idx:
        row = index_rows[idx]
        results.append(
            {
                "chunk_id": row.get("chunk_id", ""),
                "doc_id": row.get("doc_id", ""),
                "issue_date": row.get("issue_date", ""),
                "title": row.get("title", ""),
                "chunk_index": int(row.get("chunk_index", 0)),
                "chunk_chars": int(row.get("chunk_chars", len(row.get("text", "")))),
                "text_content": row.get("text", ""),
                "distance": float(scores[idx]),
            }
        )
    return results


def build_context_block(matches: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    used = 0
    for idx, item in enumerate(matches, start=1):
        block = (
            f"[SOURCE {idx}] title={item['title']} issue_date={item['issue_date']} "
            f"chunk_id={item['chunk_id']} bm25_score={item['distance']:.4f}\n"
            f"{item['text_content']}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)>\"]+", text)
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        cleaned = url.rstrip(".,;:!?)")
        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique


def _vt_url_id(url: str) -> str:
    raw = url.encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    return encoded.rstrip("=")


def tool_virustotal_url_report(cfg: AppConfig, url: str) -> dict[str, Any]:
    if not cfg.virustotal_api_key:
        return {
            "tool": "virustotal_url_report",
            "ok": False,
            "error": "Missing VIRUSTOTAL_API_KEY in .env",
            "url": url,
        }

    headers = {"x-apikey": cfg.virustotal_api_key}
    url_id = _vt_url_id(url)
    api_url = f"https://www.virustotal.com/api/v3/urls/{url_id}"
    response = requests.get(api_url, headers=headers, timeout=40)
    if response.status_code == 404:
        submit = requests.post(
            "https://www.virustotal.com/api/v3/urls",
            headers=headers,
            data={"url": url},
            timeout=40,
        )
        submit.raise_for_status()
        submit_body = submit.json()
        analysis_id = (submit_body.get("data") or {}).get("id", "")
        return {
            "tool": "virustotal_url_report",
            "ok": True,
            "url": url,
            "status": "submitted_for_analysis",
            "analysis_id": analysis_id,
            "note": "URL submitted to VirusTotal. Ask again in a few seconds for final verdict.",
        }

    response.raise_for_status()
    body = response.json()
    attrs = ((body.get("data") or {}).get("attributes") or {})
    stats = attrs.get("last_analysis_stats") or {}
    return {
        "tool": "virustotal_url_report",
        "ok": True,
        "url": url,
        "reputation": attrs.get("reputation"),
        "categories": attrs.get("categories", {}),
        "last_analysis_date": attrs.get("last_analysis_date"),
        "last_analysis_stats": {
            "malicious": stats.get("malicious", 0),
            "suspicious": stats.get("suspicious", 0),
            "harmless": stats.get("harmless", 0),
            "undetected": stats.get("undetected", 0),
            "timeout": stats.get("timeout", 0),
        },
    }


def tool_virustotal_file_report(cfg: AppConfig, file_body: bytes, filename: str) -> dict[str, Any]:
    """VirusTotal API v3 file lookup by SHA256; upload if unknown (same key as URL tool)."""
    if not cfg.virustotal_api_key:
        return {
            "tool": "virustotal_file_report",
            "ok": False,
            "error": "Missing VIRUSTOTAL_API_KEY in .env",
            "filename": filename,
        }

    max_mb = max(1, int(os.getenv("VIRUSTOTAL_MAX_FILE_MB", "32")))
    max_bytes = max_mb * 1024 * 1024
    if len(file_body) > max_bytes:
        return {
            "tool": "virustotal_file_report",
            "ok": False,
            "error": f"File exceeds VIRUSTOTAL_MAX_FILE_MB={max_mb}",
            "filename": filename,
            "size_bytes": len(file_body),
        }

    headers = {"x-apikey": cfg.virustotal_api_key}
    sha256_hex = hashlib.sha256(file_body).hexdigest()

    api_url = f"https://www.virustotal.com/api/v3/files/{sha256_hex}"
    response = requests.get(api_url, headers=headers, timeout=60)
    if response.status_code == 404:
        upload_name = filename.strip() or "upload.bin"
        submit = requests.post(
            "https://www.virustotal.com/api/v3/files",
            headers=headers,
            files={"file": (upload_name, file_body)},
            timeout=120,
        )
        submit.raise_for_status()
        submit_body = submit.json()
        file_id = ((submit_body.get("data") or {}).get("id") or sha256_hex)
        return {
            "tool": "virustotal_file_report",
            "ok": True,
            "filename": upload_name,
            "sha256": sha256_hex,
            "status": "submitted_for_analysis",
            "file_id": file_id,
            "note": "File submitted to VirusTotal. Ask again shortly for full analysis.",
        }

    response.raise_for_status()
    body = response.json()
    attrs = ((body.get("data") or {}).get("attributes") or {})
    stats = attrs.get("last_analysis_stats") or {}
    return {
        "tool": "virustotal_file_report",
        "ok": True,
        "filename": filename,
        "sha256": sha256_hex,
        "meaningful_name": attrs.get("meaningful_name"),
        "type_description": attrs.get("type_description"),
        "size": attrs.get("size"),
        "last_analysis_date": attrs.get("last_analysis_date"),
        "last_analysis_stats": {
            "malicious": stats.get("malicious", 0),
            "suspicious": stats.get("suspicious", 0),
            "harmless": stats.get("harmless", 0),
            "undetected": stats.get("undetected", 0),
            "timeout": stats.get("timeout", 0),
        },
    }


def maybe_run_tools(cfg: AppConfig, question: str) -> list[dict[str, Any]]:
    tool_results: list[dict[str, Any]] = []
    urls = extract_urls(question)
    if urls:
        for url in urls[:3]:
            try:
                result = tool_virustotal_url_report(cfg, url)
            except RequestException as exc:
                result = {
                    "tool": "virustotal_url_report",
                    "ok": False,
                    "url": url,
                    "error": str(exc),
                }
            tool_results.append(result)
    return tool_results


def print_tool_results(tool_results: list[dict[str, Any]]) -> None:
    for item in tool_results:
        tool_name = item.get("tool", "unknown_tool")
        ok = item.get("ok", False)
        label = item.get("url") or item.get("filename") or item.get("sha256") or ""
        if not ok:
            print(f"[tool] {tool_name} failed ({label}): {item.get('error', 'unknown error')}")
            continue
        stats = item.get("last_analysis_stats", {})
        if stats:
            print(
                f"[tool] {tool_name} ok ({label}) | "
                f"malicious={stats.get('malicious', 0)} "
                f"suspicious={stats.get('suspicious', 0)} "
                f"harmless={stats.get('harmless', 0)}"
            )
        else:
            print(f"[tool] {tool_name} ok ({label}) | status={item.get('status', 'done')}")


def build_tool_evidence(tool_results: list[dict[str, Any]] | None) -> str:
    if not tool_results:
        return "[]"
    return json.dumps(tool_results, ensure_ascii=False, indent=2)


def ensure_ollama_gpu_or_raise(cfg: AppConfig) -> None:
    if not cfg.force_gpu:
        return
    try:
        response = requests.get(f"{cfg.ollama_url}/api/ps", timeout=20)
        response.raise_for_status()
        body = response.json()
    except RequestException as exc:
        raise RuntimeError(
            "FORCE_GPU: cannot query Ollama /api/ps to verify GPU usage."
        ) from exc

    models = body.get("models", [])
    if not models:
        return

    target = None
    for model in models:
        model_name = model.get("model", "") or model.get("name", "")
        if model_name == cfg.ollama_model:
            target = model
            break
    if target is None:
        target = models[0]

    processor = (target.get("processor") or "").strip().lower()
    if not processor:
        return
    if "cpu" in processor and "gpu" not in processor:
        raise RuntimeError(
            f"FORCE_GPU=1 but Ollama reports processor='{processor}'. "
            "Use GPU-enabled Ollama, NVIDIA/ROCm drivers, and ensure the model fits in VRAM."
        )


def warmup_ollama_gpu_probe(cfg: AppConfig) -> None:
    """Prime the model so /api/ps reports processor; then enforce FORCE_GPU."""
    ollama_options: dict[str, Any] = {
        "num_ctx": min(cfg.ollama_num_ctx, 512),
        "num_predict": 1,
        "temperature": 0,
        "num_gpu": cfg.ollama_num_gpu,
        "num_thread": cfg.ollama_num_thread,
    }
    response = requests.post(
        f"{cfg.ollama_url}/api/chat",
        json={
            "model": cfg.ollama_model,
            "messages": [{"role": "user", "content": "."}],
            "stream": False,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            "options": ollama_options,
        },
        timeout=cfg.ollama_timeout_secs,
    )
    response.raise_for_status()
    ensure_ollama_gpu_or_raise(cfg)


def ask_ollama(
    cfg: AppConfig,
    messages: list[dict[str, str]],
) -> str:
    def extract_text(payload: dict[str, Any]) -> str:
        message = payload.get("message", {}) or {}
        return (
            message.get("content")
            or message.get("reasoning_content")
            or payload.get("response")
            or payload.get("output_text")
            or ""
        )

    ollama_options: dict[str, Any] = {
        "num_ctx": cfg.ollama_num_ctx,
        "num_predict": cfg.ollama_num_predict,
        "temperature": cfg.ollama_temperature,
        "num_gpu": cfg.ollama_num_gpu,
        "num_thread": cfg.ollama_num_thread,
    }

    last_error = None
    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            ensure_ollama_gpu_or_raise(cfg)
            response = requests.post(
                f"{cfg.ollama_url}/api/chat",
                json={
                    "model": cfg.ollama_model,
                    "messages": messages,
                    "stream": cfg.ollama_stream,
                    "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
                    "options": ollama_options,
                },
                timeout=cfg.ollama_timeout_secs,
                stream=True,
            )
            response.raise_for_status()
            full_text = ""
            print("\nAssistant> ", end="", flush=True)
            if cfg.ollama_stream:
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        body = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = extract_text(body)
                    if token:
                        full_text += token
                        print(token, end="", flush=True)
                    if body.get("done"):
                        break
            else:
                body = response.json()
                full_text = extract_text(body)
                if full_text:
                    print(full_text, end="", flush=True)
            print("\n")
            if full_text.strip():
                return full_text.strip()

            fallback = requests.post(
                f"{cfg.ollama_url}/api/chat",
                json={
                    "model": cfg.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
                    "options": ollama_options,
                },
                timeout=cfg.ollama_timeout_secs,
            )
            fallback.raise_for_status()
            fallback_body = fallback.json()
            fallback_text = extract_text(fallback_body)
            if fallback_text and fallback_text.strip():
                return fallback_text.strip()
            raise RuntimeError(
                "Ollama returned an empty response. Try smaller num_ctx/num_predict "
                "or set OLLAMA_STREAM=false."
            )
        except ReadTimeout as exc:
            last_error = exc
            print(
                f"[warn] Ollama timed out (attempt {attempt}/{cfg.ollama_retries}, "
                f"timeout={cfg.ollama_timeout_secs}s)."
            )
        except RequestException as exc:
            last_error = exc
            print(f"[warn] Ollama request failed (attempt {attempt}/{cfg.ollama_retries}): {exc}")

    raise RuntimeError(
        "Ollama request failed after retries. "
        "Check if Ollama is running, model is pulled, and OLLAMA_URL is reachable."
    ) from last_error


def ask_llm(
    cfg: AppConfig,
    question: str,
    context_block: str,
    chat_history: list[dict[str, str]],
    tool_results: list[dict[str, Any]] | None = None,
) -> str:
    messages = _build_chat_messages(question, context_block, chat_history, tool_results)
    if cfg.llm_backend in {"openai", "hf_openai", "openai_compatible"}:
        return ask_openai_compatible(cfg, messages)
    return ask_ollama(cfg, messages)


def chat_turn(
    cfg: AppConfig,
    rows: list[dict[str, Any]],
    bm25: BM25Okapi,
    question: str,
    history: list[dict[str, str]],
    uploaded_file: tuple[bytes, str] | None = None,
) -> str:
    """
    Single user question → answer using the same pipeline as CLI (BM25 + tools + LLM).
    Optional uploaded_file: (raw bytes, filename) → VirusTotal file scan (same API family as URL tool).
    Mutates history by appending user + assistant messages on success.
    """
    matches = retrieve_context_bm25(rows, bm25, cfg, question)
    context_block = build_context_block(matches, cfg.max_context_chars)
    tool_results: list[dict[str, Any]] = []
    if uploaded_file is not None:
        raw, fname = uploaded_file
        if raw:
            try:
                tool_results.append(tool_virustotal_file_report(cfg, raw, fname))
            except RequestException as exc:
                tool_results.append(
                    {
                        "tool": "virustotal_file_report",
                        "ok": False,
                        "filename": fname,
                        "error": str(exc),
                    }
                )
    tool_results.extend(maybe_run_tools(cfg, question))
    answer = ask_llm(cfg, question, context_block, history, tool_results=tool_results)
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return answer


def model_label(cfg: AppConfig) -> str:
    if cfg.llm_backend in {"openai", "hf_openai", "openai_compatible"}:
        return cfg.openai_model
    return cfg.ollama_model


def main() -> None:
    cfg = load_config()
    _validate_llm_config(cfg)
    print("Retrieval: BM25 (no sentence-transformers / torch)")
    print(f"Loading local chunks: {cfg.chunks_path}")
    rows = load_local_chunks(cfg.chunks_path)
    if not rows:
        raise ValueError("No rows found in local chunks file.")
    bm25 = build_bm25_index(rows)

    print(f"Local index rows={len(rows)}")
    if cfg.llm_backend in {"openai", "hf_openai", "openai_compatible"}:
        print(f"LLM: OpenAI-compatible {_openai_chat_url(cfg)} model={cfg.openai_model}")
    else:
        print(
            f"LLM: Ollama {cfg.ollama_model} at {cfg.ollama_url} "
            f"(num_gpu={cfg.ollama_num_gpu} — use GPU on host if Ollama has CUDA/ROCm)"
        )
        if cfg.force_gpu:
            print("FORCE_GPU=1: probing Ollama and requiring GPU (see /api/ps).")
            try:
                warmup_ollama_gpu_probe(cfg)
            except (RequestException, RuntimeError) as exc:
                raise SystemExit(f"FORCE_GPU startup check failed: {exc}") from exc
    print(
        f"LLM timeout={cfg.ollama_timeout_secs}s retries={cfg.ollama_retries} stream={cfg.ollama_stream}"
    )
    print("Type 'exit' to quit.\n")

    history: list[dict[str, str]] = []
    while True:
        question = input("You> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        matches = retrieve_context_bm25(rows, bm25, cfg, question)
        context_block = build_context_block(matches, cfg.max_context_chars)
        tool_results = maybe_run_tools(cfg, question)
        if tool_results:
            print_tool_results(tool_results)
        try:
            answer = ask_llm(cfg, question, context_block, history, tool_results=tool_results)
        except RuntimeError as exc:
            print(f"\nAssistant> {exc}\n")
            continue

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
