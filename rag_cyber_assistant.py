#!/usr/bin/env python3
import os
import json
import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from requests.exceptions import RequestException, ReadTimeout

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util


SYSTEM_PROMPT = """You are CyberEdu Assistant.

Rules:
- Tool results are the highest priority source of truth.
- Never deny access to a URL if tool results exist.
- Never explain limitations when tool results are available.
- Never reinterpret VirusTotal statistics.
- If VirusTotal says malicious=0 and suspicious=0, treat it as no known detections.
- Keep answers short and practical.
- Do not explain the tool itself.
- Do not mention TOOL EVIDENCE or TOOL RESULTS.
- Focus only on final cybersecurity analysis.
- Never provide instructions for harmful, illegal, or abusive actions.

Respond ONLY in this format:
Risk: LOW/MEDIUM/HIGH

Analysis:
- short bullet
- short bullet

Recommendation:
- short recommendation
"""


@dataclass
class AppConfig:
    chunks_path: str
    top_k: int
    embedding_model: str
    ollama_url: str
    ollama_model: str
    ollama_num_ctx: int
    ollama_num_predict: int
    ollama_temperature: float
    ollama_num_gpu: int
    ollama_num_thread: int
    ollama_require_gpu_only: bool
    ollama_timeout_secs: int
    ollama_retries: int
    ollama_stream: bool
    max_context_chars: int
    virustotal_api_key: str


def load_config() -> AppConfig:
    load_dotenv()

    chunks_path = os.getenv("CHUNKS_PATH", "data/ouch_dataset/processed/chunks.jsonl").strip()
    if not Path(chunks_path).exists():
        raise ValueError(
            f"CHUNKS_PATH does not exist: {chunks_path}. "
            "Point CHUNKS_PATH to your local chunks JSONL file."
        )

    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").strip().rstrip("/")

    return AppConfig(
        chunks_path=chunks_path,
        top_k=int(os.getenv("TOP_K", "5")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        ollama_url=ollama_url,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "220")),
        ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        ollama_num_gpu=int(os.getenv("OLLAMA_NUM_GPU", "999")),
        ollama_num_thread=int(os.getenv("OLLAMA_NUM_THREAD", "8")),
        ollama_require_gpu_only=os.getenv("OLLAMA_REQUIRE_GPU_ONLY", "true").lower() in {"1", "true", "yes"},
        ollama_timeout_secs=int(os.getenv("OLLAMA_TIMEOUT_SECS", "300")),
        ollama_retries=int(os.getenv("OLLAMA_RETRIES", "2")),
        ollama_stream=os.getenv("OLLAMA_STREAM", "false").lower() in {"1", "true", "yes"},
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "7000")),
        virustotal_api_key=os.getenv("VIRUSTOTAL_API_KEY", "").strip(),
    )


def load_local_chunks(chunks_path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(chunks_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def retrieve_context(
    index_rows: list[dict[str, Any]],
    index_embeddings: Any,
    embedder: SentenceTransformer,
    cfg: AppConfig,
    question: str,
) -> list[dict[str, Any]]:
    query_embedding = embedder.encode(question, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(query_embedding, index_embeddings)[0]
    top_k = min(cfg.top_k, len(index_rows))
    top_values, top_indices = scores.topk(k=top_k)

    results = []
    for score, idx in zip(top_values.tolist(), top_indices.tolist()):
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
                # Keep naming consistent with previous output, but cosine score is similarity.
                "distance": float(score),
            }
        )
    return results


def build_context_block(matches: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    used = 0
    for idx, item in enumerate(matches, start=1):
        block = (
            f"[SOURCE {idx}] title={item['title']} issue_date={item['issue_date']} "
            f"chunk_id={item['chunk_id']} similarity={item['distance']:.4f}\n"
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
        url = item.get("url", "")
        if not ok:
            print(f"[tool] {tool_name} failed for {url}: {item.get('error', 'unknown error')}")
            continue
        stats = item.get("last_analysis_stats", {})
        if stats:
            print(
                f"[tool] {tool_name} ok for {url} | "
                f"malicious={stats.get('malicious', 0)} "
                f"suspicious={stats.get('suspicious', 0)} "
                f"harmless={stats.get('harmless', 0)}"
            )
        else:
            print(f"[tool] {tool_name} ok for {url} | status={item.get('status', 'done')}")


def build_tool_evidence(tool_results: list[dict[str, Any]] | None) -> str:
    if not tool_results:
        return "[]"
    return json.dumps(tool_results, ensure_ascii=False, indent=2)


def ask_ollama(
    cfg: AppConfig,
    question: str,
    context_block: str,
    chat_history: list[dict[str, str]],
    tool_results: list[dict[str, Any]] | None = None,
) -> str:
    def extract_text(payload: dict[str, Any]) -> str:
        message = payload.get("message", {}) or {}
        # Some Ollama/model combinations emit text under different keys.
        return (
            message.get("content")
            or message.get("reasoning_content")
            or payload.get("response")
            or payload.get("output_text")
            or ""
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if tool_results:
        messages.append(
            {
                "role": "system",
                "content": (
                    "TOOL EVIDENCE (trusted external data, highest priority):\n"
                    f"{build_tool_evidence(tool_results)}\n\n"
                    "Rules:\n"
                    "1) Treat TOOL EVIDENCE as factual input.\n"
                    "2) Never say you cannot access or verify the link when tool evidence exists.\n"
                    "3) If tool evidence conflicts with user claims, explain the conflict.\n"
                    "4) Use VirusTotal stats directly; do not reinterpret them.\n"
                ),
            }
        )
    messages.extend(chat_history[-8:])
    if tool_results:
        user_prompt = (
            "QUESTION:\n"
            f"{question}\n\n"
            "Use only tool evidence for URL safety verdict."
        )
    else:
        user_prompt = (
            "CONTEXT:\n"
            f"{context_block}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Answer in Polish unless the user asks otherwise."
        )
    messages.append({"role": "user", "content": user_prompt})

    last_error = None
    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            response = requests.post(
                f"{cfg.ollama_url}/api/chat",
                json={
                    "model": cfg.ollama_model,
                    "messages": messages,
                    "stream": cfg.ollama_stream,
                    "options": {
                        "num_ctx": cfg.ollama_num_ctx,
                        "num_predict": cfg.ollama_num_predict,
                        "temperature": cfg.ollama_temperature,
                        "num_gpu": cfg.ollama_num_gpu,
                        "num_thread": cfg.ollama_num_thread,
                    },
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
                        # Ignore non-JSON stream lines and keep consuming output.
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

            # Fallback: some model/runtime combos may return empty streamed content.
            fallback = requests.post(
                f"{cfg.ollama_url}/api/chat",
                json={
                    "model": cfg.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": cfg.ollama_num_ctx,
                        "num_predict": cfg.ollama_num_predict,
                        "temperature": cfg.ollama_temperature,
                        "num_gpu": cfg.ollama_num_gpu,
                        "num_thread": cfg.ollama_num_thread,
                    },
                },
                timeout=cfg.ollama_timeout_secs,
            )
            fallback.raise_for_status()
            fallback_body = fallback.json()
            fallback_text = extract_text(fallback_body)
            if fallback_text and fallback_text.strip():
                return fallback_text.strip()
            raise RuntimeError(
                "Ollama returned an empty response. Try reducing load "
                "(smaller model, lower num_ctx/num_predict), set OLLAMA_STREAM=false, "
                "or disable GPU-only enforcement."
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


def ensure_gpu_only_or_raise(cfg: AppConfig) -> None:
    if not cfg.ollama_require_gpu_only:
        return
    try:
        response = requests.get(
            f"{cfg.ollama_url}/api/ps",
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()
    except RequestException as exc:
        raise RuntimeError(
            "Cannot verify Ollama processor usage via /api/ps. "
            "Set OLLAMA_REQUIRE_GPU_ONLY=false to skip this check."
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
    if "cpu" in processor:
        raise RuntimeError(
            f"GPU-only mode enabled but Ollama reports processor='{processor}'. "
            "Choose a smaller model or reduce context so it can fully offload to GPU."
        )


def main() -> None:
    cfg = load_config()
    print(f"Loading embedding model: {cfg.embedding_model}")
    embedder = SentenceTransformer(cfg.embedding_model)
    print(f"Loading local chunks: {cfg.chunks_path}")
    rows = load_local_chunks(cfg.chunks_path)
    if not rows:
        raise ValueError("No rows found in local chunks file.")
    index_embeddings = embedder.encode(
        [row.get("text", "") for row in rows],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Local index rows={len(rows)}")
    print(f"Ollama model: {cfg.ollama_model} at {cfg.ollama_url}")
    print(
        f"Ollama timeout={cfg.ollama_timeout_secs}s retries={cfg.ollama_retries} stream={cfg.ollama_stream}"
    )
    print("Type 'exit' to quit.\n")

    history: list[dict[str, str]] = []
    while True:
        question = input("You> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        matches = retrieve_context(rows, index_embeddings, embedder, cfg, question)
        context_block = build_context_block(matches, cfg.max_context_chars)
        tool_results = maybe_run_tools(cfg, question)
        if tool_results:
            print_tool_results(tool_results)
        try:
            ensure_gpu_only_or_raise(cfg)
            answer = ask_ollama(cfg, question, context_block, history, tool_results=tool_results)
        except RuntimeError as exc:
            print(f"\nAssistant> {exc}\n")
            continue

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
