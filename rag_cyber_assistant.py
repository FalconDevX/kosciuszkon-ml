#!/usr/bin/env python3
import os
import sys
import io
import json
import hashlib
import base64
import ipaddress
import re
from urllib.parse import quote
import heapq
import time
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


SYSTEM_PROMPT = """You are CyberBot — a sharp, friendly cybersecurity assistant.
You speak naturally in both Polish and English: mirror whichever language the user writes in (mixed input → mirror naturally).

You handle casual slang and technical questions alike (e.g. “hej sprawdź ten link”, “co to jest phishing?”, “check this URL”).

TOOLS AND EVIDENCE (follow exactly — this deployment does not parse fake XML tool tags in your reply text):

1) Prefetched VirusTotal JSON — When a system message contains external tool JSON with VirusTotal data, those scans already ran on the server. Each object has a `"tool"` field, e.g. `virustotal_url_report`, `virustotal_ip_report`, `virustotal_file_report`. Summarize for the user in plain language: engine counts, reputation, categories, country/ASN for IPs — never dump raw JSON. Do not invent scan outcomes.

2) Web search snippets — When JSON includes `web_search`, those are real-time search result titles/links/snippets (not your prior knowledge). Summarize and cite sources by title/URL; warn that snippets can be wrong or outdated.

3) Native API tools — When the runtime exposes function tools (e.g. Ollama tool calling), you may call `virustotal_url_report` for http(s) URLs and `virustotal_ip_report` for IPv4/IPv6 addresses when the user asks to verify safety/reputation and the value appears in the message. Call `web_search` when the user wants current web information, news, or facts not in CONTEXT. After tool results return, interpret them clearly.

4) Files — Do not try to trigger a file scan yourself; uploads are handled by the backend. If file-scan JSON is present in context, interpret it and do not claim “no file was attached.”

Behavior:
- Match user language and tone; stay conversational, not robotic.
- For vague asks with no URL/IP/file, ask one short clarifying question.
- For concepts (“what is malware?”), explain briefly with a concrete angle when helpful.
- For phishing/social engineering stories, help spot red flags practically.
- Use RAG CONTEXT when relevant; if it is thin or off-topic, say so briefly and rely on general knowledge.
- When the user wants to read or practice on this platform, suggest specific wiki articles or quiz categories using Markdown links from the separate platform-navigation system message (relative paths only).

Limits:
- Refuse offensive exploits / hacking others — one short sentence.
- Never fabricate VirusTotal results or claim safety without actual scan data in context.
- Off-topic harassment: decline in one sentence.
"""

_platform_catalog: dict[str, Any] | None = None


def normalize_ui_locale(locale: str | None) -> str:
    if not locale:
        return "pl"
    s = str(locale).strip().lower().replace("_", "-")
    if s.startswith("en"):
        return "en"
    return "pl"


def _load_platform_catalog() -> dict[str, Any]:
    global _platform_catalog
    if _platform_catalog is not None:
        return _platform_catalog
    path = Path(__file__).resolve().parent / "data" / "platform_catalog.json"
    if path.is_file():
        try:
            _platform_catalog = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            _platform_catalog = {"articles": [], "quiz_categories": []}
    else:
        _platform_catalog = {"articles": [], "quiz_categories": []}
    return _platform_catalog


def build_platform_nav_instruction(locale: str | None) -> str:
    loc = normalize_ui_locale(locale)
    cat = _load_platform_catalog()
    prefix = f"/{loc}"
    wiki_hub = "Cyber wiki" if loc == "en" else "Wiki pojęć"
    quiz_hub = "Quizzes" if loc == "en" else "Quizy"
    lines: list[str] = [
        "Kościuszkon in-app navigation (add Markdown links only when it helps; mirror the user’s language for link labels):",
        f"- Wiki index: [{wiki_hub}]({prefix}/wiki-concepts)",
        f"- Quiz index: [{quiz_hub}]({prefix}/quiz)",
        "",
        "Wiki articles (use EXACT `article` query value from backticks):",
    ]
    for article in cat.get("articles") or []:
        aid = str(article.get("id") or "").strip()
        if not aid:
            continue
        title_pl = str(article.get("title_pl") or "").strip()
        title_en = str(article.get("title_en") or title_pl).strip()
        label = title_en if loc == "en" else title_pl
        q = quote(aid, safe="")
        lines.append(f"- `{aid}` → [{label}]({prefix}/wiki-concepts?article={q})")
    lines.append("")
    lines.append("Quiz category deep links (`category` query must match exactly):")
    for qc in cat.get("quiz_categories") or []:
        cname = str(qc.get("category") or "").strip()
        if not cname:
            continue
        cq = quote(cname, safe="")
        lines.append(f"- [{cname}]({prefix}/quiz?category={cq})")
    lines.append("")
    lines.append(
        "Use relative URLs exactly as above (leading /). Do not invent article ids. "
        "Prefer wiki deep links for reading and quiz links for practice."
    )
    return "\n".join(lines)


OLLAMA_TOOL_SYSTEM_SUFFIX = """

Native tools (only when the API supplies them):
- Call `virustotal_url_report` with the full URL (scheme included) when the user wants a link checked.
- Call `virustotal_ip_report` with the IP string when the user wants an address checked.
- Call `web_search` with a short factual query when the user needs up-to-date web information not in CONTEXT.
- After results: use the returned numbers/snippets; never invent counts or URLs.
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
    ollama_tool_calling: bool


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
    openai_model = os.getenv("OPENAI_MODEL", "").strip() or os.getenv("OLLAMA_MODEL", "qwen3:8b").strip()
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
        ollama_tool_calling=os.getenv("OLLAMA_TOOL_CALLING", "true").lower()
        in {"1", "true", "yes"},
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
    *,
    locale: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": build_platform_nav_instruction(locale)},
    ]
    if tool_results:
        messages.append(
            {
                "role": "system",
                "content": (
                    "External tool results (JSON):\n"
                    f"{build_tool_evidence(tool_results)}\n\n"
                    "Summarize in the user's language; do not print raw JSON. VirusTotal objects are "
                    "malware/reputation scans; `web_search` objects are live search snippets (verify claims). "
                    "Do not fabricate data beyond this JSON."
                ),
            }
        )
    messages.extend(chat_history[-8:])
    if tool_results:
        upload_hint = ""
        for tr in tool_results:
            if tr.get("tool") == "virustotal_file_report":
                fn = tr.get("filename") or "unknown"
                upload_hint = (
                    f"\nNote: The server already scanned an uploaded file ({fn!r}). "
                    "The JSON block above is that scan — do not say no file was attached.\n"
                )
                break
        user_prompt = (
            f"User question:\n{question}{upload_hint}\n"
            "Use the external tool results in the system message. Each object has a "
            "`tool` field (virustotal_*, web_search). Answer in the user's language; "
            "summarize — do not dump raw JSON."
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
    response.encoding = "utf-8"

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


_IPV4_RE = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\w.])"
)


def extract_ips(text: str) -> list[str]:
    """Public IPv4/IPv6 literals in free text (validated); order preserved, deduped."""
    out: list[str] = []
    seen: set[str] = set()

    def add_raw(raw: str) -> None:
        raw = raw.strip()
        if not raw:
            return
        try:
            addr = ipaddress.ip_address(raw.strip("[]"))
        except ValueError:
            return
        canon = str(addr)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)

    for m in _IPV4_RE.finditer(text):
        add_raw(m.group(0))
    for m in re.finditer(r"\[([0-9a-fA-F:.%]+)\]", text):
        add_raw(m.group(1))
    for token in re.split(r"[\s,;|]+", text):
        t = token.strip().strip("()\"'")
        t = t.rstrip(".,;:!?)")
        if not t or (":" not in t and "." not in t):
            continue
        if _IPV4_RE.fullmatch(t):
            continue
        add_raw(t)
    return out


def _web_search_enabled_from_env() -> bool:
    return os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


_WEB_TRIG_LINE = re.compile(
    r"(?is)^(?:szukaj|wyszukaj|znajdź\s+w\s+sieci|przeszukaj\s+internet"
    r"|search\s+(?:the\s+)?web\s+for|search|google|look\s+up)\s*[:：]\s*(.+)$"
)
_WEB_TRIG_PREFIX = re.compile(
    r"(?is)^(?:szukaj|wyszukaj|search|google)\s+(.+)$"
)


def extract_web_search_queries(text: str) -> list[str]:
    """
    Explicit intent only (avoids firing on every message).
    Examples: "szukaj: ransomware trends", "search: CVE-2024", line-wise triggers.
    """
    if not _web_search_enabled_from_env():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in text.strip().splitlines():
        s = line.strip()
        if not s:
            continue
        m = _WEB_TRIG_LINE.match(s)
        if not m:
            m = _WEB_TRIG_PREFIX.match(s)
        if not m:
            continue
        q = (m.group(1) or "").strip().strip('"\'')
        if not q or len(q) > 500:
            continue
        key = q.casefold()
        if key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= 2:
            break
    return out


def tool_web_search(_cfg: AppConfig, query: str) -> dict[str, Any]:
    """DuckDuckGo text search — no API key; returns titled snippets for the LLM."""
    if not _web_search_enabled_from_env():
        return {
            "tool": "web_search",
            "ok": False,
            "query": query.strip(),
            "error": "web search disabled (WEB_SEARCH_ENABLED=false)",
        }
    query = query.strip()
    if not query:
        return {"tool": "web_search", "ok": False, "error": "empty query"}
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "tool": "web_search",
            "ok": False,
            "query": query,
            "error": "duckduckgo-search is not installed (pip install duckduckgo-search)",
        }
    max_results = max(1, min(15, int(os.getenv("WEB_SEARCH_MAX_RESULTS", "8"))))
    max_snippet = max(80, int(os.getenv("WEB_SEARCH_SNIPPET_CHARS", "400")))
    rows: list[dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            gen = ddgs.text(query, max_results=max_results)
            if gen:
                for r in gen:
                    if not isinstance(r, dict):
                        continue
                    title = str(r.get("title") or "").strip()
                    href = str(r.get("href") or "").strip()
                    body = str(r.get("body") or "").strip()
                    if len(body) > max_snippet:
                        body = body[: max_snippet - 3] + "..."
                    rows.append({"title": title, "href": href, "snippet": body})
    except Exception as exc:
        return {"tool": "web_search", "ok": False, "query": query, "error": str(exc)}
    if not rows:
        return {
            "tool": "web_search",
            "ok": False,
            "query": query,
            "error": "no results returned (rate limit or empty index)",
        }
    return {
        "tool": "web_search",
        "ok": True,
        "query": query,
        "results": rows,
        "note": "Third-party search snippets — verify critical facts.",
    }


def _vt_ip_path_segment(ip: str) -> str | None:
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return None
    if addr.version == 4:
        return str(addr)
    return quote(addr.compressed, safe="")


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
        try:
            _vt_poll_analysis_until_completed(cfg.virustotal_api_key, analysis_id)
        except (TimeoutError, RuntimeError, ValueError, RequestException) as exc:
            return {
                "tool": "virustotal_url_report",
                "ok": False,
                "url": url,
                "error": str(exc),
            }
        response = None
        for _ in range(8):
            response = requests.get(api_url, headers=headers, timeout=40)
            if response.status_code == 200:
                break
            if response.status_code != 404:
                response.raise_for_status()
            time.sleep(0.75)
        if response is None or response.status_code != 200:
            return {
                "tool": "virustotal_url_report",
                "ok": False,
                "url": url,
                "error": "URL report still unavailable after analysis completed",
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


def tool_virustotal_ip_report(cfg: AppConfig, ip: str) -> dict[str, Any]:
    """VirusTotal API v3: GET /ip_addresses/{ip} — separate tool id from URL/file scans."""
    if not cfg.virustotal_api_key:
        return {
            "tool": "virustotal_ip_report",
            "ok": False,
            "error": "Missing VIRUSTOTAL_API_KEY in .env",
            "ip": ip,
        }
    segment = _vt_ip_path_segment(ip)
    if not segment:
        return {
            "tool": "virustotal_ip_report",
            "ok": False,
            "error": "invalid IP address",
            "ip": ip,
        }
    headers = {"x-apikey": cfg.virustotal_api_key, "accept": "application/json"}
    api_url = f"https://www.virustotal.com/api/v3/ip_addresses/{segment}"
    try:
        response = requests.get(api_url, headers=headers, timeout=45)
    except RequestException as exc:
        return {
            "tool": "virustotal_ip_report",
            "ok": False,
            "ip": ip,
            "error": str(exc),
        }
    if response.status_code == 404:
        return {
            "tool": "virustotal_ip_report",
            "ok": False,
            "ip": ip,
            "error": "IP not found in VirusTotal (no report yet)",
        }
    try:
        response.raise_for_status()
    except RequestException as exc:
        return {
            "tool": "virustotal_ip_report",
            "ok": False,
            "ip": ip,
            "error": str(exc),
        }
    body = response.json()
    attrs = ((body.get("data") or {}).get("attributes") or {})
    stats = attrs.get("last_analysis_stats") or {}
    result: dict[str, Any] = {
        "tool": "virustotal_ip_report",
        "ok": True,
        "ip": ip.strip(),
        "reputation": attrs.get("reputation"),
        "last_analysis_date": attrs.get("last_analysis_date"),
        "last_analysis_stats": {
            "malicious": stats.get("malicious", 0),
            "suspicious": stats.get("suspicious", 0),
            "harmless": stats.get("harmless", 0),
            "undetected": stats.get("undetected", 0),
            "timeout": stats.get("timeout", 0),
        },
    }
    if attrs.get("country") is not None:
        result["country"] = attrs.get("country")
    if attrs.get("asn") is not None:
        result["asn"] = attrs.get("asn")
    if attrs.get("as_owner") is not None:
        result["as_owner"] = attrs.get("as_owner")
    return result


def _vt_file_headers(api_key: str) -> dict[str, str]:
    """VirusTotal v3: only x-apikey (+ accept). Never set Content-Type for multipart — requests sets boundary."""
    return {"x-apikey": api_key, "accept": "application/json"}


def _vt_fetch_file_comments(api_key: str, file_id: str) -> list[dict[str, Any]]:
    """
    GET https://www.virustotal.com/api/v3/files/{id}/comments (paginated).
    file_id is the VirusTotal file identifier (SHA256 for known files).
    Failures are swallowed — returns partial or empty list (never raises).
    """
    file_id = (file_id or "").strip()
    if not file_id:
        return []
    headers = _vt_file_headers(api_key)
    max_items = max(1, int(os.getenv("VIRUSTOTAL_COMMENTS_LIMIT", "40")))
    max_pages = max(1, int(os.getenv("VIRUSTOTAL_COMMENTS_MAX_PAGES", "5")))
    per_comment_chars = max(100, int(os.getenv("VIRUSTOTAL_COMMENT_MAX_CHARS", "2000")))
    out: list[dict[str, Any]] = []
    url: str | None = f"https://www.virustotal.com/api/v3/files/{file_id}/comments"
    pages = 0
    while url and pages < max_pages and len(out) < max_items:
        try:
            resp = requests.get(url, headers=headers, timeout=45)
        except RequestException:
            break
        if resp.status_code == 404:
            break
        if resp.status_code != 200:
            break
        try:
            payload = resp.json()
        except ValueError:
            break
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            attrs = item.get("attributes") or {}
            if not isinstance(attrs, dict):
                continue
            text = attrs.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if len(text) > per_comment_chars:
                text = text[: per_comment_chars - 3] + "..."
            entry: dict[str, Any] = {
                "text": text,
                "date": attrs.get("date"),
            }
            votes = attrs.get("votes")
            if isinstance(votes, dict):
                entry["votes"] = {
                    "positive": int(votes.get("positive") or 0),
                    "negative": int(votes.get("negative") or 0),
                }
            out.append(entry)
            if len(out) >= max_items:
                break
        pages += 1
        if len(out) >= max_items:
            break
        links = payload.get("links") or {}
        nxt = links.get("next") if isinstance(links, dict) else None
        url = nxt if isinstance(nxt, str) and nxt.startswith("http") else None
    return out


def _vt_analysis_poll_settings() -> tuple[float, int]:
    interval = float(os.getenv("VIRUSTOTAL_ANALYSIS_POLL_INTERVAL_SECS", "2"))
    max_wait = int(os.getenv("VIRUSTOTAL_ANALYSIS_POLL_MAX_SECS", "120"))
    return max(interval, 0.5), max(max_wait, 15)


def _vt_poll_analysis_until_completed(api_key: str, analysis_id: str) -> None:
    """Block until VirusTotal /analyses/{id} reports status completed (or fail/timeout)."""
    if not analysis_id or not str(analysis_id).strip():
        raise ValueError("empty analysis_id from VirusTotal submit response")
    headers = _vt_file_headers(api_key)
    interval, max_wait = _vt_analysis_poll_settings()
    deadline = time.monotonic() + max_wait
    poll_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
    last_status = ""
    while time.monotonic() < deadline:
        resp = requests.get(poll_url, headers=headers, timeout=45)
        resp.raise_for_status()
        body = resp.json()
        last_status = str(
            (((body.get("data") or {}).get("attributes") or {}).get("status") or "")
        ).lower()
        if last_status == "completed":
            return
        if last_status in ("failed", "aborted"):
            raise RuntimeError(f"VirusTotal analysis ended with status={last_status!r}")
        time.sleep(interval)
    raise TimeoutError(
        f"VirusTotal analysis not completed after {max_wait}s "
        f"(analysis_id={analysis_id!r}, last_status={last_status!r})"
    )


def _vt_parse_upload_url(payload: dict[str, Any]) -> str | None:
    """GET /files/upload_url returns data as URL string or nested object."""
    data = payload.get("data")
    if isinstance(data, str) and data.startswith("http"):
        return data
    if isinstance(data, dict):
        url = data.get("url") or data.get("upload_url")
        if isinstance(url, str) and url.startswith("http"):
            return url
    return None


def _vt_submit_file_multipart(
    url: str,
    api_key: str,
    upload_name: str,
    file_body: bytes,
    *,
    use_api_key_header: bool,
    timeout: int,
) -> requests.Response:
    """
    POST multipart with form field name 'file' (VirusTotal /api/v3/files).
    Do not pass Content-Type: multipart/form-data manually — boundary would be missing.
    """
    headers = _vt_file_headers(api_key) if use_api_key_header else {"accept": "application/json"}
    upload_name = upload_name.strip() or "upload.bin"
    files = {"file": (upload_name, file_body, "application/octet-stream")}
    data: dict[str, str] = {}
    zip_pw = os.getenv("VIRUSTOTAL_ZIP_PASSWORD", "").strip()
    if zip_pw:
        data["password"] = zip_pw
    return requests.post(
        url,
        headers=headers,
        files=files,
        data=data or None,
        timeout=timeout,
    )


def tool_virustotal_file_report(cfg: AppConfig, file_body: bytes, filename: str) -> dict[str, Any]:
    """VirusTotal API v3: GET file by SHA256; if 404, upload via POST /files (<32MB) or large upload URL (>=32MB)."""
    if not cfg.virustotal_api_key:
        return {
            "tool": "virustotal_file_report",
            "ok": False,
            "error": "Missing VIRUSTOTAL_API_KEY in .env",
            "filename": filename,
        }

    small_limit_mb = max(1, int(os.getenv("VIRUSTOTAL_MAX_FILE_MB", "32")))
    large_limit_mb = max(small_limit_mb, int(os.getenv("VIRUSTOTAL_LARGE_FILE_MB", "650")))
    small_limit_bytes = small_limit_mb * 1024 * 1024
    large_limit_bytes = large_limit_mb * 1024 * 1024
    size = len(file_body)
    if size > large_limit_bytes:
        return {
            "tool": "virustotal_file_report",
            "ok": False,
            "error": f"File exceeds VIRUSTOTAL_LARGE_FILE_MB={large_limit_mb}",
            "filename": filename,
            "size_bytes": size,
        }

    headers = _vt_file_headers(cfg.virustotal_api_key)
    sha256_hex = hashlib.sha256(file_body).hexdigest()
    upload_name = filename.strip() or "upload.bin"

    api_url = f"https://www.virustotal.com/api/v3/files/{sha256_hex}"
    response = requests.get(api_url, headers=headers, timeout=60)
    if response.status_code == 404:
        vt_small_url = "https://www.virustotal.com/api/v3/files"
        if size <= small_limit_bytes:
            submit = _vt_submit_file_multipart(
                vt_small_url,
                cfg.virustotal_api_key,
                upload_name,
                file_body,
                use_api_key_header=True,
                timeout=120,
            )
        else:
            up = requests.get(
                "https://www.virustotal.com/api/v3/files/upload_url",
                headers=headers,
                timeout=60,
            )
            up.raise_for_status()
            large_url = _vt_parse_upload_url(up.json())
            if not large_url:
                return {
                    "tool": "virustotal_file_report",
                    "ok": False,
                    "error": "VirusTotal upload_url response missing URL",
                    "filename": upload_name,
                    "size_bytes": size,
                }
                                                                                               
            submit = _vt_submit_file_multipart(
                large_url,
                cfg.virustotal_api_key,
                upload_name,
                file_body,
                use_api_key_header=False,
                timeout=600,
            )
        submit.raise_for_status()
        submit_body = submit.json()
        analysis_id = ((submit_body.get("data") or {}).get("id") or "")
        try:
            _vt_poll_analysis_until_completed(cfg.virustotal_api_key, analysis_id)
        except (TimeoutError, RuntimeError, ValueError, RequestException) as exc:
            return {
                "tool": "virustotal_file_report",
                "ok": False,
                "filename": upload_name,
                "sha256": sha256_hex,
                "error": str(exc),
            }
                                                                               
        response = None
        for _ in range(8):
            response = requests.get(api_url, headers=headers, timeout=60)
            if response.status_code == 200:
                break
            if response.status_code != 404:
                response.raise_for_status()
            time.sleep(0.75)
        if response is None or response.status_code != 200:
            return {
                "tool": "virustotal_file_report",
                "ok": False,
                "filename": upload_name,
                "sha256": sha256_hex,
                "error": "file report not available after analysis completed",
            }

    response.raise_for_status()
    body = response.json()
    data = body.get("data") or {}
    vt_file_id = ""
    if isinstance(data, dict):
        fid = data.get("id")
        if isinstance(fid, str) and fid.strip():
            vt_file_id = fid.strip()
    file_id_for_comments = vt_file_id or sha256_hex
    attrs = (data.get("attributes") or {}) if isinstance(data, dict) else {}
    stats = attrs.get("last_analysis_stats") or {}
    result: dict[str, Any] = {
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
    comments = _vt_fetch_file_comments(cfg.virustotal_api_key, file_id_for_comments)
    if comments:
        result["community_comments"] = comments
    return result


def maybe_run_virustotal_tools(cfg: AppConfig, question: str) -> list[dict[str, Any]]:
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
    ips = extract_ips(question)
    if ips:
        for ip in ips[:3]:
            try:
                result = tool_virustotal_ip_report(cfg, ip)
            except RequestException as exc:
                result = {
                    "tool": "virustotal_ip_report",
                    "ok": False,
                    "ip": ip,
                    "error": str(exc),
                }
            tool_results.append(result)
    return tool_results


def maybe_run_web_search_tools(cfg: AppConfig, question: str) -> list[dict[str, Any]]:
    """Runs when WEB_SEARCH_ENABLED and user uses szukaj:/search: style triggers."""
    out: list[dict[str, Any]] = []
    for q in extract_web_search_queries(question):
        out.append(tool_web_search(cfg, q))
    return out


def maybe_run_tools(cfg: AppConfig, question: str) -> list[dict[str, Any]]:
    combined = maybe_run_virustotal_tools(cfg, question)
    combined.extend(maybe_run_web_search_tools(cfg, question))
    return combined


def print_tool_results(tool_results: list[dict[str, Any]]) -> None:
    for item in tool_results:
        tool_name = item.get("tool", "unknown_tool")
        ok = item.get("ok", False)
        label = (
            item.get("url")
            or item.get("ip")
            or item.get("filename")
            or item.get("sha256")
            or item.get("query")
            or ""
        )
        if not ok:
            print(f"[tool] {tool_name} failed ({label}): {item.get('error', 'unknown error')}")
            continue
        if tool_name == "web_search":
            n = len(item.get("results") or [])
            print(f"[tool] {tool_name} ok (query={label!r}) | hits={n}")
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


def ollama_tools_schema() -> list[dict[str, Any]]:
    """OpenAI-style tools payload for Ollama /api/chat."""
    return [
        {
            "type": "function",
            "function": {
                "name": "virustotal_url_report",
                "description": (
                    "Query VirusTotal for a URL: reputation, categories, and "
                    "last_analysis_stats (malicious/suspicious/harmless/undetected counts). "
                    "Use when the user asks if a link is safe, wants a URL scanned, or provides http(s) URL to check."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL with scheme, e.g. https://example.com/path",
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "virustotal_ip_report",
                "description": (
                    "Query VirusTotal for an IP address: reputation, last_analysis_stats, "
                    "and optional country/asn fields. Use when the user asks about an IPv4/IPv6 "
                    "address, suspicious connections, or ‘what is this IP’."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ip": {
                            "type": "string",
                            "description": "IPv4 or IPv6 address as written by the user, e.g. 203.0.113.5 or 2001:db8::1",
                        },
                    },
                    "required": ["ip"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the public web via DuckDuckGo (titles, URLs, short snippets). "
                    "Use for fresh facts, news, CVE details, or anything not in CONTEXT. "
                    "Pass a concise English or Polish query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query, e.g. 'CVE-2024-1234 details' or 'phishing bank Poland'",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


def _build_ollama_tool_messages(
    question: str,
    context_block: str,
    chat_history: list[dict[str, str]],
    prefetch_tool_results: list[dict[str, Any]] | None,
    *,
    locale: str | None = None,
) -> list[dict[str, Any]]:
    """Initial messages for Ollama tool-calling loop (URLs via model; file scans still prefetched)."""
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + OLLAMA_TOOL_SYSTEM_SUFFIX},
        {"role": "system", "content": build_platform_nav_instruction(locale)},
    ]
    if prefetch_tool_results:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "External tool results already available for this turn (JSON):\n"
                    f"{build_tool_evidence(prefetch_tool_results)}\n\n"
                    "Integrate with the user question (VirusTotal vs web_search); do not claim "
                    "no file was scanned if file data is present."
                ),
            }
        )
    for turn in chat_history[-8:]:
        msgs.append({"role": turn["role"], "content": turn["content"]})
    msgs.append(
        {
            "role": "user",
            "content": (
                f"CONTEXT (retrieved excerpts):\n{context_block}\n\n"
                f"User question:\n{question}"
            ),
        }
    )
    return msgs


def _parse_tool_arguments(args_raw: Any) -> dict[str, Any]:
    if args_raw is None:
        return {}
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str):
        s = args_raw.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}
    return {}


def _dispatch_ollama_tool(tc: dict[str, Any], cfg: AppConfig) -> dict[str, Any]:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = (fn.get("name") or "").strip()
    args = _parse_tool_arguments(fn.get("arguments"))
    if name == "virustotal_url_report":
        url = str(args.get("url") or "").strip()
        if not url:
            return {"tool": name, "ok": False, "error": "missing url parameter"}
        try:
            return tool_virustotal_url_report(cfg, url)
        except RequestException as exc:
            return {"tool": name, "ok": False, "url": url, "error": str(exc)}
    if name == "virustotal_ip_report":
        ip_val = str(args.get("ip") or "").strip()
        if not ip_val:
            return {"tool": name, "ok": False, "error": "missing ip parameter"}
        try:
            return tool_virustotal_ip_report(cfg, ip_val)
        except RequestException as exc:
            return {"tool": name, "ok": False, "ip": ip_val, "error": str(exc)}
    if name == "web_search":
        q = str(args.get("query") or "").strip()
        if not q:
            return {"tool": name, "ok": False, "error": "missing query parameter"}
        return tool_web_search(cfg, q)
    return {"ok": False, "error": f"unknown tool: {name!r}"}


def ask_ollama_with_tools(
    cfg: AppConfig,
    messages: list[dict[str, Any]],
) -> str:
    """
    Multi-turn Ollama /api/chat with tools (stream disabled — tool_calls need full response).
    """
    tools = ollama_tools_schema()
    max_rounds = max(1, int(os.getenv("OLLAMA_TOOL_MAX_ROUNDS", "6")))
    ollama_options: dict[str, Any] = {
        "num_ctx": cfg.ollama_num_ctx,
        "num_predict": cfg.ollama_num_predict,
        "temperature": cfg.ollama_temperature,
        "num_gpu": cfg.ollama_num_gpu,
        "num_thread": cfg.ollama_num_thread,
    }
    last_error: Exception | None = None
    for attempt in range(1, cfg.ollama_retries + 1):
        try:
            working: list[dict[str, Any]] = [dict(m) for m in messages]
            for _round in range(max_rounds):
                ensure_ollama_gpu_or_raise(cfg)
                response = requests.post(
                    f"{cfg.ollama_url}/api/chat",
                    json={
                        "model": cfg.ollama_model,
                        "messages": working,
                        "tools": tools,
                        "stream": False,
                        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
                        "options": ollama_options,
                    },
                    timeout=cfg.ollama_timeout_secs,
                )
                response.raise_for_status()
                body = response.json()
                msg = body.get("message") if isinstance(body.get("message"), dict) else {}
                tool_calls = msg.get("tool_calls")
                if not tool_calls and isinstance(body.get("tool_calls"), list):
                    tool_calls = body.get("tool_calls")

                if (
                    isinstance(tool_calls, list)
                    and len(tool_calls) > 0
                    and any(isinstance(x, dict) for x in tool_calls)
                ):
                    assistant_out: dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.get("content") or "",
                    }
                    assistant_out["tool_calls"] = tool_calls
                    working.append(assistant_out)
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        result = _dispatch_ollama_tool(tc, cfg)
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        tname = (fn.get("name") or "virustotal_url_report").strip()
                        tool_msg: dict[str, Any] = {
                            "role": "tool",
                            "content": json.dumps(result, ensure_ascii=False),
                            "name": tname,
                        }
                        tid = tc.get("id")
                        if isinstance(tid, str) and tid.strip():
                            tool_msg["tool_call_id"] = tid.strip()
                        working.append(tool_msg)
                    continue

                text = str(msg.get("content") or "").strip()
                if text:
                    return text
                return ""

            raise RuntimeError(
                f"Ollama tool loop exceeded OLLAMA_TOOL_MAX_ROUNDS={max_rounds} without a final answer."
            )
        except ReadTimeout as exc:
            last_error = exc
            print(
                f"[warn] Ollama tool-call timed out (attempt {attempt}/{cfg.ollama_retries}, "
                f"timeout={cfg.ollama_timeout_secs}s)."
            )
        except RequestException as exc:
            last_error = exc
            print(
                f"[warn] Ollama tool-call request failed (attempt {attempt}/{cfg.ollama_retries}): {exc}"
            )

    raise RuntimeError(
        "Ollama tool-calling failed after retries. Check OLLAMA_URL, model tool support, and logs."
    ) from last_error


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
            response.encoding = "utf-8"
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
    *,
    locale: str | None = None,
) -> str:
    if cfg.llm_backend in {"openai", "hf_openai", "openai_compatible"}:
        messages = _build_chat_messages(
            question, context_block, chat_history, tool_results, locale=locale
        )
        return ask_openai_compatible(cfg, messages)
    if cfg.llm_backend == "ollama" and cfg.ollama_tool_calling:
        tool_msgs = _build_ollama_tool_messages(
            question, context_block, chat_history, tool_results, locale=locale
        )
        return ask_ollama_with_tools(cfg, tool_msgs)
    messages = _build_chat_messages(
        question, context_block, chat_history, tool_results, locale=locale
    )
    return ask_ollama(cfg, messages)


def extract_web_search_sources(
    tool_results: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Flatten web_search tool outputs into [{title, url}] for API responses."""
    if not tool_results:
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for tr in tool_results:
        if not isinstance(tr, dict):
            continue
        if tr.get("tool") != "web_search" or not tr.get("ok"):
            continue
        for r in tr.get("results") or []:
            if not isinstance(r, dict):
                continue
            url = (r.get("href") or r.get("url") or "").strip()
            title = (r.get("title") or "").strip() or url
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"title": title, "url": url})
    return out


def chat_turn(
    cfg: AppConfig,
    rows: list[dict[str, Any]],
    bm25: BM25Okapi,
    question: str,
    history: list[dict[str, str]],
    uploaded_file: tuple[bytes, str] | None = None,
    enable_web_search: bool = False,
    locale: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Single user question → (answer, tool_results) using the same pipeline as CLI (BM25 + tools + LLM).
    Optional uploaded_file: (raw bytes, filename) → VirusTotal file scan.
    enable_web_search=True → force one web_search call on the full question, in addition to
    the existing explicit-trigger extraction (e.g. "szukaj: ...").
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
    extracted_web = maybe_run_web_search_tools(cfg, question)
    tool_results.extend(extracted_web)
    if enable_web_search and not extracted_web and question.strip():
        tool_results.append(tool_web_search(cfg, question.strip()))
    if not (cfg.llm_backend == "ollama" and cfg.ollama_tool_calling):
        tool_results.extend(maybe_run_virustotal_tools(cfg, question))
    answer = ask_llm(
        cfg, question, context_block, history, tool_results=tool_results, locale=locale
    )
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return answer, tool_results


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
        if cfg.ollama_tool_calling:
            print(
                "Ollama tool calling: ON — virustotal_url_report / virustotal_ip_report / web_search."
            )
        else:
            print("Ollama tool calling: OFF — URLs are scanned via regex before the LLM (legacy).")
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
        tool_results: list[dict[str, Any]] = []
        tool_results.extend(maybe_run_web_search_tools(cfg, question))
        if not (cfg.llm_backend == "ollama" and cfg.ollama_tool_calling):
            tool_results.extend(maybe_run_virustotal_tools(cfg, question))
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
