#!/usr/bin/env python3
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


SYSTEM_PROMPT = """You are CyberEdu Assistant, a cybersecurity educator.
Your job is to explain clearly, safely, and practically for non-experts.
Use only the provided context when stating specific facts.
If the context is insufficient, say what is missing and give general safe guidance.
Never provide instructions for harmful, illegal, or abusive actions.
Prefer step-by-step defensive advice and concrete examples.
"""


@dataclass
class AppConfig:
    database_url: str
    db_schema: str
    db_table: str
    top_k: int
    embedding_model: str
    ollama_url: str
    ollama_model: str
    max_context_chars: int


def load_config() -> AppConfig:
    load_dotenv()

    database_url = os.getenv("DATABASE_URL", "").strip()
    if (not database_url or "YOUR_SUPABASE_HOST" in database_url) and Path("string.txt").exists():
        database_url = Path("string.txt").read_text(encoding="utf-8").strip().strip('"').strip("'")

    if not database_url or "YOUR_SUPABASE_HOST" in database_url:
        raise ValueError(
            "DATABASE_URL is not set correctly. Put the real Supabase URL in .env "
            "(or string.txt), not the placeholder from .env.example."
        )

    return AppConfig(
        database_url=database_url,
        db_schema=os.getenv("DB_SCHEMA", "vector"),
        db_table=os.getenv("DB_TABLE", "ouch_chunks"),
        top_k=int(os.getenv("TOP_K", "5")),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "7000")),
    )


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vector) + "]"


def retrieve_context(
    conn: psycopg.Connection,
    cfg: AppConfig,
    query_embedding: list[float],
) -> list[dict[str, Any]]:
    sql = f"""
        SELECT
            chunk_id,
            doc_id,
            issue_date,
            title,
            chunk_index,
            chunk_chars,
            text_content,
            (embedding <=> %s::vector) AS distance
        FROM {cfg.db_schema}.{cfg.db_table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    embedding_text = vector_literal(query_embedding)
    with conn.cursor() as cur:
        cur.execute(sql, (embedding_text, embedding_text, cfg.top_k))
        rows = cur.fetchall()

    items = []
    for row in rows:
        items.append(
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "issue_date": row[2],
                "title": row[3],
                "chunk_index": row[4],
                "chunk_chars": row[5],
                "text_content": row[6],
                "distance": row[7],
            }
        )
    return items


def build_context_block(matches: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    used = 0
    for idx, item in enumerate(matches, start=1):
        block = (
            f"[SOURCE {idx}] title={item['title']} issue_date={item['issue_date']} "
            f"chunk_id={item['chunk_id']} distance={item['distance']:.4f}\n"
            f"{item['text_content']}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def ask_ollama(
    cfg: AppConfig,
    question: str,
    context_block: str,
    chat_history: list[dict[str, str]],
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[-8:])
    user_prompt = (
        "Use the context below to answer the question.\n\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer in Polish unless the user asks otherwise."
    )
    messages.append({"role": "user", "content": user_prompt})

    response = requests.post(
        f"{cfg.ollama_url}/api/chat",
        json={"model": cfg.ollama_model, "messages": messages, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    message = body.get("message", {})
    return message.get("content", "").strip()


def check_table_access(conn: psycopg.Connection, cfg: AppConfig) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {cfg.db_schema}.{cfg.db_table};")
        return int(cur.fetchone()[0])


def main() -> None:
    cfg = load_config()
    print(f"Loading embedding model: {cfg.embedding_model}")
    embedder = SentenceTransformer(cfg.embedding_model)

    with psycopg.connect(cfg.database_url, autocommit=True) as conn:
        total = check_table_access(conn, cfg)
        print(f"Connected to {cfg.db_schema}.{cfg.db_table} rows={total}")
        print(f"Ollama model: {cfg.ollama_model} at {cfg.ollama_url}")
        print("Type 'exit' to quit.\n")

        history: list[dict[str, str]] = []
        while True:
            question = input("You> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            query_embedding = embedder.encode([question], normalize_embeddings=True)[0].tolist()
            matches = retrieve_context(conn, cfg, query_embedding)
            context_block = build_context_block(matches, cfg.max_context_chars)
            answer = ask_ollama(cfg, question, context_block, history)

            print(f"\nAssistant> {answer}\n")
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
