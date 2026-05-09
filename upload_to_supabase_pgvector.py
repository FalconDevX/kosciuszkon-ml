#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import psycopg
from psycopg import sql
from sentence_transformers import SentenceTransformer


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def parse_db_url(db_url: str) -> str:
    value = db_url.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        value = value[1:-1]
    return value


def db_url_from_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Connection string file is empty: {path}")
    return parse_db_url(raw)


def embed_batches(model: SentenceTransformer, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def ensure_schema_table(conn: psycopg.Connection, schema: str, table: str, dim: int) -> None:
    schema_ident = sql.Identifier(schema)
    table_ident = sql.Identifier(table)
    embedding_index_ident = sql.Identifier(f"{table}_embedding_idx")
    issue_date_index_ident = sql.Identifier(f"{table}_issue_date_idx")

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(schema_ident))
        cur.execute(
            sql.SQL(
                """
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                issue_date TEXT,
                title TEXT,
                chunk_index INTEGER,
                chunk_chars INTEGER,
                text_content TEXT,
                embedding vector({dim})
            );
            """
            ).format(schema=schema_ident, table=table_ident, dim=sql.SQL(str(dim)))
        )
        cur.execute(
            sql.SQL(
                """
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {schema}.{table}
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            ).format(
                index_name=embedding_index_ident,
                schema=schema_ident,
                table=table_ident,
            )
        )
        cur.execute(
            sql.SQL(
                """
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {schema}.{table}(issue_date);
            """
            ).format(
                index_name=issue_date_index_ident,
                schema=schema_ident,
                table=table_ident,
            )
        )
    conn.commit()


def upload_rows(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    rows: list[dict],
    embeddings: list[list[float]],
) -> None:
    upsert_sql = sql.SQL(
        """
        INSERT INTO {schema}.{table}
        (chunk_id, doc_id, issue_date, title, chunk_index, chunk_chars, text_content, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            issue_date = EXCLUDED.issue_date,
            title = EXCLUDED.title,
            chunk_index = EXCLUDED.chunk_index,
            chunk_chars = EXCLUDED.chunk_chars,
            text_content = EXCLUDED.text_content,
            embedding = EXCLUDED.embedding;
    """
    ).format(schema=sql.Identifier(schema), table=sql.Identifier(table))

    payload = []
    for row, emb in zip(rows, embeddings):
        payload.append(
            (
                row.get("chunk_id"),
                row.get("doc_id"),
                row.get("issue_date"),
                row.get("title"),
                int(row.get("chunk_index", 0)),
                int(row.get("chunk_chars", len(row.get("text", "")))),
                row.get("text", ""),
                json.dumps(emb),
            )
        )

    with conn.cursor() as cur:
        cur.executemany(upsert_sql, payload)
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload OUCH chunk embeddings to Supabase pgvector.")
    parser.add_argument(
        "--chunks-path",
        default="data/ouch_dataset/processed/chunks.jsonl",
        help="Path to chunks JSONL",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="Postgres connection string (overrides --db-url-file)",
    )
    parser.add_argument(
        "--db-url-file",
        default="string.txt",
        help="File containing Postgres connection string",
    )
    parser.add_argument("--schema", default="vector", help="Target schema")
    parser.add_argument("--table", default="ouch_chunks", help="Target table")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    args = parser.parse_args()

    db_url = parse_db_url(args.db_url) if args.db_url else db_url_from_file(Path(args.db_url_file))

    chunks = load_chunks(Path(args.chunks_path))
    if not chunks:
        raise ValueError("No chunks found to upload.")

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embedding model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    embeddings = embed_batches(model, [x.get("text", "") for x in chunks], args.batch_size)
    dim = len(embeddings[0]) if embeddings else 0
    if dim <= 0:
        raise ValueError("Invalid embedding dimension.")

    with psycopg.connect(db_url, autocommit=False) as conn:
        ensure_schema_table(conn, args.schema, args.table, dim)
        upload_rows(conn, args.schema, args.table, chunks, embeddings)

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{};").format(
                    sql.Identifier(args.schema),
                    sql.Identifier(args.table),
                )
            )
            total = cur.fetchone()[0]

    print(f"Upload complete. Rows in {args.schema}.{args.table}: {total}")


if __name__ == "__main__":
    main()
