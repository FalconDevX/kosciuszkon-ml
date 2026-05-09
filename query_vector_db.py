#!/usr/bin/env python3
import argparse

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Chroma DB with consistent embeddings.")
    parser.add_argument("--db-dir", default="data/ouch_dataset/vector_db", help="Chroma DB directory")
    parser.add_argument("--collection-name", default="ouch_chunks_pl", help="Collection name")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model (must match indexing model)",
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Top results")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db_dir)
    collection = client.get_collection(name=args.collection_name)

    model = SentenceTransformer(args.model_name)
    query_embedding = model.encode([args.query], normalize_embeddings=True).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=args.top_k,
    )

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for idx, chunk_id in enumerate(ids):
        meta = metas[idx] if idx < len(metas) else {}
        text = docs[idx] if idx < len(docs) else ""
        distance = distances[idx] if idx < len(distances) else None
        print(f"\n[{idx + 1}] chunk_id={chunk_id} distance={distance}")
        print(f"title={meta.get('title', '')} issue_date={meta.get('issue_date', '')}")
        print(text[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
