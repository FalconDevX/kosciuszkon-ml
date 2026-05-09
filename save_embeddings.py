#!/usr/bin/env python3
"""Precompute sentence embeddings for chunks JSONL -> embeddings.npy (run once).

Install: pip install -r requirements-embeddings.txt
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


def main() -> None:
    load_dotenv()

    chunks_path = os.getenv("CHUNKS_PATH", "data/ouch_dataset/processed/chunks.jsonl").strip()
    if not Path(chunks_path).exists():
        raise SystemExit(f"CHUNKS_PATH not found: {chunks_path}")

    default_out = str(Path(chunks_path).resolve().parent / "embeddings.npy")
    out_path = os.getenv("EMBEDDINGS_NPY_PATH", default_out).strip()
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))

    texts: list[str] = []
    with Path(chunks_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row.get("text", ""))

    if not texts:
        raise SystemExit("No chunks in file.")

    print(f"Model: {model_name}")
    print(f"Chunks: {len(texts)}")
    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(out_path, vectors.astype(np.float32))
    meta_path = str(Path(out_path).with_suffix(".meta.json"))
    Path(meta_path).write_text(
        json.dumps(
            {
                "chunks_path": str(Path(chunks_path).resolve()),
                "embeddings_path": str(Path(out_path).resolve()),
                "count": len(texts),
                "embedding_model": model_name,
                "dtype": "float32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {vectors.shape} -> {out_path}")
    print(f"Meta -> {meta_path}")


if __name__ == "__main__":
    main()
