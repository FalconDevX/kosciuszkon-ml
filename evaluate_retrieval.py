#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


@dataclass
class EvalQuery:
    question: str
    expected_keywords: list[str]


DEFAULT_QUERIES = [
    EvalQuery(
        question="Jak rozpoznać phishing i fałszywe maile?",
        expected_keywords=["phishing", "mail", "wiadomo", "oszust"],
    ),
    EvalQuery(
        question="Jak zabezpieczyć hasła i czy warto menedżer haseł?",
        expected_keywords=["has", "pass", "mened", "password"],
    ),
    EvalQuery(
        question="Co robić po przejęciu konta?",
        expected_keywords=["przej", "konto", "hacked", "zhak"],
    ),
    EvalQuery(
        question="Jak bronić się przed oszustwami telefonicznymi i vishingiem?",
        expected_keywords=["telefon", "vishing", "call", "głos"],
    ),
    EvalQuery(
        question="Jak bezpiecznie robić zakupy online?",
        expected_keywords=["zakup", "shopping", "online", "sklep"],
    ),
]


def load_chunks(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    return text.lower()


def hit_for_query(
    query: EvalQuery,
    rows: list[dict],
    row_embeddings,
    embedder: SentenceTransformer,
    k: int,
) -> tuple[bool, list[dict]]:
    q_emb = embedder.encode(query.question, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, row_embeddings)[0]
    k = min(k, len(rows))
    top_values, top_idx = scores.topk(k=k)

    top_hits = []
    matched = False
    for score, idx in zip(top_values.tolist(), top_idx.tolist()):
        row = rows[idx]
        title = row.get("title", "")
        text = row.get("text", "")
        haystack = normalize(f"{title}\n{text[:600]}")
        keyword_match = any(kw in haystack for kw in query.expected_keywords)
        if keyword_match:
            matched = True
        top_hits.append(
            {
                "score": float(score),
                "title": title,
                "chunk_id": row.get("chunk_id", ""),
                "keyword_match": keyword_match,
            }
        )
    return matched, top_hits


def evaluate_model(model_name: str, rows: list[dict], queries: list[EvalQuery], k: int) -> dict:
    embedder = SentenceTransformer(model_name)
    row_embeddings = embedder.encode(
        [r.get("text", "") for r in rows],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    per_query = []
    hits = 0
    for query in queries:
        matched, top_hits = hit_for_query(query, rows, row_embeddings, embedder, k)
        if matched:
            hits += 1
        per_query.append(
            {
                "question": query.question,
                "hit": matched,
                "top_hits": top_hits,
            }
        )

    return {
        "model": model_name,
        "k": k,
        "queries": len(queries),
        "hits": hits,
        "hit_rate": hits / len(queries) if queries else 0.0,
        "details": per_query,
    }


def print_summary(results: list[dict]) -> None:
    print("\n=== Retrieval Evaluation Summary ===")
    for item in results:
        print(
            f"- {item['model']}: Hit@{item['k']} = {item['hits']}/{item['queries']} "
            f"({item['hit_rate']:.2%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality across embedding models.")
    parser.add_argument(
        "--chunks-path",
        default="data/ouch_dataset/processed/chunks.jsonl",
        help="Path to local chunks JSONL",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-base",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ],
        help="Embedding model names to compare",
    )
    parser.add_argument("--k", type=int, default=5, help="Hit@k depth")
    parser.add_argument(
        "--output-json",
        default="data/ouch_dataset/processed/retrieval_eval.json",
        help="Where to save detailed JSON report",
    )
    args = parser.parse_args()

    rows = load_chunks(Path(args.chunks_path))
    if not rows:
        raise ValueError(f"No rows loaded from {args.chunks_path}")

    results = []
    for model_name in args.models:
        print(f"\nEvaluating model: {model_name}")
        result = evaluate_model(model_name, rows, DEFAULT_QUERIES, args.k)
        results.append(result)

    print_summary(results)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed report: {output_path}")


if __name__ == "__main__":
    main()
