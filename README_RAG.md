# Local Cybersecurity RAG Assistant

This app uses:
- Supabase Postgres + pgvector table (`vector.ouch_chunks`)
- local Ollama model (`qwen3:8b`)
- local embedding model (`paraphrase-multilingual-MiniLM-L12-v2`)

## 1) Install dependencies

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## 2) Prepare environment

```bash
cp .env.example .env
```

Edit `.env` and set `DATABASE_URL`.

## 3) Start Ollama model

```bash
ollama pull qwen3:8b
ollama run qwen3:8b
```

Keep Ollama running in another terminal.

## 4) Run assistant

```bash
.venv/bin/python rag_cyber_assistant.py
```

Type questions, `exit` to quit.

## Notes

- For 8GB VRAM, `qwen3:8b` quantized via Ollama is a practical choice.
- Retrieval uses cosine distance in pgvector (`embedding <=> query_vector`).
- If your table is not in schema `vector`, update `DB_SCHEMA`/`DB_TABLE` in `.env`.
