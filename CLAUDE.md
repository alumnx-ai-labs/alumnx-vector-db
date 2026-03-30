# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv venv && uv sync

# Run locally (with auto-reload)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
uv run pytest -q

# Run a single test file
uv run pytest tests/test_api.py -q

# Run with Docker
docker-compose up --build
```

**Required environment variables** — copy `.env.example` to `.env` and fill in:

- `GOOGLE_API_KEY` — Google Gemini API key
- `POSTGRES_URL` — PostgreSQL connection URL (AWS RDS or local)

## Architecture

NexVec is a FastAPI-based recruitment RAG service: ingest PDF resumes → structured parse → embed → store; retrieve via SQL-first + targeted vector search.

### Core Pipelines

**Ingest (`POST /ingest`):**
PDF upload → SHA-256 dedup → pdfplumber text extract → Gemini LLM parse into 7 sections → resolve/create user (email/phone identity) → embed 6 sections (unit-normalised) → append to flat `nex_vec.npy` → persist resume row to Postgres.

**Retrieve (`POST /retrieve`):**
Natural language query → Gemini LLM generates SQL → Postgres returns matching `resume_id`s → load only those chunk positions from flat `nex_vec.npy` → dot product (= cosine similarity, vectors are unit-normalised) → dedup by `user_id` → return ranked candidates.

### Key Components

| Layer          | Location                                  | Purpose                                            |
| -------------- | ----------------------------------------- | -------------------------------------------------- |
| API routes     | `app/routers/`                            | Thin HTTP handlers delegating to services          |
| Ingestion      | `app/services/ingestion.py`               | Orchestrates extract → parse → embed → store       |
| Retrieval      | `app/services/retrieval_service.py`       | SQL-first filter → targeted vector search → rank   |
| LLM parser     | `app/services/llm_parser.py`              | Gemini Flash: PDF text → 7-section structured JSON |
| LLM query      | `app/services/llm_query.py`               | Gemini Flash: natural language → PostgreSQL SELECT |
| Embedder       | `app/services/embedding/embedder.py`      | Google Gemini embeddings, batched                  |
| Vector store   | `app/services/store/vector_file_store.py` | Flat `.npy` files in `./vector_store/`             |
| Postgres store | `app/services/store/postgres_store.py`    | `users` + `resumes` tables on AWS RDS              |
| Config         | `config.yaml` + `app/config.py`           | Embedding model, KNN k, vector dimensions          |

### Database Schema

**`users`** — one row per unique person (identity: email + phone)

- `user_id TEXT PK`, `name`, `email`, `phone`, `location`, `created_at`

**`resumes`** — one row per uploaded resume (a person can have multiple)

- `resume_id TEXT PK`, `user_id FK`, `source_filename`, `file_hash UNIQUE`
- 7 section columns: `objectives`, `work_experience_years NUMERIC`, `work_experience_text`, `projects`, `education`, `skills TEXT[]`, `achievements`
- 6 chunk_id columns (one per embeddable section): `objectives_chunk_id`, `work_experience_text_chunk_id`, `projects_chunk_id`, `education_chunk_id`, `skills_chunk_id`, `achievements_chunk_id`
- `embedding_model`, `is_active BOOLEAN`, `created_at`

### Vector Store

Single flat file: `vector_store/nex_vec.npy` (all vectors) + `nex_vec_ids.npy` (chunk_ids) + `nex_vec.jsonl` (audit log: chunk_id + resume_id + vector).

All vectors are **unit-normalised** so dot product = cosine similarity.

### Design Decisions

- **SQL-first retrieval**: LLM generates SQL for structured filters (skills, experience, location) — only fall back to full scan on SQL exception, never on empty results.
- **Person deduplication**: `email`/`phone` → reuse `user_id` across resume versions. One result per person in retrieval output.
- **Content dedup**: SHA-256 hash — skip LLM + embedding if file already ingested.
- **Soft deletes**: `is_active = FALSE` — `ON CONFLICT (file_hash) DO UPDATE` re-activates on re-ingest.
- **GIN index on `skills TEXT[]`**: fast `@>` array containment queries.

## Testing

Tests live in `tests/` with fixtures in `conftest.py`. Tests mock the Google Gemini API and use temp directories for the vector store. Run the full suite with `uv run pytest -q` before raising a PR.

## Deployment

CI/CD is split across `.github/workflows/ci.yml`, `.github/workflows/deploy-dev.yml`, and `.github/workflows/deploy-prod.yml`.

- `CI` runs unit tests on every push and on pull requests targeting `dev` or `main`.
- `Deploy Dev` runs only after a successful `CI` workflow on `dev`.
- `Deploy Prod` runs only after a successful `CI` workflow on `main`.

The production rollout target is the Alumnx Vector DB service on EC2 port `8012`. GitHub branch protection is still required on `main` if merges must be blocked until `CI / Unit Tests` passes.
