Debug failures or poor results in the NexVec /retrieve pipeline.

## Pipeline Stages & Files

```
Query Input → SQL Generation → SQL Execution → Chunk Mapping → Query Embedding → Vector Search → Score Aggregation → Response
```

---

### Stage 1: SQL Generation
**File:** `app/services/llm_query.py`

- Gemini Flash generates a PostgreSQL SELECT from the natural language query.
- Expected output: a valid SQL string filtering on `resumes` columns (skills, location, work_experience_years).
- Symptom: LLM returns invalid SQL → service falls back to fetching ALL active resumes (check logs for `"SQL generation failed, falling back to full scan"`).
- Debug: Log the raw LLM response to see what SQL was generated.

### Stage 2: SQL Execution
**File:** `app/services/store/postgres_store.py` → `execute_filter_query()`

- SQL is executed against the `resumes` table. Returns list of `resume_id`s.
- Symptom: Empty result despite resumes existing → SQL filter is too restrictive.
  - Check: does the LLM-generated SQL use correct column names?
  - Skills filter uses GIN: `skills @> ARRAY['Python']` (must match exactly).
  - Try a simpler query that shouldn't filter (e.g. "show me all candidates").

### Stage 3: Chunk ID Mapping
**File:** `app/services/retrieval_service.py`

- Builds `chunk_id → resume_id` + section name mapping from resume rows.
- Only chunks belonging to SQL-filtered `resume_id`s are searched.
- Symptom: Correct resumes in SQL result but 0 candidates returned → chunk_ids in Postgres don't match chunk_ids in `.npy` file.
  - This happens if a resume was re-ingested without deleting old vectors.

### Stage 4: Query Embedding
**File:** `app/services/embedding/embedder.py`

- Query text is embedded using the same model as the indexed resumes.
- Symptom: Poor similarity scores → embedding model mismatch.
  - Check: `embedding_model` in `config.yaml` must match what was used during ingest.
  - Each resume stores `embedding_model` in Postgres — compare against current config.

### Stage 5: Vector Search (KNN)
**File:** `app/services/retrieval/knn.py`

- Loads the full `.npy` matrix, extracts rows matching filtered chunk_ids, computes dot product.
- All vectors must be unit-normalized for dot product = cosine similarity.
- Symptom: All scores ≈ 0 → vectors are not normalized. Check embedder's normalize step.
- Symptom: Wrong top results → check that only the filtered chunk indices are searched (not the full matrix).

### Stage 6: Score Aggregation & Deduplication
**File:** `app/services/retrieval_service.py`

- Best score per `resume_id`, then best resume per `user_id` (one result per person).
- Symptom: Same person appears twice → user deduplication not working; check `user_id` assignment in Postgres.
- Symptom: Fewer than `k` results → fewer unique users matched; this is correct behavior.

---

## Quick Diagnostic Commands

```bash
# Count searchable chunks
psql "$POSTGRES_URL" -c "
  SELECT COUNT(*) as active_resumes,
         COUNT(objectives_chunk_id) as objectives,
         COUNT(skills_chunk_id) as skills
  FROM resumes WHERE is_active=TRUE;"

# Check vector store
python -c "
import numpy as np
v = np.load('vector_store/nex_vec.npy')
ids = np.load('vector_store/nex_vec_ids.npy', allow_pickle=True)
norms = np.linalg.norm(v, axis=1)
print(f'Vectors: {v.shape}, IDs: {len(ids)}')
print(f'Norm range: [{norms.min():.4f}, {norms.max():.4f}]')
"

# Test a retrieve call
curl -s -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "software engineer", "k": 3}' | python -m json.tool
```
