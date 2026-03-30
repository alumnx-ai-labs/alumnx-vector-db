Debug failures in the NexVec /ingest pipeline.

## Pipeline Stages & Files

```
PDF Upload → PDF Extraction → LLM Parse → User Resolution → Embedding → Postgres Insert → Vector Append
```

Work through each stage in order.

---

### Stage 1: PDF Extraction
**File:** `app/services/pdf_extractor.py`

- Is the file a valid PDF? Try opening with pdfplumber directly.
- Is `min_page_text_length` too high? Check `config.yaml` (default: 20 chars).
- Are pages being skipped? The extractor filters pages below the minimum length.
- Symptom: `"No text extracted"` or empty sections downstream.

### Stage 2: LLM Resume Parsing
**File:** `app/services/llm_parser.py`

- Is `GOOGLE_API_KEY` set? Check environment.
- Is the Gemini model available? Try `gemini-1.5-flash` as a fallback model.
- Is the prompt returning malformed JSON? Log the raw LLM response before parsing.
- Expected output: 7 section fields + `name`, `email`, `phone`, `location`.
- Symptom: `KeyError` on section field, or `None` for all sections.

### Stage 3: Duplicate Detection
**File:** `app/services/store/duplicate_checker.py`

- SHA-256 hash is computed on raw file bytes.
- If a duplicate is detected, ingest returns early with the existing `resume_id`.
- Symptom: Re-uploading the same file returns a 409 or the old resume_id.

### Stage 4: User Resolution
**File:** `app/services/store/postgres_store.py` → `find_user_by_email_or_phone()`

- Checks `users` table by email first, then phone (digits only).
- If no match: creates a new user with a UUID.
- Symptom: Duplicate users created — check if email/phone extraction is empty.

### Stage 5: Embedding Generation
**File:** `app/services/embedding/embedder.py`

- Embeds 6 sections in a batch: objectives, work_experience_text, projects, education, skills, achievements.
- Sections with empty/None text are skipped (no chunk_id assigned).
- Vectors are L2-normalized before storage.
- Symptom: Missing chunk_ids in response — section text was empty after LLM parse.

### Stage 6: PostgreSQL Insert
**File:** `app/services/store/postgres_store.py` → `insert_resume()`

- Check `POSTGRES_URL` env var is set correctly.
- Check `file_hash` uniqueness constraint — same file uploaded twice triggers this.
- Symptom: `psycopg2.errors.UniqueViolation` → duplicate hash.

### Stage 7: Vector File Append
**File:** `app/services/store/vector_file_store.py` → `append_vectors()`

- Appends to `<vector_store_path>/<kb_name>.npy` and `_ids.npy`.
- Creates files on first ingest if they don't exist.
- Symptom: `FileNotFoundError` → check `vector_store_path` in `config.yaml` exists and is writable.

---

## Quick Diagnostic Commands

```bash
# Check env vars
echo "GOOGLE_API_KEY set: $([ -n "$GOOGLE_API_KEY" ] && echo YES || echo NO)"
echo "POSTGRES_URL set: $([ -n "$POSTGRES_URL" ] && echo YES || echo NO)"

# Check vector store
ls -la vector_store/
python -c "import numpy as np; v=np.load('vector_store/nex_vec.npy'); print('Vectors:', v.shape)"

# Check postgres
psql "$POSTGRES_URL" -c "SELECT COUNT(*) FROM resumes WHERE is_active=TRUE;"
```
