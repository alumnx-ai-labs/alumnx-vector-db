Upload a PDF resume to the /ingest endpoint and display the full response.

Steps:
1. Check if a file path argument was provided (e.g. `/ingest path/to/resume.pdf`). If not, look for `test.pdf` in the project root.
2. Determine the API base URL: use the `API_URL` environment variable if set, otherwise default to `http://localhost:8000`.
3. Run the upload using curl:
   ```
   curl -s -X POST "$API_URL/ingest" \
     -F "file=@<pdf_path>" | python -m json.tool
   ```
4. Parse and display the `IngestResponse` fields clearly:
   - `resume_id`
   - `user_id`
   - `ingested_sections` (list each section name + chunk_id)
   - `embedding_model`
5. If the response contains an `error` field, explain what went wrong and suggest a fix based on common failure points in `app/services/ingestion.py`.
