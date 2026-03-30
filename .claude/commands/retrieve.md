Run a semantic candidate search against the /retrieve endpoint.

Usage: `/retrieve <natural language query>` (e.g. `/retrieve senior Python engineer with Kubernetes experience`)

Steps:
1. Take everything after `/retrieve` as the query string. If no query is provided, ask the user for one.
2. Determine the API base URL: use `API_URL` env var or default to `http://localhost:8000`.
3. POST the query:
   ```
   curl -s -X POST "$API_URL/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"query": "<query>", "k": 5}' | python -m json.tool
   ```
4. Display the `RetrieveResponse` in a readable table:
   - Rank | Name | Email | Score | Matched Sections | Experience (years)
5. If zero candidates are returned, explain possible reasons:
   - No resumes ingested yet
   - SQL filter too restrictive (check `llm_query.py` output)
   - Vector store empty or mismatched embedding model
6. If an error occurs, point to the relevant file in `app/services/retrieval_service.py`.
