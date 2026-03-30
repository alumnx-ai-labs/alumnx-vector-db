Check the API health and display the current runtime configuration.

Steps:
1. Determine the API base URL: use `API_URL` env var or default to `http://localhost:8000`.
2. Hit the health endpoint:
   ```
   curl -s "$API_URL/health"
   ```
3. Report the server status (ok / error).
4. Read `config.yaml` from the project root and display the active configuration:
   - `embedding_model`
   - `output_dimensionality` / `vector_size`
   - `knn_k`
   - `vector_store_path`
   - `min_page_text_length`
5. Check that `GOOGLE_API_KEY` and `POSTGRES_URL` environment variables are set (without printing their values). Warn if either is missing.
6. If the server is unreachable, suggest:
   - `uv run uvicorn main:app --reload` for local dev
   - `docker compose up` for Docker setup
   - Check PM2 status on EC2 with `pm2 list`
