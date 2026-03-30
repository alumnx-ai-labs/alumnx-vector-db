Run the project test suite and summarize the results.

Steps:
1. Check that `GOOGLE_API_KEY` is set in the environment. If not, warn the user — tests will be skipped automatically by `tests/conftest.py`.
2. Run the full test suite:
   ```
   uv run pytest -v 2>&1
   ```
3. Parse the output and display a summary:
   - Total tests: passed / failed / skipped / errors
   - List each failed test with: test name, file path, line number, and the assertion error message
   - List any errors (not assertion failures) separately
4. If all tests pass, confirm and note the total count.
5. If there are failures, for each one:
   - Identify which module is under test (API, ingestion, KNN, store)
   - Point to the relevant source file based on the test file name:
     - `test_api.py` → `app/routers/`
     - `test_ingestion.py` → `app/services/ingestion.py`
     - `test_knn.py` → `app/services/retrieval/knn.py`
     - `test_store.py` → `app/services/store/`
   - Suggest a likely fix based on the error message

To run a single test file: `uv run pytest tests/test_api.py -v`
To run a single test: `uv run pytest tests/test_api.py::test_function_name -v`
