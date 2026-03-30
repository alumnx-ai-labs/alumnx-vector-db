Scaffold a new FastAPI router for this project.

Usage: `/new-router <router-name>` (e.g. `/new-router analytics`)

Steps:
1. Take the router name from the argument. If not provided, ask the user for the name and what endpoints it should expose.
2. Read `app/routers/documents.py` to understand the existing router pattern (imports, APIRouter instantiation, response models, error handling via `app/errors.py`).
3. Read `app/models.py` to understand existing Pydantic models before creating new ones.
4. Read `app/main.py` to see how routers are registered.

Then create the following:

**`app/routers/<name>.py`**
- Import `APIRouter` from fastapi
- Import relevant models from `app/models`
- Import `format_error` from `app/errors`
- Define `router = APIRouter()`
- Add at least one stub endpoint with a docstring
- Include proper HTTP exception handling

**Update `app/main.py`**
- Import the new router: `from app.routers.<name> import router as <name>_router`
- Register it: `app.include_router(<name>_router)`

**`tests/test_<name>.py`**
- Import fixtures from `tests/conftest.py`
- Add one passing stub test for each new endpoint

5. Show a summary of all files created/modified.
