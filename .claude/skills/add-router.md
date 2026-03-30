Add a new FastAPI router to the NexVec API following project conventions.

## Step-by-Step

### 1. Read existing patterns first
- Read `app/routers/documents.py` — the cleanest example of the router pattern
- Read `app/models.py` — check for reusable request/response models before creating new ones
- Read `app/errors.py` — use `format_error()` for all error responses
- Read `app/main.py` — see how routers are registered

### 2. Create `app/routers/<name>.py`

```python
from fastapi import APIRouter, HTTPException
from app.models import YourRequestModel, YourResponseModel
from app.errors import format_error

router = APIRouter()

@router.get("/<name>", response_model=YourResponseModel)
async def your_endpoint():
    """Docstring describing the endpoint."""
    try:
        # business logic here
        pass
    except SomeException as e:
        raise HTTPException(status_code=400, detail=format_error("ERROR_CODE", str(e)))
```

### 3. Add models to `app/models.py` (only if not already covered)

Follow Pydantic v2 style. Use `model_config = ConfigDict(...)` if needed.

### 4. Register in `app/main.py`

```python
from app.routers.<name> import router as <name>_router
app.include_router(<name>_router)
```

### 5. Write a stub test in `tests/test_<name>.py`

Read `tests/conftest.py` first for available fixtures.

## Checklist
- [ ] Router file created in `app/routers/`
- [ ] New Pydantic models added to `app/models.py` (if needed)
- [ ] Router registered in `app/main.py`
- [ ] At least one test added in `tests/`
- [ ] Error responses use `format_error()` from `app/errors.py`
- [ ] No bare `except:` clauses
