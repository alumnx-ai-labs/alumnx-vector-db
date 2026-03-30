Add a new service layer module to the NexVec application.

## Step-by-Step

### 1. Read existing service patterns first
- Read `app/services/ingestion.py` — shows how to orchestrate `PostgresStore`, `VectorFileStore`, and the embedder
- Read `app/services/retrieval_service.py` — shows the retrieval pattern
- Read `app/config.py` — for accessing runtime config (`get_config()`)

### 2. Create `app/services/<name>.py`

```python
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.config import get_config
from app.exceptions import SomeRelevantException

async def your_service_function(
    param: str,
    store: PostgresStore,
    vector_store: VectorFileStore,
) -> dict:
    """
    Brief description of what this does.

    Args:
        param: Description
        store: PostgreSQL store instance
        vector_store: Vector file store instance

    Returns:
        Description of return value

    Raises:
        SomeRelevantException: When X happens
    """
    cfg = get_config()
    # implementation
```

### 3. Wire into a router

In the relevant `app/routers/<name>.py`, call the service function. Do not put business logic in routers — routers handle HTTP, services handle logic.

### 4. Handle errors via `app/exceptions.py`

Check `app/exceptions.py` for existing custom exceptions before creating new ones.

## Checklist
- [ ] Service file created in `app/services/`
- [ ] Function is `async`
- [ ] Takes `PostgresStore` / `VectorFileStore` as parameters (not globals)
- [ ] Uses `get_config()` for runtime config
- [ ] Custom exceptions in `app/exceptions.py` (not inline `raise Exception`)
- [ ] Called from a router, not from another service (avoid deep call chains)
- [ ] Tests added using `MockPostgresStore`/`MockVectorFileStore` from `tests/helpers.py`
