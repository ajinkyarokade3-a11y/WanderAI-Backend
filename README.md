# WanderAI-Backend

FastAPI backend for Traveller-App and Operator-Web. It owns authentication, API routes, business logic, AI integrations, external providers, SQLAlchemy models, Alembic migrations, and database seed data.

Operator authentication currently preserves the existing behavior but does not appear to enforce durable authorization on every backend request. This remains a future security task.

The `legacy-node/` directory is reference-only and is not part of normal startup.

## Validation

- `backend.main` imports successfully from this repository.
- `pytest tests -q` passes: 95 tests passed.
- The frontend repositories build and type-check independently.
- Operator authentication retains the existing behavior but does not yet enforce durable authorization on every backend request.

## Run

```text
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
