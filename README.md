# NexVec

NexVec is a Phase 1 FastAPI service for ingesting PDF documents, chunking them, generating embeddings, storing them in JSONL files, and retrieving relevant chunks with KNN search.

## Requirements

- Python 3.12+
- `uv` for dependency management
- A valid `GOOGLE_API_KEY` for local runtime usage

## Project Layout

```text
nexvec/
  app/
  tests/
  config.yaml
  .env
  requirements.txt
  main.py
```

## Setup

1. Open a terminal in the `nexvec/` directory.
2. Create or activate your virtual environment.

On Windows PowerShell:

```powershell
.\env\Scripts\python.exe --version
```

If you need to create the virtual environment:

```powershell
python -m venv env
```

3. Install dependencies:

```bash
uv sync
```

## Configure `.env`

Create a file named `.env` in the `nexvec/` directory.

Example:

```env
GOOGLE_API_KEY=your_api_key_here
```

The app loads `.env` automatically on startup.

Important:
- Do not commit real secrets to git.
- The repository already ignores `.env`.

## Run the API

Start the server:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## Run Tests

Run the test suite from the `nexvec/` directory:

```bash
uv run pytest -q
```

The test suite uses dummy environment defaults in `tests/conftest.py`, so CI can run without live credentials.

## Endpoints

- `GET /chunking-strategies`
- `GET /retrieval-strategies`
- `GET /knowledgebases`
- `POST /ingest`
- `POST /retrieve`

## Notes

- The vector store is created lazily when ingest or retrieve first touches storage.
- `ann` is not supported in Phase 1 and returns HTTP 400.
- `excludevectors=true` can be sent to `/retrieve` to omit `embedding_vector` from the response.

## Project Objective

This project aims to build a vector database system where data is converted into embeddings and stored efficiently for similarity search.

## What We Are Trying to Achieve

- Convert raw data into chunks
- Generate embeddings for each chunk
- Store embeddings as vectors
- Separate metadata and store it in MySQL
- Enable efficient search using vector similarity

## Features

- Chunking of data
- Embedding generation
- Vector storage
- Metadata storage in MySQL
- Fast retrieval system

## My Contribution

- Updated README documentation
- Understood project workflow
- Tested project setup
