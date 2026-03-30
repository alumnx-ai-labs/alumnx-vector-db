# Alumnx Vector DB - Local Setup & Usage Guide

This guide provides instructions for setting up and running the Alumnx Vector DB locally for development and testing.

## Prerequisites
- **Docker** and **Docker Compose** installed. (See [Docker Setup Guide](docker_setup.md))
- **Python 3.12+** (if running without Docker).
- **uv** (recommended for Python dependency management).

---

## 🚀 Running Locally with Docker (Recommended)

The easiest way to get started is using Docker Compose. This ensures all system dependencies (like NLTK and PDF libraries) are correctly configured.

### 1. Build and Start
Run the following command in the project root:
```bash
docker-compose up --build
```

### 2. Access the API
- **Swagger Documentation**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **ReDoc**: [http://localhost:8001/redoc](http://localhost:8001/redoc)

### 3. Persistent Storage
The Vector Store is persisted in the `./vector_store` directory on your host machine.

---

## 🐍 Running Locally with Python (Native)

If you prefer to run the application directly on your machine:

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment and Sync Dependencies
```bash
uv venv
uv sync
```

### 3. Start the Application
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/ingest/pdf` | Upload and process a PDF file into the vector store. |
| `GET` | `/chunking/strategies` | List available text chunking strategies. |
| `GET` | `/status` | Check the health status of the API. |

---

## 🛠 Deployment & CI/CD

The service is deployed on EC2 and managed via **PM2**.

- **Dev Endpoint**: [http://13.126.130.56:8001/docs](http://13.126.130.56:8001/docs)
- **Prod Endpoint**: [http://13.205.59.184:8012/docs](http://13.205.59.184:8012/docs)
- **CI Workflow**: Unit tests run in `.github/workflows/ci.yml`.
- **Dev Deploy**: `.github/workflows/deploy-dev.yml` runs after a successful CI run on `dev`.
- **Prod Deploy**: `.github/workflows/deploy-prod.yml` runs after a successful CI run on `main`.
- **Merge Protection**: GitHub branch protection on `main` must require `CI / Unit Tests` if production merges should be blocked until tests pass.
- **Manual Restart**: 
  ```bash
  pm2 restart alumnx-vector-db-prod
  ```
- **View Logs**:
  ```bash
  pm2 logs alumnx-vector-db-prod
  ```

---
---

## 📞 Support

For any issues, contact the Alumnx engineering team.
