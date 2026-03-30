# NexVec

NexVec is a high-performance Phase 1 FastAPI service for document intelligence. It ingests PDF documents (resumes), extracts structured data and semantic chunks, and performs **Hybrid Retrieval** using PostgreSQL and Vector Similarity.

## 🚀 Quick Start (Docker)

The fastest way to get NexVec running locally:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/alumnx-vector-db.git
    cd alumnx-vector-db
    ```

2.  **Configure environment**:
    Copy `.env.example` to `.env` and add your **Google Gemini API Key**.
    ```bash
    cp .env.example .env
    ```

3.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```

NexVec will be available at:
-   **API**: [http://localhost:8000](http://localhost:8000)
-   **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Local Development Setup

If you prefer running without Docker:

### Prerequisites
-   Python 3.12+
-   [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
-   PostgreSQL 16+

### 1. Install Dependencies
Using **uv** (fastest):
```bash
uv sync
```
Or using **pip**:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up Database
Ensure you have a PostgreSQL database running and update your `.env` file:
```env
POSTGRES_URL=postgresql://your_user:your_password@localhost:5432/your_db
GOOGLE_API_KEY=your_gemini_api_key
```

### 3. Run the Service
```bash
uv run uvicorn main:app --reload --port 8000
```

---

## 🏗️ Architecture: Hybrid Retrieval

NexVec uses a two-stage retrieval pipeline:

1.  **Stage 1: SQL Filtering (PostgreSQL)**
    LLM (Gemini) analyzes the natural language query and generates optimized SQL to filter candidates by metadata (experience years, skills, location, etc.).
2.  **Stage 2: Vector Reranking (NumPy)**
    Semantic similarity is computed against chunks of the documents filtered in Stage 1 to find the most relevant context.

## 📁 Project Layout

```text
alumnx-vector-db/
├── app/                  # FastAPI Application
│   ├── routers/          # API Endpoints
│   ├── services/         # Core Logic (Ingestion, Retrieval, Parsing)
│   └── models.py         # Pydantic Schemas
├── vector_store/         # Metadata & Embeddings (Stored locally)
├── config.yaml           # Global Constants
├── .env                  # Secrets (Ignored by Git)
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container orchestration (App + DB)
├── main.py               # Entrypoint
└── requirements.txt      # Dependencies
```

## 🛠️ Key Features

-   **Deep Ingestion**: Chunking PDF resumes using advanced extraction.
-   **AI Parsing**: Uses Gemini to extract structured user profiles (experience, skills, contact).
-   **Structured Search**: Filter candidates by complex criteria via natural language.
-   **Semantic Search**: KNN similarity search on extracted document sections.
-   **Deduplication**: Automatically handles profile updates and file hash checking.

## 🧪 Running Tests

```bash
uv run pytest
```
*Note: Requires valid GOOGLE_API_KEY for tests involving LLM/Embeddings.*

## 🔀 PR Process

Before raising any PR, run the unit tests locally:

```bash
python -m pytest -q
```

- For `dev`: use PRs for active feature and integration work.
- For `main`: only raise PRs after the change is validated and ready for production.
- In both cases, wait for GitHub Actions to pass before merge.

Team workflow details are documented in [docs/pr-process.md](docs/pr-process.md).

---
*Developed for Document-Intelligence and Vector-Search excellence.*
