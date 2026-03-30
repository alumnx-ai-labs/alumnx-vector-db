# Stage 1: Build stage
FROM ghcr.io/astral-sh/uv:latest AS uv_setup
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv from the official image
COPY --from=uv_setup /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into a portable directory
RUN uv sync --frozen

# Copy the application code
COPY . .

# PATH setup to use the venv
ENV PATH="/app/.venv/bin:$PATH"

# Install NLTK punkt data in runtime
RUN uv run python -m nltk.downloader punkt punkt_tab

# Expose port (can be overridden by docker-compose)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
