from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from tests.helpers import MockPostgresStore, MockVectorFileStore
from app.services.llm_parser import ParsedResume
from app.services.llm_query import QueryClassification


FAKE_HASH = "aabbcc112233" * 4

FAKE_PARSED = ParsedResume(
    name="John Smith",
    email="john@techcorp.com",
    phone="9876543210",
    location="Mumbai, India",
    objectives="Senior engineer seeking leadership role.",
    work_experience_years=5.0,
    work_experience_text="Senior Engineer at TechCorp 2019-2024.",
    projects="Built real-time analytics platform using Python and React.",
    education="B.Tech Computer Science, IIT Bombay, 2018.",
    skills=["Python", "React", "PostgreSQL"],
    achievements="Tech lead of the year 2022.",
)


class FakeEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


@pytest.fixture()
def client(monkeypatch):
    cfg = SimpleNamespace(
        knn_k=5,
        default_retrieval_strategy={"algorithm": "knn", "distance_metric": "cosine"},
        embedding_model="models/gemini-mock",
        output_dimensionality=3,
        vector_size=3,
        min_page_text_length=1,
        postgres_url="postgresql://mock",
    )

    pg = MockPostgresStore()
    vfs = MockVectorFileStore()

    monkeypatch.setattr("app.services.ingestion.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.retrieval_service.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.embedding.embedder.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.store.vector_file_store.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.ingestion.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr("app.services.retrieval_service.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr("app.services.ingestion.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.services.ingestion.VectorFileStore", lambda: vfs)
    monkeypatch.setattr("app.services.retrieval_service.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.services.retrieval_service.VectorFileStore", lambda: vfs)
    monkeypatch.setattr("app.routers.candidates.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.routers.documents.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.routers.documents.VectorFileStore", lambda: vfs)
    monkeypatch.setattr(
        "app.services.ingestion.extract_pdf_pages",
        lambda _: [SimpleNamespace(text="Full resume text content.")],
    )
    monkeypatch.setattr("app.services.ingestion.parse_resume", lambda _: FAKE_PARSED)
    monkeypatch.setattr("app.services.ingestion._hash_file", lambda _: FAKE_HASH)
    # Text-to-SQL: return a fixed classification (execution handled by mock pg)
    monkeypatch.setattr(
        "app.services.retrieval_service.classify_and_generate_sql",
        lambda _: QueryClassification(
            sql="SELECT resume_id FROM resumes WHERE is_active = TRUE",
            needs_vector=True,
            reason="Mock: always use vector search in tests.",
            hypothetical_doc="Worked as Python engineer building ML systems.",
        ),
    )
    monkeypatch.setattr(
        "app.services.ingestion.chunk_section",
        lambda section, text: [text],
    )

    class FakeHNSW:
        def __init__(self, dims): pass
        def build(self, vectors, ids): pass
        def search_filtered(self, query, allowed_ids, k):
            return [(cid, 0.85) for cid in allowed_ids[:k]]

    fake_hnsw = FakeHNSW(3)
    monkeypatch.setattr(
        "app.services.retrieval_service.get_hnsw_store",
        lambda dims: fake_hnsw,
    )

    yield TestClient(app)


def test_ingest_pdf_returns_resume_id_and_sections(client):
    resp = client.post(
        "/ingest",
        files={"file": ("john_resume.pdf", b"%PDF-1.4 fake content", "application/pdf")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "resume_id" in payload
    assert "user_id" in payload
    assert payload["source_filename"] == "john_resume.pdf"
    assert payload["name"] == "John Smith"
    assert "Python" in payload["skills"]
    # With chunking, work_experience_text AND projects are both embedded (>= 1 section)
    assert len(payload["sections_ingested"]) >= 1


def test_ingest_non_pdf_returns_400(client):
    resp = client.post(
        "/ingest",
        files={"file": ("doc.docx", b"fake docx", "application/vnd.openxmlformats")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "INVALID_FILE_TYPE"


def test_ingest_duplicate_hash_returns_existing_resume(client, monkeypatch):
    first = client.post(
        "/ingest",
        files={"file": ("resume.pdf", b"%PDF-1.4 content", "application/pdf")},
    )
    first_id = first.json()["resume_id"]

    second = client.post(
        "/ingest",
        files={"file": ("cv.pdf", b"%PDF-1.4 content", "application/pdf")},
    )
    assert second.json()["resume_id"] == first_id


def test_retrieve_returns_candidates_no_duplicates(client):
    client.post(
        "/ingest",
        files={"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    resp = client.post("/retrieve", json={"query": "Python engineer", "k": 5})
    assert resp.status_code == 200
    payload = resp.json()
    assert "candidates" in payload
    assert payload["k_used"] == 5

    # No duplicate resume_ids
    ids = [c["resume_id"] for c in payload["candidates"]]
    assert len(ids) == len(set(ids))


def test_retrieve_candidate_has_expected_fields(client):
    client.post(
        "/ingest",
        files={"file": ("resume.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    resp = client.post("/retrieve", json={"query": "engineer"})
    candidates = resp.json()["candidates"]
    if candidates:
        c = candidates[0]
        assert "resume_id" in c
        assert "user_id" in c
        assert "similarity_score" in c
        assert "matched_sections" in c
        assert isinstance(c["skills"], list)


def test_list_documents_returns_ingested_resume(client):
    client.post(
        "/ingest",
        files={"file": ("john.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    resp = client.get("/documents")
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert any(d["source_filename"] == "john.pdf" for d in docs)


def test_delete_document_removes_it(client):
    client.post(
        "/ingest",
        files={"file": ("todelete.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    resp = client.delete("/documents/todelete.pdf")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "todelete.pdf"


def test_empty_query_returns_400(client):
    resp = client.post("/retrieve", json={"query": "   "})
    assert resp.status_code == 400
    assert resp.json()["error"] == "EMPTY_QUERY"
