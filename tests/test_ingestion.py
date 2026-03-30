from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.services.ingestion import ingest_file, EMBEDDABLE_SECTIONS, UNIVERSAL_VECTOR_STORE
from app.models import IngestResponse
from app.services.llm_parser import ParsedResume
from tests.helpers import MockPostgresStore, MockVectorFileStore


FAKE_HASH = "abc123def456" * 4  # 48-char fake SHA-256


class FakeEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.5, 0.0] for t in texts]


FAKE_PARSED = ParsedResume(
    name="Jane Doe",
    email="jane@acme.com",
    phone="9123456789",
    location="Bangalore, India",
    objectives="Seeking a senior engineering role in a product company.",
    work_experience_years=3.0,
    work_experience_text="Worked at Acme Corp as software engineer 2021-2024.",
    projects="Built a distributed caching system using Redis and Python.",
    education="B.Tech Computer Science, NIT Trichy, 2020.",
    skills=["Python", "FastAPI", "PostgreSQL"],
    achievements="Won hackathon 2022. Led team of 5 engineers.",
)


@pytest.fixture
def deps(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        vector_store_path=tmp_path / "vs",
        embedding_model="models/gemini-mock",
        vector_size=3,
        output_dimensionality=3,
        postgres_url="postgresql://mock",
    )
    pg = MockPostgresStore()
    vfs = MockVectorFileStore()

    monkeypatch.setattr("app.services.ingestion.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.store.vector_file_store.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.ingestion.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr("app.services.ingestion.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.services.ingestion.VectorFileStore", lambda: vfs)
    monkeypatch.setattr(
        "app.services.ingestion.extract_pdf_pages",
        lambda _: [SimpleNamespace(text="Full resume text here.")],
    )
    monkeypatch.setattr("app.services.ingestion.parse_resume", lambda _: FAKE_PARSED)
    monkeypatch.setattr("app.services.ingestion._hash_file", lambda _: FAKE_HASH)
    monkeypatch.setattr("app.services.ingestion.chunk_section", lambda section, text: [text])

    return SimpleNamespace(pg=pg, vfs=vfs, cfg=cfg)


def test_ingest_new_resume_creates_sections_and_user(deps):
    resp = ingest_file(file_name="resume.pdf", file_path="/tmp/resume.pdf", embedding_model=None)

    assert isinstance(resp, IngestResponse)
    assert resp.source_filename == "resume.pdf"
    assert resp.name == "Jane Doe"
    assert "Python" in resp.skills
    assert resp.resume_id is not None
    assert resp.user_id is not None

    # With chunking, 2 sections (work_experience_text + projects) are embedded
    assert len(resp.sections_ingested) == len(EMBEDDABLE_SECTIONS)

    # 2 vectors in the flat store (one chunk per section with passthrough mock)
    _, ids = deps.vfs.read(UNIVERSAL_VECTOR_STORE)
    assert len(ids) == 2

    # Resume row written to postgres
    assert len(deps.pg._resumes) == 1
    assert len(deps.pg._users) == 1


def test_ingest_duplicate_hash_skips_processing(deps):
    first = ingest_file(file_name="resume.pdf", file_path="/tmp/resume.pdf", embedding_model=None)
    second = ingest_file(file_name="different_name.pdf", file_path="/tmp/resume.pdf", embedding_model=None)

    assert first.resume_id == second.resume_id
    # Still only 2 vectors — duplicate was skipped
    _, ids = deps.vfs.read(UNIVERSAL_VECTOR_STORE)
    assert len(ids) == 2


def test_ingest_different_content_creates_new_resume(deps, monkeypatch):
    first = ingest_file(file_name="resume.pdf", file_path="/tmp/a.pdf", embedding_model=None)

    monkeypatch.setattr("app.services.ingestion._hash_file", lambda _: "different_hash" * 3)
    second = ingest_file(file_name="resume2.pdf", file_path="/tmp/b.pdf", embedding_model=None)

    assert first.resume_id != second.resume_id
    # 4 vectors total — 2 chunks × 2 resumes, all in one flat file
    _, ids = deps.vfs.read(UNIVERSAL_VECTOR_STORE)
    assert len(ids) == 4


def test_ingest_same_person_reuses_user_id(deps, monkeypatch):
    first = ingest_file(file_name="resume.pdf", file_path="/tmp/a.pdf", embedding_model=None)

    monkeypatch.setattr("app.services.ingestion._hash_file", lambda _: "different_hash" * 3)
    second = ingest_file(file_name="resume2.pdf", file_path="/tmp/b.pdf", embedding_model=None)

    # Same email/phone → same user_id
    assert first.user_id == second.user_id
    assert len(deps.pg._users) == 1


def test_ingest_section_names_correct(deps):
    resp = ingest_file(file_name="resume.pdf", file_path="/tmp/resume.pdf", embedding_model=None)
    section_names = [s.section_name for s in resp.sections_ingested]
    for expected in EMBEDDABLE_SECTIONS:
        assert expected in section_names


def test_ingest_no_extractable_text_raises(deps, monkeypatch):
    monkeypatch.setattr("app.services.ingestion.extract_pdf_pages", lambda _: [])
    with pytest.raises(LookupError):
        ingest_file(file_name="empty.pdf", file_path="/tmp/empty.pdf", embedding_model=None)


def test_ingest_chunks_stored_in_postgres(deps):
    resp = ingest_file(file_name="resume.pdf", file_path="/tmp/resume.pdf", embedding_model=None)

    assert len(deps.pg._chunks) > 0
    for chunk in deps.pg._chunks.values():
        assert chunk["resume_id"] == resp.resume_id
        assert chunk["section"] in EMBEDDABLE_SECTIONS
        assert "chunk_text" in chunk
        assert "chunk_id" in chunk


def test_ingest_multi_chunk_section_creates_multiple_vectors(deps, monkeypatch):
    """Mock chunk_section to produce 3 chunks for work_experience and 2 for projects → 5 total."""
    def fake_chunk_section(section, text):
        if section == "work_experience_text":
            return [text + f" [job {i}]" for i in range(3)]
        if section == "projects":
            return [text + f" [project {i}]" for i in range(2)]
        return [text]

    monkeypatch.setattr("app.services.ingestion.chunk_section", fake_chunk_section)

    ingest_file(file_name="resume.pdf", file_path="/tmp/resume.pdf", embedding_model=None)

    _, ids = deps.vfs.read(UNIVERSAL_VECTOR_STORE)
    assert len(ids) == 5  # 3 work_experience + 2 projects
