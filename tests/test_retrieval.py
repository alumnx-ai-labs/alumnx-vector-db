from __future__ import annotations

"""End-to-end tests for app.services.retrieval_service.retrieve_documents."""

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from app.models import RetrieveRequest
from app.services.llm_query import QueryClassification
from app.services.retrieval_service import retrieve_documents
from tests.helpers import MockPostgresStore, MockVectorFileStore


# ---------------------------------------------------------------------------
# Fake HNSW store
# ---------------------------------------------------------------------------

class FakeHNSW:
    def __init__(self, dims: int = 3) -> None:
        pass

    def build(self, vectors, ids):
        pass

    def search_filtered(self, query, allowed_ids, k):
        return [(cid, 0.85) for cid in allowed_ids[:k]]


# ---------------------------------------------------------------------------
# Fake embedder with call tracking
# ---------------------------------------------------------------------------

class TrackingEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model
        self.embed_texts_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls.append(texts)
        return [[0.5, 0.5, 0.707] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return [0.5, 0.5, 0.707]


# ---------------------------------------------------------------------------
# Fixture — wires up a full retrieval environment with one ingested resume
# ---------------------------------------------------------------------------

def _make_cfg():
    return SimpleNamespace(
        knn_k=5,
        embedding_model="models/gemini-mock",
        vector_size=3,
        output_dimensionality=3,
        postgres_url="postgresql://mock",
        default_retrieval_strategy={"algorithm": "knn", "distance_metric": "cosine"},
        min_page_text_length=1,
    )


def _seed_resume(pg: MockPostgresStore, vfs: MockVectorFileStore, *, user_id: str | None = None) -> dict:
    """Insert one active resume + 2 chunks into pg and vfs. Returns the resume row."""
    uid = user_id or str(uuid.uuid4())
    rid = str(uuid.uuid4())
    cid_work = str(uuid.uuid4())
    cid_proj = str(uuid.uuid4())

    pg.upsert_user({
        "user_id": uid,
        "name": "Test User",
        "email": f"{uid[:8]}@test.com",
        "phone": "9999999999",
        "location": "Bangalore, India",
        "created_at": "2024-01-01T00:00:00",
    })
    pg.insert_resume({
        "resume_id": rid,
        "user_id": uid,
        "source_filename": f"{rid[:8]}.pdf",
        "file_hash": rid,
        "objectives": "Engineer seeking roles.",
        "work_experience_years": 3.0,
        "work_experience_text": "Worked as Python engineer building ML systems.",
        "projects": "Built NLP chatbot using Python.",
        "education": "B.Tech CS",
        "skills": ["Python", "FastAPI"],
        "achievements": "Tech lead award.",
        "objectives_chunk_id": None,
        "work_experience_text_chunk_id": cid_work,
        "projects_chunk_id": cid_proj,
        "education_chunk_id": None,
        "skills_chunk_id": None,
        "achievements_chunk_id": None,
        "embedding_model": "models/gemini-mock",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00",
    })
    pg.insert_chunks([
        {
            "chunk_id": cid_work,
            "resume_id": rid,
            "section": "work_experience_text",
            "chunk_index": 0,
            "chunk_text": "Worked as Python engineer building ML systems.",
        },
        {
            "chunk_id": cid_proj,
            "resume_id": rid,
            "section": "projects",
            "chunk_index": 0,
            "chunk_text": "Built NLP chatbot using Python.",
        },
    ])

    vec = np.array([[0.5, 0.5, 0.707], [0.5, 0.5, 0.707]], dtype=np.float32)
    vfs.append("nex_vec", [cid_work, cid_proj], vec)

    return {"resume_id": rid, "user_id": uid, "cid_work": cid_work, "cid_proj": cid_proj}


@pytest.fixture()
def env(monkeypatch):
    cfg = _make_cfg()
    pg = MockPostgresStore()
    vfs = MockVectorFileStore()
    embedder = TrackingEmbedder("models/gemini-mock")
    fake_hnsw = FakeHNSW()

    monkeypatch.setattr("app.services.retrieval_service.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.retrieval_service.PostgresStore", lambda: pg)
    monkeypatch.setattr("app.services.retrieval_service.VectorFileStore", lambda: vfs)
    monkeypatch.setattr("app.services.retrieval_service.GeminiEmbedder", lambda model: embedder)
    monkeypatch.setattr("app.services.retrieval_service.get_hnsw_store", lambda dims: fake_hnsw)
    monkeypatch.setattr(
        "app.services.retrieval_service.classify_and_generate_sql",
        lambda q: QueryClassification(
            sql="SELECT resume_id FROM resumes WHERE is_active = TRUE",
            needs_vector=True,
            reason="Mock vector search",
            hypothetical_doc="Worked as Python engineer building ML systems.",
        ),
    )

    return SimpleNamespace(pg=pg, vfs=vfs, embedder=embedder, cfg=cfg, fake_hnsw=fake_hnsw)


# ---------------------------------------------------------------------------
# Test 1: HyDE hypothetical_doc → embed_texts is called, not embed_query
# ---------------------------------------------------------------------------

def test_retrieve_uses_hyde_hypothetical_doc_for_embedding(env):
    _seed_resume(env.pg, env.vfs)

    retrieve_documents(RetrieveRequest(query="Python ML engineer"))

    # HyDE path uses embed_texts([hypothetical_doc])
    assert len(env.embedder.embed_texts_calls) >= 1
    assert len(env.embedder.embed_query_calls) == 0


# ---------------------------------------------------------------------------
# Test 2: needs_vector=False → RDS-only, similarity_score is None
# ---------------------------------------------------------------------------

def test_retrieve_rds_only_skips_vector_path(env, monkeypatch):
    monkeypatch.setattr(
        "app.services.retrieval_service.classify_and_generate_sql",
        lambda q: QueryClassification(
            sql="SELECT resume_id FROM resumes WHERE is_active = TRUE",
            needs_vector=False,
            reason="RDS only.",
            hypothetical_doc=None,
        ),
    )
    _seed_resume(env.pg, env.vfs)

    resp = retrieve_documents(RetrieveRequest(query="Python developer 3 years"))

    assert resp.logs.routing_decision == "rds_only"
    for c in resp.candidates:
        assert c.similarity_score is None
        assert c.match_type == "rds"


# ---------------------------------------------------------------------------
# Test 3: needs_vector=True → candidates have non-None similarity_score
# ---------------------------------------------------------------------------

def test_retrieve_vector_path_sets_similarity_score(env):
    _seed_resume(env.pg, env.vfs)

    resp = retrieve_documents(RetrieveRequest(query="Python ML engineer"))

    assert len(resp.candidates) >= 1
    for c in resp.candidates:
        assert c.similarity_score is not None


# ---------------------------------------------------------------------------
# Test 4: Two resumes with same user_id → only one candidate returned
# ---------------------------------------------------------------------------

def test_retrieve_deduplicates_by_user_id(env):
    shared_uid = str(uuid.uuid4())
    _seed_resume(env.pg, env.vfs, user_id=shared_uid)
    # Patch hash to produce a different resume_id for the same user
    _seed_resume(env.pg, env.vfs, user_id=shared_uid)

    resp = retrieve_documents(RetrieveRequest(query="Python ML engineer"))

    user_ids = [c.user_id for c in resp.candidates]
    assert len(user_ids) == len(set(user_ids))


# ---------------------------------------------------------------------------
# Test 5: Empty query raises ValueError("EMPTY_QUERY")
# ---------------------------------------------------------------------------

def test_retrieve_empty_query_raises_value_error(env):
    with pytest.raises(ValueError, match="EMPTY_QUERY"):
        retrieve_documents(RetrieveRequest(query="   "))


# ---------------------------------------------------------------------------
# Test 6: matched_chunk_text comes from chunks table, not full section text
# ---------------------------------------------------------------------------

def test_retrieve_matched_chunk_text_from_chunks_table(env):
    seeded = _seed_resume(env.pg, env.vfs)

    # The chunk text stored in the chunks table
    expected_chunk_text = "Worked as Python engineer building ML systems."

    resp = retrieve_documents(RetrieveRequest(query="Python ML engineer"))

    assert len(resp.candidates) >= 1
    winning = resp.candidates[0]
    assert winning.matched_chunk_text == expected_chunk_text


# ---------------------------------------------------------------------------
# Test 7: k parameter limits candidates returned
# ---------------------------------------------------------------------------

def test_retrieve_k_respected(env):
    for _ in range(5):
        _seed_resume(env.pg, env.vfs)

    resp = retrieve_documents(RetrieveRequest(query="Python ML engineer", k=2))

    assert len(resp.candidates) <= 2


# ---------------------------------------------------------------------------
# Test 8: op_counts["hyde_used"] == 1 when hypothetical_doc present
# ---------------------------------------------------------------------------

def test_retrieve_logs_contain_hyde_flag(env):
    _seed_resume(env.pg, env.vfs)

    resp = retrieve_documents(RetrieveRequest(query="Python ML engineer"))

    assert resp.logs is not None
    assert resp.logs.op_counts.get("hyde_used") == 1
