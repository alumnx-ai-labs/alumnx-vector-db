from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeEmbedder:
    def __init__(self, model: str) -> None:
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        base = float(len(text))
        return [base, 1.0, 0.0]

    def embed_image(self, path: str, mime_type: str | None = None) -> dict:
        desc = f"Fake description of image at {path}"
        return {"description": desc, "embedding": self.embed_query(desc)}

    def embed_audio(self, path: str, mime_type: str | None = None) -> dict:
        desc = f"Fake description of audio at {path}"
        return {"description": desc, "embedding": self.embed_query(desc)}

    def embed_video(self, path: str, mime_type: str | None = None) -> dict:
        desc = f"Fake description of video at {path}"
        return {"description": desc, "embedding": self.embed_query(desc)}

    def embed_text_file(self, path: str) -> dict:
        desc = f"Fake content of text file at {path}"
        return {"description": desc, "embedding": self.embed_query(desc)}


def fake_config(tmp_path: Path):
    return SimpleNamespace(
        chunk_size=10,
        overlap_size=2,
        default_chunking_strategy="fixed_length",
        max_paragraph_size=20,
        knn_k=5,
        default_retrieval_strategy={"algorithm": "knn", "distance_metric": "cosine"},
        embedding_model="models/gemini-embedding-001",
        output_dimensionality=3,
        vector_size=3,
        vector_store_path=tmp_path / "vector_store",
        min_page_text_length=1,
    )


@pytest.fixture()
def client(monkeypatch):
    cfg = fake_config(Path("."))
    monkeypatch.setattr("app.routers.ingest.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.ingestion.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.retrieval_service.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.embedding.embedder.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.store.jsonl_store.get_config", lambda: cfg)
    monkeypatch.setattr("app.services.ingestion.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr("app.services.retrieval_service.GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr(
        "app.services.ingestion.extract_pdf_pages",
        lambda _: [
            SimpleNamespace(page_number=1, text="alpha beta gamma delta epsilon"),
            SimpleNamespace(page_number=2, text="zeta eta theta iota kappa"),
        ],
    )

    data_store: dict[str, list[dict]] = {}

    class DummyPath:
        def __init__(self, kb_name: str) -> None:
            self._kb_name = kb_name

        @property
        def stem(self) -> str:
            return self._kb_name

        def exists(self) -> bool:
            return self._kb_name in data_store

    def _kb_path(kb_name: str):
        return DummyPath(kb_name)

    def _list_kb_files():
        return [_kb_path(kb_name) for kb_name in sorted(data_store)]

    def _read_rows(kb_name: str):
        return [row.copy() for row in data_store.get(kb_name, [])]

    def _write_rows(kb_name: str, rows):
        data_store[kb_name] = _read_rows(kb_name) + [row.copy() for row in rows]

    def _update_rows(kb_name: str, rows):
        data_store[kb_name] = [row.copy() for row in rows]

    monkeypatch.setattr("app.services.store.jsonl_store.JSONLStore.ensure_store_path", lambda self: None)
    monkeypatch.setattr("app.services.store.jsonl_store.JSONLStore.kb_path", lambda self, kb_name: _kb_path(kb_name))
    monkeypatch.setattr("app.services.store.jsonl_store.JSONLStore.list_kb_files", lambda self: _list_kb_files())
    monkeypatch.setattr("app.services.store.jsonl_store.JSONLStore.read_rows", lambda self, kb_name: _read_rows(kb_name))
    monkeypatch.setattr("app.services.store.jsonl_store.JSONLStore.write_rows", lambda self, kb_name, rows: _write_rows(kb_name, rows))
    monkeypatch.setattr("app.services.store.jsonl_store.JSONLStore.update_rows", lambda self, kb_name, rows: _update_rows(kb_name, rows))

    test_client = TestClient(app)
    test_client.vector_store_path = cfg.vector_store_path
    yield test_client


def test_retrieval_strategies_lists_ann_as_unsupported(client):
    response = client.get("/retrieval-strategies")
    assert response.status_code == 200
    payload = response.json()
    assert any(item["name"] == "ann" for item in payload["algorithms"])


def test_ann_is_rejected_with_400_warning(client):
    response = client.post(
        "/retrieve",
        json={"query": "hello", "retrieval_strategy": {"algorithm": "ann", "distance_metric": "cosine"}},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "INVALID_RETRIEVAL_STRATEGY"
    assert "ann" in payload["message"].lower()


def test_ingest_creates_vector_store_and_retrieve_groups_results(client):
    ingest = client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Finance KB", "chunking_strategy": "fixed_length"},
    )
    assert ingest.status_code == 200
    payload = ingest.json()
    assert payload["kb_name"] == "finance_kb"
    assert client.vector_store_path.exists()

    retrieve = client.post("/retrieve", json={"query": "alpha", "kb_name": "finance_kb"})
    assert retrieve.status_code == 200
    result = retrieve.json()
    assert result["results"][0]["kb_name"] == "finance_kb"
    assert result["results"][0]["strategy_results"]


def test_ingest_uses_config_defaults_when_strategy_is_omitted(client):
    ingest = client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Defaults KB"},
    )
    assert ingest.status_code == 200
    payload = ingest.json()
    strategies = [item["strategy_name"] for item in payload["strategies_processed"]]
    assert strategies == ["fixed_length"]


def test_list_knowledgebases_returns_created_kb(client):
    ingest = client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Engineering KB", "chunking_strategy": "fixed_length"},
    )
    assert ingest.status_code == 200

    response = client.get("/knowledgebases")
    assert response.status_code == 200
    assert "engineering_kb" in response.json()["knowledgebases"]


def test_retrieve_can_exclude_embedding_vectors(client):
    ingest = client.post(
        "/ingest",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"kb_name": "Vectors KB", "chunking_strategy": "fixed_length"},
    )
    assert ingest.status_code == 200

    response = client.post(
        "/retrieve",
        json={
            "query": "alpha",
            "kb_name": "vectors_kb",
            "excludevectors": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    chunks = payload["results"][0]["strategy_results"][0]["chunks"]
    assert chunks
    assert "embedding_vector" not in chunks[0]
