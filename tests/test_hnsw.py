from __future__ import annotations

"""Tests for app.services.store.hnsw_store.HNSWStore and get_hnsw_store."""

import numpy as np
import pytest

from app.services.store.hnsw_store import HNSWStore, get_hnsw_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def _make_store(n: int, dims: int = 3) -> tuple[HNSWStore, np.ndarray, list[str]]:
    """Build an HNSWStore with n random unit vectors."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n, dims)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms
    ids = [f"chunk_{i}" for i in range(n)]
    store = HNSWStore(dims=dims)
    store.build(vectors, ids)
    return store, vectors, ids


# ---------------------------------------------------------------------------
# Test 1: Build and search — top result is the identical vector
# ---------------------------------------------------------------------------

def test_hnsw_build_and_search_exact_match():
    store, vectors, ids = _make_store(5)
    query = vectors[2].copy()  # identical to chunk_2
    results = store.search_filtered(query, ids, k=1)

    assert len(results) == 1
    assert results[0][0] == "chunk_2"
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 2: search_filtered respects allowed_ids
# ---------------------------------------------------------------------------

def test_hnsw_search_filtered_respects_allowed_ids():
    store, vectors, ids = _make_store(10)
    allowed = ["chunk_1", "chunk_3", "chunk_7"]
    query = vectors[7].copy()  # closest to chunk_7

    results = store.search_filtered(query, allowed, k=5)

    returned_ids = [r[0] for r in results]
    for rid in returned_ids:
        assert rid in allowed


# ---------------------------------------------------------------------------
# Test 3: Empty allowed_ids returns empty list
# ---------------------------------------------------------------------------

def test_hnsw_empty_allowed_returns_empty():
    store, vectors, ids = _make_store(5)
    results = store.search_filtered(vectors[0], [], k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Test 4: k limit respected
# ---------------------------------------------------------------------------

def test_hnsw_k_limit_respected():
    store, vectors, ids = _make_store(10)
    results = store.search_filtered(vectors[0], ids, k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# Test 5: Appended vectors are searchable
# ---------------------------------------------------------------------------

def test_hnsw_append_new_vectors_searchable():
    store, vectors, ids = _make_store(3)

    # Create 2 new unit vectors
    rng = np.random.default_rng(99)
    new_raw = rng.standard_normal((2, 3)).astype(np.float32)
    norms = np.linalg.norm(new_raw, axis=1, keepdims=True)
    new_vecs = new_raw / norms
    new_ids = ["appended_0", "appended_1"]

    store.append(new_vecs, new_ids)

    all_ids = ids + new_ids
    results = store.search_filtered(new_vecs[0], all_ids, k=5)
    returned_ids = [r[0] for r in results]

    assert "appended_0" in returned_ids


# ---------------------------------------------------------------------------
# Test 6: Identical unit vectors → score ≈ 1.0
# ---------------------------------------------------------------------------

def test_hnsw_scores_are_cosine_similarity():
    vec = _unit([1.0, 0.0, 0.0])
    store = HNSWStore(dims=3)
    store.build(np.array([vec]), ["only"])

    results = store.search_filtered(vec.copy(), ["only"], k=1)

    assert len(results) == 1
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 7: Subset < 500 uses exact KNN path (verify via correct results)
# ---------------------------------------------------------------------------

def test_hnsw_exact_knn_path_for_small_subset():
    store, vectors, ids = _make_store(10)
    # Only 3 allowed — well under EXACT_KNN_THRESHOLD (500) → exact KNN
    allowed = ["chunk_0", "chunk_4", "chunk_9"]
    query = vectors[4].copy()  # closest to chunk_4

    results = store.search_filtered(query, allowed, k=3)

    returned_ids = [r[0] for r in results]
    # chunk_4 should be the best match
    assert returned_ids[0] == "chunk_4"
    # All results are within the allowed set
    assert all(rid in allowed for rid in returned_ids)


# ---------------------------------------------------------------------------
# Test 8: get_hnsw_store singleton — same object returned on two calls
# ---------------------------------------------------------------------------

def test_hnsw_singleton_get_hnsw_store(monkeypatch):
    # Reset the module-level singleton so this test is self-contained
    import app.services.store.hnsw_store as hnsw_module
    original = hnsw_module._store_instance
    hnsw_module._store_instance = None

    try:
        store_a = get_hnsw_store(3)
        store_b = get_hnsw_store(3)
        assert store_a is store_b
    finally:
        hnsw_module._store_instance = original
