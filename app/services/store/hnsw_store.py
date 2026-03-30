from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False

logger = logging.getLogger("nexvec.hnsw")

# For filtered subsets smaller than this, exact KNN is faster than HNSW post-filtering
_EXACT_KNN_THRESHOLD = 500

# HNSW build params — higher M/ef_construction = better recall, slower build
_HNSW_M = 32
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH = 100


class HNSWStore:
    """
    In-memory HNSW index over the universal vector store.

    - Built at startup from the existing .npy files.
    - Updated incrementally when new vectors are appended.
    - search_filtered() uses exact KNN for small subsets and HNSW for large/full scans.
    - Falls back to exact KNN on all operations if hnswlib is not installed.
    """

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self._lock = threading.Lock()
        self._index: hnswlib.Index | None = None  # type: ignore[name-defined]
        # Bidirectional maps: internal int label ↔ chunk_id string
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label: int = 0
        # Flat arrays kept for exact-KNN fallback and subset scoring
        self._vectors: np.ndarray = np.empty((0, dims), dtype=np.float32)
        self._ids: list[str] = []

    # ── Build / load ──────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        """Build the index from scratch given all existing vectors and their ids."""
        if not chunk_ids:
            logger.info("HNSW: no vectors to index")
            return

        with self._lock:
            self._ids = list(chunk_ids)
            self._vectors = np.asarray(vectors, dtype=np.float32)
            self._label_to_id = {i: cid for i, cid in enumerate(chunk_ids)}
            self._id_to_label = {cid: i for i, cid in enumerate(chunk_ids)}
            self._next_label = len(chunk_ids)

            if _HNSWLIB_AVAILABLE:
                idx = hnswlib.Index(space="ip", dim=self.dims)
                idx.init_index(
                    max_elements=max(len(chunk_ids) * 2, 1000),
                    M=_HNSW_M,
                    ef_construction=_HNSW_EF_CONSTRUCTION,
                    random_seed=42,
                )
                idx.set_ef(_HNSW_EF_SEARCH)
                labels = np.arange(len(chunk_ids), dtype=np.int32)
                idx.add_items(self._vectors, labels)
                self._index = idx
                logger.info("HNSW index built: %d vectors, dims=%d", len(chunk_ids), self.dims)
            else:
                logger.warning("hnswlib not installed — HNSW disabled, using exact KNN")

    def append(self, new_vectors: np.ndarray, new_ids: list[str]) -> None:
        """Add new vectors to both the flat store and the HNSW index."""
        if not new_ids:
            return
        with self._lock:
            new_arr = np.asarray(new_vectors, dtype=np.float32)
            start_label = self._next_label

            for i, cid in enumerate(new_ids):
                label = start_label + i
                self._label_to_id[label] = cid
                self._id_to_label[cid] = label

            self._next_label += len(new_ids)
            self._ids.extend(new_ids)
            self._vectors = (
                np.vstack([self._vectors, new_arr]) if len(self._ids) > len(new_ids)
                else new_arr.copy()
            )

            if _HNSWLIB_AVAILABLE and self._index is not None:
                # Resize if needed
                if self._next_label >= self._index.get_max_elements():
                    self._index.resize_index(self._next_label * 2)
                labels = np.arange(start_label, self._next_label, dtype=np.int32)
                self._index.add_items(new_arr, labels)

    # ── Search ────────────────────────────────────────────────────────

    def search_filtered(
        self,
        query_vector: np.ndarray,
        allowed_chunk_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """
        Return up to k (chunk_id, score) pairs from within allowed_chunk_ids,
        sorted by descending cosine similarity.

        Strategy:
          - Small subsets (< EXACT_KNN_THRESHOLD): exact matrix multiply — always fast.
          - Large subsets: HNSW over-fetch then filter to allowed set.
          - If hnswlib is unavailable: exact KNN always.
        """
        if not allowed_chunk_ids:
            return []

        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if len(allowed_chunk_ids) < _EXACT_KNN_THRESHOLD or not _HNSWLIB_AVAILABLE or self._index is None:
            return self._exact_knn(q, allowed_chunk_ids, k)

        return self._hnsw_filtered(q, allowed_chunk_ids, k)

    def _exact_knn(
        self,
        query: np.ndarray,
        allowed_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """Exact cosine KNN on a subset of vectors identified by allowed_ids."""
        with self._lock:
            id_to_label = self._id_to_label
            vectors = self._vectors

        positions = [id_to_label[cid] for cid in allowed_ids if cid in id_to_label]
        if not positions:
            return []

        subset = vectors[positions]
        scores: np.ndarray = subset @ query

        actual_k = min(k, len(positions))
        top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        with self._lock:
            label_to_id = self._label_to_id

        return [
            (allowed_ids[positions.index(positions[i])], float(scores[i]))
            for i in top_idx
        ]

    def _hnsw_filtered(
        self,
        query: np.ndarray,
        allowed_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """HNSW search with post-filtering to allowed_ids."""
        allowed_set = set(allowed_ids)
        # Over-fetch: retrieve more candidates to account for post-filter loss
        ef = min(len(allowed_ids), max(k * 8, 200))

        with self._lock:
            index = self._index
            label_to_id = self._label_to_id

        if index is None:
            return self._exact_knn(query, allowed_ids, k)

        index.set_ef(ef)
        labels, distances = index.knn_query(query.reshape(1, -1), k=ef)

        results: list[tuple[str, float]] = []
        for label, dist in zip(labels[0], distances[0]):
            cid = label_to_id.get(int(label))
            if cid and cid in allowed_set:
                # hnswlib inner-product space: distance = 1 - dot_product
                results.append((cid, float(1.0 - dist)))
                if len(results) == k:
                    break

        # If HNSW didn't yield enough results (aggressive filter), fall back to exact KNN
        if len(results) < k and len(results) < len(allowed_ids):
            logger.debug("HNSW post-filter yield too low (%d/%d) — falling back to exact KNN", len(results), k)
            return self._exact_knn(query, allowed_ids, k)

        return results


# ---------------------------------------------------------------------------
# Module-level singleton — built once at startup, reused across requests
# ---------------------------------------------------------------------------
_store_instance: HNSWStore | None = None
_store_lock = threading.Lock()


def get_hnsw_store(dims: int) -> HNSWStore:
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = HNSWStore(dims=dims)
    return _store_instance
