from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from app.config import get_config


class VectorFileStore:
    """
    Four storage formats for each knowledge base:

      <kb_name>.npy               — float32 vectors (N, dims)         — 3.1 binary compressed numpy
      <kb_name>_ids.npy           — chunk_id strings (N,)
      <kb_name>.jsonl             — one JSON line per chunk            — 3.2 JSON lines
                                    {chunk_id, resume_id, vector}
      <kb_name>_normalized.json.gz — gzip JSON array of all chunks    — 3.3 normalized compressed JSON
                                    [{chunk_id, vector}, ...]
      <kb_name>_index.pkl         — dict chunk_id → np.ndarray        — 3.4 indexed hash map (O(1) lookup)
    """

    def __init__(self) -> None:
        self.config = get_config()

    def _ensure_path(self) -> Path:
        path = self.config.vector_store_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _vec_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}.npy"

    def _ids_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_ids.npy"

    def _jsonl_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}.jsonl"

    def _normalized_gz_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_normalized.json.gz"

    def _index_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_index.pkl"

    def _raw_vec_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_raw.npy"

    def _raw_ids_path(self, kb_name: str) -> Path:
        return self._ensure_path() / f"{kb_name}_raw_ids.npy"

    # ── 3.1: Binary NPY operations ────────────────────────────────────

    def read(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        """Return (vectors, chunk_ids). Empty arrays if KB does not exist."""
        vec_path = self._vec_path(kb_name)
        ids_path = self._ids_path(kb_name)
        if not vec_path.exists() or not ids_path.exists():
            return np.empty((0, self.config.vector_size), dtype=np.float32), []
        vectors = np.load(vec_path, mmap_mode="r")
        chunk_ids = np.load(ids_path, allow_pickle=False).tolist()
        return vectors, chunk_ids

    def append(
        self,
        kb_name: str,
        chunk_ids: list[str],
        vectors: np.ndarray,
        text_records: list[dict] | None = None,
    ) -> None:
        """
        Append new chunk_ids and their normalised vectors to all 4 storage formats.
        If text_records is provided, also append those lines to the .jsonl file.
        Each record in text_records must have at minimum: chunk_id, resume_id, vector.
        """
        existing_vectors, existing_ids = self.read(kb_name)
        new_vectors = (
            np.vstack([existing_vectors, vectors]).astype(np.float32)
            if existing_ids
            else vectors.astype(np.float32)
        )
        new_ids = existing_ids + chunk_ids
        self._write_npy(kb_name, new_ids, new_vectors)

        if text_records:
            self._append_jsonl(kb_name, text_records)

        # 3.3: normalized compressed JSON — incremental append
        self._append_normalized_gz(kb_name, chunk_ids, vectors)

        # 3.4: indexed hash map — incremental update
        self._update_index(kb_name, chunk_ids, vectors)

    def remove_chunk_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        """Remove specific chunk_ids from all 4 storage formats."""
        existing_vectors, existing_ids = self.read(kb_name)
        if not existing_ids:
            return
        mask = np.array([cid not in ids_to_remove for cid in existing_ids])
        kept_ids = [cid for cid, keep in zip(existing_ids, mask) if keep]
        kept_vectors = existing_vectors[mask]
        self._write_npy(kb_name, kept_ids, kept_vectors)
        self._remove_jsonl_ids(kb_name, ids_to_remove)
        self._remove_normalized_gz_ids(kb_name, ids_to_remove)
        self._remove_index_ids(kb_name, ids_to_remove)
        self._remove_raw_ids(kb_name, ids_to_remove)

    def list_kb_names(self) -> list[str]:
        path = self._ensure_path()
        return sorted(p.stem for p in path.glob("*.npy") if not p.stem.endswith("_ids"))

    def append_raw(
        self,
        kb_name: str,
        chunk_ids: list[str],
        raw_vectors: np.ndarray,
    ) -> None:
        """Append raw (unnormalized) vectors from the embedding model."""
        raw_path = self._raw_vec_path(kb_name)
        ids_path = self._raw_ids_path(kb_name)
        if raw_path.exists() and ids_path.exists():
            existing = np.load(raw_path, mmap_mode="r")
            existing_ids = np.load(ids_path, allow_pickle=False).tolist()
            combined = np.vstack([existing, raw_vectors]).astype(np.float32)
            combined_ids = existing_ids + chunk_ids
        else:
            combined = raw_vectors.astype(np.float32)
            combined_ids = chunk_ids
        np.save(raw_path, combined)
        np.save(ids_path, np.array(combined_ids, dtype=str))

    def read_raw(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        """Return (raw_vectors, chunk_ids) — unnormalized, as received from the embedding model."""
        raw_path = self._raw_vec_path(kb_name)
        ids_path = self._raw_ids_path(kb_name)
        if not raw_path.exists() or not ids_path.exists():
            return np.empty((0, self.config.vector_size), dtype=np.float32), []
        vectors = np.load(raw_path, mmap_mode="r")
        chunk_ids = np.load(ids_path, allow_pickle=False).tolist()
        return vectors, chunk_ids

    def delete_kb(self, kb_name: str) -> None:
        for p in (
            self._vec_path(kb_name),
            self._ids_path(kb_name),
            self._jsonl_path(kb_name),
            self._normalized_gz_path(kb_name),
            self._index_path(kb_name),
            self._raw_vec_path(kb_name),
            self._raw_ids_path(kb_name),
        ):
            if p.exists():
                p.unlink()

    # ── 3.2: JSONL operations ─────────────────────────────────────────

    def read_jsonl(self, kb_name: str) -> list[dict]:
        """Return all records from the .jsonl file."""
        path = self._jsonl_path(kb_name)
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ── 3.3: Normalized Compressed JSON operations ────────────────────

    def read_normalized_gz(self, kb_name: str) -> list[dict]:
        """Return all records from the normalized gzip JSON file.
        Each record: {chunk_id: str, vector: list[float]}
        """
        path = self._normalized_gz_path(kb_name)
        if not path.exists():
            return []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    # ── 3.4: Indexed Hash Map operations ─────────────────────────────

    def read_index(self, kb_name: str) -> dict[str, np.ndarray]:
        """Return the indexed dict mapping chunk_id → float32 vector."""
        path = self._index_path(kb_name)
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Startup migration ─────────────────────────────────────────────

    def sync_alternate_formats(self, kb_name: str) -> None:
        """Build 3.3 and 3.4 files from existing NPY data if they don't exist.
        Call once at startup to migrate existing vector stores.
        """
        gz_path = self._normalized_gz_path(kb_name)
        idx_path = self._index_path(kb_name)

        if gz_path.exists() and idx_path.exists():
            return

        vectors, chunk_ids = self.read(kb_name)
        if not chunk_ids:
            return

        if not gz_path.exists():
            records = [
                {"chunk_id": cid, "vector": vectors[i].tolist()}
                for i, cid in enumerate(chunk_ids)
            ]
            with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                json.dump(records, f)

        if not idx_path.exists():
            index = {cid: vectors[i].copy() for i, cid in enumerate(chunk_ids)}
            with open(idx_path, "wb") as f:
                pickle.dump(index, f)

    # ── Internal helpers ──────────────────────────────────────────────

    def _write_npy(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        np.save(self._vec_path(kb_name), vectors)
        np.save(self._ids_path(kb_name), np.array(chunk_ids, dtype=str))

    def _append_jsonl(self, kb_name: str, records: list[dict]) -> None:
        path = self._jsonl_path(kb_name)
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _remove_jsonl_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        path = self._jsonl_path(kb_name)
        if not path.exists():
            return
        kept_lines: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("chunk_id") not in ids_to_remove:
                        kept_lines.append(line)
                except json.JSONDecodeError:
                    kept_lines.append(line)
        with path.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

    def _append_normalized_gz(
        self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray
    ) -> None:
        gz_path = self._normalized_gz_path(kb_name)
        existing: list[dict] = []
        if gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                existing = json.load(f)
        new_records = [
            {"chunk_id": cid, "vector": vectors[i].tolist()}
            for i, cid in enumerate(chunk_ids)
        ]
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(existing + new_records, f)

    def _remove_normalized_gz_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        gz_path = self._normalized_gz_path(kb_name)
        if not gz_path.exists():
            return
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            records = json.load(f)
        kept = [r for r in records if r.get("chunk_id") not in ids_to_remove]
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(kept, f)

    def _update_index(
        self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray
    ) -> None:
        idx_path = self._index_path(kb_name)
        index: dict[str, np.ndarray] = {}
        if idx_path.exists():
            with open(idx_path, "rb") as f:
                index = pickle.load(f)
        for i, cid in enumerate(chunk_ids):
            index[cid] = vectors[i].copy()
        with open(idx_path, "wb") as f:
            pickle.dump(index, f)

    def _remove_index_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        idx_path = self._index_path(kb_name)
        if not idx_path.exists():
            return
        with open(idx_path, "rb") as f:
            index = pickle.load(f)
        for cid in ids_to_remove:
            index.pop(cid, None)
        with open(idx_path, "wb") as f:
            pickle.dump(index, f)

    def _remove_raw_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        raw_path = self._raw_vec_path(kb_name)
        ids_path = self._raw_ids_path(kb_name)
        if not raw_path.exists() or not ids_path.exists():
            return
        existing = np.load(raw_path, mmap_mode="r")
        existing_ids = np.load(ids_path, allow_pickle=False).tolist()
        mask = np.array([cid not in ids_to_remove for cid in existing_ids])
        kept_ids = [cid for cid, keep in zip(existing_ids, mask) if keep]
        kept_vectors = existing[mask]
        np.save(raw_path, kept_vectors.astype(np.float32))
        np.save(ids_path, np.array(kept_ids, dtype=str))
