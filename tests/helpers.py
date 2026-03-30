from __future__ import annotations

"""In-memory mock implementations of PostgresStore and VectorFileStore.
Used across the test suite — no real DB or filesystem required.
"""

import numpy as np


class MockPostgresStore:
    """In-memory PostgresStore matching the current interface."""

    def __init__(self) -> None:
        self._users: dict[str, dict] = {}    # user_id → row
        self._resumes: dict[str, dict] = {}  # resume_id → row

    def ensure_table(self) -> None:
        pass

    # ── Identity ──────────────────────────────────────────────────────

    def get_resume_id_by_hash(self, file_hash: str) -> str | None:
        for r in self._resumes.values():
            if r.get("file_hash") == file_hash:
                return r["resume_id"]
        return None

    def get_user_id_by_contact(self, email: str | None, phone: str | None) -> str | None:
        if not email and not phone:
            return None
        for u in self._users.values():
            if email and u.get("email") == email:
                return u["user_id"]
            if phone and u.get("phone") == phone:
                return u["user_id"]
        return None

    # ── User operations ───────────────────────────────────────────────

    def upsert_user(self, row: dict) -> None:
        uid = row["user_id"]
        if uid in self._users:
            existing = self._users[uid]
            existing["name"] = row.get("name") or existing.get("name")
            existing["email"] = row.get("email") or existing.get("email")
            existing["phone"] = row.get("phone") or existing.get("phone")
            existing["location"] = row.get("location") or existing.get("location")
        else:
            self._users[uid] = row.copy()

    # ── Resume operations ─────────────────────────────────────────────

    def insert_resume(self, row: dict) -> None:
        self._resumes[row["resume_id"]] = row.copy()

    def get_resume_by_id(self, resume_id: str) -> dict | None:
        r = self._resumes.get(resume_id)
        if not r:
            return None
        u = self._users.get(r.get("user_id", ""), {})
        return {**r, "name": u.get("name"), "email": u.get("email"),
                "phone": u.get("phone"), "location": u.get("location")}

    def get_resumes_by_ids(self, resume_ids: list[str]) -> list[dict]:
        result = []
        for rid in resume_ids:
            r = self._resumes.get(rid)
            if r and r.get("is_active"):
                u = self._users.get(r.get("user_id", ""), {})
                result.append({**r, "name": u.get("name"), "email": u.get("email"),
                                "phone": u.get("phone"), "location": u.get("location")})
        return result

    def get_all_active_resumes(self) -> list[dict]:
        result = []
        for r in self._resumes.values():
            if r.get("is_active"):
                u = self._users.get(r.get("user_id", ""), {})
                result.append({**r, "name": u.get("name"), "email": u.get("email"),
                                "phone": u.get("phone"), "location": u.get("location")})
        return result

    def get_active_resume_count(self) -> int:
        return sum(1 for r in self._resumes.values() if r.get("is_active"))

    # ── Text-to-SQL ───────────────────────────────────────────────────

    def execute_sql_query(self, sql: str) -> list[str]:
        """In tests, return all active resume_ids (simulates unfiltered SQL)."""
        return [r["resume_id"] for r in self._resumes.values() if r.get("is_active")]

    # ── Document operations ───────────────────────────────────────────

    def list_documents(self) -> list[dict]:
        result = []
        for r in self._resumes.values():
            if r.get("is_active"):
                u = self._users.get(r.get("user_id", ""), {})
                result.append({
                    "resume_id": r["resume_id"],
                    "source_filename": r["source_filename"],
                    "uploaded_at": r.get("created_at", ""),
                    "name": u.get("name"),
                })
        return result

    def get_document(self, source_filename: str) -> dict | None:
        for r in self._resumes.values():
            if r.get("source_filename") == source_filename and r.get("is_active"):
                u = self._users.get(r.get("user_id", ""), {})
                return {
                    "resume_id": r["resume_id"],
                    "source_filename": source_filename,
                    "uploaded_at": r.get("created_at", ""),
                    "name": u.get("name"),
                    "work_experience_years": r.get("work_experience_years"),
                    "skills": list(r.get("skills") or []),
                }
        return None

    def delete_document(self, source_filename: str) -> list[tuple[str, str]]:
        section_names = [
            "objectives", "work_experience_text", "projects",
            "education", "skills", "achievements",
        ]
        result: list[tuple[str, str]] = []
        for r in self._resumes.values():
            if r.get("source_filename") == source_filename and r.get("is_active"):
                r["is_active"] = False
                for s in section_names:
                    cid = r.get(f"{s}_chunk_id")
                    if cid:
                        result.append((s, cid))
        return result


class MockVectorFileStore:
    """In-memory VectorFileStore matching the current interface."""

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        self._ids: dict[str, list[str]] = {}
        self._raw_vectors: dict[str, np.ndarray] = {}
        self._raw_ids: dict[str, list[str]] = {}
        self._jsonl_records: dict[str, list[dict]] = {}

    def read(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        if kb_name not in self._ids:
            return np.empty((0, 3), dtype=np.float32), []
        return self._vectors[kb_name].copy(), list(self._ids[kb_name])

    def append(
        self,
        kb_name: str,
        chunk_ids: list[str],
        vectors: np.ndarray,
        text_records: list[dict] | None = None,
    ) -> None:
        if kb_name in self._ids:
            self._vectors[kb_name] = np.vstack([self._vectors[kb_name], vectors]).astype(np.float32)
            self._ids[kb_name].extend(chunk_ids)
        else:
            self._vectors[kb_name] = vectors.astype(np.float32)
            self._ids[kb_name] = list(chunk_ids)
        if text_records:
            self._jsonl_records.setdefault(kb_name, []).extend(record.copy() for record in text_records)

    def append_raw(self, kb_name: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if kb_name in self._raw_ids:
            self._raw_vectors[kb_name] = np.vstack([self._raw_vectors[kb_name], vectors]).astype(np.float32)
            self._raw_ids[kb_name].extend(chunk_ids)
        else:
            self._raw_vectors[kb_name] = vectors.astype(np.float32)
            self._raw_ids[kb_name] = list(chunk_ids)

    def read_raw(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        if kb_name not in self._raw_ids:
            return np.empty((0, 3), dtype=np.float32), []
        return self._raw_vectors[kb_name].copy(), list(self._raw_ids[kb_name])

    def read_jsonl(self, kb_name: str) -> list[dict]:
        return [record.copy() for record in self._jsonl_records.get(kb_name, [])]

    def read_normalized_gz(self, kb_name: str) -> list[dict]:
        vectors, ids = self.read(kb_name)
        return [
            {"chunk_id": chunk_id, "vector": vectors[idx].tolist()}
            for idx, chunk_id in enumerate(ids)
        ]

    def read_index(self, kb_name: str) -> dict[str, np.ndarray]:
        vectors, ids = self.read(kb_name)
        return {chunk_id: vectors[idx].copy() for idx, chunk_id in enumerate(ids)}

    def remove_chunk_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        if kb_name not in self._ids:
            return
        old_ids = self._ids[kb_name]
        old_vecs = self._vectors[kb_name]
        mask = np.array([cid not in ids_to_remove for cid in old_ids])
        self._ids[kb_name] = [cid for cid, keep in zip(old_ids, mask) if keep]
        self._vectors[kb_name] = old_vecs[mask].astype(np.float32)
        if kb_name in self._raw_ids:
            raw_ids = self._raw_ids[kb_name]
            raw_vecs = self._raw_vectors[kb_name]
            raw_mask = np.array([cid not in ids_to_remove for cid in raw_ids])
            self._raw_ids[kb_name] = [cid for cid, keep in zip(raw_ids, raw_mask) if keep]
            self._raw_vectors[kb_name] = raw_vecs[raw_mask].astype(np.float32)
        if kb_name in self._jsonl_records:
            self._jsonl_records[kb_name] = [
                record for record in self._jsonl_records[kb_name]
                if record.get("chunk_id") not in ids_to_remove
            ]

    def list_kb_names(self) -> list[str]:
        return sorted(self._ids.keys())

    def delete_kb(self, kb_name: str) -> None:
        self._vectors.pop(kb_name, None)
        self._ids.pop(kb_name, None)
        self._raw_vectors.pop(kb_name, None)
        self._raw_ids.pop(kb_name, None)
        self._jsonl_records.pop(kb_name, None)
