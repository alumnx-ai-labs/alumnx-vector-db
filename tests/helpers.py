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

    def read(self, kb_name: str) -> tuple[np.ndarray, list[str]]:
        if kb_name not in self._ids:
            return np.empty((0, 3), dtype=np.float32), []
        return self._vectors[kb_name].copy(), list(self._ids[kb_name])

    def read_index(self, kb_name: str) -> dict[str, np.ndarray]:
        if kb_name not in self._ids:
            return {}
        return {cid: self._vectors[kb_name][i].copy() for i, cid in enumerate(self._ids[kb_name])}

    def read_jsonl(self, kb_name: str) -> list[dict]:
        return []

    def read_normalized_gz(self, kb_name: str) -> list[dict]:
        return []

    def sync_alternate_formats(self, kb_name: str) -> None:
        return

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

    def append_raw(self, kb_name: str, chunk_ids: list[str], raw_vectors: np.ndarray) -> None:
        self.append(kb_name, chunk_ids, raw_vectors)

    def remove_chunk_ids(self, kb_name: str, ids_to_remove: set[str]) -> None:
        if kb_name not in self._ids:
            return
        old_ids = self._ids[kb_name]
        old_vecs = self._vectors[kb_name]
        mask = np.array([cid not in ids_to_remove for cid in old_ids])
        self._ids[kb_name] = [cid for cid, keep in zip(old_ids, mask) if keep]
        self._vectors[kb_name] = old_vecs[mask].astype(np.float32)

    def list_kb_names(self) -> list[str]:
        return sorted(self._ids.keys())

    def delete_kb(self, kb_name: str) -> None:
        self._vectors.pop(kb_name, None)
        self._ids.pop(kb_name, None)
