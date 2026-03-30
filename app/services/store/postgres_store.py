from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from app.config import get_config

logger = logging.getLogger("nexvec.postgres_store")

# ---------------------------------------------------------------------------
# Module-level connection pool (shared across all PostgresStore instances)
# ---------------------------------------------------------------------------
_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool(dsn: str) -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pg_pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=dsn)
                logger.info("Postgres connection pool initialized (min=2, max=10)")
    return _pool


# ---------------------------------------------------------------------------
# DDL — users table
# ---------------------------------------------------------------------------
_USERS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    name       TEXT,
    email      TEXT,
    phone      TEXT,
    location   TEXT,
    created_at TEXT NOT NULL
);
"""

_USERS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);
"""

# ---------------------------------------------------------------------------
# DDL — resumes table
# ---------------------------------------------------------------------------
_RESUMES_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS resumes (
    resume_id                     TEXT    PRIMARY KEY,
    user_id                       TEXT    NOT NULL REFERENCES users(user_id),
    source_filename               TEXT    NOT NULL,
    file_hash                     TEXT    UNIQUE NOT NULL,
    objectives                    TEXT,
    work_experience_years         NUMERIC,
    work_experience_text          TEXT,
    projects                      TEXT,
    education                     TEXT,
    skills                        TEXT[],
    achievements                  TEXT,
    objectives_chunk_id           TEXT,
    work_experience_text_chunk_id TEXT,
    projects_chunk_id             TEXT,
    education_chunk_id            TEXT,
    skills_chunk_id               TEXT,
    achievements_chunk_id         TEXT,
    embedding_model               TEXT    NOT NULL,
    is_active                     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at                    TEXT    NOT NULL
);
"""

_RESUMES_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_resumes_user_id      ON resumes(user_id);
CREATE INDEX IF NOT EXISTS idx_resumes_hash         ON resumes(file_hash);
CREATE INDEX IF NOT EXISTS idx_resumes_active       ON resumes(is_active);
CREATE INDEX IF NOT EXISTS idx_resumes_source       ON resumes(source_filename);
CREATE INDEX IF NOT EXISTS idx_resumes_skills       ON resumes USING gin(skills);
CREATE INDEX IF NOT EXISTS idx_resumes_exp_years    ON resumes(work_experience_years);
CREATE INDEX IF NOT EXISTS idx_resumes_active_exp   ON resumes(is_active, work_experience_years);
"""

# ---------------------------------------------------------------------------
# DDL — resume_chunks table (one row per embedded chunk)
# ---------------------------------------------------------------------------
_CHUNKS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS resume_chunks (
    chunk_id    TEXT PRIMARY KEY,
    resume_id   TEXT NOT NULL,
    section     TEXT NOT NULL,
    chunk_index INT  NOT NULL DEFAULT 0,
    chunk_text  TEXT
);
"""

_CHUNKS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_chunks_resume_id ON resume_chunks(resume_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section   ON resume_chunks(section);
"""

# Trigram indexes for fast ILIKE '%keyword%' search on text columns.
# Requires pg_trgm extension (available on AWS RDS PostgreSQL by default).
_TRGM_INDEX_DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_resumes_work_exp_trgm  ON resumes USING gin(work_experience_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_resumes_projects_trgm  ON resumes USING gin(projects gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_resumes_objectives_trgm ON resumes USING gin(objectives gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_resumes_education_trgm ON resumes USING gin(education gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_resumes_achievements_trgm ON resumes USING gin(achievements gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_users_location_trgm    ON users USING gin(location gin_trgm_ops);
"""

# ---------------------------------------------------------------------------
# Migration — backfill and schema fixes
# ---------------------------------------------------------------------------
_MIGRATION_STEPS = [
    # Ensure all columns exist on users table
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS name     TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS email    TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone    TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS location TEXT",
    # Ensure all columns exist on resumes table
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS user_id                       TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS source_filename               TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS file_hash                     TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS objectives                    TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS work_experience_years         NUMERIC",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS work_experience_text          TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS projects                      TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS education                     TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS skills                        TEXT[]",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS achievements                  TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS objectives_chunk_id           TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS work_experience_text_chunk_id TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS projects_chunk_id             TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS education_chunk_id            TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS skills_chunk_id               TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS achievements_chunk_id         TEXT",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS embedding_model               TEXT",
    "UPDATE resumes SET embedding_model = 'unknown' WHERE embedding_model IS NULL",
    "ALTER TABLE resumes ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE",
    "UPDATE resumes SET is_active = TRUE WHERE is_active IS NULL",
    # Fix skills column type
    """
    DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'resumes'
            AND column_name = 'skills' AND data_type = 'text'
        ) THEN
            ALTER TABLE resumes DROP COLUMN skills;
            ALTER TABLE resumes ADD COLUMN skills TEXT[];
        END IF;
    END $$
    """,
    # Backfill resume_chunks from existing *_chunk_id columns on resumes table
    # (handles resumes ingested before the chunks table existed)
    """
    INSERT INTO resume_chunks (chunk_id, resume_id, section, chunk_index)
    SELECT work_experience_text_chunk_id, resume_id, 'work_experience_text', 0
    FROM resumes
    WHERE work_experience_text_chunk_id IS NOT NULL
    ON CONFLICT (chunk_id) DO NOTHING
    """,
    """
    INSERT INTO resume_chunks (chunk_id, resume_id, section, chunk_index)
    SELECT projects_chunk_id, resume_id, 'projects', 0
    FROM resumes
    WHERE projects_chunk_id IS NOT NULL
    ON CONFLICT (chunk_id) DO NOTHING
    """,
]


class PostgresStore:
    """Stores user profiles and resume sections in PostgreSQL."""

    def __init__(self) -> None:
        self.config = get_config()

    @contextmanager
    def _conn(self):
        """Borrow a connection from the pool; return it when done."""
        pool = _get_pool(self.config.postgres_url)
        conn = pool.getconn()
        try:
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            pool.putconn(conn)

    def ensure_table(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_USERS_TABLE_DDL)
                cur.execute(_RESUMES_TABLE_DDL)
                cur.execute(_CHUNKS_TABLE_DDL)
                for step in _MIGRATION_STEPS:
                    cur.execute(step)
                cur.execute(_USERS_INDEX_DDL)
                cur.execute(_RESUMES_INDEX_DDL)
                cur.execute(_CHUNKS_INDEX_DDL)
                cur.execute(_TRGM_INDEX_DDL)
            conn.commit()

    # ── Identity ──────────────────────────────────────────────────────

    def get_resume_id_by_hash(self, file_hash: str) -> str | None:
        sql = "SELECT resume_id FROM resumes WHERE file_hash = %s AND is_active = TRUE LIMIT 1"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (file_hash,))
                row = cur.fetchone()
                return row[0] if row else None

    def get_user_id_by_contact(self, email: str | None, phone: str | None) -> str | None:
        if not email and not phone:
            return None
        conditions, params = [], []
        if email:
            conditions.append("email = %s")
            params.append(email)
        if phone:
            conditions.append("phone = %s")
            params.append(phone)
        sql = f"SELECT user_id FROM users WHERE ({' OR '.join(conditions)}) LIMIT 1"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                return row[0] if row else None

    # ── User operations ───────────────────────────────────────────────

    def upsert_user(self, row: dict) -> None:
        sql = """
            INSERT INTO users (user_id, name, email, phone, location, created_at)
            VALUES (%(user_id)s, %(name)s, %(email)s, %(phone)s, %(location)s, %(created_at)s)
            ON CONFLICT (user_id) DO UPDATE SET
                name       = EXCLUDED.name,
                email      = COALESCE(EXCLUDED.email, users.email),
                phone      = COALESCE(EXCLUDED.phone, users.phone),
                location   = COALESCE(EXCLUDED.location, users.location)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, row)
            conn.commit()

    # ── Resume operations ─────────────────────────────────────────────

    def insert_resume(self, row: dict) -> None:
        sql = """
            INSERT INTO resumes (
                resume_id, user_id, source_filename, file_hash,
                objectives, work_experience_years, work_experience_text,
                projects, education, skills, achievements,
                objectives_chunk_id, work_experience_text_chunk_id,
                projects_chunk_id, education_chunk_id,
                skills_chunk_id, achievements_chunk_id,
                embedding_model, is_active, created_at
            ) VALUES (
                %(resume_id)s, %(user_id)s, %(source_filename)s, %(file_hash)s,
                %(objectives)s, %(work_experience_years)s, %(work_experience_text)s,
                %(projects)s, %(education)s, %(skills)s, %(achievements)s,
                %(objectives_chunk_id)s, %(work_experience_text_chunk_id)s,
                %(projects_chunk_id)s, %(education_chunk_id)s,
                %(skills_chunk_id)s, %(achievements_chunk_id)s,
                %(embedding_model)s, %(is_active)s, %(created_at)s
            )
            ON CONFLICT (file_hash) DO UPDATE SET
                resume_id                     = EXCLUDED.resume_id,
                user_id                       = EXCLUDED.user_id,
                source_filename               = EXCLUDED.source_filename,
                objectives                    = EXCLUDED.objectives,
                work_experience_years         = EXCLUDED.work_experience_years,
                work_experience_text          = EXCLUDED.work_experience_text,
                projects                      = EXCLUDED.projects,
                education                     = EXCLUDED.education,
                skills                        = EXCLUDED.skills,
                achievements                  = EXCLUDED.achievements,
                objectives_chunk_id           = EXCLUDED.objectives_chunk_id,
                work_experience_text_chunk_id = EXCLUDED.work_experience_text_chunk_id,
                projects_chunk_id             = EXCLUDED.projects_chunk_id,
                education_chunk_id            = EXCLUDED.education_chunk_id,
                skills_chunk_id               = EXCLUDED.skills_chunk_id,
                achievements_chunk_id         = EXCLUDED.achievements_chunk_id,
                embedding_model               = EXCLUDED.embedding_model,
                is_active                     = TRUE,
                created_at                    = EXCLUDED.created_at
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, row)
            conn.commit()

    def get_resume_by_id(self, resume_id: str) -> dict | None:
        sql = """
            SELECT r.resume_id, r.user_id, r.source_filename, r.file_hash,
                   r.objectives, r.work_experience_years, r.work_experience_text,
                   r.projects, r.education, r.skills, r.achievements,
                   r.objectives_chunk_id, r.work_experience_text_chunk_id,
                   r.projects_chunk_id, r.education_chunk_id,
                   r.skills_chunk_id, r.achievements_chunk_id,
                   r.embedding_model, r.created_at,
                   u.name, u.email, u.phone, u.location
            FROM resumes r
            JOIN users u ON u.user_id = r.user_id
            WHERE r.resume_id = %s
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (resume_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    def get_resumes_by_ids(self, resume_ids: list[str]) -> list[dict]:
        if not resume_ids:
            return []
        sql = """
            SELECT r.resume_id, r.user_id, r.source_filename,
                   r.objectives, r.work_experience_years, r.work_experience_text,
                   r.projects, r.education, r.skills, r.achievements,
                   r.objectives_chunk_id, r.work_experience_text_chunk_id,
                   r.projects_chunk_id, r.education_chunk_id,
                   r.skills_chunk_id, r.achievements_chunk_id,
                   u.name, u.email, u.phone, u.location
            FROM resumes r
            JOIN users u ON u.user_id = r.user_id
            WHERE r.resume_id = ANY(%s) AND r.is_active = TRUE
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (resume_ids,))
                return [dict(row) for row in cur.fetchall()]

    def get_all_active_resumes(self) -> list[dict]:
        sql = """
            SELECT r.resume_id, r.user_id, r.source_filename,
                   r.objectives, r.work_experience_years, r.work_experience_text,
                   r.projects, r.education, r.skills, r.achievements,
                   r.objectives_chunk_id, r.work_experience_text_chunk_id,
                   r.projects_chunk_id, r.education_chunk_id,
                   r.skills_chunk_id, r.achievements_chunk_id,
                   u.name, u.email, u.phone, u.location
            FROM resumes r
            JOIN users u ON u.user_id = r.user_id
            WHERE r.is_active = TRUE
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                return [dict(row) for row in cur.fetchall()]

    # ── Chunk operations ──────────────────────────────────────────────

    def insert_chunks(self, chunks: list[dict]) -> None:
        """
        Insert chunk records into resume_chunks.
        Each dict must have: chunk_id, resume_id, section, chunk_index, chunk_text.
        """
        if not chunks:
            return
        sql = """
            INSERT INTO resume_chunks (chunk_id, resume_id, section, chunk_index, chunk_text)
            VALUES (%(chunk_id)s, %(resume_id)s, %(section)s, %(chunk_index)s, %(chunk_text)s)
            ON CONFLICT (chunk_id) DO NOTHING
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, chunks)
            conn.commit()

    def get_chunks_for_resume_ids(self, resume_ids: list[str]) -> list[dict]:
        """Return all chunk records (chunk_id, resume_id, section, chunk_text) for a list of resume_ids."""
        if not resume_ids:
            return []
        sql = """
            SELECT chunk_id, resume_id, section, chunk_index, chunk_text
            FROM resume_chunks
            WHERE resume_id = ANY(%s)
            ORDER BY resume_id, section, chunk_index
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (resume_ids,))
                return [dict(row) for row in cur.fetchall()]

    def delete_chunks_for_resume(self, resume_id: str) -> list[str]:
        """Delete all chunks for a resume and return their chunk_ids (for vector store cleanup)."""
        sql = "DELETE FROM resume_chunks WHERE resume_id = %s RETURNING chunk_id"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (resume_id,))
                rows = cur.fetchall()
            conn.commit()
        return [row[0] for row in rows]

    # ── Text-to-SQL execution ─────────────────────────────────────────

    _DANGEROUS_SQL = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|EXECUTE|EXEC|COPY|VACUUM)\b",
        re.IGNORECASE,
    )

    def execute_sql_query(self, sql: str) -> list[str]:
        """
        Execute an LLM-generated SELECT query and return the first column of each row.

        Defence-in-depth:
          1. Must start with SELECT
          2. Must not contain any DML/DDL keywords
        """
        stripped = sql.strip()
        if not stripped.upper().startswith("SELECT"):
            logger.warning("Rejected non-SELECT SQL: %s", sql[:100])
            return []
        if self._DANGEROUS_SQL.search(stripped):
            logger.warning("Rejected SQL containing dangerous keywords: %s", sql[:200])
            return []
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(stripped)
                rows = cur.fetchall()
                return [str(row[0]) for row in rows if row[0]]

    # ── Document operations ───────────────────────────────────────────

    def get_active_resume_count(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM resumes WHERE is_active = TRUE")
                return cur.fetchone()[0]

    def list_documents(self) -> list[dict]:
        sql = """
            SELECT r.resume_id, r.source_filename, r.created_at,
                   u.name
            FROM resumes r
            JOIN users u ON u.user_id = r.user_id
            WHERE r.is_active = TRUE
            ORDER BY r.created_at DESC
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                return [dict(row) for row in cur.fetchall()]

    def get_document(self, source_filename: str) -> dict | None:
        sql = """
            SELECT r.resume_id, r.source_filename, r.created_at,
                   r.work_experience_years, r.skills,
                   u.name
            FROM resumes r
            JOIN users u ON u.user_id = r.user_id
            WHERE r.source_filename = %s AND r.is_active = TRUE
            LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (source_filename,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "resume_id": row["resume_id"],
                    "source_filename": row["source_filename"],
                    "uploaded_at": row["created_at"],
                    "name": row["name"],
                    "work_experience_years": row["work_experience_years"],
                    "skills": list(row["skills"] or []),
                }

    def delete_document(self, source_filename: str) -> list[tuple[str, str]]:
        """
        Deactivate the resume for this filename.
        Returns list of (section_name, chunk_id) for vector store cleanup.
        """
        sql = """
            UPDATE resumes SET is_active = FALSE
            WHERE source_filename = %s AND COALESCE(is_active, TRUE) = TRUE
            RETURNING
                resume_id,
                objectives_chunk_id,
                work_experience_text_chunk_id,
                projects_chunk_id,
                education_chunk_id,
                skills_chunk_id,
                achievements_chunk_id
        """
        section_names = [
            "objectives", "work_experience_text", "projects",
            "education", "skills", "achievements",
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source_filename,))
                rows = cur.fetchall()
            conn.commit()

        result: list[tuple[str, str]] = []
        for row in rows:
            resume_id = row[0]
            # Also clean up all fine-grained chunks for this resume
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM resume_chunks WHERE resume_id = %s RETURNING chunk_id",
                        (resume_id,),
                    )
                    extra_chunk_ids = [r[0] for r in cur.fetchall()]
                conn.commit()
            for cid in extra_chunk_ids:
                result.append(("chunk", cid))

            for section, chunk_id in zip(section_names, row[1:]):
                if chunk_id:
                    result.append((section, chunk_id))
        return result
