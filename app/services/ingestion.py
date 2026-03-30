from __future__ import annotations

import hashlib
import logging
import uuid

import numpy as np

from app.config import get_config
from app.models import IngestResponse, SectionResult
from app.services.chunking.job_entry_chunker import chunk_section
from app.services.embedding.embedder import GeminiEmbedder
from app.services.llm_parser import ParsedResume, parse_resume
from app.services.pdf_extractor import extract_pdf_pages
from app.services.store.postgres_store import PostgresStore
from app.services.store.vector_file_store import VectorFileStore
from app.utils import now_ist_iso

logger = logging.getLogger("nexvec.ingestion")

# Sections that are embedded into vectors — all others are SQL-only
EMBEDDABLE_SECTIONS = [
    "work_experience_text",
    "projects",
]

UNIVERSAL_VECTOR_STORE = "nex_vec"


def _hash_file(file_path: str) -> str:
    """SHA-256 of file bytes — content-based identity, filename-independent."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _normalise(arr: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(arr)
    return (arr / norm) if norm > 0 else arr


def _section_text(parsed: ParsedResume, section: str) -> str | None:
    """Return embeddable text for a section. Skills list is joined to a string."""
    if section == "skills":
        return ", ".join(parsed.skills) if parsed.skills else None
    return getattr(parsed, section, None)


def ingest_file(
    file_name: str,
    file_path: str,
    embedding_model: str | None,
) -> IngestResponse:
    """
    Ingest a PDF resume:

      1. SHA-256 content hash — skip if already ingested
      2. Extract text → LLM parses into 7 fixed sections + identity fields
      3. Resolve or create user (email/phone identity)
      4. Smart-chunk each embeddable section into job-entry/project-level chunks
      5. Embed all chunks in one batched Gemini call (unit-normalised)
      6. Persist chunks to resume_chunks + vectors to vector store
      7. Persist resume row with section texts + first chunk_ids to Postgres
    """
    config = get_config()
    pg = PostgresStore()
    vfs = VectorFileStore()
    active_model = embedding_model or config.embedding_model

    # ── Content-hash deduplication ────────────────────────────────────
    file_hash = _hash_file(file_path)
    existing_resume_id = pg.get_resume_id_by_hash(file_hash)

    if existing_resume_id:
        logger.info(
            "Duplicate content (hash=%s...), returning existing resume_id=%s",
            file_hash[:16], existing_resume_id,
        )
        resume = pg.get_resume_by_id(existing_resume_id)
        r = resume or {}
        sections_ingested = [
            SectionResult(section_name=s, chunk_id=r[f"{s}_chunk_id"])
            for s in EMBEDDABLE_SECTIONS
            if r.get(f"{s}_chunk_id")
        ]
        return IngestResponse(
            resume_id=existing_resume_id,
            user_id=r.get("user_id", ""),
            source_filename=r.get("source_filename", file_name),
            sections_ingested=sections_ingested,
            name=r.get("name"),
            skills=list(r.get("skills") or []),
            work_experience_years=r.get("work_experience_years"),
            embedding_model=active_model,
            ingested_at=r.get("created_at", now_ist_iso()),
        )

    # ── Extract text ──────────────────────────────────────────────────
    pages = extract_pdf_pages(file_path)
    if not pages:
        raise LookupError("NO_EXTRACTABLE_TEXT")
    full_text = "\n\n".join(p.text for p in pages if p.text.strip())
    logger.info("Extracted %d pages (%d chars) from %s", len(pages), len(full_text), file_name)

    # ── LLM parse → 7 fixed sections + identity ───────────────────────
    parsed = parse_resume(full_text)

    # ── Resolve or create user (email/phone identity) ──────────────────
    user_id = pg.get_user_id_by_contact(parsed.email, parsed.phone) or str(uuid.uuid4())
    created_at = now_ist_iso()

    pg.upsert_user({
        "user_id": user_id,
        "name": parsed.name,
        "email": parsed.email,
        "phone": parsed.phone,
        "location": parsed.location,
        "created_at": created_at,
    })
    logger.info("User: user_id=%s (%s)", user_id,
                "existing" if pg.get_user_id_by_contact(parsed.email, parsed.phone) else "new")

    # ── Smart-chunk each section → collect all chunks ─────────────────
    resume_id = str(uuid.uuid4())
    embedder = GeminiEmbedder(active_model)

    # all_chunks: list of (section, chunk_index, chunk_text)
    all_chunks: list[tuple[str, int, str]] = []
    for section in EMBEDDABLE_SECTIONS:
        text = _section_text(parsed, section)
        if not text:
            continue
        chunks = chunk_section(section, text)
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append((section, idx, chunk_text))

    # ── Embed all chunks in one batched call ──────────────────────────
    # chunk_ids: section → first chunk_id (for backward-compat columns on resumes table)
    chunk_ids: dict[str, str] = {}

    if all_chunks:
        texts = [ct for _, _, ct in all_chunks]
        raw_vecs = embedder.embed_texts(texts)
        cids = [str(uuid.uuid4()) for _ in all_chunks]

        normed_vecs = [_normalise(np.asarray(v, dtype=np.float32)) for v in raw_vecs]
        raw_arrs = [np.asarray(v, dtype=np.float32) for v in raw_vecs]

        vfs.append(
            UNIVERSAL_VECTOR_STORE,
            cids,
            np.stack(normed_vecs).astype(np.float32),
            text_records=[
                {"chunk_id": cid, "resume_id": resume_id, "vector": normed.tolist()}
                for cid, normed in zip(cids, normed_vecs)
            ],
        )
        vfs.append_raw(UNIVERSAL_VECTOR_STORE, cids, np.stack(raw_arrs))

        # Build section → first_chunk_id mapping and insert chunk rows
        chunk_rows: list[dict] = []
        for (section, idx, chunk_text), cid in zip(all_chunks, cids):
            if idx == 0:
                chunk_ids[section] = cid  # first chunk = backward-compat column
            chunk_rows.append({
                "chunk_id": cid,
                "resume_id": resume_id,
                "section": section,
                "chunk_index": idx,
                "chunk_text": chunk_text,
            })
        pg.insert_chunks(chunk_rows)

        logger.info(
            "Embedded %d chunks across %d sections for resume_id=%s",
            len(all_chunks), len(chunk_ids), resume_id,
        )

    # ── Persist resume row ────────────────────────────────────────────
    pg.insert_resume({
        "resume_id": resume_id,
        "user_id": user_id,
        "source_filename": file_name,
        "file_hash": file_hash,
        "objectives": parsed.objectives,
        "work_experience_years": parsed.work_experience_years,
        "work_experience_text": parsed.work_experience_text,
        "projects": parsed.projects,
        "education": parsed.education,
        "skills": parsed.skills or [],
        "achievements": parsed.achievements,
        "objectives_chunk_id": chunk_ids.get("objectives"),
        "work_experience_text_chunk_id": chunk_ids.get("work_experience_text"),
        "projects_chunk_id": chunk_ids.get("projects"),
        "education_chunk_id": chunk_ids.get("education"),
        "skills_chunk_id": chunk_ids.get("skills"),
        "achievements_chunk_id": chunk_ids.get("achievements"),
        "embedding_model": embedder.model,
        "is_active": True,
        "created_at": created_at,
    })

    logger.info(
        "Ingested file=%s resume_id=%s user_id=%s chunks=%d model=%s",
        file_name, resume_id, user_id, len(all_chunks), embedder.model,
    )

    sections_ingested = [
        SectionResult(section_name=s, chunk_id=chunk_ids[s])
        for s in EMBEDDABLE_SECTIONS
        if s in chunk_ids
    ]
    return IngestResponse(
        resume_id=resume_id,
        user_id=user_id,
        source_filename=file_name,
        sections_ingested=sections_ingested,
        name=parsed.name,
        skills=parsed.skills,
        work_experience_years=parsed.work_experience_years,
        embedding_model=embedder.model,
        ingested_at=created_at,
    )
